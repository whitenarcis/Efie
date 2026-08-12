"""
efi/dashboard/server.py

`DashboardServer` — сборка всего дашборда: маршруты API, отдача статики и
доступ по токену. Владелец жизненного цикла — `efi.app.EfiApp`, поэтому
интерфейс здесь такой же, как у остальных подсистем: `start()`/`stop()`.

Про доступ. По умолчанию дашборд слушает все интерфейсы — иначе основной
сценарий (Эфи в Termux на телефоне, дашборд смотрят с ноутбука) не работает
вовсе. Токен при этом не обязателен в домашней сети и обязателен, если
`host` — публично маршрутизируемый адрес (проверка в `DashboardSettings`).
Токен принимается заголовком, cookie или query-параметром — последнее нужно,
чтобы открыть дашборд по ссылке с другого устройства и остаться
авторизованным: cookie ставится сразу, и SSE-поток (EventSource не умеет
свои заголовки) дальше работает сам.
"""

from __future__ import annotations

import hmac
import logging
import socket
from pathlib import Path

import aiofiles
import aiofiles.os

from efi.config.schema import DashboardSettings
from efi.dashboard.api import build_routes
from efi.dashboard.http import AnyResponse, Handler, HttpServer, Request, Response
from efi.dashboard.snapshot import DashboardContext

logger = logging.getLogger(__name__)

_STATIC_DIR = Path(__file__).resolve().parent / "static"
_COOKIE_NAME = "efi_dashboard_token"

_CONTENT_TYPES = {
    ".html": "text/html; charset=utf-8",
    ".css": "text/css; charset=utf-8",
    ".js": "text/javascript; charset=utf-8",
    ".json": "application/json; charset=utf-8",
    ".svg": "image/svg+xml",
    ".ico": "image/x-icon",
    ".png": "image/png",
    ".woff2": "font/woff2",
}

_TOKEN_PAGE = """<!doctype html>
<html lang="ru"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>EFI · доступ</title>
<style>
  :root {{ color-scheme: light dark; }}
  body {{ margin:0; min-height:100vh; display:flex; align-items:center; justify-content:center;
         background:#f2f2f0; color:#1a1a1a;
         font:300 15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif; }}
  form {{ width:min(92vw, 380px); text-align:center; }}
  h1 {{ font-size:14px; font-weight:400; letter-spacing:.42em; text-transform:uppercase; margin:0 0 40px; }}
  p {{ color:#8a8a86; font-size:12px; letter-spacing:.16em; text-transform:uppercase; margin:0 0 18px; }}
  input {{ width:100%; padding:14px 0; border:0; border-bottom:1px solid #c9c9c5; background:transparent;
           font:300 16px/1 inherit; color:inherit; text-align:center; letter-spacing:.1em; outline:none; }}
  input:focus {{ border-bottom-color:#1a1a1a; }}
  button {{ margin-top:28px; padding:13px 34px; border:1px solid #1a1a1a; background:transparent; color:inherit;
            font:400 11px/1 inherit; letter-spacing:.28em; text-transform:uppercase; cursor:pointer; }}
  button:hover {{ background:#1a1a1a; color:#f2f2f0; }}
  @media (prefers-color-scheme: dark) {{
    body {{ background:#121211; color:#e8e8e4; }}
    input {{ border-bottom-color:#33332f; }}
    input:focus {{ border-bottom-color:#e8e8e4; }}
    button {{ border-color:#e8e8e4; }}
    button:hover {{ background:#e8e8e4; color:#121211; }}
  }}
</style>
</head><body>
<form method="get" action="/">
  <h1>EFI<span style="opacity:.45">.DASHBOARD</span></h1>
  <p>{message}</p>
  <input type="password" name="token" autofocus autocomplete="current-password" placeholder="токен доступа">
  <button type="submit">Войти</button>
</form>
</body></html>
"""


class DashboardServer:
    """HTTP-дашборд Эфи: `/` — интерфейс, `/api/*` — данные, `/static/*` — ресурсы страницы."""

    def __init__(self, context: DashboardContext, settings: DashboardSettings) -> None:
        self._context = context
        self._settings = settings
        self._routes: dict[str, Handler] = build_routes(context)
        self._http = HttpServer(settings.host, settings.port, self._handle)
        #: путь -> (mtime, содержимое). Файлы крошечные, но перечитывать их
        #: на каждый запрос незачем; проверка mtime оставляет возможность
        #: править вёрстку, не перезапуская Эфи.
        self._static_cache: dict[Path, tuple[float, bytes]] = {}

    @property
    def url(self) -> str:
        """Адрес для самой машины, где запущена Эфи."""
        host = self._settings.host
        display_host = "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host  # noqa: S104 — не bind, а показ ссылки
        return f"http://{display_host}:{self._http.port}/"

    @property
    def lan_url(self) -> str | None:
        """
        Адрес для ОСТАЛЬНЫХ устройств сети — то, что realistically и нужно
        открыть: Эфи крутится в Termux на телефоне, а смотрят на неё с
        ноутбука. Без этой строки в логе пользователю пришлось бы отдельно
        выяснять адрес телефона в сети.

        БЕЗ токена в адресе, хотя с ним было бы удобнее кликать прямо из
        терминала. Эта строка уходит в лог, а логи люди вставляют в issue,
        когда просят помощи, — и вместе с логом уезжал бы ключ от собственной
        переписки, дневника и профилей людей. Токен спрашивает сама страница:
        форма для этого уже есть, а куки живут неделю.
        """
        if self._settings.is_local_only:
            return None
        address = _primary_lan_address()
        if address is None:
            return None
        return f"http://{address}:{self._http.port}/"

    async def start(self) -> None:
        await self._http.start()
        logger.info("dashboard: listening on %s (%s)", self._http.sockets_description, self.url)

        lan_url = self.lan_url
        if lan_url is not None:
            logger.info("dashboard: с других устройств этой сети — %s", lan_url)
            if self._settings.token is not None:
                logger.info("dashboard: токен спросит сама страница (он в dashboard.token)")
        if self._settings.token is None and not self._settings.is_local_only:
            logger.warning(
                "dashboard: токен не задан — дашборд открыт любому устройству вашей сети. "
                "Если сеть не только ваша, задайте EFI_DASHBOARD__TOKEN"
            )

    async def stop(self) -> None:
        await self._http.stop()
        logger.info("dashboard: stopped")

    # -- обработка запроса --------------------------------------------------

    async def _handle(self, request: Request) -> AnyResponse:
        if request.method not in {"GET", "HEAD"}:
            return Response.error(405, "дашборд доступен только на чтение")

        authorized, token_from_query = self._check_token(request)
        if not authorized:
            return self._unauthorized(request)

        response = await self._route(request)
        if token_from_query and isinstance(response, Response):
            # Ставим cookie ровно один раз — на том запросе, где токен пришёл
            # ссылкой. Дальше страница и её EventSource ходят уже с cookie.
            response.headers["Set-Cookie"] = (
                f"{_COOKIE_NAME}={token_from_query}; Path=/; SameSite=Strict; Max-Age=604800; HttpOnly"
            )
        return response

    async def _route(self, request: Request) -> AnyResponse:
        handler = self._routes.get(request.path)
        if handler is not None:
            return await handler(request)

        if request.path in {"/", "/index.html"}:
            return await self._static_response(_STATIC_DIR / "index.html")

        if request.path.startswith("/static/"):
            name = request.path[len("/static/") :]
            path = self._resolve_static(name)
            if path is None:
                return Response.error(404, "файл не найден")
            return await self._static_response(path)

        if request.path == "/favicon.ico":
            # Отдельного .ico нет и не нужно: браузер запрашивает его сам,
            # и 404 в логах на каждой вкладке только зашумлял бы ленту.
            return Response(status=204)

        if request.path.startswith("/api/"):
            return Response.error(404, "неизвестный маршрут API")

        # Клиентская маршрутизация страницы — по хэшу, поэтому любой другой
        # путь это опечатка, а не «глубокая ссылка», которую надо отдать SPA.
        return Response.error(404, "страница не найдена")

    # -- доступ ------------------------------------------------------------

    def _check_token(self, request: Request) -> tuple[bool, str | None]:
        """Возвращает (авторизован, токен-из-query-если-он-и-подошёл)."""
        expected = self._settings.token.get_secret_value() if self._settings.token is not None else ""
        if not expected:
            return True, None

        from_query = request.param("token")
        if from_query and hmac.compare_digest(from_query, expected):
            return True, from_query
        header = request.headers.get("x-efi-token", "")
        if header and hmac.compare_digest(header, expected):
            return True, None
        cookie = request.cookie(_COOKIE_NAME) or ""
        if cookie and hmac.compare_digest(cookie, expected):
            return True, None
        return False, None

    def _unauthorized(self, request: Request) -> Response:
        if request.accepts_html and not request.path.startswith("/api/"):
            message = "нужен токен доступа" if not request.param("token") else "токен не подошёл"
            return Response.html(_TOKEN_PAGE.format(message=message), status=401)
        return Response.error(401, "нужен токен доступа")

    # -- статика -----------------------------------------------------------

    def _resolve_static(self, name: str) -> Path | None:
        """
        Путь к файлу статики или None, если имя выглядит подозрительно.

        Проверяется и само имя (без разделителей пути), и итоговый
        разрешённый путь: одной проверки имени мало против символических
        ссылок внутри каталога статики.
        """
        if not name or "/" in name or "\\" in name or name.startswith("."):
            return None
        candidate = (_STATIC_DIR / name).resolve()
        if candidate.parent != _STATIC_DIR.resolve():
            return None
        return candidate

    async def _static_response(self, path: Path) -> Response:
        try:
            stat = await aiofiles.os.stat(path)
        except OSError:
            return Response.error(404, "файл не найден")

        cached = self._static_cache.get(path)
        if cached is not None and cached[0] == stat.st_mtime:
            content = cached[1]
        else:
            async with aiofiles.open(path, mode="rb") as handle:
                content = await handle.read()
            self._static_cache[path] = (stat.st_mtime, content)

        return Response(
            body=content,
            content_type=_CONTENT_TYPES.get(path.suffix, "application/octet-stream"),
        )


def _primary_lan_address() -> str | None:
    """
    Адрес этой машины в локальной сети.

    Через UDP-сокет, а не через `socket.gethostbyname(gethostname())`:
    последний на Android/Termux и на большинстве Linux-систем отдаёт
    127.0.0.1 и толку от него нет. UDP-`connect` пакетов не отправляет — он
    только заставляет ядро выбрать исходящий интерфейс, чей адрес нам и
    нужен; адрес назначения при этом недостижим и не важен.
    """
    for probe in ("192.168.255.255", "10.255.255.255", "8.8.8.8"):
        probe_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            probe_socket.connect((probe, 9))
            address = str(probe_socket.getsockname()[0])
        except OSError:
            continue
        finally:
            probe_socket.close()
        if address and not address.startswith("127."):
            return address
    return None


__all__ = ["DashboardServer"]
