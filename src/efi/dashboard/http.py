"""
efi/dashboard/http.py

Минимальный асинхронный HTTP/1.1-сервер поверх `asyncio.start_server`.

Зачем свой, а не веб-фреймворк — см. докстринг пакета: Эфи должна
запускаться в том числе в Termux, а дашборду нужны ровно три вещи — GET с
query-параметрами, отдача нескольких статических файлов и один поток
Server-Sent Events. Это укладывается в один модуль, зато не добавляет ни
одной зависимости в проект, который сознательно держит их список коротким.

Что поддерживается: GET/HEAD, keep-alive, `Content-Length`, SSE-ответы
потоком, ограничения на размер заголовков и тела, таймауты чтения. Чего нет
сознательно: chunked-запросы, upgrade/WebSocket, сжатие, HTTPS. Сервер
слушает петлевой интерфейс (см. `DashboardSettings`), а не публичную сеть.

Потоковый ответ (`StreamResponse`) всегда закрывает соединение по
завершении: без `Content-Length` длину ответа иначе не обозначить, а
городить chunked-кодирование ради одного SSE-эндпоинта не стоит —
`Connection: close` для потока событий полностью корректен и понятен любому
браузеру.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from email.utils import formatdate
from http.cookies import SimpleCookie
from typing import Any
from urllib.parse import parse_qs, unquote, urlsplit

logger = logging.getLogger(__name__)

#: Потолок на строку запроса вместе со всеми заголовками.
_MAX_HEADER_BYTES = 32 * 1024
#: Потолок на тело запроса. Дашборд читающий, тело приходит разве что пустое.
_MAX_BODY_BYTES = 256 * 1024
#: Сколько ждать заголовки на уже установленном соединении (в т.ч. keep-alive).
_READ_TIMEOUT_SECONDS = 30.0

_STATUS_TEXT = {
    200: "OK",
    204: "No Content",
    302: "Found",
    400: "Bad Request",
    401: "Unauthorized",
    404: "Not Found",
    405: "Method Not Allowed",
    408: "Request Timeout",
    413: "Payload Too Large",
    431: "Request Header Fields Too Large",
    500: "Internal Server Error",
    503: "Service Unavailable",
}


class BadRequestError(Exception):
    """Запрос не удалось разобрать — соединение закрывается с коротким ответом."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


@dataclass(slots=True, frozen=True)
class Request:
    """Разобранный HTTP-запрос. Заголовки — в нижнем регистре, как принято для регистронезависимых имён."""

    method: str
    path: str
    query: dict[str, list[str]]
    headers: dict[str, str]
    body: bytes = b""

    def param(self, name: str, default: str = "") -> str:
        values = self.query.get(name)
        return values[0] if values else default

    def int_param(self, name: str, default: int, *, minimum: int = 0, maximum: int | None = None) -> int:
        """
        Числовой query-параметр с зажимом в допустимый диапазон.

        Мусор (`limit=abc`) не считается ошибкой запроса и не ломает страницу:
        берётся значение по умолчанию — для read-only дашборда это удобнее,
        чем 400 в ответ на криво собранную ссылку.
        """
        raw = self.param(name)
        try:
            value = int(raw)
        except ValueError:
            return default
        value = max(minimum, value)
        if maximum is not None:
            value = min(maximum, value)
        return value

    def cookie(self, name: str) -> str | None:
        raw = self.headers.get("cookie")
        if not raw:
            return None
        jar = SimpleCookie()
        try:
            jar.load(raw)
        except Exception:
            return None
        morsel = jar.get(name)
        return morsel.value if morsel is not None else None

    @property
    def accepts_html(self) -> bool:
        """Пришёл ли запрос из адресной строки браузера, а не из fetch() дашборда."""
        return "text/html" in self.headers.get("accept", "")


@dataclass(slots=True)
class Response:
    """Обычный ответ целиком в памяти."""

    status: int = 200
    body: bytes = b""
    content_type: str = "text/plain; charset=utf-8"
    headers: dict[str, str] = field(default_factory=dict)

    @classmethod
    def json(cls, payload: Any, *, status: int = 200) -> Response:
        raw = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
        return cls(status=status, body=raw, content_type="application/json; charset=utf-8")

    @classmethod
    def text(cls, message: str, *, status: int = 200) -> Response:
        return cls(status=status, body=message.encode("utf-8"))

    @classmethod
    def html(cls, markup: str, *, status: int = 200) -> Response:
        return cls(status=status, body=markup.encode("utf-8"), content_type="text/html; charset=utf-8")

    @classmethod
    def error(cls, status: int, message: str = "") -> Response:
        return cls.json({"error": message or _STATUS_TEXT.get(status, "error"), "status": status}, status=status)

    @classmethod
    def redirect(cls, location: str) -> Response:
        return cls(status=302, headers={"Location": location})


@dataclass(slots=True)
class StreamResponse:
    """
    Потоковый ответ (SSE). `stream` отдаёт уже готовые к отправке байты;
    сервер не добавляет к ним ни разметки, ни длины.
    """

    stream: AsyncIterator[bytes]
    status: int = 200
    content_type: str = "text/event-stream; charset=utf-8"
    headers: dict[str, str] = field(default_factory=dict)


AnyResponse = Response | StreamResponse
Handler = Callable[[Request], Awaitable[AnyResponse]]


class HttpServer:
    """
    Обёртка над `asyncio.start_server`: разбирает запрос, зовёт единственный
    `handler` (маршрутизацию делает вызывающая сторона — см. `server.py`) и
    сериализует ответ.
    """

    def __init__(self, host: str, port: int, handler: Handler) -> None:
        self._host = host
        self._port = port
        self._handler = handler
        self._server: asyncio.Server | None = None
        #: Задачи обслуживания открытых соединений. `Server.close()` закрывает
        #: только слушающий сокет и об уже принятых соединениях ничего не
        #: знает, поэтому их приходится вести самим — см. `stop()`.
        self._connections: set[asyncio.Task[None]] = set()

    @property
    def sockets_description(self) -> str:
        """Человекочитаемые адреса, на которых сервер реально слушает (порт 0 → выбранный ядром)."""
        if self._server is None or not self._server.sockets:
            return f"{self._host}:{self._port}"
        parts = []
        for sock in self._server.sockets:
            address = sock.getsockname()
            if isinstance(address, tuple) and len(address) >= 2:
                parts.append(f"{address[0]}:{address[1]}")
        return ", ".join(parts) or f"{self._host}:{self._port}"

    @property
    def port(self) -> int:
        """Фактический порт (важен при `port = 0`, когда его выбирает ядро — например, в тестах)."""
        if self._server is not None and self._server.sockets:
            address = self._server.sockets[0].getsockname()
            if isinstance(address, tuple) and len(address) >= 2:
                return int(address[1])
        return self._port

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._serve_connection, self._host, self._port)

    async def stop(self) -> None:
        """
        Закрывает слушающий сокет и снимает все открытые соединения.

        Снимать обязательно: SSE-поток живёт до тех пор, пока клиент его
        читает, а обычный запрос может в этот момент держать соединение с
        SQLite. Брошенная (а не отменённая) задача обслуживания уносит это
        соединение с собой — aiosqlite держит под каждое из них
        НЕ-daemon-поток, и такой поток потом не даёт процессу завершиться.
        """
        if self._server is None:
            return
        self._server.close()
        await self._server.wait_closed()
        self._server = None

        connections = tuple(self._connections)
        for task in connections:
            task.cancel()
        if connections:
            await asyncio.gather(*connections, return_exceptions=True)
        self._connections.clear()

    async def _serve_connection(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        task = asyncio.current_task()
        if task is not None:
            self._connections.add(task)
        try:
            while True:
                try:
                    request = await self._read_request(reader)
                except BadRequestError as exc:
                    await self._write_response(writer, None, Response.error(exc.status, exc.message))
                    return
                if request is None:  # клиент закрыл соединение — штатное завершение
                    return

                response = await self._dispatch(request)
                keep_alive = await self._write_response(writer, request, response)
                if not keep_alive:
                    return
        except (ConnectionResetError, BrokenPipeError, TimeoutError):
            # Обычная жизнь браузера: вкладку закрыли, SSE оборвался, клиент
            # ушёл по таймауту. Логировать это уровнем выше debug незачем.
            logger.debug("dashboard.http: connection closed by peer")
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("dashboard.http: unexpected failure while serving a connection")
        finally:
            if task is not None:
                self._connections.discard(task)
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
                # CancelledError здесь — это остановка сервера (см. stop()):
                # соединение всё равно закрывается, дожидаться подтверждения
                # от уже уходящего клиента не нужно.
                pass

    async def _dispatch(self, request: Request) -> AnyResponse:
        try:
            return await self._handler(request)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("dashboard.http: handler failed for %s %s", request.method, request.path)
            return Response.error(500, "внутренняя ошибка дашборда")

    async def _read_request(self, reader: asyncio.StreamReader) -> Request | None:
        try:
            head = await asyncio.wait_for(reader.readuntil(b"\r\n\r\n"), timeout=_READ_TIMEOUT_SECONDS)
        except asyncio.IncompleteReadError as exc:
            if not exc.partial:
                return None
            raise BadRequestError(400, "неполный запрос") from exc
        except asyncio.LimitOverrunError as exc:
            raise BadRequestError(431, "слишком длинные заголовки") from exc
        except TimeoutError as exc:
            raise BadRequestError(408, "истекло время ожидания запроса") from exc

        if len(head) > _MAX_HEADER_BYTES:
            raise BadRequestError(431, "слишком длинные заголовки")

        method, target, headers = _parse_head(head)
        body = await self._read_body(reader, headers)
        split = urlsplit(target)
        return Request(
            method=method,
            path=unquote(split.path) or "/",
            query=parse_qs(split.query, keep_blank_values=True),
            headers=headers,
            body=body,
        )

    async def _read_body(self, reader: asyncio.StreamReader, headers: Mapping[str, str]) -> bytes:
        raw_length = headers.get("content-length")
        if not raw_length:
            return b""
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise BadRequestError(400, "некорректный Content-Length") from exc
        if length < 0:
            raise BadRequestError(400, "некорректный Content-Length")
        if length > _MAX_BODY_BYTES:
            raise BadRequestError(413, "слишком большое тело запроса")
        if length == 0:
            return b""
        try:
            return await asyncio.wait_for(reader.readexactly(length), timeout=_READ_TIMEOUT_SECONDS)
        except asyncio.IncompleteReadError as exc:
            raise BadRequestError(400, "тело запроса оборвалось") from exc
        except TimeoutError as exc:
            raise BadRequestError(408, "истекло время ожидания тела запроса") from exc

    async def _write_response(
        self,
        writer: asyncio.StreamWriter,
        request: Request | None,
        response: AnyResponse,
    ) -> bool:
        """Отправляет ответ и сообщает, можно ли переиспользовать соединение."""
        if isinstance(response, StreamResponse):
            await self._write_stream(writer, response)
            return False

        keep_alive = _client_wants_keep_alive(request)
        headers = {
            "Content-Type": response.content_type,
            "Content-Length": str(len(response.body)),
            "Date": formatdate(usegmt=True),
            "Connection": "keep-alive" if keep_alive else "close",
            # Дашборд показывает живое состояние; закэшированный ответ здесь
            # всегда вреден — вплоть до "Эфи офлайн" на странице живого бота.
            "Cache-Control": "no-store",
            **response.headers,
        }
        writer.write(_render_head(response.status, headers))
        if request is None or request.method != "HEAD":
            writer.write(response.body)
        await writer.drain()
        return keep_alive

    async def _write_stream(self, writer: asyncio.StreamWriter, response: StreamResponse) -> None:
        headers = {
            "Content-Type": response.content_type,
            "Date": formatdate(usegmt=True),
            "Connection": "close",
            "Cache-Control": "no-store",
            # Промежуточные прокси любят копить SSE в буфере; заголовок ниже —
            # общепринятая просьба этого не делать.
            "X-Accel-Buffering": "no",
            **response.headers,
        }
        writer.write(_render_head(response.status, headers))
        await writer.drain()
        try:
            async for chunk in response.stream:
                writer.write(chunk)
                await writer.drain()
        finally:
            aclose = getattr(response.stream, "aclose", None)
            if aclose is not None:
                await aclose()


def _parse_head(head: bytes) -> tuple[str, str, dict[str, str]]:
    # По RFC заголовки — байты, а не текст, и формально в них только ASCII.
    # На практике UTF-8 туда всё же попадает (например, значение cookie,
    # выставленное строкой с кириллицей), поэтому сначала пробуем UTF-8, а
    # latin-1 остаётся запасным вариантом — он не падает никогда.
    try:
        text = head.decode("utf-8")
    except UnicodeDecodeError:
        text = head.decode("latin-1")
    lines = text.split("\r\n")
    request_line = lines[0]
    parts = request_line.split(" ")
    if len(parts) != 3:
        raise BadRequestError(400, "некорректная строка запроса")
    method, target, version = parts
    if not version.startswith("HTTP/"):
        raise BadRequestError(400, "неизвестная версия протокола")

    headers: dict[str, str] = {}
    for line in lines[1:]:
        if not line:
            continue
        name, separator, value = line.partition(":")
        if not separator:
            raise BadRequestError(400, "некорректный заголовок")
        headers[name.strip().lower()] = value.strip()
    return method.upper(), target, headers


def _client_wants_keep_alive(request: Request | None) -> bool:
    if request is None:
        return False
    connection = request.headers.get("connection", "").lower()
    return "close" not in connection


def _render_head(status: int, headers: Mapping[str, str]) -> bytes:
    reason = _STATUS_TEXT.get(status, "OK")
    lines = [f"HTTP/1.1 {status} {reason}"]
    lines.extend(f"{name}: {value}" for name, value in headers.items())
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")


def sse_event(data: Any, *, event: str | None = None) -> bytes:
    """
    Сериализует одно событие Server-Sent Events.

    Перевод строки внутри данных разбивается на несколько `data:` — иначе
    браузер обрежет событие на первом же `\\n` (частая причина "SSE работает,
    но приходит мусор" на трассировках исключений в логах).
    """
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False, default=str)
    chunks = [f"event: {event}"] if event else []
    chunks.extend(f"data: {line}" for line in payload.split("\n"))
    return ("\n".join(chunks) + "\n\n").encode("utf-8")


__all__ = [
    "AnyResponse",
    "BadRequestError",
    "Handler",
    "HttpServer",
    "Request",
    "Response",
    "StreamResponse",
    "sse_event",
]
