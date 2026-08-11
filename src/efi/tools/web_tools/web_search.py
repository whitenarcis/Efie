"""
efi/tools/web_tools/web_search.py

Веб-поиск без ключей и оплаты: несколько бесплатных движков, опрашиваемых по
очереди до первого, который реально что-то вернул.

Почему по очереди, а не один. Бесплатный поиск скрейпингом — заведомо
хрупкая опора: страница не документирована, отдаётся то лайт-вёрсткой, то
страницей «похоже, вы бот», а иногда просто 403 по User-Agent. Один источник
означает, что в такой момент Эфи слепнет целиком — а слепнет она молча, и со
стороны это выглядит как «поиск не работает никогда».

Порядок: html-версия DuckDuckGo -> lite-версия -> Mojeek (независимый индекс,
то есть отдельная точка отказа, а не третье зеркало того же). Если ни один не
разобрался знакомой вёрсткой, вторым проходом идёт грубый разбор уже
полученных страниц (`_parse_generic`) — переименованный CSS-класс не должен
ослеплять Эфи целиком.

ИСТОРИЯ БАГА, из-за которого поиск не работал ВООБЩЕ. Прошлый парсер
выбрасывал каждую ссылку, в адресе которой встречалось "duckduckgo.com" —
защита от внутренней навигации. Но DuckDuckGo отдаёт результаты не прямыми
ссылками, а редиректом вида `//duckduckgo.com/l/?uddg=<адрес>`, поэтому под
эту проверку попадали ровно все результаты: список получался пустым, поиск
всегда возвращал «ничего не нашлось», а фоновые потребители писали в лог
«yielded nothing useful, skipping». Теперь редирект РАСПАКОВЫВАЕТСЯ
(`_resolve_url`), а отбрасывается только настоящая внутренняя навигация.

Честная граница: для ЖИВЫХ числовых данных (курс валют, счёт матча прямо
сейчас) сниппеты поиска — ненадёжный источник, они бывают закэшированы. Для
погоды есть отдельный get_weather (прямой API, без посредников).
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol
from urllib.parse import parse_qs, urlsplit

import httpx
from bs4 import BeautifulSoup, Tag

from efi.tools.base import Tool, ToolContext

logger = logging.getLogger(__name__)


class SearchJournal(Protocol):
    """
    Куда откладывается сам факт похода в интернет. Конкретная реализация —
    efi.memory.social_memory.SocialInteractionStore.

    Протокол объявлен здесь, а не импортируется из memory/, намеренно:
    инструмент не должен знать про подсистему памяти (см. принцип изоляции
    инструментов в efi/tools/base.py) — ему достаточно знать, что кто-то
    умеет принять «искала X, нашла Y».
    """

    async def record_web_lookup(
        self, *, query: str, digest: str, chat_id: int | None = None, thread_id: int | None = None
    ) -> int: ...


_REQUEST_TIMEOUT_SECONDS = 12.0
_MAX_RESULTS = 5
_SNIPPET_MAX_LENGTH = 220

#: User-Agent реального браузера. Прошлый ("EfiBot/1.0") — честный, но
#: поисковики массово отвечают на такие 403 или страницей-заглушкой, и
#: честность оборачивалась тем, что поиска не было вовсе.
_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
)
_HEADERS = {
    "User-Agent": _USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}

#: Признаки страницы «похоже, вы робот» вместо результатов. Приходят с кодом
#: 200, поэтому по статусу их не отличить — только по содержимому.
_ANOMALY_MARKERS = (
    "unfortunately, bots use duckduckgo too",
    "anomaly.js",
    "detected unusual activity",
)


@dataclass(slots=True, frozen=True)
class SearchResult:
    """Один результат выдачи."""

    title: str
    snippet: str
    url: str

    def render(self) -> str:
        return f"- {self.title}: {self.snippet} ({self.url})"


@dataclass(slots=True, frozen=True)
class SearchOutcome:
    """
    Итог поиска в структурированном виде.

    Нужен фоновым потребителям (efi.behavior.researcher,
    efi.behavior.life_engine): раньше они разбирали текстовый ответ
    инструмента подстроками ("error:" / "ничего не нашлось") и поэтому не
    могли отличить сломанный поиск от честно пустой выдачи — в лог уходило
    одно и то же «nothing useful», по которому нельзя было понять, что
    вообще произошло.
    """

    query: str
    results: list[SearchResult] = field(default_factory=list)
    error: str = ""
    backend: str = ""

    @property
    def failed(self) -> bool:
        """True — техническая неудача (сеть, блокировка, нечитаемая вёрстка), а не «ничего не найдено»."""
        return bool(self.error)

    def render(self) -> str:
        """Текст для модели — то, что возвращает Tool.execute."""
        if self.failed:
            return f"error: поиск не удался: {self.error}"
        if not self.results:
            return f"По запросу {self.query!r} ничего не нашлось."
        return "Результаты поиска:\n" + "\n".join(result.render() for result in self.results)

    def digest(self) -> str:
        return "\n".join(result.render() for result in self.results)


@dataclass(slots=True, frozen=True)
class _Backend:
    """Один поисковый источник: как спросить и как разобрать ответ."""

    name: str
    url: str
    method: str
    parse: Callable[[str], list[SearchResult]]


class WebSearchTool(Tool):
    """Ищет в интернете через бесплатные движки без ключа, перебирая их до первого рабочего."""

    name = "web_search"
    description = (
        "Ищет в интернете по запросу и возвращает список результатов (заголовок + краткое описание + ссылка). "
        "Подходит для общих фактических вопросов, новостей, 'погугли X'. НЕ подходит для точных live-данных "
        "вроде текущей погоды (для погоды используй get_weather) или курсов валют — сниппеты поиска могут "
        "быть устаревшими."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Поисковый запрос"},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, *, client: httpx.AsyncClient | None = None, journal: SearchJournal | None = None) -> None:
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(_REQUEST_TIMEOUT_SECONDS),
            headers=_HEADERS,
            # Движки любят отвечать 301/302 на «правильный» адрес; без этого
            # ответ был бы пустой страницей редиректа, а не выдачей.
            follow_redirects=True,
        )
        self._journal = journal
        self._backends = _default_backends()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        query = str(arguments.get("query", "")).strip()
        if not query:
            return "error: query must not be empty"

        return (await self.search(query, journal_context=context)).render()

    async def search(self, query: str, *, journal_context: ToolContext | None = None) -> SearchOutcome:
        """
        Опрашивает движки по очереди и возвращает первый непустой результат.

        Пустая выдача — НЕ повод остановиться: у бесплатных зеркал регулярно
        бывает, что один отдал страницу без результатов, а следующий по тому
        же запросу отвечает нормально. Останавливаемся только на успехе;
        `error` заполняется, лишь если ни один источник не смог даже
        ответить.

        `journal_context` — куда отнести факт похода в интернет (см.
        `_journal_lookup`). Необязателен, потому что вызывающая сторона
        бывает разной: инструмент внутри разговора всегда передаёт свой
        контекст, а фоновые исследователи — синтетический, без чата.
        """
        errors: list[str] = []
        #: Страницы, которые ответили, но знакомым разбором ничего не дали.
        #: Грубый разбор по ним идёт ВТОРЫМ проходом, а не сразу: иначе
        #: первый же движок «успешно» отдавал бы ссылки без описаний,
        #: подобранные со всей страницы, и до движка, чью вёрстку мы
        #: действительно умеем читать, дело бы не дошло.
        unparsed: list[tuple[str, str]] = []

        for backend in self._backends:
            try:
                html = await self._fetch(backend, query)
            except httpx.HTTPError as exc:
                errors.append(f"{backend.name}: {exc}")
                logger.warning("web_search: %s failed for %r: %s", backend.name, query, exc)
                continue

            if _looks_like_anomaly_page(html):
                errors.append(f"{backend.name}: страница проверки на робота")
                logger.warning("web_search: %s returned an anti-bot page for %r", backend.name, query)
                continue

            results = backend.parse(html)[:_MAX_RESULTS]
            if results:
                logger.debug("web_search: %s returned %d results for %r", backend.name, len(results), query)
                return await self._finish(query, results, backend.name, journal_context)
            unparsed.append((backend.name, html))
            logger.debug("web_search: %s returned no results for %r, trying the next backend", backend.name, query)

        for name, html in unparsed:
            results = _parse_generic(BeautifulSoup(html, "html.parser"))[:_MAX_RESULTS]
            if results:
                logger.warning(
                    "web_search: %s answered, но знакомая вёрстка не распозналась — разобрано грубым фолбэком "
                    "(%d ссылок). Если это повторяется, разметка движка изменилась",
                    name, len(results),
                )
                return await self._finish(query, results, f"{name}-generic", journal_context)

        if errors and len(errors) == len(self._backends):
            # Ни один источник не ответил — это поломка, а не «ничего не нашлось».
            return SearchOutcome(query=query, error="; ".join(errors))
        return SearchOutcome(query=query)

    async def _finish(
        self,
        query: str,
        results: list[SearchResult],
        backend: str,
        journal_context: ToolContext | None,
    ) -> SearchOutcome:
        outcome = SearchOutcome(query=query, results=results, backend=backend)
        if journal_context is not None:
            await self._journal_lookup(outcome, journal_context)
        return outcome

    async def _fetch(self, backend: _Backend, query: str) -> str:
        if backend.method == "POST":
            response = await self._client.post(backend.url, data={"q": query})
        else:
            response = await self._client.get(backend.url, params={"q": query})
        response.raise_for_status()
        return response.text

    async def _journal_lookup(self, outcome: SearchOutcome, context: ToolContext) -> None:
        """
        Откладывает поход в интернет в память. Пустой результат сюда не
        доходит (см. вызывающую сторону): «ничего не нашлось» — не опыт.

        Сбой журнала не должен превращаться в ошибку поиска: модель уже
        получила результаты, и терять их из-за проблемы с записью в память —
        худший из возможных обменов.
        """
        if self._journal is None:
            return
        thread_id = context.notification.payload.get("thread_id")
        try:
            await self._journal.record_web_lookup(
                query=outcome.query,
                digest=outcome.digest(),
                chat_id=context.chat_id,
                thread_id=thread_id if isinstance(thread_id, int) else None,
            )
        except Exception:
            logger.warning("web_search: failed to journal lookup for %r", outcome.query, exc_info=True)


# ---------------------------------------------------------------------------
# Источники и разбор их вёрстки
# ---------------------------------------------------------------------------


def _default_backends() -> list[_Backend]:
    return [
        _Backend("ddg-html", "https://html.duckduckgo.com/html/", "POST", parse_duckduckgo_html),
        _Backend("ddg-lite", "https://lite.duckduckgo.com/lite/", "POST", parse_duckduckgo_lite),
        _Backend("mojeek", "https://www.mojeek.com/search", "GET", parse_mojeek),
    ]


def parse_duckduckgo_html(html: str) -> list[SearchResult]:
    """Разбор html.duckduckgo.com: блок результата, ссылка `a.result__a`, сниппет `.result__snippet`."""
    soup = BeautifulSoup(html, "html.parser")
    results: list[SearchResult] = []
    for block in soup.select("div.result, div.web-result"):
        link = block.select_one("a.result__a")
        if link is None:
            continue
        url = _resolve_url(link.get("href"))
        title = link.get_text(strip=True)
        if not url or not title:
            continue
        snippet_tag = block.select_one(".result__snippet")
        snippet = snippet_tag.get_text(" ", strip=True)[:_SNIPPET_MAX_LENGTH] if snippet_tag else ""
        results.append(SearchResult(title=title, snippet=snippet, url=url))
    return results


def parse_duckduckgo_lite(html: str) -> list[SearchResult]:
    """Разбор lite.duckduckgo.com: таблица, где ссылка и её сниппет лежат в соседних строках."""
    soup = BeautifulSoup(html, "html.parser")
    results: list[SearchResult] = []
    for link in soup.select("a.result-link"):
        url = _resolve_url(link.get("href"))
        title = link.get_text(strip=True)
        if not url or not title:
            continue
        results.append(SearchResult(title=title, snippet=_lite_snippet(link), url=url))
    return results


def parse_mojeek(html: str) -> list[SearchResult]:
    """Разбор mojeek.com: независимый индекс с простой и стабильной вёрсткой — `li > h2 > a` и `p.s`."""
    soup = BeautifulSoup(html, "html.parser")
    results: list[SearchResult] = []
    for item in soup.select("ul.results-standard li, li.result, div.results li"):
        link = item.select_one("h2 a, a.title")
        if link is None:
            continue
        url = _resolve_url(link.get("href"))
        title = link.get_text(strip=True)
        if not url or not title:
            continue
        snippet_tag = item.select_one("p.s, p.snippet")
        snippet = snippet_tag.get_text(" ", strip=True)[:_SNIPPET_MAX_LENGTH] if snippet_tag else ""
        results.append(SearchResult(title=title, snippet=snippet, url=url))
    return results


def _lite_snippet(link: Tag) -> str:
    """
    Сниппет лайт-вёрстки лежит не рядом со ссылкой, а в одной из следующих
    строк таблицы. Ищем именно ячейку `td.result-snippet`, а не «текст
    следующей строки»: между ними бывает строка с адресом (`link-text`), и
    слепой «следующий tr» приносил её вместо описания.
    """
    row = link.find_parent("tr")
    if row is None:
        return ""
    first_text = ""
    for sibling in row.find_next_siblings("tr"):
        if sibling.find("a", class_="result-link") is not None:
            break  # дошли до следующего результата
        cell = sibling.find("td", class_="result-snippet")
        if isinstance(cell, Tag):
            return cell.get_text(" ", strip=True)[:_SNIPPET_MAX_LENGTH]
        # Запоминаем первую попавшуюся строку на случай, если размеченной
        # ячейки со сниппетом в вёрстке не окажется вовсе: описание без
        # класса лучше, чем пустое описание.
        first_text = first_text or sibling.get_text(" ", strip=True)[:_SNIPPET_MAX_LENGTH]
    return first_text


def _parse_generic(soup: BeautifulSoup) -> list[SearchResult]:
    """
    Грубый фолбэк на случай, если вёрстка знакомого движка поменялась: любые
    внешние ссылки страницы. Хуже по качеству, но лучше, чем внезапно
    ослепнуть целиком из-за переименованного CSS-класса.
    """
    results: list[SearchResult] = []
    seen: set[str] = set()
    for tag in soup.find_all("a", href=True):
        if not isinstance(tag, Tag):
            continue
        url = _resolve_url(tag.get("href"))
        title = tag.get_text(strip=True)
        if not url or len(title) < 8 or url in seen:
            continue
        seen.add(url)
        results.append(SearchResult(title=title, snippet="", url=url))
    return results


def _resolve_url(href: object) -> str:
    """
    Настоящий адрес результата.

    DuckDuckGo (и в html-, и в lite-вёрстке) отдаёт результаты не прямой
    ссылкой, а редиректом `//duckduckgo.com/l/?uddg=<адрес>&rut=...`. Именно
    поэтому прошлая проверка «в адресе есть duckduckgo.com — значит, это
    внутренняя навигация, пропускаем» выбрасывала ВСЕ результаты подряд.
    Распаковываем параметр `uddg`, а отбрасываем только те внутренние
    ссылки, за которыми ничего не стоит (настройки, «следующая страница»).
    """
    # str(): у многозначных атрибутов bs4 отдаёт список, и тогда любая
    # проверка вхождения молча работала бы с элементом списка, а не с адресом.
    raw = str(href or "").strip()
    if not raw:
        return ""
    if raw.startswith("//"):
        raw = "https:" + raw

    parts = urlsplit(raw)
    if parts.netloc.endswith("duckduckgo.com"):
        target = (parse_qs(parts.query).get("uddg") or [""])[0].strip()
        return target if target.startswith(("http://", "https://")) else ""
    if raw.startswith(("http://", "https://")):
        return raw
    return ""


def _looks_like_anomaly_page(html: str) -> bool:
    lowered = html[:4000].lower()
    return any(marker in lowered for marker in _ANOMALY_MARKERS)


__all__ = ["SearchJournal", "SearchOutcome", "SearchResult", "WebSearchTool"]
