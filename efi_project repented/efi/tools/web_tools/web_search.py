"""
efi/tools/web_tools/web_search.py

Веб-поиск через HTML lite-интерфейс DuckDuckGo (lite.duckduckgo.com) —
реальные результаты поиска (заголовок + сниппет + ссылка), а не только редкие
instant-answer инфобоксы, как было в предыдущей версии на Instant Answer API.
Всё ещё без API-ключа и без оплаты, но заметно шире по охвату: подходит для
"погугли X", новостей, общих фактических вопросов — не только для
энциклопедических определений.

Честная граница: для ЖИВЫХ числовых данных (текущая погода, курсы валют,
счёт матча прямо сейчас) сниппеты поиска — ненадёжный источник, они могут
быть закэшированы/устаревшими. Для погоды есть отдельный get_weather
(прямой бесплатный API без посредников, см. efi/tools/web_tools/get_weather.py).

ВАЖНАЯ ОГОВОРКА: разметка lite.duckduckgo.com — неофициальная HTML-страница,
её структура нигде не документирована и может измениться без предупреждения.
Парсинг ниже (_parse_results) целится в структуру на момент написания и имеет
грубый фолбэк на случай, если основной селектор перестанет находить
результаты — но проверить вживую в этой среде было невозможно (нет сети).
Если поиск начнёт стабильно возвращать "ничего не нашлось" при явно
существующих результатах — смотреть сюда в первую очередь.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

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

_SEARCH_URL = "https://lite.duckduckgo.com/lite/"
_REQUEST_TIMEOUT_SECONDS = 10.0
_MAX_RESULTS = 5
_SNIPPET_MAX_LENGTH = 220
_USER_AGENT = "Mozilla/5.0 (compatible; EfiBot/1.0)"


class WebSearchTool(Tool):
    """Ищет в интернете через DuckDuckGo (HTML lite-интерфейс, без ключа) — реальный список результатов."""

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
            headers={"User-Agent": _USER_AGENT},
        )
        self._journal = journal

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        query = str(arguments.get("query", "")).strip()
        if not query:
            return "error: query must not be empty"

        try:
            response = await self._client.post(_SEARCH_URL, data={"q": query})
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("web_search: request failed for %r: %s", query, exc)
            return f"error: web search request failed: {exc}"

        results = _parse_results(response.text)
        if not results:
            return f"По запросу {query!r} ничего не нашлось."

        lines = [f"- {title}: {snippet} ({url})" for title, snippet, url in results[:_MAX_RESULTS]]
        digest = "\n".join(lines)
        await self._journal_lookup(query, digest, context)
        return "Результаты поиска:\n" + digest

    async def _journal_lookup(self, query: str, digest: str, context: ToolContext) -> None:
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
                query=query,
                digest=digest,
                chat_id=context.chat_id,
                thread_id=thread_id if isinstance(thread_id, int) else None,
            )
        except Exception:
            logger.warning("web_search: failed to journal lookup for %r", query, exc_info=True)


def _parse_results(html: str) -> list[tuple[str, str, str]]:
    soup = BeautifulSoup(html, "html.parser")
    results: list[tuple[str, str, str]] = []

    anchors = soup.select("a.result-link")
    if not anchors:
        anchors = _fallback_result_anchors(soup)

    for link in anchors:
        title = link.get_text(strip=True)
        url = link.get("href", "")
        if not title or not url or "duckduckgo.com" in url:
            continue
        results.append((title, _extract_snippet_near(link), url))

    return results


def _fallback_result_anchors(soup: BeautifulSoup) -> list[Tag]:
    """Если основной селектор не сработал (вёрстка изменилась) — грубый фолбэк: любые внешние ссылки на странице."""
    return [
        tag
        for tag in soup.find_all("a", href=True)
        if isinstance(tag, Tag) and str(tag["href"]).startswith("http") and "duckduckgo.com" not in str(tag["href"])
    ]


def _extract_snippet_near(link: Tag) -> str:
    row = link.find_parent("tr")
    if row is None:
        return ""
    next_row = row.find_next_sibling("tr")
    if next_row is None:
        return ""
    return next_row.get_text(strip=True)[:_SNIPPET_MAX_LENGTH]


__all__ = ["SearchJournal", "WebSearchTool"]
