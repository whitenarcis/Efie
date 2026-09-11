"""
efi/tools/web_tools/web_search.py

Веб-поиск для Efie через ddgs (DDGS).
Работает в пуле потоков через asyncio.to_thread, чтобы не блокировать
асинхронный event loop бота и корректно использовать актуальный API библиотеки.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from ddgs import DDGS

from efi.tools.base import Tool, ToolContext

logger = logging.getLogger(__name__)


class SearchJournal(Protocol):
    """Куда откладывается сам факт похода в интернет."""

    async def record_web_lookup(
        self, *, query: str, digest: str, chat_id: int | None = None, thread_id: int | None = None
    ) -> int: ...


_REQUEST_TIMEOUT_SECONDS = 12.0
_MAX_RESULTS = 5
_SNIPPET_MAX_LENGTH = 220


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
    """Итог поиска в структурированном виде."""

    query: str
    results: list[SearchResult] = field(default_factory=list)
    error: str = ""
    backend: str = ""

    @property
    def failed(self) -> bool:
        return bool(self.error)

    def render(self) -> str:
        if self.failed:
            return f"error: поиск не удался: {self.error}"
        if not self.results:
            return f"По запросу {self.query!r} ничего не нашлось."
        return "Результаты поиска:\n" + "\n".join(result.render() for result in self.results)

    def digest(self) -> str:
        return "\n".join(result.render() for result in self.results)


class WebSearchTool(Tool):
    """Ищет в интернете по запросу через DuckDuckGo (пакет ddgs)."""

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

    def __init__(self, *, journal: SearchJournal | None = None) -> None:
        self._journal = journal

    async def aclose(self) -> None:
        pass

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        query = str(arguments.get("query", "")).strip()
        if not query:
            return "error: query must not be empty"

        return (await self.search(query, journal_context=context)).render()

    async def search(self, query: str, *, journal_context: ToolContext | None = None) -> SearchOutcome:
        def _fetch() -> list[dict[str, Any]]:
            with DDGS(timeout=_REQUEST_TIMEOUT_SECONDS) as ddgs:
                return list(
                    ddgs.text(
                        query,
                        region="ru-ru",
                        safesearch="moderate",
                        max_results=_MAX_RESULTS,
                    )
                )

        try:
            raw_results = await asyncio.to_thread(_fetch)

            results: list[SearchResult] = []
            for item in raw_results:
                url = item.get("href", "").strip()
                title = item.get("title", "").strip()
                snippet = item.get("body", "").strip()[:_SNIPPET_MAX_LENGTH]

                if url and title:
                    results.append(SearchResult(title=title, snippet=snippet, url=url))

            if results:
                logger.debug("web_search: ddgs returned %d results for %r", len(results), query)
                return await self._finish(query, results, "ddgs-api", journal_context)

            logger.debug("web_search: ddgs returned empty results for %r", query)
            return SearchOutcome(query=query, backend="ddgs-api")

        except Exception as exc:
            logger.warning("web_search: ddgs failed for %r: %s", query, exc)
            return SearchOutcome(query=query, error=str(exc), backend="ddgs-api")

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

    async def _journal_lookup(self, outcome: SearchOutcome, context: ToolContext) -> None:
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


__all__ = ["SearchJournal", "SearchOutcome", "SearchResult", "WebSearchTool"]