"""
efi/tools/web_tools/web_search.py

Веб-поиск через DuckDuckGo Instant Answer API — не требует API-ключа, что
важно для инфраструктуры на Termux/VPN без доступа к платёжным сервисам (см.
известные ограничения проекта в памяти).

Честное предупреждение: это НЕ полноценная поисковая выдача — DuckDuckGo
Instant Answer API в основном отдаёт "абстракты"/инфобоксы (как в панели
сбоку у обычного поиска), а не список из десяти ссылок. Для более широкого
поиска потребовался бы платный API (SerpAPI, Brave Search и т.п.) —
сознательно не стал закладывать платную зависимость без явного запроса.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from efi.tools.base import Tool, ToolContext

logger = logging.getLogger(__name__)

_SEARCH_URL = "https://api.duckduckgo.com/"
_REQUEST_TIMEOUT_SECONDS = 10.0
_MAX_RELATED_TOPICS = 5


class WebSearchTool(Tool):
    """Ищет краткую справочную информацию в интернете (DuckDuckGo Instant Answer API, без ключа)."""

    name = "web_search"
    description = (
        "Ищет краткую справочную информацию в интернете — определения, факты, сводки по теме. "
        "Не заменяет полноценный поиск: лучше всего работает для энциклопедических вопросов, "
        "а не для 'последних новостей' или узких запросов."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Поисковый запрос"},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, *, client: httpx.AsyncClient | None = None) -> None:
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(_REQUEST_TIMEOUT_SECONDS))

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        query = str(arguments.get("query", "")).strip()
        if not query:
            return "error: query must not be empty"

        try:
            response = await self._client.get(
                _SEARCH_URL,
                params={"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"},
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            logger.warning("web_search: request failed for %r: %s", query, exc)
            return f"error: web search request failed: {exc}"
        except ValueError as exc:
            logger.warning("web_search: could not parse response for %r: %s", query, exc)
            return "error: web search returned an unparseable response"

        return _format_results(query, data)


def _format_results(query: str, data: dict[str, Any]) -> str:
    abstract = (data.get("AbstractText") or "").strip()
    abstract_source = (data.get("AbstractSource") or "").strip()
    related_topics = data.get("RelatedTopics") or []

    lines: list[str] = []
    if abstract:
        source_suffix = f" (источник: {abstract_source})" if abstract_source else ""
        lines.append(f"{abstract}{source_suffix}")

    related_texts = [
        topic["Text"] for topic in related_topics if isinstance(topic, dict) and topic.get("Text")
    ][:_MAX_RELATED_TOPICS]
    if related_texts:
        lines.append("Связанные темы:\n" + "\n".join(f"- {text}" for text in related_texts))

    if not lines:
        return f"По запросу {query!r} краткой справочной информации не нашлось."
    return "\n\n".join(lines)


__all__ = ["WebSearchTool"]
