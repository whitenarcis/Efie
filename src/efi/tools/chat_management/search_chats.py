"""
efi/tools/chat_management/search_chats.py

Инструмент поиска среди известных диалогов (по названию/юзернейму) — чтобы
модель могла найти chat_id нужного чата, не зная его заранее (например,
чтобы переслать туда сообщение через forward_message).
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from efi.tools.base import Tool, ToolContext

logger = logging.getLogger(__name__)


class ChatSearcher(Protocol):
    """Абстракция поиска среди диалогов. Конкретная реализация — efi.telegram.client.TelegramClientWrapper."""

    async def search_chats(self, query: str, *, limit: int = 10) -> list[tuple[int, str]]: ...


class SearchChatsTool(Tool):
    """Ищет среди известных диалогов по названию/юзернейму, возвращает подходящие chat_id."""

    name = "search_chats"
    description = "Ищет среди твоих диалогов по названию или юзернейму — полезно, чтобы найти chat_id нужного чата."
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Часть названия или юзернейма чата"},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, searcher: ChatSearcher) -> None:
        self._searcher = searcher

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        query = str(arguments.get("query", "")).strip()
        if not query:
            return "error: query must not be empty"

        try:
            results = await self._searcher.search_chats(query)
        except Exception as exc:
            logger.warning("search_chats: failed for %r: %s", query, exc)
            return f"error: search failed: {exc}"

        if not results:
            return f"Ничего не нашлось по запросу {query!r}."

        lines = "\n".join(f"- {chat_id}: {title}" for chat_id, title in results)
        return f"Найдено:\n{lines}"


__all__ = ["ChatSearcher", "SearchChatsTool"]
