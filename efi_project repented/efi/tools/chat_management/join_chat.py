"""
efi/tools/chat_management/join_chat.py

Инструмент вступления в чат/канал по ссылке или юзернейму. Доступен модели,
только если явно разрешено в конфиге (telegram.can_join_chats) — по
умолчанию False (см. efi/config/schema.py::TelegramSettings), самостоятельно
вступать куда попало Эфи не должна.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from efi.tools.base import Tool, ToolContext

logger = logging.getLogger(__name__)


class ChatJoiner(Protocol):
    """Абстракция вступления в чат. Конкретная реализация — efi.telegram.client.TelegramClientWrapper."""

    async def join_chat(self, chat_identifier: str) -> None: ...


class JoinChatTool(Tool):
    """Вступает в публичный чат или канал по ссылке (t.me/...) или юзернейму (@name)."""

    name = "join_chat"
    description = "Вступает в публичный чат или канал по ссылке (t.me/...) или юзернейму (@name)."
    parameters = {
        "type": "object",
        "properties": {
            "chat_identifier": {"type": "string", "description": "Ссылка t.me/... или юзернейм @name чата/канала"},
        },
        "required": ["chat_identifier"],
        "additionalProperties": False,
    }

    def __init__(self, joiner: ChatJoiner, *, enabled: bool) -> None:
        self._joiner = joiner
        self._enabled = enabled

    def is_available(self, context: ToolContext) -> bool:
        return self._enabled

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        chat_identifier = str(arguments.get("chat_identifier", "")).strip()
        if not chat_identifier:
            return "error: chat_identifier must not be empty"

        try:
            await self._joiner.join_chat(chat_identifier)
        except Exception as exc:
            logger.warning("join_chat: failed for %r: %s", chat_identifier, exc)
            return f"error: could not join {chat_identifier!r}: {exc}"

        logger.info("join_chat: joined %r", chat_identifier)
        return f"Вступила в {chat_identifier}."


__all__ = ["ChatJoiner", "JoinChatTool"]
