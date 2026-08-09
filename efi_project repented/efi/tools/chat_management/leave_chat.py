"""
efi/tools/chat_management/leave_chat.py

Инструмент выхода из чата/канала. Доступен модели, только если явно
разрешено в конфиге (telegram.can_leave_chats, по умолчанию True — в отличие
от can_join_chats: покинуть чат менее рискованно, чем вступить в незнакомый).
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from efi.tools.base import Tool, ToolContext

logger = logging.getLogger(__name__)


class ChatLeaver(Protocol):
    """Абстракция выхода из чата. Конкретная реализация — efi.telegram.client.TelegramClientWrapper."""

    async def leave_chat(self, chat_id: int) -> None: ...


class LeaveChatTool(Tool):
    """Покидает чат/канал по его ID."""

    name = "leave_chat"
    description = (
        "Выходит из чата или канала по его ID. Использовать осторожно — действие необратимо "
        "без нового приглашения."
    )
    parameters = {
        "type": "object",
        "properties": {
            "chat_id": {"type": "integer", "description": "ID чата/канала, который нужно покинуть"},
        },
        "required": ["chat_id"],
        "additionalProperties": False,
    }

    def __init__(self, leaver: ChatLeaver, *, enabled: bool) -> None:
        self._leaver = leaver
        self._enabled = enabled

    def is_available(self, context: ToolContext) -> bool:
        return self._enabled

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        try:
            chat_id = int(arguments["chat_id"])
        except (KeyError, TypeError, ValueError):
            return "error: chat_id must be an integer"

        try:
            await self._leaver.leave_chat(chat_id)
        except Exception as exc:
            logger.warning("leave_chat: failed for chat_id=%s: %s", chat_id, exc)
            return f"error: could not leave chat_id={chat_id}: {exc}"

        logger.info("leave_chat: left chat_id=%s", chat_id)
        return f"Покинула чат {chat_id}."


__all__ = ["ChatLeaver", "LeaveChatTool"]
