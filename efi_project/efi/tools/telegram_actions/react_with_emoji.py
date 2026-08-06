"""
efi/tools/telegram_actions/react_with_emoji.py

Инструмент реакции эмодзи на сообщение собеседника — часто более уместный
ответ, чем текст (например, на шутку или короткую реплику).
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from efi.tools.base import Tool, ToolContext

logger = logging.getLogger(__name__)


class MessageReactor(Protocol):
    """Абстракция реакции на сообщение. Конкретная реализация — efi.telegram.client.TelegramClientWrapper."""

    async def react(self, chat_id: int, message_id: int, emoji: str) -> None: ...


class ReactWithEmojiTool(Tool):
    """Ставит эмодзи-реакцию на конкретное сообщение в чате."""

    name = "react_with_emoji"
    description = (
        "Ставит эмодзи-реакцию на сообщение собеседника — уместно вместо текстового "
        "ответа на короткие реплики/шутки/что-то, на что не нужно отвечать словами."
    )
    parameters = {
        "type": "object",
        "properties": {
            "message_id": {"type": "integer", "description": "ID сообщения, на которое нужно отреагировать"},
            "emoji": {"type": "string", "description": "Эмодзи реакции, например '❤️' или '😂'"},
        },
        "required": ["message_id", "emoji"],
        "additionalProperties": False,
    }

    def __init__(self, reactor: MessageReactor) -> None:
        self._reactor = reactor

    def is_available(self, context: ToolContext) -> bool:
        return context.chat_id is not None

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        emoji = str(arguments.get("emoji", "")).strip()
        if not emoji:
            return "error: emoji must not be empty"
        if context.chat_id is None:
            return "error: no chat_id in the current context"

        try:
            message_id = int(arguments["message_id"])
        except (KeyError, TypeError, ValueError):
            return "error: message_id must be an integer"

        try:
            await self._reactor.react(context.chat_id, message_id, emoji)
        except Exception as exc:
            logger.warning("react_with_emoji: failed to react: %s", exc)
            return f"error: could not react: {exc}"

        logger.info("react_with_emoji: reacted %s to message_id=%s in chat_id=%s", emoji, message_id, context.chat_id)
        return "Реакция поставлена."


__all__ = ["MessageReactor", "ReactWithEmojiTool"]
