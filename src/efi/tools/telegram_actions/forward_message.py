"""
efi/tools/telegram_actions/forward_message.py

Инструмент пересылки сообщения из текущего чата в другой — например, чтобы
поделиться чем-то интересным.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from efi.tools.base import Tool, ToolContext

logger = logging.getLogger(__name__)


class MessageForwarder(Protocol):
    """Абстракция пересылки сообщения. Конкретная реализация — efi.telegram.client.TelegramClientWrapper."""

    async def forward_message(self, from_chat_id: int, message_id: int, to_chat_id: int) -> None: ...


class ForwardMessageTool(Tool):
    """Пересылает сообщение из текущего чата в указанный."""

    name = "forward_message"
    description = "Пересылает сообщение из текущего чата в другой чат — например, поделиться с кем-то ещё."
    parameters = {
        "type": "object",
        "properties": {
            "message_id": {"type": "integer", "description": "ID сообщения, которое нужно переслать"},
            "to_chat_id": {"type": "integer", "description": "ID чата, куда переслать сообщение"},
        },
        "required": ["message_id", "to_chat_id"],
        "additionalProperties": False,
    }

    def __init__(self, forwarder: MessageForwarder) -> None:
        self._forwarder = forwarder

    def is_available(self, context: ToolContext) -> bool:
        return context.chat_id is not None

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        if context.chat_id is None:
            return "error: no chat_id in the current context"
        try:
            message_id = int(arguments["message_id"])
            to_chat_id = int(arguments["to_chat_id"])
        except (KeyError, TypeError, ValueError):
            return "error: message_id and to_chat_id must be integers"

        try:
            await self._forwarder.forward_message(context.chat_id, message_id, to_chat_id)
        except Exception as exc:
            logger.warning("forward_message: failed: %s", exc)
            return f"error: could not forward message: {exc}"

        logger.info("forward_message: forwarded message_id=%s from %s to %s", message_id, context.chat_id, to_chat_id)
        return "Сообщение переслано."


__all__ = ["MessageForwarder", "ForwardMessageTool"]
