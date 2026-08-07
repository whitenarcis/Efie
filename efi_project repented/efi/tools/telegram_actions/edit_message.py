"""
efi/tools/telegram_actions/edit_message.py

Инструмент редактирования уже отправленного сообщения — например, чтобы
исправить опечатку или дополнить ответ, не заваливая чат новым сообщением.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from efi.tools.base import Tool, ToolContext

logger = logging.getLogger(__name__)


class MessageEditor(Protocol):
    """Абстракция редактирования сообщения. Конкретная реализация — efi.telegram.client.TelegramClientWrapper."""

    async def edit_message(self, chat_id: int, message_id: int, text: str) -> None: ...


class EditMessageTool(Tool):
    """Редактирует ранее отправленное сообщение по его message_id."""

    name = "edit_telegram_message"
    description = "Редактирует уже отправленное тобой сообщение — исправляет текст, не отправляя новое сообщение."
    parameters = {
        "type": "object",
        "properties": {
            "message_id": {"type": "integer", "description": "ID сообщения, которое нужно отредактировать"},
            "text": {"type": "string", "description": "Новый текст сообщения"},
        },
        "required": ["message_id", "text"],
        "additionalProperties": False,
    }

    def __init__(self, editor: MessageEditor) -> None:
        self._editor = editor

    def is_available(self, context: ToolContext) -> bool:
        return context.chat_id is not None

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        text = str(arguments.get("text", "")).strip()
        if not text:
            return "error: text must not be empty"
        if context.chat_id is None:
            return "error: no chat_id in the current context"

        try:
            message_id = int(arguments["message_id"])
        except (KeyError, TypeError, ValueError):
            return "error: message_id must be an integer"

        try:
            await self._editor.edit_message(context.chat_id, message_id, text)
        except Exception as exc:
            logger.warning("edit_message: failed for message_id=%s: %s", message_id, exc)
            return f"error: could not edit message: {exc}"

        logger.info("edit_message: edited message_id=%s in chat_id=%s", message_id, context.chat_id)
        return "Сообщение отредактировано."


__all__ = ["MessageEditor", "EditMessageTool"]
