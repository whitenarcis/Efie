"""
efi/tools/telegram_actions/stickers.py

Отправка стикера в текущий чат — простой, "человеческий" способ отреагировать
без слов. Принимает file_id конкретного стикера (Telegram file_id, не сам
файл): подбор "какой именно стикер из набора подходит по смыслу" — отдельная
задача (индекс стикеров с описаниями, аналог tools/stickers.h у референса,
где подбор встроен в сам инструмент), не реализована в этом шаге. Сейчас
инструмент рассчитан на то, что модель знает file_id заранее (например, из
списка в системном промпте) или переиспользует ранее увиденный в истории.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from efi.tools.base import Tool, ToolContext

logger = logging.getLogger(__name__)


class StickerSender(Protocol):
    """Абстракция отправки стикера. Конкретная реализация — efi.telegram.client.TelegramClientWrapper."""

    async def send_sticker(self, chat_id: int, sticker_file_id: str) -> None: ...


class SendStickerTool(Tool):
    """Отправляет стикер по его Telegram file_id в текущий чат."""

    name = "send_sticker"
    description = "Отправляет стикер в текущий чат по его Telegram file_id."
    parameters = {
        "type": "object",
        "properties": {
            "sticker_file_id": {"type": "string", "description": "Telegram file_id стикера"},
        },
        "required": ["sticker_file_id"],
        "additionalProperties": False,
    }

    def __init__(self, sender: StickerSender) -> None:
        self._sender = sender

    def is_available(self, context: ToolContext) -> bool:
        return context.chat_id is not None

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        sticker_file_id = str(arguments.get("sticker_file_id", "")).strip()
        if not sticker_file_id:
            return "error: sticker_file_id must not be empty"
        if context.chat_id is None:
            return "error: no chat_id in the current context, nowhere to send the sticker"

        await self._sender.send_sticker(context.chat_id, sticker_file_id)
        logger.info("send_sticker: sent %s to chat_id=%s", sticker_file_id, context.chat_id)
        return "Стикер отправлен."


__all__ = ["StickerSender", "SendStickerTool"]
