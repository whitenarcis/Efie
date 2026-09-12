"""
efi/tools/telegram_actions/stickers.py

Отправка стикера в текущий чат — простой, "человеческий" способ отреагировать
без слов.

Модель НЕ получает Telegram-иды стикеров: стикеры доходят до неё как описание
от vision с коротким id («[стикер: кот смеётся (id 7)]», см. efi/telegram/
handlers.py). Инструмент принимает ровно этот короткий id и сам достаёт из
кэша описаний (efi/db/sticker_descriptions.py) актуальный Telegram file_id —
стабильного file_id для отправки у модели нет и не должно быть.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from efi.tools.base import Tool, ToolContext

logger = logging.getLogger(__name__)

#: Сколько известных стикеров перечислить в ошибке про неизвестный id — чтобы
#: модель могла подобрать правильный, а не гадать вслепую.
_MAX_SUGGESTED_STICKERS = 7


class StickerSender(Protocol):
    """Абстракция отправки стикера. Конкретная реализация — efi.telegram.client.TelegramClientWrapper."""

    async def send_sticker(self, chat_id: int, sticker_file_id: str) -> None: ...


class SendStickerTool(Tool):
    """Отправляет известный стикер по короткому id (см. описание входящего стикера или блок «[Известные стикеры]»)."""

    name = "send_sticker"
    description = (
        "Отправляет известный стикер по его короткому id — числу в скобках у "
        "описания стикера («[стикер: кот смеётся (id 7)]») или из блока "
        "«[Известные стикеры]»."
    )
    parameters = {
        "type": "object",
        "properties": {
            "sticker_id": {
                "type": "integer",
                "description": "Короткий id стикера из списка известных (например, 7)",
            },
        },
        "required": ["sticker_id"],
        "additionalProperties": False,
    }

    def __init__(self, sender: StickerSender, sticker_store: Any) -> None:
        self._sender = sender
        # Дак-тайпинг (обычно efi.db.sticker_descriptions.StickerDescriptionStore):
        # нужны методы by_id(sticker_id) и recent(limit) — см. efi/db/sticker_descriptions.py.
        self._sticker_store = sticker_store

    def is_available(self, context: ToolContext) -> bool:
        return context.chat_id is not None and self._sticker_store is not None

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        sticker_id = arguments.get("sticker_id")
        if not isinstance(sticker_id, int):
            return "error: sticker_id must be an integer id известного стикера (например, 7)"
        if context.chat_id is None:
            return "error: no chat_id in the current context, nowhere to send the sticker"

        if self._sticker_store is None:
            return "error: список известных стикеров недоступен"

        sticker = await self._sticker_store.by_id(sticker_id)
        if sticker is None or not sticker.file_id:
            known = await self._sticker_store.recent(limit=_MAX_SUGGESTED_STICKERS)
            known_ids = ", ".join(str(item.sticker_id) for item in known)
            hint = f" Известные id: {known_ids}." if known_ids else " Известных стикеров пока нет."
            return f"error: стикер с id {sticker_id} неизвестен.{hint}"

        await self._sender.send_sticker(context.chat_id, sticker.file_id)
        logger.info("send_sticker: sent sticker_id=%s to chat_id=%s", sticker_id, context.chat_id)
        return f"Стикер отправлен (id {sticker_id})."


__all__ = ["StickerSender", "SendStickerTool"]
