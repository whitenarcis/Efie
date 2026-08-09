"""
efi/telegram/typing_tracker.py

Отслеживание статуса "печатает.../записывает голосовое.../загружает фото..."
собеседника через сырые MTProto-апдейты Pyrogram (UpdateUserTyping —
приватные чаты, UpdateChatUserTyping — обычные группы, UpdateChannelUserTyping
— супергруппы/каналы). Используется MessageDebouncer (efi/telegram/debounce.py),
чтобы реагировать сразу после того, как собеседник ВИДИМО закончил печатать, а
не по фиксированному таймеру тишины после отправки сообщения.

Любой из статусов composing (не только текстовый "печатает", но и запись
голосового/загрузка медиа) засчитывается одинаково — цель одна: "человек ещё
что-то готовит, рано отвечать".

ВАЖНАЯ ОГОВОРКА: конвертация "сырых" MTProto id (channel_id/chat_id из
апдейтов) в привычный Pyrogram chat_id (знак минус для обычных групп,
префикс "-100" для супергрупп/каналов через pyrogram.utils.get_channel_id)
сверена по документированному поведению Pyrogram/Pyrofork на момент
написания — если после обновления библиотеки статус перестанет совпадать с
нужным chat_id, стоит перепроверить именно эту функцию в первую очередь; без
сети/установленного пакета в этой среде проверить вживую было невозможно.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from pyrogram import Client
from pyrogram.handlers import RawUpdateHandler
from pyrogram.raw.types import UpdateChannelUserTyping, UpdateChatUserTyping, UpdateUserTyping
from pyrogram.utils import get_channel_id

logger = logging.getLogger(__name__)

#: Telegram-клиенты обычно обновляют статус "печатает" каждые ~5-6с, пока
#: человек продолжает набирать текст — если апдейта давно не было, считаем,
#: что печатать перестали (Telegram не шлёт отдельное "typing stopped" событие).
_DEFAULT_TYPING_TTL_SECONDS = 6.0


class TypingTracker:
    """Держит момент последнего замеченного composing-статуса по каждому chat_id."""

    def __init__(self, *, ttl_seconds: float = _DEFAULT_TYPING_TTL_SECONDS) -> None:
        self._ttl_seconds = ttl_seconds
        self._last_seen_at: dict[int, float] = {}

    def register(self, client: Client) -> None:
        """Подписывается на сырые апдейты клиента. Вызывается один раз при сборке приложения (efi/app.py)."""
        client.add_handler(RawUpdateHandler(self._on_raw_update))

    def is_typing(self, chat_id: int) -> bool:
        """True, если composing-статус для этого чата видели за последние ttl_seconds."""
        last_seen = self._last_seen_at.get(chat_id)
        if last_seen is None:
            return False
        return (time.monotonic() - last_seen) < self._ttl_seconds

    async def _on_raw_update(self, client: Client, update: Any, users: Any, chats: Any) -> None:
        chat_id = _extract_chat_id(update)
        if chat_id is not None:
            self._last_seen_at[chat_id] = time.monotonic()


def _extract_chat_id(update: object) -> int | None:
    """Возвращает Pyrogram-совместимый chat_id из сырого typing-апдейта, либо None, если апдейт не про typing."""
    if isinstance(update, UpdateUserTyping):
        return update.user_id
    if isinstance(update, UpdateChatUserTyping):
        return -update.chat_id
    if isinstance(update, UpdateChannelUserTyping):
        try:
            return get_channel_id(update.channel_id)
        except Exception:
            logger.debug("typing_tracker: could not convert channel_id=%s", update.channel_id, exc_info=True)
            return None
    return None


__all__ = ["TypingTracker"]
