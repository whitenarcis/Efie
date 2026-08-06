"""
efi/telegram/handlers.py

Обработчики входящих событий Telegram: текст, фото, голосовые, видео-кружки.
Каждый обработчик:
    1. Проверяет доступ (efi.security.access_control) — молча игнорирует
       событие, если чат не проходит проверку Lockdown/allowlist'а.
    2. Для медиа — скачивает файл и превращает его в текст (транскрипция
       голоса/видео-кружка, описание фото через LLMRouter/VISION, см.
       efi/telegram/media/) ДО того, как событие попадёт в Worker: Worker и
       вся остальная система работают только с текстом.
    3. Санитайзинг текста и отображаемого имени отправителя
       (efi.security.sanitize) — первая точка, где внешний ввод попадает в систему.
    4. Отмечает активность в SilenceMonitor (см. efi/behavior/silence_monitor.py).
    5. Собирает Notification(USER_MESSAGE) и кладёт его в NotificationManager.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any

from pyrogram import Client, filters
from pyrogram.enums import ChatType
from pyrogram.handlers import MessageHandler
from pyrogram.types import Message as PyrogramMessage

from efi.config.schema import TelegramSettings
from efi.llm.router import LLMRouter
from efi.notifications.manager import NotificationManager
from efi.notifications.schemas import Notification, NotificationType
from efi.security.access_control import ChatAccessInfo, is_chat_accessible
from efi.telegram.formatting import format_user_message
from efi.telegram.media.image import describe_photo
from efi.telegram.media.video import transcribe_video_note
from efi.telegram.media.voice import transcribe_voice_message

logger = logging.getLogger(__name__)

_HandlerCallback = Callable[[Client, PyrogramMessage], Awaitable[None]]


class TelegramEventHandlers:
    """
    Регистрирует обработчики входящих событий на переданном Pyrogram Client и
    транслирует их в Notification -> NotificationManager.put().

    `activity_recorder` — необязательная зависимость (обычно
    efi.behavior.silence_monitor.SilenceMonitor); передана через дак-тайпинг
    (нужен только метод `record_activity(chat_id: int)`), чтобы этот модуль
    не зависел от efi.behavior напрямую.
    """

    def __init__(
        self,
        manager: NotificationManager,
        telegram_settings: TelegramSettings,
        router: LLMRouter,
        media_cache_dir: Path,
        *,
        activity_recorder: Any | None = None,
    ) -> None:
        self._manager = manager
        self._telegram_settings = telegram_settings
        self._router = router
        self._media_cache_dir = media_cache_dir
        self._activity_recorder = activity_recorder

    def register(self, client: Client) -> None:
        """Регистрирует все обработчики на клиенте. Вызывается один раз при сборке приложения (efi/app.py)."""
        own_messages_filter = ~filters.bot & ~filters.me
        client.add_handler(_make_handler(filters.text & own_messages_filter, self._handle_text))
        client.add_handler(_make_handler(filters.photo & own_messages_filter, self._handle_photo))
        client.add_handler(_make_handler(filters.voice & own_messages_filter, self._handle_voice))
        client.add_handler(_make_handler(filters.video_note & own_messages_filter, self._handle_video_note))

    async def _handle_text(self, client: Client, message: PyrogramMessage) -> None:
        await self._dispatch_user_message(message, text=message.text or "")

    async def _handle_photo(self, client: Client, message: PyrogramMessage) -> None:
        caption = message.caption or ""
        downloaded_path = await self._download_media(client, message)
        description = (
            await describe_photo(self._router, downloaded_path)
            if downloaded_path is not None
            else "[фото — не удалось загрузить файл]"
        )
        text = f"[прислал(а) фото] {description}"
        if caption:
            text += f" (подпись: {caption})"
        await self._dispatch_user_message(message, text=text, payload={"media_type": "photo"})

    async def _handle_voice(self, client: Client, message: PyrogramMessage) -> None:
        downloaded_path = await self._download_media(client, message)
        transcript = (
            await transcribe_voice_message(self._router, downloaded_path)
            if downloaded_path is not None
            else "[голосовое сообщение — не удалось загрузить файл]"
        )
        await self._dispatch_user_message(
            message, text=f"[прислал(а) голосовое] {transcript}", payload={"media_type": "voice"}
        )

    async def _handle_video_note(self, client: Client, message: PyrogramMessage) -> None:
        downloaded_path = await self._download_media(client, message)
        transcript = (
            await transcribe_video_note(self._router, downloaded_path)
            if downloaded_path is not None
            else "[видео-кружок — не удалось загрузить файл]"
        )
        await self._dispatch_user_message(
            message, text=f"[прислал(а) видео-кружок] {transcript}", payload={"media_type": "video_note"}
        )

    async def _download_media(self, client: Client, message: PyrogramMessage) -> Path | None:
        try:
            result = await client.download_media(message, file_name=f"{self._media_cache_dir}/")
        except Exception:
            logger.exception("telegram: failed to download media for message_id=%s", message.id)
            return None
        return Path(result) if result else None

    async def _dispatch_user_message(
        self,
        message: PyrogramMessage,
        *,
        text: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if message.from_user is None or message.chat is None:
            return  # анонимные админы, каналы и т.п. — вне текущего скоупа

        access_info = ChatAccessInfo(
            chat_id=message.chat.id,
            is_owner=message.from_user.id == self._telegram_settings.owner_id,
            is_contact=bool(getattr(message.from_user, "is_contact", False)),
            is_private_chat=message.chat.type == ChatType.PRIVATE,
        )
        allowed, reason = is_chat_accessible(access_info, self._telegram_settings)
        if not allowed:
            logger.debug("telegram: message from chat_id=%s dropped (%s)", access_info.chat_id, reason)
            return

        if self._activity_recorder is not None:
            self._activity_recorder.record_activity(access_info.chat_id)

        notification = Notification(
            type=NotificationType.USER_MESSAGE,
            priority=0,  # входящее сообщение пользователя — наивысший приоритет обработки
            chat_id=access_info.chat_id,
            message=format_user_message(message, text),
            payload={"telegram_message_id": message.id, **(payload or {})},
        )
        await self._manager.put(notification)


def _make_handler(message_filter: filters.Filter, callback: _HandlerCallback) -> MessageHandler:
    return MessageHandler(callback, message_filter)


__all__ = ["TelegramEventHandlers"]
