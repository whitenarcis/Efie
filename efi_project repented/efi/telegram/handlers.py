"""
efi/telegram/handlers.py

Обработчики входящих событий Telegram: текст, фото, голосовые, видео-кружки.

Путь одного сообщения:
    1. Единая точка допуска — `_authorize()`: Lockdown/allowlist
       (efi.security.access_control), а для GROUP/SUPERGROUP — ЕЩЁ И
       адресность (упоминание бота или reply на его сообщение, см.
       `_is_addressed_to_bot`). Личка адресности не требует — там и так
       общаются один на один. Вызывается ДО любой дорогостоящей работы
       (скачивание медиа, STT, VISION-описание фото), чтобы групповой чат с
       активным трафиком не тратил его на сообщения, не адресованные Эфи.
    2. Для медиа — скачивание + превращение в текст (транскрипция голоса/
       видео-кружка, описание фото через LLMRouter/VISION, см.
       efi/telegram/media/). Транскрипция голоса/кружка предпочитает прямой
       Groq STT (efi.media.stt_groq.GroqSTT, если настроен ключ) и
       откатывается на LLMRouter (роль VISION), если ключа нет или Groq не
       дал текста.
    3. Не улетает в очередь немедленно — сначала в MessageDebouncer
       (efi/telegram/debounce.py): несколько быстрых сообщений подряд от
       одного собеседника группируются в одно событие ("anti-interrupt" —
       без этого Worker реагировал бы на каждую реплику по отдельности,
       заново пересобирая весь контекст на каждую).
    4. По истечении паузы тишины — сборка одного Notification(USER_MESSAGE):
       санитайзинг текста/имени (efi.security.sanitize), отметка активности
       в SilenceMonitor, контекст чата (личка/группа — см. _build_chat_context)
       кладётся в payload, чтобы SystemPromptBuilder мог сообщить модели,
       что она сейчас не в приватной переписке один на один.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

from pyrogram import Client, filters
from pyrogram.enums import ChatType
from pyrogram.handlers import MessageHandler
from pyrogram.types import Message as PyrogramMessage

from efi.config.schema import HumanizerSettings, TelegramSettings
from efi.llm.router import LLMRouter
from efi.media.stt_groq import GroqSTT
from efi.notifications.manager import NotificationManager
from efi.notifications.schemas import Notification, NotificationType
from efi.security.access_control import ChatAccessInfo, is_chat_accessible
from efi.telegram.debounce import MessageDebouncer
from efi.telegram.formatting import format_user_message
from efi.telegram.media.image import describe_photo
from efi.telegram.media.video import transcribe_video_note
from efi.telegram.media.voice import transcribe_voice_message
from efi.telegram.typing_tracker import TypingTracker

logger = logging.getLogger(__name__)

_HandlerCallback = Callable[[Client, PyrogramMessage], Awaitable[None]]
_AudioTranscriber = Callable[[LLMRouter, Path], Awaitable[str]]

#: Групповые типы чата, в которых Эфи отвечает ТОЛЬКО когда к ней обращаются
#: напрямую — см. _is_addressed_to_bot и докстринг модуля.
_GROUP_CHAT_TYPES = (ChatType.GROUP, ChatType.SUPERGROUP)


@dataclass(slots=True, frozen=True)
class _PendingMessage:
    """Один элемент буфера дебаунса: уже готовый (медиа превращено в текст) фрагмент + исходное сообщение Pyrogram."""

    message: PyrogramMessage
    text: str
    payload: dict[str, Any] = field(default_factory=dict)


class ReadReceiptSender(Protocol):
    """Абстракция отметки чата прочитанным. Конкретная реализация — efi.telegram.client.TelegramClientWrapper."""

    async def mark_as_read(self, chat_id: int) -> None: ...


class TelegramEventHandlers:
    """
    Регистрирует обработчики входящих событий на переданном Pyrogram Client и
    транслирует их (через MessageDebouncer) в Notification -> NotificationManager.put().

    `activity_recorder` — необязательная зависимость (обычно
    efi.behavior.silence_monitor.SilenceMonitor); передана через дак-тайпинг
    (нужен только метод `record_activity(chat_id: int)`), чтобы этот модуль
    не зависел от efi.behavior напрямую.

    `affinity_recorder` — аналогичная необязательная зависимость (обычно
    efi.behavior.affinity.AffinityTracker; нужен асинхронный метод
    `record_message(chat_id: int, text: str)`), которым текст реплики
    классифицируется и сдвигает близость/уважение к чату ДО того, как
    Worker соберёт по ней системный промпт — иначе сдвиг применился бы
    постфактум, уже после ответа на это же сообщение.

    `curiosity_recorder` — аналогичный необязательный дак-тайпинг (обычно
    efi.behavior.curiosity.CuriosityTracker; асинхронный метод
    `consider_message(chat_id: int | None, text: str)`), которым из текста
    реплики извлекаются темы-кандидаты на фоновое исследование.

    `organic_ping_recorder` — аналогичный необязательный дак-тайпинг (обычно
    efi.behavior.organic_ping.OrganicPingGenerator; асинхронный метод
    `handle_reply(chat_id: int)`), которым отмечается, что собеседник
    ответил в чате, где недавно был органический пинг.

    `stt` — необязательный efi.media.stt_groq.GroqSTT: если задан, голосовые
    и видео-кружки транскрибируются им в первую очередь (см. `_transcribe_audio`),
    с откатом на LLMRouter, если Groq не настроен или вернул пустой результат.
    """

    def __init__(
        self,
        manager: NotificationManager,
        telegram_settings: TelegramSettings,
        router: LLMRouter,
        media_cache_dir: Path,
        humanizer_settings: HumanizerSettings,
        typing_tracker: TypingTracker | None,
        *,
        activity_recorder: Any | None = None,
        affinity_recorder: Any | None = None,
        curiosity_recorder: Any | None = None,
        organic_ping_recorder: Any | None = None,
        stt: GroqSTT | None = None,
        read_receipt_sender: ReadReceiptSender | None = None,
    ) -> None:
        self._manager = manager
        self._telegram_settings = telegram_settings
        self._router = router
        self._media_cache_dir = media_cache_dir
        self._activity_recorder = activity_recorder
        self._affinity_recorder = affinity_recorder
        self._curiosity_recorder = curiosity_recorder
        self._organic_ping_recorder = organic_ping_recorder
        self._stt = stt
        self._read_receipt_sender = read_receipt_sender
        self._debouncer: MessageDebouncer[_PendingMessage] = MessageDebouncer(
            self._flush_debounced,
            typing_tracker=typing_tracker,
            post_typing_delay_range=(
                humanizer_settings.debounce_post_typing_min_seconds,
                humanizer_settings.debounce_post_typing_max_seconds,
            ),
            typing_poll_interval_seconds=humanizer_settings.debounce_typing_poll_interval_seconds,
            fallback_delay_seconds=humanizer_settings.debounce_fallback_delay_seconds,
            max_wait_seconds=humanizer_settings.debounce_max_wait_seconds,
        )

    def register(self, client: Client) -> None:
        """Регистрирует все обработчики на клиенте. Вызывается один раз при сборке приложения (efi/app.py)."""
        own_messages_filter = ~filters.bot & ~filters.me
        client.add_handler(_make_handler(filters.text & own_messages_filter, self._handle_text))
        client.add_handler(_make_handler(filters.photo & own_messages_filter, self._handle_photo))
        client.add_handler(_make_handler(filters.voice & own_messages_filter, self._handle_voice))
        client.add_handler(_make_handler(filters.video_note & own_messages_filter, self._handle_video_note))

    async def flush_pending(self) -> None:
        """Принудительно сбрасывает все накопленные в дебаунсере сообщения. Вызывается при graceful shutdown (efi/app.py)."""
        await self._debouncer.flush_all()

    async def _handle_text(self, client: Client, message: PyrogramMessage) -> None:
        access_info = await self._authorize(client, message)
        if access_info is None:
            return
        await self._dispatch_user_message(access_info, message, text=message.text or "")

    async def _handle_photo(self, client: Client, message: PyrogramMessage) -> None:
        access_info = await self._authorize(client, message)
        if access_info is None:
            return

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
        await self._dispatch_user_message(access_info, message, text=text, payload={"media_type": "photo"})

    async def _handle_voice(self, client: Client, message: PyrogramMessage) -> None:
        access_info = await self._authorize(client, message)
        if access_info is None:
            return

        downloaded_path = await self._download_media(client, message)
        transcript = await self._transcribe_audio(
            downloaded_path,
            fallback_transcriber=transcribe_voice_message,
            missing_file_placeholder="[голосовое сообщение — не удалось загрузить файл]",
        )
        await self._dispatch_user_message(
            access_info, message, text=f"[прислал(а) голосовое] {transcript}", payload={"media_type": "voice"}
        )

    async def _handle_video_note(self, client: Client, message: PyrogramMessage) -> None:
        access_info = await self._authorize(client, message)
        if access_info is None:
            return

        downloaded_path = await self._download_media(client, message)
        transcript = await self._transcribe_audio(
            downloaded_path,
            fallback_transcriber=transcribe_video_note,
            missing_file_placeholder="[видео-кружок — не удалось загрузить файл]",
        )
        await self._dispatch_user_message(
            access_info, message, text=f"[прислал(а) видео-кружок] {transcript}", payload={"media_type": "video_note"}
        )

    async def _transcribe_audio(
        self,
        downloaded_path: Path | None,
        *,
        fallback_transcriber: _AudioTranscriber,
        missing_file_placeholder: str,
    ) -> str:
        """
        Единая точка STT для голосовых и видео-кружков: если настроен прямой
        Groq-клиент — используем его первым (быстрее и дешевле общего
        VISION-роута LLMRouter); при отсутствии настройки ИЛИ пустом
        результате от Groq — откатываемся на `fallback_transcriber` (через
        LLMRouter), чтобы распознавание речи не переставало работать целиком
        из-за того, что для Groq STT не задан ключ или конкретный запрос к
        нему не удался.
        """
        if downloaded_path is None:
            return missing_file_placeholder

        if self._stt is not None:
            text = await self._stt.transcribe(downloaded_path)
            if text:
                return text
            logger.info(
                "telegram: Groq STT gave no text for %s, falling back to LLMRouter transcription", downloaded_path
            )

        return await fallback_transcriber(self._router, downloaded_path)

    async def _download_media(self, client: Client, message: PyrogramMessage) -> Path | None:
        try:
            result = await client.download_media(message, file_name=f"{self._media_cache_dir}/")
        except Exception:
            logger.exception("telegram: failed to download media for message_id=%s", message.id)
            return None
        return Path(result) if result else None

    async def _authorize(self, client: Client, message: PyrogramMessage) -> ChatAccessInfo | None:
        """
        Единая точка допуска сообщения к обработке — см. докстринг модуля.
        Возвращает готовый ChatAccessInfo, если сообщение разрешено к
        обработке, иначе None (дальше вызывающая сторона просто выходит,
        не тратясь на скачивание медиа/STT/VISION).
        """
        if message.from_user is None or message.chat is None:
            return None  # анонимные админы, каналы и т.п. — вне текущего скоупа

        access_info = ChatAccessInfo(
            chat_id=message.chat.id,
            is_owner=message.from_user.id == self._telegram_settings.owner_id,
            is_contact=bool(getattr(message.from_user, "is_contact", False)),
            is_private_chat=message.chat.type == ChatType.PRIVATE,
        )
        allowed, reason = is_chat_accessible(access_info, self._telegram_settings)
        if not allowed:
            logger.debug("telegram: message from chat_id=%s dropped (%s)", access_info.chat_id, reason)
            return None

        if message.chat.type in _GROUP_CHAT_TYPES and not _is_addressed_to_bot(message, client):
            logger.debug(
                "telegram: group message from chat_id=%s ignored (no mention/reply to the bot)", access_info.chat_id
            )
            return None

        return access_info

    async def _dispatch_user_message(
        self,
        access_info: ChatAccessInfo,
        message: PyrogramMessage,
        *,
        text: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        """
        Отмечает активность/близость/любопытство, дальше НЕ кладёт
        Notification в очередь напрямую — передаёт в MessageDebouncer. Сама
        постановка в NotificationManager происходит позже, в
        _flush_debounced(), когда пройдёт пауза тишины (или сработает
        потолок max_wait_seconds).
        """
        if self._activity_recorder is not None:
            self._activity_recorder.record_activity(access_info.chat_id)

        if self._affinity_recorder is not None:
            await self._affinity_recorder.record_message(access_info.chat_id, text)

        if self._curiosity_recorder is not None:
            await self._curiosity_recorder.consider_message(access_info.chat_id, text)

        if self._organic_ping_recorder is not None:
            await self._organic_ping_recorder.handle_reply(access_info.chat_id)

        if self._read_receipt_sender is not None:
            await self._read_receipt_sender.mark_as_read(access_info.chat_id)

        await self._debouncer.add(
            access_info.chat_id,
            _PendingMessage(message=message, text=text, payload=payload or {}),
        )

    async def _flush_debounced(self, chat_id: int, items: list[_PendingMessage]) -> None:
        """
        Callback дебаунсера: несколько накопленных сообщений одного
        собеседника превращаются в один Notification. Имя отправителя и
        reply-контекст (formatting.format_user_message) берутся из ПОСЛЕДНЕГО
        сообщения пачки — это самый свежий контекст на момент реакции.
        """
        last_message = items[-1].message
        combined_text = "\n".join(item.text for item in items)

        merged_payload: dict[str, Any] = {"telegram_message_ids": [item.message.id for item in items]}
        for item in items:
            merged_payload.update(item.payload)
        merged_payload.update(_build_chat_context(last_message))
        merged_payload["sender_is_owner"] = (
            last_message.from_user is not None and last_message.from_user.id == self._telegram_settings.owner_id
        )

        notification = Notification(
            type=NotificationType.USER_MESSAGE,
            priority=0,  # входящее сообщение пользователя — наивысший приоритет обработки
            chat_id=chat_id,
            message=format_user_message(last_message, combined_text),
            payload=merged_payload,
        )
        await self._manager.put(notification)


def _is_addressed_to_bot(message: PyrogramMessage, client: Client) -> bool:
    """
    В группах/супергруппах Эфи отвечает ТОЛЬКО когда к ней обращаются
    напрямую — иначе она реагировала бы на каждую реплику в чате, что для
    юзербота выглядит как спам, а не как участие в разговоре.

    Два независимых признака адресности, проверяются оба:
        - `message.mentioned` — флаг из самого Telegram (явное @username-
          упоминание и некоторые reply-случаи);
        - явный reply на прошлое сообщение самой Эфи, сверенный по id
          отправителя через `client.me` — не полагается на то, корректно ли
          Telegram проставил `mentioned`, поэтому надёжнее держать обе
          проверки, а не только одну.
    """
    if bool(message.mentioned):
        return True

    reply = message.reply_to_message
    me = client.me
    return reply is not None and reply.from_user is not None and me is not None and reply.from_user.id == me.id


def _build_chat_context(message: PyrogramMessage) -> dict[str, Any]:
    """
    Тип/название чата и имя отправителя — попадает в Notification.payload,
    откуда его читает EfiSystemPromptBuilder: тип чата — чтобы сообщить
    модели, что она не всегда в приватной переписке один на один, имя
    отправителя — как источник {user_name} для шаблона personality.md,
    когда пишет владелец (см. efi/prompts/builder.py).
    """
    chat = message.chat
    sender_name = message.from_user.first_name if message.from_user else None
    return {
        "chat_type": chat.type.name if chat.type else None,
        "chat_title": chat.title,
        "sender_name": sender_name,
    }


def _make_handler(message_filter: filters.Filter, callback: _HandlerCallback) -> MessageHandler:
    return MessageHandler(callback, message_filter)


__all__ = ["TelegramEventHandlers"]
