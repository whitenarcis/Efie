"""
efi/telegram/formatting.py

Форматирование объектов Pyrogram в текст, понятный LLM. Вынесено из
efi/telegram/handlers.py в отдельный модуль, чтобы форматирование (что
именно и как показывать модели) можно было менять и тестировать отдельно от
маршрутизации событий в NotificationManager.
"""

from __future__ import annotations

from pyrogram.types import Message as PyrogramMessage

from efi.security.sanitize import sanitize_display_name, sanitize_text

_REPLY_PREVIEW_MAX_LENGTH = 200


def format_sender_name(message: PyrogramMessage) -> str:
    """Санитайзированное отображаемое имя отправителя (имя, а не username — обычно естественнее для диалога)."""
    if message.from_user is None:
        return "неизвестный собеседник"
    return sanitize_display_name(message.from_user.first_name or "собеседник")


def format_reply_context(message: PyrogramMessage) -> str:
    """
    Короткий контекст "в ответ на" — если сообщение является ответом на
    другое, добавляет краткую цитату, чтобы модель понимала связь реплик,
    даже если отвеченное сообщение уже выпало из последних N сообщений истории.

    Отдельно отмечает случай, когда собеседник свайпнул именно ЕЁ СОБСТВЕННОЕ
    сообщение (Pyrogram User.is_self у отвеченного сообщения) — это не то же
    самое, что ответ кому-то ещё в групповом чате, и должно читаться моделью
    иначе ("ты сказала это, а он ответил конкретно на ЭТО", а не просто
    "какая-то реплика где-то рядом по смыслу").
    """
    replied = message.reply_to_message
    if replied is None:
        return ""
    replied_text = (replied.text or replied.caption or "").strip()
    if not replied_text:
        return ""
    preview = sanitize_text(replied_text)[:_REPLY_PREVIEW_MAX_LENGTH]

    replied_from_self = bool(replied.from_user and getattr(replied.from_user, "is_self", False))
    if replied_from_self:
        return f' (это ответ конкретно на ТВОЁ сообщение: "{preview}")'
    return f' (в ответ на: "{preview}")'


def format_user_message(message: PyrogramMessage, body_text: str) -> str:
    """
    Собирает итоговую строку для Notification.message: имя отправителя +
    контекст ответа + санитайзированный текст. Единая точка форматирования —
    используется efi.telegram.handlers для всех типов входящих событий
    (текст, фото, голос, видео-кружок; для медиа body_text уже содержит
    готовое описание/транскрипцию из efi/telegram/media/).
    """
    sender_name = format_sender_name(message)
    reply_context = format_reply_context(message)
    return f"{sender_name}{reply_context}: {sanitize_text(body_text)}"


__all__ = ["format_sender_name", "format_reply_context", "format_user_message"]
