"""
efi/security/access_control.py

Проверка прав доступа: решает, может ли Эфи реагировать на событие в данном
чате, исходя из режима Lockdown (efi.config.schema.LockdownMode) и allowlist'а.
Прямой аналог util/is_accessible_from_lockdown.h у референса.

Чистая функция без I/O — источник данных о чате (владелец ли, контакт ли)
собирается заранее вызывающей стороной (efi/telegram/, ещё не реализован) и
передаётся сюда как готовый ChatAccessInfo.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from efi.config.schema import LockdownMode, TelegramSettings


@dataclass(slots=True, frozen=True)
class ChatAccessInfo:
    """Всё, что нужно знать про чат, чтобы принять решение о доступе."""

    chat_id: int
    is_owner: bool
    is_contact: bool
    is_private_chat: bool


class AccessDeniedReason(str, Enum):
    """Причина отказа — для логирования/дебага. НИКОГДА не должна попадать в текст, который видит пользователь."""

    LOCKDOWN_OWNER_ONLY = "lockdown_owner_only"
    LOCKDOWN_CONTACTS_ONLY = "lockdown_contacts_only"
    NOT_IN_ALLOWLIST = "not_in_allowlist"


def is_chat_accessible(chat: ChatAccessInfo, telegram_settings: TelegramSettings) -> tuple[bool, AccessDeniedReason | None]:
    """
    Решает, разрешено ли реагировать на событие в данном чате.

    Владелец (`telegram_settings.owner_id`, отражённый в `chat.is_owner`)
    всегда проходит проверку, независимо от режима Lockdown и allowlist'а —
    Lockdown ограничивает доступ ДЛЯ ДРУГИХ, не для владельца.

    Возвращает (allowed, reason); reason заполнен только при отказе.
    """
    if chat.is_owner:
        return True, None

    # Чаты сообщества — ЯВНЫЙ opt-in владельца (telegram.community_chats):
    # он сам перечислил, где Эфи участвует как обычный участник, поэтому
    # lockdown их не закрывает. Всё, чего в этом списке нет, lockdown
    # закрывает как раньше — список сужает исключение, а не отменяет режим.
    if chat.chat_id in telegram_settings.community_chats:
        return True, None

    if telegram_settings.lockdown_mode is LockdownMode.OWNER_ONLY:
        return False, AccessDeniedReason.LOCKDOWN_OWNER_ONLY

    if telegram_settings.lockdown_mode is LockdownMode.CONTACTS_ONLY and not chat.is_contact:
        return False, AccessDeniedReason.LOCKDOWN_CONTACTS_ONLY

    if telegram_settings.allowed_chats and chat.chat_id not in telegram_settings.allowed_chats:
        return False, AccessDeniedReason.NOT_IN_ALLOWLIST

    return True, None


__all__ = ["ChatAccessInfo", "AccessDeniedReason", "is_chat_accessible"]
