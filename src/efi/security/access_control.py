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
from enum import StrEnum

from efi.config.schema import LockdownMode, TelegramSettings


@dataclass(slots=True, frozen=True)
class ChatAccessInfo:
    """Всё, что нужно знать про чат, чтобы принять решение о доступе."""

    chat_id: int
    is_owner: bool
    is_contact: bool
    is_private_chat: bool


class AccessDeniedReason(StrEnum):
    """Причина отказа — для логирования/дебага. НИКОГДА не должна попадать в текст, который видит пользователь."""

    LOCKDOWN_OWNER_ONLY = "lockdown_owner_only"
    LOCKDOWN_CONTACTS_ONLY = "lockdown_contacts_only"

    def hint(self) -> str:
        """Что именно поменять в конфигурации, чтобы отказа не было."""
        return _DENIAL_HINTS[self]


_DENIAL_HINTS: dict[AccessDeniedReason, str] = {
    AccessDeniedReason.LOCKDOWN_OWNER_ONLY: (
        'telegram.lockdown_mode = "owner_only" — отвечает только владельцу. Для ответов остальным поставьте '
        '"contacts_only" (только контакты аккаунта) или "none" (всем, включая незнакомых)'
    ),
    AccessDeniedReason.LOCKDOWN_CONTACTS_ONLY: (
        'telegram.lockdown_mode = "contacts_only" — отвечает только контактам аккаунта. Добавьте человека в '
        'контакты, впишите его чат в telegram.allowed_chats или поставьте lockdown_mode = "none"'
    ),
}


def is_chat_accessible(
    chat: ChatAccessInfo, telegram_settings: TelegramSettings
) -> tuple[bool, AccessDeniedReason | None]:
    """
    Решает, разрешено ли реагировать на событие в данном чате.

    Владелец (`telegram_settings.owner_id`, отражённый в `chat.is_owner`)
    всегда проходит проверку, независимо от режима Lockdown и allowlist'а —
    Lockdown ограничивает доступ ДЛЯ ДРУГИХ, не для владельца.

    Возвращает (allowed, reason); reason заполнен только при отказе.
    """
    if chat.is_owner:
        return True, None

    # Оба списка — ЯВНЫЙ opt-in владельца, и оба РАЗРЕШАЮТ, а не запрещают:
    #   community_chats — где Эфи участвует как обычный участник сообщества;
    #   allowed_chats   — «свои» чаты помимо лички владельца.
    # Перечисленное здесь проходит независимо от lockdown; всё остальное
    # решает сам lockdown ниже.
    if chat.chat_id in telegram_settings.community_chats:
        return True, None
    if chat.chat_id in telegram_settings.allowed_chats:
        return True, None

    # Раньше здесь была ещё одна проверка: непустой `allowed_chats` работал
    # как ЧЁРНЫЙ список — «есть список, и тебя в нём нет, значит молчим», —
    # причём ПОСЛЕ проверки режима. Это противоречило и названию поля
    # («allowlist ... помимо владельца»), и описанию режима NONE («публичный
    # режим, отвечает в любом чате»): выставив none, владелец всё равно не
    # получал ответов посторонним, пока список был непуст.
    #
    # На практике это стало тупиком: с тех пор как `allowed_chats` задаёт
    # ещё и чаты, где Эфи вправе писать первой (см. ConversationLifecycle.
    # allows_proactive_ping_to_chat), заполнить его было ОБЯЗАТЕЛЬНО ради
    # проактивности — и тем самым намертво закрывались все ЛС посторонних.
    # Одновременно и то и другое было недостижимо в принципе.
    if telegram_settings.lockdown_mode is LockdownMode.OWNER_ONLY:
        return False, AccessDeniedReason.LOCKDOWN_OWNER_ONLY

    if telegram_settings.lockdown_mode is LockdownMode.CONTACTS_ONLY and not chat.is_contact:
        return False, AccessDeniedReason.LOCKDOWN_CONTACTS_ONLY

    return True, None


def describe_access_policy(telegram_settings: TelegramSettings) -> str:
    """
    Одна строка о том, кто фактически может с ней говорить.

    Пишется в лог при старте. Причина простая: «почему она не отвечает» —
    вопрос про сочетание трёх настроек, и выяснять его по конфигу вручную
    неудобно ровно в тот момент, когда что-то не работает.
    """
    mode = telegram_settings.lockdown_mode
    if mode is LockdownMode.OWNER_ONLY:
        base = "только владелец"
    elif mode is LockdownMode.CONTACTS_ONLY:
        base = "владелец и контакты аккаунта"
    else:
        base = "владелец и кто угодно, включая незнакомых в ЛС"

    extras: list[str] = []
    if telegram_settings.allowed_chats:
        extras.append(f"+{len(telegram_settings.allowed_chats)} чат(ов) из allowed_chats")
    if telegram_settings.community_chats:
        extras.append(f"+{len(telegram_settings.community_chats)} канал(ов) сообщества")
    suffix = f" ({', '.join(extras)})" if extras else ""
    return f"{base}{suffix}"


__all__ = ["ChatAccessInfo", "AccessDeniedReason", "describe_access_policy", "is_chat_accessible"]
