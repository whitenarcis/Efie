"""
Тесты допуска сообщений (efi.security.access_control).

Регрессия на тупик: `allowed_chats` работал как ЧЁРНЫЙ список — «список
непуст, тебя в нём нет, значит молчим», — причём проверялся ПОСЛЕ режима
lockdown. Это противоречило и названию поля («allowlist ... помимо
владельца»), и описанию режима NONE («публичный режим, отвечает в любом
чате»).

Практическое следствие было хуже теоретического: с тех пор как
`allowed_chats` задаёт ещё и чаты, где Эфи вправе писать первой, заполнить
его стало обязательно ради проактивности — и это намертво закрывало все ЛС
посторонних. «Пишет первой в своих чатах» и «отвечает незнакомым в ЛС»
оказались взаимоисключающими, хотя ни одна настройка об этом не сообщала.
"""

from __future__ import annotations

import pytest

from efi.config.schema import LockdownMode, TelegramSettings
from efi.security.access_control import (
    AccessDeniedReason,
    ChatAccessInfo,
    describe_access_policy,
    is_chat_accessible,
)

_OWNER_ID = 2129889949
_MY_GROUP = -4404120219
_STRANGER_DM = 777123
_COMMUNITY = -1001234


def _settings(**overrides: object) -> TelegramSettings:
    defaults: dict[str, object] = {
        "api_id": 1,
        "api_hash": "x",
        "owner_id": _OWNER_ID,
        "lockdown_mode": LockdownMode.OWNER_ONLY,
    }
    defaults.update(overrides)
    return TelegramSettings(**defaults)  # type: ignore[arg-type]


def _dm(chat_id: int = _STRANGER_DM, *, is_contact: bool = False) -> ChatAccessInfo:
    return ChatAccessInfo(chat_id=chat_id, is_owner=False, is_contact=is_contact, is_private_chat=True)


# -- сам тупик ---------------------------------------------------------------


def test_stranger_dm_passes_in_open_mode_even_with_allowed_chats_filled() -> None:
    """
    Главная регрессия: раньше здесь был отказ NOT_IN_ALLOWLIST, и включить
    ответы посторонним было невозможно, не сломав проактивность.
    """
    settings = _settings(lockdown_mode=LockdownMode.NONE, allowed_chats=[_OWNER_ID, _MY_GROUP])

    allowed, reason = is_chat_accessible(_dm(), settings)

    assert allowed is True, "режим none обязан означать то, что написано в его описании"
    assert reason is None


def test_proactive_chats_and_open_dms_are_compatible() -> None:
    """Обе настройки должны работать одновременно — ровно этого не хватало."""
    settings = _settings(lockdown_mode=LockdownMode.NONE, allowed_chats=[_MY_GROUP])

    assert is_chat_accessible(_dm(), settings)[0] is True
    assert is_chat_accessible(_dm(_MY_GROUP), settings)[0] is True


# -- lockdown по-прежнему закрывает ------------------------------------------


def test_owner_only_blocks_strangers() -> None:
    allowed, reason = is_chat_accessible(_dm(), _settings(allowed_chats=[_MY_GROUP]))

    assert allowed is False
    assert reason is AccessDeniedReason.LOCKDOWN_OWNER_ONLY


def test_contacts_only_blocks_non_contacts_but_lets_contacts_through() -> None:
    settings = _settings(lockdown_mode=LockdownMode.CONTACTS_ONLY, allowed_chats=[_MY_GROUP])

    assert is_chat_accessible(_dm(is_contact=False), settings)[0] is False
    assert is_chat_accessible(_dm(is_contact=True), settings)[0] is True


def test_owner_always_passes_regardless_of_mode() -> None:
    owner_chat = ChatAccessInfo(chat_id=_OWNER_ID, is_owner=True, is_contact=False, is_private_chat=True)

    for mode in LockdownMode:
        assert is_chat_accessible(owner_chat, _settings(lockdown_mode=mode))[0] is True


# -- списки разрешают, а не запрещают ----------------------------------------


def test_allowed_chats_bypasses_a_closed_lockdown() -> None:
    """Владелец сам вписал этот чат в «свои» — режим не должен его закрывать."""
    settings = _settings(lockdown_mode=LockdownMode.OWNER_ONLY, allowed_chats=[_MY_GROUP])

    assert is_chat_accessible(_dm(_MY_GROUP), settings)[0] is True


def test_community_chats_still_bypass_lockdown() -> None:
    settings = _settings(community_chats=[_COMMUNITY])

    assert is_chat_accessible(_dm(_COMMUNITY), settings)[0] is True
    assert is_chat_accessible(_dm(-9999), settings)[0] is False


# -- диагностика --------------------------------------------------------------


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        (LockdownMode.OWNER_ONLY, "только владелец"),
        (LockdownMode.CONTACTS_ONLY, "контакты"),
        (LockdownMode.NONE, "кто угодно"),
    ],
)
def test_policy_description_names_who_can_talk(mode: LockdownMode, expected: str) -> None:
    """Строка в лог при старте: «почему она не отвечает» не должно требовать чтения конфига."""
    assert expected in describe_access_policy(_settings(lockdown_mode=mode))


def test_policy_description_counts_explicit_lists() -> None:
    description = describe_access_policy(_settings(allowed_chats=[1, 2], community_chats=[3]))

    assert "2 чат" in description
    assert "1 канал" in description


def test_every_denial_reason_explains_how_to_fix_it() -> None:
    """Отказ без подсказки — это ровно та тишина, из-за которой пришлось лезть в код."""
    for reason in AccessDeniedReason:
        assert "lockdown_mode" in reason.hint()
