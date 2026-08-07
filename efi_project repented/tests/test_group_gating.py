"""Тесты для efi.telegram.handlers._is_addressed_to_bot — фикс групповых ответов."""

from __future__ import annotations

from types import SimpleNamespace
from typing import cast

from pyrogram import Client
from pyrogram.types import Message, User

from efi.telegram.handlers import _is_addressed_to_bot

_BOT_ID = 111
_OTHER_ID = 222


def _fake_client(*, me_id: int | None = _BOT_ID) -> Client:
    me = User(id=me_id) if me_id is not None else None
    return cast(Client, SimpleNamespace(me=me))


def test_mentioned_flag_is_addressed() -> None:
    message = Message(id=1, mentioned=True)
    assert _is_addressed_to_bot(message, _fake_client()) is True


def test_reply_to_bot_message_is_addressed() -> None:
    reply = Message(id=1, from_user=User(id=_BOT_ID))
    message = Message(id=2, mentioned=False, reply_to_message=reply)
    assert _is_addressed_to_bot(message, _fake_client()) is True


def test_reply_to_other_user_is_not_addressed() -> None:
    reply = Message(id=1, from_user=User(id=_OTHER_ID))
    message = Message(id=2, mentioned=False, reply_to_message=reply)
    assert _is_addressed_to_bot(message, _fake_client()) is False


def test_plain_group_message_is_not_addressed() -> None:
    message = Message(id=1, mentioned=False)
    assert _is_addressed_to_bot(message, _fake_client()) is False


def test_reply_without_from_user_is_not_addressed() -> None:
    reply = Message(id=1, from_user=None)
    message = Message(id=2, mentioned=False, reply_to_message=reply)
    assert _is_addressed_to_bot(message, _fake_client()) is False


def test_reply_to_bot_when_client_me_unavailable_is_not_addressed() -> None:
    reply = Message(id=1, from_user=User(id=_BOT_ID))
    message = Message(id=2, mentioned=False, reply_to_message=reply)
    assert _is_addressed_to_bot(message, _fake_client(me_id=None)) is False
