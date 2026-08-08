"""
Тесты на два взаимосвязанных дефекта, из-за которых Эфи фактически не могла
пользоваться стикерами/реакциями:

1. efi.telegram.handlers._build_sticker_text_and_payload — раньше входящие
   стикеры вообще не обрабатывались (не было обработчика на filters.sticker),
   поэтому ни в истории, ни в контексте модели о них не оставалось следа, и
   у модели не было ни одного известного ей file_id для send_sticker.
2. efi.tools.telegram_actions.react_with_emoji.ReactWithEmojiTool — раньше
   требовал message_id ПАРАМЕТРОМ от модели, а взять его было неоткуда (ни
   один message_id нигде не показывается в тексте промпта), из-за чего
   инструмент был фактически недоступен для реального использования.
"""

from __future__ import annotations

from pyrogram.types import Message, Sticker

from efi.notifications.schemas import Notification, NotificationType
from efi.telegram.handlers import _build_sticker_text_and_payload
from efi.tools.base import ToolContext
from efi.tools.telegram_actions.react_with_emoji import ReactWithEmojiTool


def _sticker(*, emoji: str | None = "😂", file_id: str | None = "AAAA_file_id") -> Sticker:
    return Sticker(
        file_id=file_id or "",
        file_unique_id="unique",
        width=512,
        height=512,
        is_animated=False,
        is_video=False,
        emoji=emoji,
    )


def test_sticker_text_includes_emoji_and_file_id() -> None:
    message = Message(id=1, sticker=_sticker(emoji="😂", file_id="AAAA_file_id"))
    text, payload = _build_sticker_text_and_payload(message)

    assert "😂" in text
    assert "AAAA_file_id" in text
    assert payload["media_type"] == "sticker"
    assert payload["sticker_file_id"] == "AAAA_file_id"


def test_sticker_text_falls_back_when_no_emoji() -> None:
    message = Message(id=1, sticker=_sticker(emoji=None))
    text, _payload = _build_sticker_text_and_payload(message)
    assert "?" in text


def test_sticker_payload_omits_file_id_when_missing() -> None:
    message = Message(id=1, sticker=_sticker(file_id=None))
    _text, payload = _build_sticker_text_and_payload(message)
    assert "sticker_file_id" not in payload


class _FakeReactor:
    def __init__(self) -> None:
        self.calls: list[tuple[int, int, str]] = []

    async def react(self, chat_id: int, message_id: int, emoji: str) -> None:
        self.calls.append((chat_id, message_id, emoji))


def _context(*, message_ids: list[int] | None) -> ToolContext:
    payload = {"telegram_message_ids": message_ids} if message_ids is not None else {}
    return ToolContext(
        notification=Notification(type=NotificationType.USER_MESSAGE, chat_id=42, message="x", payload=payload)
    )


async def test_react_resolves_message_id_from_context_without_model_input() -> None:
    reactor = _FakeReactor()
    tool = ReactWithEmojiTool(reactor)
    context = _context(message_ids=[10, 11, 12])

    result = await tool.execute({"emoji": "❤️"}, context)

    assert "Реакция поставлена" in result
    assert reactor.calls == [(42, 12, "❤️")]  # реагирует на ПОСЛЕДНЕЕ сообщение пачки


async def test_react_is_unavailable_without_a_current_message() -> None:
    tool = ReactWithEmojiTool(_FakeReactor())
    context = _context(message_ids=None)
    assert tool.is_available(context) is False


async def test_react_reports_error_without_a_current_message() -> None:
    tool = ReactWithEmojiTool(_FakeReactor())
    context = _context(message_ids=None)
    result = await tool.execute({"emoji": "❤️"}, context)
    assert result.startswith("error:")


async def test_react_requires_non_empty_emoji() -> None:
    tool = ReactWithEmojiTool(_FakeReactor())
    context = _context(message_ids=[1])
    result = await tool.execute({"emoji": ""}, context)
    assert result.startswith("error:")
