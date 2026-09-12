"""
Тесты на обработку стикеров и эмодзи-реакций:

1. efi.telegram.handlers._format_sticker_message — входящий стикер доходит до
   модели КАК ОПИСАНИЕ от vision с коротким id («[стикер: кот смеётся (id 7)]»),
   а не как file_id+эмодзи. Telegram-иды остаются только в payload, никогда в
   тексте промпта.
2. efi.tools.telegram_actions.SendStickerTool — отправка по короткому id из
   кэша описаний (efi/db/sticker_descriptions.py): инструмент сам достаёт
   актуальный file_id, модель ничего про Telegram-иды не знает.
3. efi.tools.telegram_actions.react_with_emoji.ReactWithEmojiTool — раньше
   требовал message_id ПАРАМЕТРОМ от модели, а взять его было неоткуда (ни
   один message_id нигде не показывается в тексте промпта), из-за чего
   инструмент был фактически недоступен для реального использования.
"""

from __future__ import annotations

from efi.notifications.schemas import Notification, NotificationType
from efi.telegram.handlers import _format_sticker_message
from efi.tools.base import ToolContext
from efi.tools.telegram_actions.react_with_emoji import (
    _ALLOWED_REACTIONS,
    ReactWithEmojiTool,
    normalize_reaction_emoji,
)
from efi.tools.telegram_actions.send_message import SendMessageTool
from efi.tools.telegram_actions.stickers import SendStickerTool


def test_sticker_text_is_a_vision_description_with_short_id() -> None:
    text, payload = _format_sticker_message(
        description="кот смеётся", sticker_id=7, file_unique_id="U1", file_id="F1", emoji="😂"
    )

    assert text == "[прислал(а) стикер: кот смеётся (id 7)]"
    # Никаких Telegram-идов в тексте: у модели их нет и быть не должно.
    assert "F1" not in text
    assert "U1" not in text
    assert payload["media_type"] == "sticker"
    assert payload["sticker_id"] == 7
    assert payload["sticker_file_id"] == "F1"
    assert payload["sticker_file_unique_id"] == "U1"


def test_sticker_without_id_keeps_description_only() -> None:
    text, payload = _format_sticker_message(description="озабоченный пёс")

    assert text == "[прислал(а) стикер: озабоченный пёс]"
    assert "sticker_id" not in payload


def test_sticker_falls_back_to_emoji_when_undescribed() -> None:
    """Vision не сработал — остаётся эмодзи как последняя доступная информация, честно помеченная."""
    text, payload = _format_sticker_message(description=None, emoji="😂", file_id="F1")

    assert "не удалось распознать" in text
    assert "😂" in text
    assert payload["sticker_file_id"] == "F1"
    assert "sticker_id" not in payload


def test_animated_sticker_fallback_note() -> None:
    text, _payload = _format_sticker_message(description=None, emoji="🔥", degraded_note="анимированный стикер")
    assert "анимированный стикер" in text
    assert "🔥" in text


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
    # Реагирует на ПОСЛЕДНЕЕ сообщение пачки, и эмодзи уходит уже каноничным
    # (без U+FE0F) — иначе Telegram примет вызов молча, не поставив ничего.
    assert reactor.calls == [(42, 12, "❤")]


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


# -- нормализация эмодзи под набор реакций Telegram --------------------------------
#
# Регрессия ровно с тем симптомом, который её и скрывал: MTProto-вызов с
# нештатным эмодзи проходит без ошибки, Pyrogram возвращает True, в логах
# честное "реакция поставлена" — а в чате не появляется ничего.


def test_variation_selector_is_stripped() -> None:
    """'❤️' (U+2764 U+FE0F) — то, что пишет модель; в списке реакций лежит '❤' (U+2764)."""
    assert normalize_reaction_emoji("❤️") == "❤"


def test_skin_tone_is_stripped() -> None:
    assert normalize_reaction_emoji("👍🏻") == "👍"


def test_plain_reaction_passes_through() -> None:
    assert normalize_reaction_emoji("🔥") == "🔥"


def test_composite_reactions_survive_normalization() -> None:
    """ZWJ трогать нельзя: в наборе есть составные реакции, разбор по частям их уничтожит."""
    assert normalize_reaction_emoji("❤‍🔥") == "❤‍🔥"
    assert normalize_reaction_emoji("🤷‍♂️") == "🤷‍♂"


def test_non_standard_emoji_is_rejected() -> None:
    assert normalize_reaction_emoji("🫠") is None
    assert normalize_reaction_emoji("не эмодзи") is None


async def test_unsupported_emoji_never_reaches_the_network() -> None:
    """
    Ключевое: молчаливый no-op заменён внятной ошибкой, по которой модель
    может выбрать другую реакцию, — вместо доклада об успехе без результата.
    """
    reactor = _FakeReactor()
    tool = ReactWithEmojiTool(reactor)

    result = await tool.execute({"emoji": "🫠"}, _context(message_ids=[7]))

    assert result.startswith("error:")
    assert reactor.calls == []


def test_every_allowed_reaction_survives_its_own_normalization() -> None:
    """
    Защита от опечатки в самом списке: запись с U+FE0F или тоном кожи внутри
    стала бы недостижимой — нормализация превращала бы её в другую строку, и
    штатная реакция молча отвергалась бы как нештатная.
    """
    assert [item for item in _ALLOWED_REACTIONS if normalize_reaction_emoji(item) != item] == []


async def test_rejection_suggests_valid_alternatives() -> None:
    tool = ReactWithEmojiTool(_FakeReactor())
    result = await tool.execute({"emoji": "🫠"}, _context(message_ids=[7]))
    assert "👍" in result and "🔥" in result


# -- обязательный reply для публичного комментария --------------------------------
#
# Комментарий под постом в канале — это РЕПЛАЙ на экземпляр поста в группе
# обсуждения (см. efi/telegram/comments.py). Свободное сообщение в ту же группу
# комментарием под постом не становится, поэтому решать, ставить ли реплай, здесь
# нельзя оставлять на усмотрение модели — за неё это делает payload.


class _RecordingSender:
    """Запоминает, с каким reply_to_message_id её позвали."""

    def __init__(self) -> None:
        self.reply_to_message_id: int | None = None

    async def send_message(self, chat_id: int, text: str, **kwargs: object) -> None:
        self.reply_to_message_id = kwargs.get("reply_to_message_id")  # type: ignore[assignment]
        on_bubble_sent = kwargs.get("on_bubble_sent")
        if callable(on_bubble_sent):
            on_bubble_sent(text)


def _send_context(**payload: object) -> ToolContext:
    return ToolContext(
        notification=Notification(
            type=NotificationType.PUBLIC_COMMENT,
            priority=7,
            chat_id=-1002,
            message="повод",
            payload={"telegram_message_ids": [77], **payload},
        )
    )


async def test_force_reply_makes_the_comment_a_reply_without_the_model_asking() -> None:
    sender = _RecordingSender()
    tool = SendMessageTool(sender)  # type: ignore[arg-type]

    await tool.execute({"text": "интересная мысль"}, _send_context(force_reply=True))

    assert sender.reply_to_message_id == 77


async def test_without_force_reply_the_model_still_decides() -> None:
    sender = _RecordingSender()
    tool = SendMessageTool(sender)  # type: ignore[arg-type]

    await tool.execute({"text": "просто реплика"}, _send_context())

    assert sender.reply_to_message_id is None


# -- send_sticker по короткому id из кэша описаний --------------------------
# Модель ничего не знает про Telegram-иды: ей виден «id 7» из описания
# входящего стикера или из блока «[Известные стикеры]», а инструмент сам
# достаёт из кэша efi/db/sticker_descriptions.py актуальный file_id.


class _FakeStickerSender:
    def __init__(self) -> None:
        self.calls: list[tuple[int, str]] = []

    async def send_sticker(self, chat_id: int, sticker_file_id: str) -> None:
        self.calls.append((chat_id, sticker_file_id))


class _FakeStickerStore:
    """Дак-тайпинг StickerDescriptionStore: только то, что нужно инструменту."""

    def __init__(self, known: dict[int, object] | None = None) -> None:
        self.known: dict[int, object] = known or {}

    async def by_id(self, sticker_id: int) -> object | None:
        return self.known.get(sticker_id)

    async def recent(self, limit: int) -> list[object]:
        items = list(self.known.items())[:limit]
        return [item[1] for item in items]


def _sticker_context() -> ToolContext:
    return ToolContext(notification=Notification(type=NotificationType.USER_MESSAGE, chat_id=42, message="x"))


async def test_send_sticker_resolves_short_id_to_file_id() -> None:
    sender = _FakeStickerSender()
    store = _FakeStickerStore({7: _Known(7, "F7")})
    tool = SendStickerTool(sender, store)  # type: ignore[arg-type]

    result = await tool.execute({"sticker_id": 7}, _sticker_context())

    assert "Стикер отправлен" in result
    assert sender.calls == [(42, "F7")]


class _Known:
    """Минимальный объект записи: инструменту нужны только sticker_id и file_id."""

    def __init__(self, sticker_id: int, file_id: str) -> None:
        self.sticker_id = sticker_id
        self.file_id = file_id


def _fake_known(id_: int) -> _Known:
    return _Known(id_, f"FILE_{id_}")


async def test_send_sticker_rejects_unknown_id_with_hint() -> None:
    sender = _FakeStickerSender()
    store = _FakeStickerStore({5: _fake_known(5)})
    tool = SendStickerTool(sender, store)  # type: ignore[arg-type]

    result = await tool.execute({"sticker_id": 99}, _sticker_context())

    assert result.startswith("error:")
    assert "99" in result
    # Подсказка должна перечислить реально известные id, чтобы модель
    # могла выбрать правильный, а не гадать.
    assert "5" in result
    assert sender.calls == []


async def test_send_sticker_is_unavailable_without_a_current_chat() -> None:
    tool = SendStickerTool(_FakeStickerSender(), _FakeStickerStore({1: _fake_known(1)}))  # type: ignore[arg-type]
    context = ToolContext(notification=Notification(type=NotificationType.USER_MESSAGE, chat_id=None, message="x"))
    assert not tool.is_available(context)


async def test_send_sticker_rejects_non_integer_id() -> None:
    tool = SendStickerTool(_FakeStickerSender(), _FakeStickerStore())  # type: ignore[arg-type]
    result = await tool.execute({"sticker_id": "7"}, _sticker_context())
    assert result.startswith("error:")
