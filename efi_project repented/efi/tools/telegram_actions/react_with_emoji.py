"""
efi/tools/telegram_actions/react_with_emoji.py

Инструмент реакции эмодзи на сообщение собеседника — часто более уместный
ответ, чем текст (например, на шутку или короткую реплику).

Реагировать можно ТОЛЬКО на сообщение(я), которые реально вызвали текущий
Notification (их telegram_message_id кладёт efi.telegram.handlers в payload
как "telegram_message_ids") — тот же принцип и то же ограничение, что и у
reply_to_current в efi.tools.telegram_actions.send_message.SendMessageTool:
history/Session не хранят исходные Telegram message_id, поэтому модель
физически не может знать id произвольного сообщения из более старой истории.
Раньше инструмент требовал message_id ПАРАМЕТРОМ от модели — но ей неоткуда
было его узнать (ни один message_id нигде не показывается в тексте промпта),
из-за чего инструмент был фактически недоступен для реального использования.
Теперь, как и у SendMessageTool, id резолвится автоматически из контекста.

ПРО НОРМАЛИЗАЦИЮ ЭМОДЗИ (вторая причина, по которой инструмент выглядел
рабочим, не будучи им). Telegram принимает в качестве реакции не любой
эмодзи, а строку, ТОЧНО совпадающую с одной из своих штатных реакций — вплоть
до кодовых точек. Штатный список хранит "❤" как U+2764, без variation
selector U+FE0F; модель же почти всегда пишет "❤️" (U+2764 U+FE0F), потому
что именно так эмодзи выглядит в обычном тексте. Такие же расхождения дают
модификаторы тона кожи ("👍🏻" вместо "👍").

Симптом при этом самый неприятный из возможных: MTProto-вызов проходит без
ошибки, Pyrogram возвращает True, в логах честная запись "реакция
поставлена" — а в чате не появляется ничего. Поэтому эмодзи здесь
приводится к каноничному виду (снимаются U+FE0F и тон кожи) и сверяется со
списком штатных реакций ДО обращения к сети: лучше вернуть модели внятную
ошибку с подсказкой, чем тихо не сделать ничего.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from efi.tools.base import Tool, ToolContext

logger = logging.getLogger(__name__)

#: Variation Selector-16 — "нарисуй предыдущий символ как цветной эмодзи".
#: В обычном тексте он есть почти всегда, в списке реакций Telegram — нет.
_VARIATION_SELECTOR_16 = "️"

#: Модификаторы тона кожи (U+1F3FB..U+1F3FF). Реакции Telegram — без тона.
_SKIN_TONE_MODIFIERS = frozenset("\U0001f3fb\U0001f3fc\U0001f3fd\U0001f3fe\U0001f3ff")

#: Штатные (бесплатные) реакции Telegram. Кастомные эмодзи-реакции требуют
#: Premium и другого типа TL (ReactionCustomEmoji), поэтому сюда не входят.
#: Записаны в каноничном виде — том, в котором их ждёт сервер: без U+FE0F.
_ALLOWED_REACTIONS = frozenset(
    {
        "👍", "👎", "❤", "🔥", "🥰", "👏", "😁", "🤔", "🤯", "😱", "🤬", "😢", "🎉", "🤩",
        "🤮", "💩", "🙏", "👌", "🕊", "🤡", "🥱", "🥴", "😍", "🐳", "❤‍🔥", "🌚", "🌭", "💯",
        "🤣", "⚡", "🍌", "🏆", "💔", "🤨", "😐", "🍓", "🍾", "💋", "🖕", "😈", "😴", "😭",
        "🤓", "👻", "👨‍💻", "👀", "🎃", "🙈", "😇", "😨", "🤝", "✍", "🤗", "🫡", "🎅", "🎄",
        "☃", "💅", "🤪", "🗿", "🆒", "💘", "🙉", "🦄", "😘", "💊", "🙊", "😎", "👾",
        "🤷‍♂", "🤷", "🤷‍♀", "😡",
    }
)

#: Что показать модели, когда она попросила реакцию, которой у Telegram нет.
#: Короткая выборка, а не весь список: цель — дать пересобрать вызов, а не
#: занять половину контекста перечислением.
_SUGGESTED_REACTIONS = ("👍", "❤", "🔥", "😁", "🤔", "😢", "🤯", "👏", "🤣", "💯")


def normalize_reaction_emoji(emoji: str) -> str | None:
    """
    Приводит эмодзи к тому виду, в котором его ждёт Telegram, и возвращает
    None, если такой штатной реакции не существует.

    Снимается U+FE0F и модификаторы тона кожи — именно на них расходятся
    "эмодзи как его пишут в тексте" и "эмодзи как он лежит в списке реакций".
    ZWJ-последовательности (U+200D) НЕ трогаем: в списке есть составные
    реакции вроде "❤‍🔥" и "🤷‍♂", разбор которых по частям их бы уничтожил.
    """
    stripped = "".join(
        char for char in emoji.strip() if char != _VARIATION_SELECTOR_16 and char not in _SKIN_TONE_MODIFIERS
    )
    return stripped if stripped in _ALLOWED_REACTIONS else None


class MessageReactor(Protocol):
    """Абстракция реакции на сообщение. Конкретная реализация — efi.telegram.client.TelegramClientWrapper."""

    async def react(self, chat_id: int, message_id: int, emoji: str) -> None: ...


class ReactWithEmojiTool(Tool):
    """Ставит эмодзи-реакцию на сообщение собеседника, которое сейчас вызвало твой ответ."""

    name = "react_with_emoji"
    description = (
        "Ставит эмодзи-реакцию на сообщение собеседника, которое СЕЙЧАС вызвало твой ответ (не на произвольное "
        "старое сообщение из истории) — уместно вместо текстового ответа на короткие реплики/шутки/что-то, на "
        "что не нужно отвечать словами. Если собеседник прислал несколько сообщений подряд одним блоком — "
        "реакция ставится на последнее из них."
    )
    parameters = {
        "type": "object",
        "properties": {
            "emoji": {
                "type": "string",
                "description": (
                    "Эмодзи реакции. Годятся ТОЛЬКО штатные реакции Telegram, любой другой эмодзи "
                    "не поставится: " + " ".join(_SUGGESTED_REACTIONS) + " и подобные им."
                ),
            },
        },
        "required": ["emoji"],
        "additionalProperties": False,
    }

    def __init__(self, reactor: MessageReactor) -> None:
        self._reactor = reactor

    def is_available(self, context: ToolContext) -> bool:
        return context.chat_id is not None and self._current_message_id(context) is not None

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        requested = str(arguments.get("emoji", "")).strip()
        if not requested:
            return "error: emoji must not be empty"
        if context.chat_id is None:
            return "error: no chat_id in the current context"

        message_id = self._current_message_id(context)
        if message_id is None:
            return "error: no message in the current context to react to"

        # Проверка ДО сети: Telegram принимает такой вызов молча, ничего не
        # ставит и не возвращает ошибки — см. докстринг модуля.
        emoji = normalize_reaction_emoji(requested)
        if emoji is None:
            logger.info("react_with_emoji: %r is not a standard Telegram reaction, refusing to send it", requested)
            return (
                f"error: {requested!r} не входит в набор реакций Telegram — такая реакция не поставится. "
                f"Выбери другую, например: {' '.join(_SUGGESTED_REACTIONS)}"
            )

        try:
            await self._reactor.react(context.chat_id, message_id, emoji)
        except Exception as exc:
            logger.warning("react_with_emoji: failed to react: %s", exc)
            return f"error: could not react: {exc}"

        logger.info("react_with_emoji: reacted %s to message_id=%s in chat_id=%s", emoji, message_id, context.chat_id)
        return "Реакция поставлена."

    def _current_message_id(self, context: ToolContext) -> int | None:
        message_ids = context.notification.payload.get("telegram_message_ids")
        if not message_ids:
            return None
        return int(message_ids[-1])


__all__ = ["MessageReactor", "ReactWithEmojiTool", "normalize_reaction_emoji"]
