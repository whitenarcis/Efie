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
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from efi.tools.base import Tool, ToolContext

logger = logging.getLogger(__name__)


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
            "emoji": {"type": "string", "description": "Эмодзи реакции, например '❤️' или '😂'"},
        },
        "required": ["emoji"],
        "additionalProperties": False,
    }

    def __init__(self, reactor: MessageReactor) -> None:
        self._reactor = reactor

    def is_available(self, context: ToolContext) -> bool:
        return context.chat_id is not None and self._current_message_id(context) is not None

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        emoji = str(arguments.get("emoji", "")).strip()
        if not emoji:
            return "error: emoji must not be empty"
        if context.chat_id is None:
            return "error: no chat_id in the current context"

        message_id = self._current_message_id(context)
        if message_id is None:
            return "error: no message in the current context to react to"

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


__all__ = ["MessageReactor", "ReactWithEmojiTool"]
