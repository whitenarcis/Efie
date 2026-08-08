"""
efi/tools/memory_tools/remember_person.py

Инструмент, которым Эфи фиксирует сложившееся отношение к КОНКРЕТНОМУ
собеседнику (efi.memory.people.PeopleStore.set_impression).

Числовые affinity/respect_level копятся сами, по эвристике на каждое
сообщение — но они не умеют хранить СМЫСЛ ("этот вечно спорит, но по делу",
"с этим весело, но на него нельзя положиться"). Это как раз то, что делает
память о человеке человеческой, а не счётчиком, поэтому такой вывод пишет
сама модель, когда он у неё сложился.

user_id не запрашивается параметром — он берётся из контекста уведомления
(payload["sender_id"], кладёт efi.telegram.handlers): модели неоткуда узнать
чужой Telegram user_id, ровно как и message_id в
efi.tools.telegram_actions.react_with_emoji.
"""

from __future__ import annotations

import logging
from typing import Any

from efi.memory.people import PeopleStore
from efi.tools.base import Tool, ToolContext

logger = logging.getLogger(__name__)


class RememberPersonTool(Tool):
    """Записывает сложившееся мнение о собеседнике, который пишет прямо сейчас."""

    name = "remember_person"
    description = (
        "Запоминает твоё личное отношение/впечатление о собеседнике, который пишет сейчас — то, что ты о нём "
        "поняла (например: 'вечно спорит по мелочам, но по делу', 'легко обещает и не делает'). Используй, "
        "когда о человеке действительно что-то сложилось, а не после каждой реплики. Это переживёт рестарт "
        "и будет напоминать тебе, с кем ты имеешь дело, при следующих встречах — в том числе в других чатах."
    )
    parameters = {
        "type": "object",
        "properties": {
            "impression": {
                "type": "string",
                "description": "Что ты о нём поняла, коротко и от первого лица",
            },
        },
        "required": ["impression"],
        "additionalProperties": False,
    }

    def __init__(self, people: PeopleStore) -> None:
        self._people = people

    def is_available(self, context: ToolContext) -> bool:
        return _sender_id(context) is not None

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        impression = str(arguments.get("impression", "")).strip()
        if not impression:
            return "error: impression must not be empty"

        sender_id = _sender_id(context)
        if sender_id is None:
            return "error: no sender in the current context to remember"

        profile = await self._people.set_impression(sender_id, impression)
        if profile is None:
            return "error: this person is not known yet, nothing to attach the impression to"

        logger.info("remember_person: impression saved for user_id=%s", sender_id)
        return f"Запомнила про {profile.display_name or sender_id}: {impression!r}"


def _sender_id(context: ToolContext) -> int | None:
    sender_id = context.notification.payload.get("sender_id")
    return sender_id if isinstance(sender_id, int) else None


__all__ = ["RememberPersonTool"]
