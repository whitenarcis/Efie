"""
efi/tools/dev_tools/start_project.py

«Договорились — берусь»: перевод обсуждения в фоновую задачу разработки.

Ключевое здесь — `is_available`. Инструмент показывается модели ТОЛЬКО когда
в этом чате есть обсуждаемое предложение И обсуждение реально состоялось
(efi.behavior.collab_coding.CollabCodingDesk.may_start). Пока стороны не
поговорили, инструмента для модели не существует: запретить соглашаться
сразу словами в промпте невозможно — она соглашается, потому что это самый
вероятный ответ на предложение. А вот вызвать инструмент, которого ей не
показали, она не может (реестр проверяет доступность дважды — и при показе,
и при исполнении, см. efi/tools/registry.py).

Проверка `may_start` повторяется внутри `execute` не для симметрии: между
показом списка инструментов и вызовом проходит целый ход модели, и состояние
за это время может измениться (например, задачу уже завели).
"""

from __future__ import annotations

from typing import Any

from efi.behavior.collab_coding import CollabCodingDesk
from efi.tools.base import Tool, ToolContext


class StartDevProjectTool(Tool):
    """Запускает работу над проектом, о котором договорились в чате."""

    name = "start_dev_project"
    description = (
        "Берёшься за проект, который вы с собеседником только что обсудили. Вызывай ТОЛЬКО когда вы "
        "договорились по существу: что это за штука, на чём пишем и что она должна уметь. Не вызывай в "
        "ответ на первое же «давай напишем» — сначала обсуди замысел, стек и подводные камни, как "
        "обсуждают с человеком, с которым будете это делать. После вызова ты уходишь писать код в "
        "фоне и сама расскажешь, когда будет что показать."
    )
    parameters = {
        "type": "object",
        "properties": {
            "idea": {
                "type": "string",
                "description": (
                    "Итоговая формулировка замысла своими словами: что за инструмент, какую проблему "
                    "решает, на чём пишем и что решили НЕ делать"
                ),
            }
        },
        "required": ["idea"],
        "additionalProperties": False,
    }

    def __init__(self, desk: CollabCodingDesk) -> None:
        self._desk = desk

    def is_available(self, context: ToolContext) -> bool:
        return self._desk.may_start(context.chat_id)

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        chat_id = context.chat_id
        if chat_id is None:
            return "error: проект можно начать только в конкретном чате"

        idea = str(arguments.get("idea", "")).strip()
        if len(idea) < 16:
            # Пустая или односложная формулировка означает, что договорённости
            # на самом деле нет, — а спека по «сделай бота» получится ровно
            # такой же пустой.
            return "error: сформулируй замысел подробнее — что за инструмент и что он должен уметь"

        task = await self._desk.start(chat_id, idea=idea)
        if task is None:
            return "error: сначала обсудите замысел — стек, структуру, что именно эта штука должна делать"

        return (
            f"Взяла в работу (задача #{task.id}). Пишешь её в фоне; когда будет что показать — "
            "скажешь сама. Сейчас просто ответь собеседнику по-человечески, что берёшься."
        )


__all__ = ["StartDevProjectTool"]
