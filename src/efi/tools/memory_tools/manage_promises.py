"""
efi/tools/memory_tools/manage_promises.py

Инструменты для открытых обещаний/напоминаний в рабочей памяти
(efi.memory.working_memory.WorkingMemory). Без них блок "[Текущее состояние]
... открытые задачи/обещания" в системном промпте (efi.prompts.builder.
_build_working_memory_block) никогда не заполнялся бы: структура готова с
самого начала, но не было ни одного способа её населить. personality.md
прямо ссылается на этот раздел ("ТО, ЧТО ТЫ НЕДАВНО УПОМЯНУЛА") как на место,
куда нужно заглянуть, прежде чем честно признаться, что не помнишь, — без
этих инструментов ссылка вела в пустоту.
"""

from __future__ import annotations

from typing import Any

from efi.memory.working_memory import WorkingMemory
from efi.tools.base import Tool, ToolContext


class RememberPromiseTool(Tool):
    """Сохраняет обещание/напоминание/незавершённую задачу в рабочую память — то, к чему нужно будет вернуться."""

    name = "remember_promise"
    description = (
        "Запоминает обещание, напоминание или незавершённую задачу, которую ты дала себе или собеседнику "
        "(например, 'скину ссылку позже', 'напомнить спросить как прошло собеседование'). Используй, когда "
        "ты сказала, что что-то сделаешь ПОЗЖЕ, а не прямо сейчас — иначе в следующий раз ты об этом не вспомнишь."
    )
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Что именно обещано/нужно не забыть, коротко, от первого лица"},
        },
        "required": ["text"],
        "additionalProperties": False,
    }

    def __init__(self, working_memory: WorkingMemory) -> None:
        self._working_memory = working_memory

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        text = str(arguments.get("text", "")).strip()
        if not text:
            return "error: text must not be empty"
        await self._working_memory.add_item(text)
        return f"Запомнила: {text!r}"


class CompletePromiseTool(Tool):
    """Отмечает ранее сохранённое обещание/напоминание выполненным."""

    name = "complete_promise"
    description = (
        "Отмечает выполненным обещание/напоминание из списка 'открытые задачи/обещания' в твоём текущем "
        "состоянии — используй, когда ты только что сделала то, что обещала раньше. Передай часть исходного "
        "текста, по которой его можно узнать."
    )
    parameters = {
        "type": "object",
        "properties": {
            "text_query": {"type": "string", "description": "Часть текста обещания, по которой его найти"},
        },
        "required": ["text_query"],
        "additionalProperties": False,
    }

    def __init__(self, working_memory: WorkingMemory) -> None:
        self._working_memory = working_memory

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        text_query = str(arguments.get("text_query", "")).strip()
        if not text_query:
            return "error: text_query must not be empty"
        item = await self._working_memory.find_and_mark_done(text_query)
        if item is None:
            return f"Не нашла открытого обещания, похожего на {text_query!r}."
        return f"Отметила выполненным: {item.text!r}"


__all__ = ["RememberPromiseTool", "CompletePromiseTool"]
