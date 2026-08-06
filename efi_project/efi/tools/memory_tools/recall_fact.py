"""
efi/tools/memory_tools/recall_fact.py

Инструмент поиска фактов о сущности — точечного (по конкретному ключу) или
полного (все факты про entity_id разом). Дополняет remember_fact.py: пишет
туда, читает отсюда.
"""

from __future__ import annotations

from typing import Any

from efi.memory.facts import FactStore
from efi.tools.base import Tool, ToolContext


class RecallFactTool(Tool):
    """Позволяет модели вспомнить сохранённые факты о сущности — все разом или один конкретный."""

    name = "recall_fact"
    description = (
        "Ищет ранее сохранённые структурированные факты о ком-то или о чём-то. "
        "Если указать key — вернётся только этот факт; если key не указан — все известные факты об entity_id."
    )
    parameters = {
        "type": "object",
        "properties": {
            "entity_id": {
                "type": "string",
                "description": "Кого/чего касается факт, например 'user:625207005' или 'self'",
            },
            "key": {"type": "string", "description": "Конкретный факт, который нужно вспомнить (необязательно)"},
        },
        "required": ["entity_id"],
        "additionalProperties": False,
    }

    def __init__(self, facts: FactStore) -> None:
        self._facts = facts

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        entity_id = str(arguments.get("entity_id", "")).strip()
        if not entity_id:
            return "error: entity_id must not be empty"

        key = arguments.get("key")
        if key:
            value = await self._facts.get(entity_id, str(key).strip())
            if value is None:
                return f"Факт {entity_id}.{key} не найден."
            return f"{entity_id}.{key} = {value}"

        all_facts = await self._facts.get_all(entity_id)
        if not all_facts:
            return f"О {entity_id} пока ничего не известно."
        lines = "\n".join(f"- {fact_key}: {fact_value}" for fact_key, fact_value in all_facts.items())
        return f"Известные факты о {entity_id}:\n{lines}"


__all__ = ["RecallFactTool"]
