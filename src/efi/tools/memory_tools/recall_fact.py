"""
efi/tools/memory_tools/recall_fact.py

Инструмент поиска фактов о сущности — точечного (по конкретному ключу) или
полного (все факты про entity_id разом). Дополняет remember_fact.py: пишет
туда, читает отсюда.

Читает из строгого хранилища знаний (efi/memory/dedup.py::KnowledgeStore) —
того же, куда пишет remember_fact после валидации. Вместе со значением
возвращает и число подтверждений: «упомянуто 9 раз» — часть факта, а не
служебная метрика, и без неё модель не отличает устойчивую черту от
однажды услышанного.
"""

from __future__ import annotations

from typing import Any

from efi.memory.dedup import KnowledgeStore
from efi.memory.validator import FactValidator
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

    def __init__(self, store: KnowledgeStore, validator: FactValidator) -> None:
        self._store = store
        self._validator = validator

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        raw_entity = str(arguments.get("entity_id", "")).strip()
        if not raw_entity:
            return "error: entity_id must not be empty"

        # Через тот же нормализатор, что и запись: модель зовёт сущность то
        # «Рома», то «user:625207005», то «я» — и без приведения к общему
        # виду чтение промахивалось бы мимо собственной же записи.
        entity_id = self._validator.normalize_entity_for_lookup(raw_entity)
        facts = await self._store.recall(entity_ids=[entity_id], limit=30)

        requested_key = str(arguments.get("key", "") or "").strip()
        if requested_key:
            normalized_key = self._validator.normalize_attribute_for_lookup(requested_key)
            matching = [fact for fact in facts if fact.attribute == normalized_key]
            if not matching:
                return f"Факт {raw_entity}.{requested_key} не найден."
            return "\n".join(fact.render_for_prompt() for fact in matching)

        if not facts:
            return f"О {raw_entity} пока ничего не известно."
        lines = "\n".join(fact.render_for_prompt() for fact in facts)
        return f"Известные факты о {raw_entity}:\n{lines}"


__all__ = ["RecallFactTool"]
