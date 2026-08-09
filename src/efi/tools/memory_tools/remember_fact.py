"""
efi/tools/memory_tools/remember_fact.py

Инструмент сохранения структурированного факта о сущности (entity_id/key/value)
в FactStore. В отличие от Diary (свободный текст, семантический поиск),
факты — короткие, точные, легко перезаписываемые значения ("любимый цвет",
"дата рождения", "как зовут кота"), которые дешевле хранить и читать как пары
ключ-значение, чем каждый раз искать по смыслу.
"""

from __future__ import annotations

from typing import Any

from efi.memory.facts import FactStore
from efi.tools.base import Tool, ToolContext


class RememberFactTool(Tool):
    """Позволяет модели сохранить (или обновить) конкретный факт о сущности."""

    name = "remember_fact"
    description = (
        "Сохраняет короткий структурированный факт о ком-то или о чём-то — например, "
        "любимый цвет, день рождения, кличку питомца. Используй для точных, легко "
        "формулируемых фактов; для свободных воспоминаний и историй используй дневник (ask_diary)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "entity_id": {
                "type": "string",
                "description": "Кого/чего касается факт, например 'user:625207005' или 'self' для фактов о себе",
            },
            "key": {"type": "string", "description": "Название факта, например 'любимый_цвет'"},
            "value": {"type": "string", "description": "Значение факта"},
            "confidence": {
                "type": "number",
                "description": "Насколько ты уверена в этом факте, от 0 до 1 (по умолчанию 1.0 — точно знаешь)",
            },
        },
        "required": ["entity_id", "key", "value"],
        "additionalProperties": False,
    }

    def __init__(self, facts: FactStore) -> None:
        self._facts = facts

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        entity_id = str(arguments.get("entity_id", "")).strip()
        key = str(arguments.get("key", "")).strip()
        value = str(arguments.get("value", "")).strip()
        if not entity_id or not key or not value:
            return "error: entity_id, key and value must all be non-empty"

        confidence = _coerce_confidence(arguments.get("confidence", 1.0))
        await self._facts.upsert(entity_id, key, value, confidence=confidence)
        return f"Запомнила: {entity_id}.{key} = {value!r} (confidence={confidence:.2f})"


def _coerce_confidence(raw: Any) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 1.0
    return max(0.0, min(value, 1.0))


__all__ = ["RememberFactTool"]
