"""
efi/tools/memory_tools/remember_diary_entry.py

Инструмент осознанного "запомни это". В отличие от автоматической ночной
новеллизации (efi.memory.consolidation.DiaryConsolidator.novelize_recent_history,
которая раз в сутки сама решает, что стоит запомнить), этот инструмент даёт
модели способ сохранить что-то в дневник СРАЗУ, в момент разговора — если
она сама понимает, что это важно и не стоит дожидаться ночи.
"""

from __future__ import annotations

from typing import Any

from efi.memory.rag import RAGMemory
from efi.tools.base import Tool, ToolContext


class RememberDiaryEntryTool(Tool):
    """Сохраняет свободный текст в долгосрочный дневник — для моментов, которые стоит запомнить надолго."""

    name = "remember_diary_entry"
    description = (
        "Сохраняет что-то важное в твой долгосрочный дневник ПРЯМО СЕЙЧАС — используй, когда происходит "
        "что-то, что ты сама хочешь запомнить надолго (важный факт о собеседнике, значимое событие, "
        "договорённость, эмоциональный момент), не дожидаясь ночной сводки. Не используй для мелочей и "
        "обычной болтовни — только для того, что реально имеет значение через дни/недели."
    )
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Что запомнить, от первого лица, коротко"},
            "confidence": {
                "type": "number",
                "description": (
                    "Насколько это точно/уверенно, от 0 (догадка/теория) до 1 (точный факт). По умолчанию 0.7."
                ),
            },
        },
        "required": ["text"],
        "additionalProperties": False,
    }

    def __init__(self, rag: RAGMemory) -> None:
        self._rag = rag

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        text = str(arguments.get("text", "")).strip()
        if not text:
            return "error: text must not be empty"
        confidence = _coerce_confidence(arguments.get("confidence", 0.7))

        entry = await self._rag.remember(text, confidence=confidence)
        if entry is None:
            return "Похожее уже есть в дневнике — не стала дублировать."
        return f"Запомнила в дневник: {text!r}"


def _coerce_confidence(raw: Any) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 0.7
    return max(0.0, min(value, 1.0))


__all__ = ["RememberDiaryEntryTool"]
