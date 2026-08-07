"""
efi/tools/memory_tools/manage_belief.py

Инструмент осознанного закрепления/обновления позиции по теме — граф
убеждений (efi.memory.beliefs.BeliefStore) иначе никогда не пополняется:
без него блок текущего состояния промпта (efi.prompts.builder) не находит
чего "отстаивать" при попытке переубеждения. Модель вызывает этот инструмент,
когда осознаёт, что у неё сформировалось (или изменилось) чёткое мнение по
теме, а не для мимолётных реакций на каждую реплику.
"""

from __future__ import annotations

from typing import Any

from efi.memory.beliefs import BeliefStore
from efi.tools.base import Tool, ToolContext


class UpdateBeliefTool(Tool):
    """Записывает или обновляет твою позицию (stance) по теме в графе убеждений, с уверенностью в ней."""

    name = "update_belief"
    description = (
        "Закрепляет ИЛИ обновляет твоё личное мнение (stance) по конкретной теме в долгосрочном графе "
        "убеждений — используй, когда у тебя реально сложилось чёткое мнение по теме (не по мелочи) или "
        "когда тебя переубедили ПОСЛЕ того, как ты честно взвесила новые аргументы. Чем выше confidence_score, "
        "тем упрямее ты будешь отстаивать эту позицию в будущих разговорах — не завышай уверенность просто так, "
        "и не меняй устоявшееся мнение только потому, что собеседник настаивает без новых доводов."
    )
    parameters = {
        "type": "object",
        "properties": {
            "topic": {
                "type": "string",
                "description": "Короткая тема убеждения, по которой можно её узнать позже (например 'vim vs vscode')",
            },
            "stance": {"type": "string", "description": "Твоя позиция по теме, от первого лица, коротко"},
            "confidence_score": {
                "type": "number",
                "description": "Насколько твёрдо ты в этом убеждена, от 0 (лёгкая склонность) до 1 (несгибаемая "
                "позиция). По умолчанию 0.5.",
            },
        },
        "required": ["topic", "stance"],
        "additionalProperties": False,
    }

    def __init__(self, beliefs: BeliefStore) -> None:
        self._beliefs = beliefs

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        topic = str(arguments.get("topic", "")).strip()
        stance = str(arguments.get("stance", "")).strip()
        if not topic or not stance:
            return "error: topic and stance must not be empty"
        confidence_score = _coerce_confidence(arguments.get("confidence_score", 0.5))

        await self._beliefs.upsert(topic, stance, confidence_score=confidence_score)
        return f"Убеждение по теме {topic!r} закреплено (уверенность {confidence_score:.2f})"


def _coerce_confidence(raw: Any) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 0.5
    return max(0.0, min(value, 1.0))


__all__ = ["UpdateBeliefTool"]
