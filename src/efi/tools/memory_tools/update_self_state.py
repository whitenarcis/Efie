"""
efi/tools/memory_tools/update_self_state.py

Инструмент обновления собственного эмоционального/физического состояния и
уровня энергии (efi.memory.working_memory.WorkingMemory). Без него
WorkingMemorySnapshot.energy навсегда остаётся дефолтным значением: вход для
efi.behavior.busy_engine.BusyEngine (низкая энергия удлиняет ignore_delay) и
для блока "[Текущее состояние]" в системном промпте (efi/prompts/builder.py)
существует, но нечем его двигать — модель должна иметь способ сама сказать
"я устала"/"я взбодрилась", чтобы это отразилось на её собственном поведении
в следующих разговорах, а не только в рамках одной реплики.
"""

from __future__ import annotations

from typing import Any

from efi.memory.working_memory import WorkingMemory
from efi.tools.base import Tool, ToolContext


class UpdateSelfStateTool(Tool):
    """Обновляет твоё текущее эмоциональное/физическое состояние и уровень энергии."""

    name = "update_self_state"
    description = (
        "Обновляет твоё текущее состояние — эмоциональное, физическое и/или уровень энергии (0..1, где 0 — "
        "совсем без сил, 1 — на пике бодрости). Используй, когда твоё состояние заметно поменялось (разговор "
        "вымотал, что-то взбодрило, задолбалась от чего-то) — не на каждую реплику, а когда это правда важно "
        "запомнить до следующего разговора. Любое из полей можно не указывать — обновится только заданное."
    )
    parameters = {
        "type": "object",
        "properties": {
            "emotional_state": {"type": "string", "description": "Текущее эмоциональное состояние, коротко"},
            "physical_state": {"type": "string", "description": "Текущее физическое состояние/самочувствие, коротко"},
            "energy": {"type": "number", "description": "Уровень бодрости от 0 до 1"},
        },
        "additionalProperties": False,
    }

    def __init__(self, working_memory: WorkingMemory) -> None:
        self._working_memory = working_memory

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        emotional_state = _coerce_optional_str(arguments.get("emotional_state"))
        physical_state = _coerce_optional_str(arguments.get("physical_state"))
        energy = _coerce_optional_float(arguments.get("energy"))

        if emotional_state is None and physical_state is None and energy is None:
            return "error: at least one of emotional_state/physical_state/energy must be provided"

        await self._working_memory.update_state(
            emotional_state=emotional_state, physical_state=physical_state, energy=energy
        )
        return "Обновила своё состояние."


def _coerce_optional_str(raw: Any) -> str | None:
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _coerce_optional_float(raw: Any) -> float | None:
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


__all__ = ["UpdateSelfStateTool"]
