"""
efi/tools/memory_tools/update_self_state.py

Способ Эфи назвать своё состояние СВОИМИ словами
(efi.memory.working_memory.WorkingMemory.update_state).

Раньше это был единственный способ вообще: пока инструмент не вызван,
`energy` стоял на дефолтных 0.7, а состояние оставалось пустой строкой. И
именно поэтому он не работал — у модели нет причины тратить ход на служебное
действие, которого от неё в этой реплике никто не ждёт, и на практике она не
вызывала его почти никогда. Энергия месяцами не двигалась, а
efi.behavior.busy_engine, ради которого она и заведена, работал с константой.

Теперь базовое самочувствие живёт само (efi/behavior/energy.py): падает к
ночи, тратится на разговоры, восстанавливается со временем. Этот инструмент
остался для того, чего механика знать не может, — «этот разговор меня
конкретно вымотал», «новость взбесила». Его слова сильнее модели и становятся
новым якорем, от которого всё считается дальше; молчание модели больше не
означает, что состояния нет.
"""

from __future__ import annotations

from typing import Any

from efi.memory.working_memory import WorkingMemory
from efi.tools.base import Tool, ToolContext


class UpdateSelfStateTool(Tool):
    """Обновляет твоё текущее эмоциональное/физическое состояние и уровень энергии."""

    name = "update_self_state"
    description = (
        "Называет твоё состояние СВОИМИ словами — эмоциональное, физическое и/или уровень энергии (0..1, "
        "где 0 — совсем без сил, 1 — на пике бодрости). Базовое самочувствие у тебя есть всегда: оно само "
        "падает к ночи, тратится на разговоры и восстанавливается за время. Этот инструмент нужен, когда "
        "случилось что-то, чего эта механика знать не может: разговор конкретно вымотал, новость взбесила, "
        "что-то резко взбодрило. Не на каждую реплику. Любое из полей можно не указывать."
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
