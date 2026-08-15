"""
efi/tools/dev_tools/project_status.py

«Как там твой проект?» — ответ на этот вопрос фактами, а не выдумкой.

Без инструмента модель отвечает на такой вопрос правдоподобной прозой: она
не знает ни статуса задачи, ни адреса репозитория, но признаться в этом
языковой модели тяжелее, чем сочинить. Здесь она получает то, что реально
записано в базе: чем занята сейчас и что уже выложено, со ссылками.

Ссылки берутся из БД, а не из памяти разговора: пересказанный по памяти
адрес репозитория — это адрес, которого не существует.
"""

from __future__ import annotations

from typing import Any

from efi.dev.store import DevTaskStore
from efi.tools.base import Tool, ToolContext

#: Сколько прошлых проектов показывать. Это ответ на вопрос «чем занималась»,
#: а не выгрузка портфолио.
_RECENT_LIMIT = 3


class DevProjectStatusTool(Tool):
    """Показывает, над чем Эфи работает сейчас и что уже выложила."""

    name = "check_my_projects"
    description = (
        "Смотрит, над каким проектом ты сейчас работаешь и какие уже выложила на GitHub (со ссылками). "
        "Вызывай, когда собеседник спрашивает про твой код, проекты или GitHub, — и когда сама хочешь "
        "сослаться на свою наработку. Не выдумывай статус и ссылки по памяти: здесь они настоящие."
    )

    def __init__(self, store: DevTaskStore) -> None:
        self._store = store

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        active = await self._store.active()
        released = await self._store.recent_releases(limit=_RECENT_LIMIT)

        if not active and not released:
            return "Сейчас ты ничего не пишешь и выложенных проектов пока нет."

        parts: list[str] = []
        if active:
            parts.append("В работе:\n" + "\n".join(f"- {task.render_for_prompt()}" for task in active))
        if released:
            parts.append(
                "Выложено:\n"
                + "\n".join(
                    f"- {task.spec.title if task.spec else task.idea}: {task.repo_url}" for task in released
                )
            )
        return "\n\n".join(parts)


__all__ = ["DevProjectStatusTool"]
