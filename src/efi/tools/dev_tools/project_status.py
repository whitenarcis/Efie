"""
efi/tools/dev_tools/project_status.py

«Как там твой проект?» — ответ на этот вопрос фактами, а не выдумкой.

Без инструмента модель отвечает на такой вопрос правдоподобной прозой: она
не знает ни статуса задачи, ни адреса репозитория, но признаться в этом
языковой модели тяжелее, чем сочинить. Здесь она получает то, что реально
записано в базе: чем занята сейчас и что уже выложено, со ссылками.

Ссылки берутся из БД, а не из памяти разговора: пересказанный по памяти
адрес репозитория — это адрес, которого не существует.

Здесь же — и то, что НЕ вышло, с причинами. «Почему ты забросила ту штуку?»
— такой же вопрос про её работу, как «что ты сейчас пишешь?», и отвечать на
него выдумкой ещё хуже: выдуманная причина превращает рабочую неудачу в
несуществующую историю, которую потом обсуждают всерьёз.
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
        "Смотрит, над каким проектом ты сейчас работаешь, какие уже выложила на GitHub (со ссылками) и "
        "что не вышло — с настоящей причиной. Вызывай, когда собеседник спрашивает про твой код, "
        "проекты или GitHub, включая «а что там с той штукой?» и «почему забросила», — и когда сама "
        "хочешь сослаться на свою наработку. Не выдумывай статус, ссылки и причины по памяти: здесь "
        "они настоящие."
    )

    def __init__(self, store: DevTaskStore) -> None:
        self._store = store

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        active = await self._store.active()
        released = await self._store.recent_releases(limit=_RECENT_LIMIT)
        abandoned = await self._store.recent_failures(limit=_RECENT_LIMIT)

        if not active and not released and not abandoned:
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
        if abandoned:
            parts.append(
                "Не вышло:\n"
                + "\n".join(
                    f"- {task.spec.title if task.spec else task.idea}: "
                    f"{task.error or 'причина не записана'}"
                    for task in abandoned
                )
            )
        return "\n\n".join(parts)


__all__ = ["DevProjectStatusTool"]
