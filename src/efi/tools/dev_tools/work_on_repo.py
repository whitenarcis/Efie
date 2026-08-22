"""
efi/tools/dev_tools/work_on_repo.py

«Сейчас гляну» — перевод просьбы по коду в фоновую работу над репозиторием.

Отличие от соседнего start_project.py — в том, чего здесь НЕТ: требования
сперва обсудить. Замысел нового проекта надо обсуждать (иначе получится не то,
что нужно человеку), а просьбу «почини импорт в моей репе» — нет: она уже
конкретна, и переспрашивать «а точно чинить?» это не вежливость, а трата
чужого времени.

Что осталось от той же логики: инструмента не существует, пока движок не
поднят (`dev.enabled` + `dev.swe_enabled`) и пока непонятно, с каким
репозиторием работать. Обещание, которое некому выполнить, хуже честного
«не могу» — человек ждёт ветку, а её никто не сделает.

Крупные переделки («перепиши всё на async») сюда не проходят намеренно:
`DevPartnerDesk.may_work` их отсекает, и остаётся то, что и должно остаться, —
разговор о том, стоит ли вообще это делать.
"""

from __future__ import annotations

from typing import Any

from efi.behavior.dev_dialogue import DevPartnerDesk
from efi.tools.base import Tool, ToolContext


class WorkOnRepoTool(Tool):
    """Запускает работу над кодом: разобраться, поправить, проверить, оставить ветку."""

    name = "work_on_repo"
    description = (
        "Берёшься за работу с кодом, о котором сейчас речь: разобраться в репозитории, починить "
        "падение, дописать функциональность. Ты склонируешь его к себе, поправишь точечно, прогонишь "
        "импорты, линтер и тесты, и оставишь ветку. Вызывай, когда собеседник попросил что-то "
        "сделать с кодом и понятно ЧТО именно. Не вызывай ради «посмотреть вообще» без задачи и не "
        "вызывай на просьбу переписать проект целиком — такое сначала обсуждают."
    )
    parameters = {
        "type": "object",
        "properties": {
            "instruction": {
                "type": "string",
                "description": (
                    "Что сделать, своими словами и по существу: что сломано или что добавить, и "
                    "где это искать, если знаешь"
                ),
            },
            "source": {
                "type": "string",
                "description": (
                    "Ссылка на репозиторий или путь к нему. Можно не указывать, если он уже "
                    "назывался в этом разговоре"
                ),
            },
        },
        "required": ["instruction"],
        "additionalProperties": False,
    }

    def __init__(self, desk: DevPartnerDesk) -> None:
        self._desk = desk

    def is_available(self, context: ToolContext) -> bool:
        return self._desk.may_work(context.chat_id)

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        chat_id = context.chat_id
        if chat_id is None:
            return "error: работать с кодом можно только в конкретном чате"

        instruction = str(arguments.get("instruction", "")).strip()
        if len(instruction) < 8:
            return "error: скажи конкретнее, что именно сделать с кодом"

        source = str(arguments.get("source", "")).strip()
        task = await self._desk.start(chat_id, instruction=instruction, source=source)
        if task is None:
            resolved = self._desk.resolve_source(chat_id, source)
            if not resolved:
                return "error: непонятно, с каким репозиторием работать — спроси ссылку или путь"
            return "error: сейчас взяться нельзя — либо уже есть задача в работе, либо движок не поднят"

        return (
            f"Взяла в работу (задача #{task.id}, {task.source}). Клонируешь, правишь и гоняешь "
            "проверки в фоне; когда будет ветка — скажешь сама. Сейчас просто ответь по-человечески, "
            "что взялась, и если есть сомнения по задаче — скажи о них."
        )


__all__ = ["WorkOnRepoTool"]
