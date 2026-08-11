"""
efi/tools/memory_tools/manage_promises.py

Инструменты для открытых обещаний/напоминаний в рабочей памяти
(efi.memory.working_memory.WorkingMemory). Без них блок "[Текущее состояние]
... открытые задачи/обещания" в системном промпте (efi.prompts.builder.
_build_working_memory_block) никогда не заполнялся бы: структура готова с
самого начала, но не было ни одного способа её населить. personality.md
прямо ссылается на этот раздел ("ТО, ЧТО ТЫ НЕДАВНО УПОМЯНУЛА") как на место,
куда нужно заглянуть, прежде чем честно признаться, что не помнишь.

СРОК. Обещание со сроком («напиши мне через 10 минут») здесь не только
записывается, но и СТАВИТСЯ НА ТАЙМЕР через efi/behavior/reminders.py.
Раньше инструмент умел только записать текст: человек просил, Эфи
соглашалась, обещание повисало в состоянии навсегда — и ничего не
происходило. Поэтому `remind_in_minutes` появился в схеме инструмента и
описан как обязательный для просьб «через сколько-то»: модель не догадается
поставить таймер, если ей не сказать, что таймер вообще существует.

Инструмент честно сообщает, поставлено ли напоминание. Молчаливый отказ
(«записала» при том, что напомнить она не сможет) — ровно та поломка, из-за
которой всё это переписывалось: снаружи выглядит согласием, по факту не
делает ничего.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

from efi.behavior.reminders import MAX_DELAY, MIN_DELAY, ReminderStore, resolve_due_at
from efi.memory.working_memory import WorkingMemory
from efi.tools.base import Tool, ToolContext

logger = logging.getLogger(__name__)

#: Предикат «можно ли в этом чате писать первой» — реализация
#: efi.behavior.conversation_lifecycle.ConversationLifecycle.
#: allows_proactive_ping_to_chat. Проверяется В МОМЕНТ ОБЕЩАНИЯ, а не при
#: срабатывании: обещать то, чего не сможешь сделать, — хуже, чем сразу
#: сказать, что не сможешь.
CanScheduleCheck = Callable[[int | None], bool]


class RememberPromiseTool(Tool):
    """Сохраняет обещание/напоминание/незавершённую задачу и, если назван срок, ставит его на таймер."""

    name = "remember_promise"
    description = (
        "Запоминает обещание, напоминание или незавершённую задачу, которую ты дала себе или собеседнику "
        "(например, 'скину ссылку позже', 'напомнить спросить как прошло собеседование'). Используй, когда "
        "ты сказала, что что-то сделаешь ПОЗЖЕ, а не прямо сейчас — иначе в следующий раз ты об этом не вспомнишь. "
        "ЕСЛИ НАЗВАН СРОК ('через 10 минут', 'через час', 'вечером') — ОБЯЗАТЕЛЬНО передай remind_in_minutes: "
        "только тогда ты действительно напишешь сама в нужный момент. Без этого параметра ты просто запишешь "
        "обещание и ничего не сделаешь."
    )
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Что именно обещано/нужно не забыть, коротко, от первого лица"},
            "remind_in_minutes": {
                "type": "number",
                "description": (
                    "Через сколько минут написать самой. Указывай, когда назван срок: "
                    "'через 10 минут' -> 10, 'через час' -> 60, 'завтра утром' -> примерное число минут. "
                    "Не указывай, если срока нет ('скину, как найду')."
                ),
            },
        },
        "required": ["text"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        working_memory: WorkingMemory,
        *,
        reminders: ReminderStore | None = None,
        can_schedule: CanScheduleCheck | None = None,
    ) -> None:
        self._working_memory = working_memory
        self._reminders = reminders
        self._can_schedule = can_schedule

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        text = str(arguments.get("text", "")).strip()
        if not text:
            return "error: text must not be empty"

        minutes = _coerce_minutes(arguments.get("remind_in_minutes"))
        chat_id = context.chat_id

        if minutes is None:
            await self._working_memory.add_item(text, chat_id=chat_id)
            return f"Запомнила: {text!r} (без срока — напомнить сама не смогу, вспомню, когда зайдёт речь)"

        blocked = self._scheduling_blocked(chat_id)
        if blocked is not None:
            # Записываем всё равно: обещание остаётся её обещанием, даже если
            # напомнить о нём она не сможет. Но говорим об этом прямо, чтобы
            # модель могла честно предупредить собеседника в том же ходу.
            await self._working_memory.add_item(text, chat_id=chat_id)
            return f"Записала {text!r}, но напомнить сама не смогу: {blocked}. Скажи об этом собеседнику."

        assert self._reminders is not None  # гарантировано _scheduling_blocked
        assert chat_id is not None  # гарантировано _scheduling_blocked
        due_at = resolve_due_at(minutes)
        await self._reminders.schedule(chat_id, text, due_at=due_at)
        await self._working_memory.add_item(text, due_at=due_at, chat_id=chat_id)
        return f"Обещала и поставила себе напоминание: {text!r} — напишу через {_describe_delay(minutes)}."

    def _scheduling_blocked(self, chat_id: int | None) -> str | None:
        """Причина, по которой таймер поставить нельзя, либо None."""
        if self._reminders is None:
            return "напоминания не подключены"
        if chat_id is None:
            return "непонятно, в какой чат писать"
        if self._can_schedule is not None and not self._can_schedule(chat_id):
            return "в этом чате мне нельзя писать первой"
        return None


class CompletePromiseTool(Tool):
    """Отмечает ранее сохранённое обещание/напоминание выполненным и снимает его таймер."""

    name = "complete_promise"
    description = (
        "Отмечает выполненным обещание/напоминание из списка 'открытые задачи/обещания' в твоём текущем "
        "состоянии — используй, когда ты только что сделала то, что обещала раньше. Передай часть исходного "
        "текста, по которой его можно узнать. Если на обещание было поставлено напоминание, оно тоже снимется."
    )
    parameters = {
        "type": "object",
        "properties": {
            "text_query": {"type": "string", "description": "Часть текста обещания, по которой его найти"},
        },
        "required": ["text_query"],
        "additionalProperties": False,
    }

    def __init__(self, working_memory: WorkingMemory, *, reminders: ReminderStore | None = None) -> None:
        self._working_memory = working_memory
        self._reminders = reminders

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        text_query = str(arguments.get("text_query", "")).strip()
        if not text_query:
            return "error: text_query must not be empty"

        item = await self._working_memory.find_and_mark_done(text_query)
        if item is None:
            return f"Не нашла открытого обещания, похожего на {text_query!r}."

        # Снимаем таймер вместе с обещанием: напомнить о том, что уже сделано,
        # — та же ошибка, что и не напомнить о несделанном, только заметнее.
        if self._reminders is not None and context.chat_id is not None:
            cancelled = await self._reminders.cancel_matching(context.chat_id, text_query)
            if cancelled:
                logger.debug("promises: снято %d напоминание(ий) по %r", cancelled, text_query)
        return f"Отметила выполненным: {item.text!r}"


def _coerce_minutes(raw: Any) -> float | None:
    """
    Разбирает срок. None означает «срока нет», а не «ноль»: это разные вещи,
    и путать их нельзя — обещание без срока не должно превращаться в
    немедленное сообщение.
    """
    if raw is None or raw == "":
        return None
    try:
        minutes = float(str(raw).strip().replace(",", "."))
    except (TypeError, ValueError):
        logger.debug("promises: не разобрала remind_in_minutes=%r, считаю, что срока нет", raw)
        return None
    if minutes <= 0:
        return None
    return minutes


def _describe_delay(minutes: float) -> str:
    """Человекочитаемый срок для ответа модели — с учётом того, что он мог быть зажат границами."""
    clamped = max(MIN_DELAY.total_seconds() / 60, min(minutes, MAX_DELAY.total_seconds() / 60))
    if clamped < 60:
        return f"{round(clamped)} мин"
    if clamped < 60 * 24:
        return f"{clamped / 60:.1f} ч".replace(".0", "")
    return f"{clamped / (60 * 24):.1f} дн".replace(".0", "")


__all__ = ["CanScheduleCheck", "CompletePromiseTool", "RememberPromiseTool"]
