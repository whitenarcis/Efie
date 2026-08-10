"""
Тесты отложенных напоминаний — «напиши мне через 10 минут».

Регрессия на баг, где механизм выглядел собранным, но был разорван в трёх
местах сразу: `remember_promise` записывал только текст (ни срока, ни чата),
`SilenceMonitor.schedule_follow_up` умел планировать, но его никто никогда не
вызывал, а очередь follow-up'ов жила в памяти процесса и проверялась раз в
15 минут. Снаружи это выглядело так: человек попросил, Эфи согласилась,
обещание повисло навсегда, и ничего не произошло.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from efi.behavior.reminders import (
    MAX_DELAY,
    MIN_DELAY,
    STALE_AFTER,
    ReminderScheduler,
    ReminderStore,
    resolve_due_at,
)
from efi.db.core import Database
from efi.db.models import MIGRATIONS
from efi.memory.working_memory import WorkingMemory
from efi.notifications.manager import NotificationManager
from efi.notifications.schemas import Notification, NotificationType
from efi.tools.base import ToolContext
from efi.tools.memory_tools.manage_promises import CompletePromiseTool, RememberPromiseTool

_CHAT_ID = 625207005


def _store(tmp_path: Path) -> ReminderStore:
    return ReminderStore(Database(tmp_path / "efi.db", migrations=MIGRATIONS))


def _context(chat_id: int | None = _CHAT_ID) -> ToolContext:
    return ToolContext(
        notification=Notification(type=NotificationType.USER_MESSAGE, chat_id=chat_id, message="напиши через 10 минут")
    )


async def _drain(manager: NotificationManager) -> list[Notification]:
    """Забирает всё, что лежит в очередях, не блокируясь на пустых."""
    collected: list[Notification] = []
    for index in range(manager.worker_count):
        while manager.qsize(index):
            collected.append(await manager.get(index))
            manager.task_done(index)
    return collected


# -- срок ---------------------------------------------------------------------


def test_delay_is_clamped_to_sane_bounds() -> None:
    now = datetime(2026, 8, 9, 12, 0, tzinfo=UTC)

    assert resolve_due_at(10, now=now) == now + timedelta(minutes=10)
    assert resolve_due_at(0, now=now) == now + MIN_DELAY, "«прямо сейчас» не должно бить поверх текущего ответа"
    assert resolve_due_at(10**9, now=now) == now + MAX_DELAY


# -- хранилище ----------------------------------------------------------------


async def test_scheduled_reminder_is_pending_until_due(tmp_path: Path) -> None:
    store = _store(tmp_path)
    now = datetime.now(UTC)
    await store.schedule(_CHAT_ID, "написать про собеседование", due_at=now + timedelta(minutes=10))

    assert await store.due(now) == []
    assert len(await store.due(now + timedelta(minutes=11))) == 1


async def test_reminder_survives_a_restart(tmp_path: Path) -> None:
    """
    Главная причина, по которой напоминания в SQLite, а не в памяти: «через
    10 минут» обязано пережить перезапуск, иначе оно тихо исчезает ровно
    тогда, когда человек на него рассчитывает.
    """
    first = _store(tmp_path)
    await first.schedule(_CHAT_ID, "написать про собеседование", due_at=datetime.now(UTC) + timedelta(minutes=10))

    reborn = ReminderStore(Database(tmp_path / "efi.db", migrations=MIGRATIONS))
    pending = await reborn.pending(_CHAT_ID)

    assert [reminder.text for reminder in pending] == ["написать про собеседование"]


async def test_repeated_request_replaces_instead_of_duplicating(tmp_path: Path) -> None:
    """Человек, повторивший просьбу, уточняет срок, а не просит написать дважды."""
    store = _store(tmp_path)
    now = datetime.now(UTC)
    await store.schedule(_CHAT_ID, "написать про собеседование", due_at=now + timedelta(minutes=10))
    await store.schedule(_CHAT_ID, "написать про собеседование", due_at=now + timedelta(minutes=30))

    pending = await store.pending(_CHAT_ID)
    assert len(pending) == 1
    assert pending[0].scheduled_at > now + timedelta(minutes=20)


async def test_cancel_matching_removes_by_substring(tmp_path: Path) -> None:
    store = _store(tmp_path)
    await store.schedule(_CHAT_ID, "написать про собеседование", due_at=datetime.now(UTC) + timedelta(minutes=10))

    assert await store.cancel_matching(_CHAT_ID, "собеседование") == 1
    assert await store.pending(_CHAT_ID) == []


# -- планировщик --------------------------------------------------------------


async def test_due_reminder_is_queued_as_follow_up(tmp_path: Path) -> None:
    """Главная регрессия: срок наступил — событие обязано появиться в очереди."""
    store = _store(tmp_path)
    manager = NotificationManager(worker_count=1)
    await store.schedule(_CHAT_ID, "написать про собеседование", due_at=datetime.now(UTC) - timedelta(seconds=1))

    queued = await ReminderScheduler(manager, store).tick()

    assert queued == 1
    notifications = await _drain(manager)
    assert len(notifications) == 1
    assert notifications[0].type is NotificationType.FOLLOW_UP
    assert notifications[0].chat_id == _CHAT_ID
    assert notifications[0].payload["promise_text"] == "написать про собеседование"
    assert "написать про собеседование" in notifications[0].message


async def test_reminder_fires_once(tmp_path: Path) -> None:
    """Иначе оно ставилось бы в очередь каждые полминуты, бесконечно."""
    store = _store(tmp_path)
    manager = NotificationManager(worker_count=1)
    await store.schedule(_CHAT_ID, "написать", due_at=datetime.now(UTC) - timedelta(seconds=1))
    scheduler = ReminderScheduler(manager, store)

    assert await scheduler.tick() == 1
    assert await scheduler.tick() == 0


async def test_reminder_has_higher_priority_than_idle_pings(tmp_path: Path) -> None:
    """У напоминания есть срок, названный человеком; опоздать с ним хуже, чем с «просто написать первой»."""
    store = _store(tmp_path)
    manager = NotificationManager(worker_count=1)
    await store.schedule(_CHAT_ID, "написать", due_at=datetime.now(UTC) - timedelta(seconds=1))
    await ReminderScheduler(manager, store).tick()

    notification = (await _drain(manager))[0]
    assert notification.priority < 6


async def test_stale_reminder_is_dropped_not_delivered(tmp_path: Path) -> None:
    """Приложение стояло сутки — «ты просил напомнить» после этого страннее, чем молчание."""
    store = _store(tmp_path)
    manager = NotificationManager(worker_count=1)
    await store.schedule(_CHAT_ID, "написать", due_at=datetime.now(UTC) - STALE_AFTER - timedelta(hours=1))

    assert await ReminderScheduler(manager, store).tick() == 0
    assert await store.pending(_CHAT_ID) == []


async def test_broken_row_does_not_stop_the_scheduler(tmp_path: Path) -> None:
    """Мусорная строка в таблице не должна ронять цикл — иначе не сработают и все остальные напоминания."""
    database = Database(tmp_path / "efi.db", migrations=MIGRATIONS)
    store = ReminderStore(database)
    manager = NotificationManager(worker_count=1)
    await database.execute(
        """
        INSERT INTO proactive_tasks (chat_id, task_type, scheduled_at, payload, status, created_at)
        VALUES (?, 'follow_up', ?, 'не json', 'pending', ?)
        """,
        (_CHAT_ID, datetime.now(UTC).isoformat(), datetime.now(UTC).isoformat()),
    )
    await store.schedule(_CHAT_ID, "нормальное напоминание", due_at=datetime.now(UTC) - timedelta(seconds=1))

    assert await ReminderScheduler(manager, store).tick() == 1


# -- инструмент ---------------------------------------------------------------


@pytest.fixture
def working_memory(tmp_path: Path) -> WorkingMemory:
    return WorkingMemory(tmp_path / "wm.json")


async def test_promise_with_a_deadline_schedules_a_reminder(
    tmp_path: Path, working_memory: WorkingMemory
) -> None:
    store = _store(tmp_path)
    tool = RememberPromiseTool(working_memory, reminders=store, can_schedule=lambda chat_id: True)

    answer = await tool.execute({"text": "написать про собеседование", "remind_in_minutes": 10}, _context())

    assert "напишу через 10 мин" in answer
    assert len(await store.pending(_CHAT_ID)) == 1
    item = (await working_memory.load()).items[0]
    assert item.due_at is not None
    assert item.chat_id == _CHAT_ID


async def test_promise_without_a_deadline_stays_a_note(tmp_path: Path, working_memory: WorkingMemory) -> None:
    """«скину, как найду» — не таймер: срока нет, и придумывать его нельзя."""
    store = _store(tmp_path)
    tool = RememberPromiseTool(working_memory, reminders=store, can_schedule=lambda chat_id: True)

    answer = await tool.execute({"text": "скину ссылку, как найду"}, _context())

    assert "без срока" in answer
    assert await store.pending(_CHAT_ID) == []
    assert (await working_memory.load()).items[0].due_at is None


async def test_tool_says_plainly_when_it_cannot_schedule(tmp_path: Path, working_memory: WorkingMemory) -> None:
    """
    Молчаливое «записала» при невозможности напомнить — ровно та поломка,
    из-за которой всё это переписывалось: снаружи согласие, по факту ничего.
    """
    store = _store(tmp_path)
    tool = RememberPromiseTool(working_memory, reminders=store, can_schedule=lambda chat_id: False)

    answer = await tool.execute({"text": "написать через час", "remind_in_minutes": 60}, _context())

    assert "напомнить сама не смогу" in answer
    assert await store.pending(_CHAT_ID) == []
    assert len((await working_memory.load()).items) == 1, "обещание всё равно остаётся её обещанием"


async def test_completing_a_promise_cancels_its_reminder(tmp_path: Path, working_memory: WorkingMemory) -> None:
    """Напомнить о сделанном — та же ошибка, что и не напомнить о несделанном."""
    store = _store(tmp_path)
    await RememberPromiseTool(working_memory, reminders=store, can_schedule=lambda chat_id: True).execute(
        {"text": "написать про собеседование", "remind_in_minutes": 10}, _context()
    )

    await CompletePromiseTool(working_memory, reminders=store).execute(
        {"text_query": "собеседование"}, _context()
    )

    assert await store.pending(_CHAT_ID) == []
    assert (await working_memory.load()).items[0].done is True


async def test_garbage_minutes_are_treated_as_no_deadline(
    tmp_path: Path, working_memory: WorkingMemory
) -> None:
    """«срока нет» и «ноль» — разные вещи: обещание без срока не должно уходить сообщением сразу."""
    store = _store(tmp_path)
    tool = RememberPromiseTool(working_memory, reminders=store, can_schedule=lambda chat_id: True)

    await tool.execute({"text": "что-нибудь", "remind_in_minutes": "потом"}, _context())

    assert await store.pending(_CHAT_ID) == []


# -- закрытие обещания после доставки ----------------------------------------


async def test_overdue_promise_is_marked_in_the_prompt(working_memory: WorkingMemory) -> None:
    """
    Если напоминание сработало, а сообщение не ушло, пункт остаётся открытым —
    и в промпте обязан читаться как просроченный, иначе спохватиться ей не с
    чего.
    """
    from efi.prompts.builder import _build_working_memory_block

    await working_memory.add_item(
        "написать про собеседование", due_at=datetime.now(UTC) - timedelta(minutes=5), chat_id=_CHAT_ID
    )
    block = _build_working_memory_block(await working_memory.load())

    assert "СРОК УЖЕ ПРОШЁЛ" in block


async def test_pending_promise_shows_its_deadline(working_memory: WorkingMemory) -> None:
    from efi.prompts.builder import _build_working_memory_block

    await working_memory.add_item(
        "написать про собеседование", due_at=datetime.now(UTC) + timedelta(hours=1), chat_id=_CHAT_ID
    )
    block = _build_working_memory_block(await working_memory.load())

    assert "к " in block
    assert "СРОК УЖЕ ПРОШЁЛ" not in block
