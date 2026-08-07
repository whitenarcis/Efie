"""
Тесты для efi.memory.working_memory.WorkingMemory.find_and_mark_done и
инструментов, замыкающих цикл рабочей памяти (efi.tools.memory_tools.
update_self_state.UpdateSelfStateTool, efi.tools.memory_tools.manage_promises).
Регрессия на дефект: рабочая память читалась в промпт/BusyEngine, но ничем
не заполнялась — ни один инструмент не давал модели способа её населить.
"""

from __future__ import annotations

from pathlib import Path

from efi.memory.working_memory import WorkingMemory
from efi.notifications.schemas import Notification, NotificationType
from efi.tools.base import ToolContext
from efi.tools.memory_tools.manage_promises import CompletePromiseTool, RememberPromiseTool
from efi.tools.memory_tools.update_self_state import UpdateSelfStateTool


def _context() -> ToolContext:
    return ToolContext(notification=Notification(type=NotificationType.USER_MESSAGE, chat_id=1, message="x"))


async def test_find_and_mark_done_matches_substring_case_insensitive(tmp_path: Path) -> None:
    wm = WorkingMemory(tmp_path / "wm.json")
    await wm.add_item("скинуть Роме ссылку на статью")

    found = await wm.find_and_mark_done("РОМЕ ссылку")
    assert found is not None
    assert found.done is True

    snapshot = await wm.load()
    assert snapshot.items[0].done is True


async def test_find_and_mark_done_returns_none_without_match(tmp_path: Path) -> None:
    wm = WorkingMemory(tmp_path / "wm.json")
    await wm.add_item("скинуть ссылку")
    assert await wm.find_and_mark_done("совершенно другое") is None


async def test_find_and_mark_done_skips_already_done_items(tmp_path: Path) -> None:
    wm = WorkingMemory(tmp_path / "wm.json")
    await wm.add_item("напомнить про фильм")
    await wm.find_and_mark_done("напомнить про фильм")
    # Уже выполненный пункт не должен находиться повторно.
    assert await wm.find_and_mark_done("напомнить про фильм") is None


async def test_update_self_state_tool_updates_energy(tmp_path: Path) -> None:
    wm = WorkingMemory(tmp_path / "wm.json")
    tool = UpdateSelfStateTool(wm)

    result = await tool.execute({"energy": 0.2, "physical_state": "устала"}, _context())
    assert "Обновила" in result

    snapshot = await wm.load()
    assert snapshot.energy == 0.2
    assert snapshot.physical_state == "устала"


async def test_update_self_state_tool_requires_at_least_one_field(tmp_path: Path) -> None:
    wm = WorkingMemory(tmp_path / "wm.json")
    tool = UpdateSelfStateTool(wm)
    result = await tool.execute({}, _context())
    assert result.startswith("error:")


async def test_remember_and_complete_promise_round_trip(tmp_path: Path) -> None:
    wm = WorkingMemory(tmp_path / "wm.json")
    remember = RememberPromiseTool(wm)
    complete = CompletePromiseTool(wm)

    remember_result = await remember.execute({"text": "скинуть плейлист"}, _context())
    assert "Запомнила" in remember_result

    snapshot = await wm.load()
    assert len(snapshot.items) == 1
    assert snapshot.items[0].done is False

    complete_result = await complete.execute({"text_query": "плейлист"}, _context())
    assert "Отметила выполненным" in complete_result

    snapshot = await wm.load()
    assert snapshot.items[0].done is True


async def test_complete_promise_tool_reports_when_not_found(tmp_path: Path) -> None:
    wm = WorkingMemory(tmp_path / "wm.json")
    complete = CompletePromiseTool(wm)
    result = await complete.execute({"text_query": "чего-то нет"}, _context())
    assert "Не нашла" in result
