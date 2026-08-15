"""
Тесты совместного проектирования: «давай напишем X» и что происходит дальше.

Главное, что здесь проверяется, — что Эфи НЕ соглашается сразу. Это не
вопрос вкуса: согласиться в ту же реплику («отличная идея, уже приступаю») —
самый вероятный ответ языковой модели на предложение, и без механического
запрета разговор о том, ЧТО именно писать, не случается никогда.

Запрет проверяется с двух сторон: стол переговоров не разрешает старт
(`may_start`), а инструмент запуска физически не показывается модели
(`is_available`) — промпт может быть проигнорирован, отсутствующий
инструмент вызвать нельзя.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from efi.behavior.collab_coding import CollabCodingDesk, detect_proposal
from efi.db.core import Database
from efi.db.models import MIGRATIONS
from efi.dev.schemas import DevTaskStatus
from efi.dev.store import DevTaskStore
from efi.notifications.schemas import Notification, NotificationType
from efi.prompts.builder import _build_collab_block
from efi.tools.base import ToolContext
from efi.tools.dev_tools.start_project import StartDevProjectTool

_CHAT_ID = 4242
_IDEA = "давай напишем cli-клиент для отслеживания релизов в репозиториях"


def _desk(tmp_path: Path) -> tuple[CollabCodingDesk, DevTaskStore]:
    store = DevTaskStore(Database(tmp_path / "efi.db", migrations=MIGRATIONS))
    return CollabCodingDesk(store), store


def _context(chat_id: int | None = _CHAT_ID) -> ToolContext:
    return ToolContext(
        notification=Notification(type=NotificationType.USER_MESSAGE, chat_id=chat_id, message="ок")
    )


# -- распознавание предложения ------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "давай напишем парсер для выгрузки статистики",
        "а давай напишем telegram-бота для напоминаний",
        "может сделаем утилиту для бэкапов",
        "давай запилим cli для конвертации csv",
    ],
)
def test_project_proposals_are_recognized(text: str) -> None:
    assert detect_proposal(text) is not None


@pytest.mark.parametrize(
    "text",
    [
        "давай сделаем паузу",
        "давай сделаем это завтра",
        "напиши мне через час",
        "бот какой-то тупой",
        "",
    ],
)
def test_ordinary_talk_is_not_a_proposal(text: str) -> None:
    """
    Двойное условие (маркер предложения И технический предмет) существует
    ровно затем, чтобы «давай сделаем паузу» не заводило проект.
    """
    assert detect_proposal(text) is None


# -- «не соглашайся слепо» ----------------------------------------------------


async def test_first_message_does_not_allow_starting_work(tmp_path: Path) -> None:
    desk, _store = _desk(tmp_path)

    await desk.consider_message(_CHAT_ID, _IDEA)

    assert desk.pending(_CHAT_ID) is not None, "предложение замечено"
    assert desk.may_start(_CHAT_ID) is False, "но браться за работу ещё нельзя"


async def test_the_tool_is_hidden_until_the_idea_is_discussed(tmp_path: Path) -> None:
    """
    Промпт можно проигнорировать, отсутствующий инструмент вызвать нельзя.
    Реестр проверяет доступность и при показе, и при исполнении.
    """
    desk, _store = _desk(tmp_path)
    tool = StartDevProjectTool(desk)
    await desk.consider_message(_CHAT_ID, _IDEA)

    assert tool.is_available(_context()) is False

    await desk.consider_message(_CHAT_ID, "на python, без внешних зависимостей")

    assert tool.is_available(_context()) is True


async def test_starting_before_the_discussion_is_refused_by_the_tool(tmp_path: Path) -> None:
    """Даже если модель как-то дотянулась до инструмента — состояние проверяется ещё раз при исполнении."""
    desk, _store = _desk(tmp_path)
    tool = StartDevProjectTool(desk)
    await desk.consider_message(_CHAT_ID, _IDEA)

    result = await tool.execute({"idea": "клиент для отслеживания релизов на python"}, _context())

    assert result.startswith("error:")
    assert "обсудите" in result


async def test_discussed_idea_becomes_a_task(tmp_path: Path) -> None:
    desk, store = _desk(tmp_path)
    tool = StartDevProjectTool(desk)
    await desk.consider_message(_CHAT_ID, _IDEA)
    await desk.consider_message(_CHAT_ID, "давай на python и без внешних зависимостей")

    result = await tool.execute(
        {"idea": "CLI для отслеживания релизов в отслеживаемых репозиториях, python, без зависимостей"},
        _context(),
    )

    assert not result.startswith("error:")
    tasks = await store.active()
    assert len(tasks) == 1
    assert tasks[0].is_collab is True
    assert tasks[0].chat_id == _CHAT_ID
    assert tasks[0].status is DevTaskStatus.PENDING
    assert desk.pending(_CHAT_ID) is None, "обсуждение закрыто, второй раз ту же идею не берём"


async def test_vague_idea_is_refused(tmp_path: Path) -> None:
    """«Сделай бота» — это не договорённость, и спека по нему получится такой же пустой."""
    desk, store = _desk(tmp_path)
    tool = StartDevProjectTool(desk)
    await desk.consider_message(_CHAT_ID, _IDEA)
    await desk.consider_message(_CHAT_ID, "ну давай на python")

    result = await tool.execute({"idea": "бота"}, _context())

    assert result.startswith("error:")
    assert await store.active() == []


async def test_notes_from_the_discussion_reach_the_task(tmp_path: Path) -> None:
    """То, о чём договорились по ходу (стек, ограничения), — часть замысла, а не забытая деталь."""
    desk, store = _desk(tmp_path)
    await desk.consider_message(_CHAT_ID, _IDEA)
    await desk.consider_message(_CHAT_ID, "на python, без внешних зависимостей")

    task = await desk.start(_CHAT_ID)

    assert task is not None
    assert "без внешних зависимостей" in task.idea
    stored = await store.get(task.id)
    assert stored is not None and stored.idea == task.idea


async def test_second_proposal_is_ignored_while_a_project_is_running(tmp_path: Path) -> None:
    """Три начатых проекта в одном чате — верный способ не доделать ни одного."""
    desk, store = _desk(tmp_path)
    await store.create("уже пишется", chat_id=_CHAT_ID, is_collab=True)

    await desk.consider_message(_CHAT_ID, "давай напишем ещё и парсер логов")

    assert desk.pending(_CHAT_ID) is None


# -- блок промпта -------------------------------------------------------------


async def test_prompt_tells_her_what_to_argue_about(tmp_path: Path) -> None:
    """
    Одного запрета мало: модель, которой сказали «не соглашайся», начинает
    спорить о смысле жизни. В блоке перечислено, о чём именно спрашивать.
    """
    desk, _store = _desk(tmp_path)
    await desk.consider_message(_CHAT_ID, _IDEA)

    block = _build_collab_block(desk.pending(_CHAT_ID))

    assert "НЕ соглашайся с ходу" in block
    assert "на чём писать" in block
    assert "отговорить" in block
    assert "нельзя" in block


async def test_prompt_switches_to_starting_after_the_discussion(tmp_path: Path) -> None:
    desk, _store = _desk(tmp_path)
    await desk.consider_message(_CHAT_ID, _IDEA)
    await desk.consider_message(_CHAT_ID, "на python")

    block = _build_collab_block(desk.pending(_CHAT_ID))

    assert "start_dev_project" in block


def test_no_proposal_means_no_block() -> None:
    assert _build_collab_block(None) == ""
