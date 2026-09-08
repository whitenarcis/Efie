"""
Тесты разговора о коде: как обычная речь становится работой.

Никаких команд и префиксов — разбирается то, как люди пишут: «глянь репу»,
«тест падает, в чём дело», «допиши туда экспорт». Проверяется и обратное:
на разговор, который к коду отношения не имеет, она не должна кидаться
клонировать репозитории.

Отдельно — граница между «делай» и «сначала обсудим». Починить импорт можно
молча, переписать проект на async — нет, и это не вежливость: у первого цена
ошибки в одну строчку, у второго — чужой вечер.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from efi.behavior.dev_dialogue import DevIntentKind, DevPartnerDesk, detect_dev_intent
from efi.db.core import Database
from efi.db.models import MIGRATIONS
from efi.dev.schemas import DevTaskKind, DevTaskStatus
from efi.dev.store import DevTaskStore
from efi.notifications.schemas import Notification, NotificationType
from efi.prompts.builder import _build_dev_partner_block
from efi.tools.base import ToolContext
from efi.tools.dev_tools.work_on_repo import WorkOnRepoTool

_CHAT = 42


# -- разбор речи --------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("глянь репу https://github.com/someone/tool", DevIntentKind.LOOK),
        ("тест падает, в чём дело", DevIntentKind.FIX),
        ("почини импорт в парсере", DevIntentKind.FIX),
        ("допиши в модуль флаг --json", DevIntentKind.CHANGE),
        ("давай перепишем это всё на async", DevIntentKind.HEAVY),
    ],
)
def test_ordinary_speech_is_understood(text: str, kind: DevIntentKind) -> None:
    intent = detect_dev_intent(text)

    assert intent is not None, text
    assert intent.kind is kind


@pytest.mark.parametrize(
    "text",
    [
        "посмотри в окно, там снег",
        "давай сходим за кофе",
        "у меня всё падает из рук сегодня",
        "",
    ],
)
def test_life_outside_code_is_not_a_work_request(text: str) -> None:
    """
    Цена ложного срабатывания — она полезет клонировать репозиторий вместо
    ответа на вопрос. Это заметнее, чем пропущенная просьба: пропущенную
    человек повторит.
    """
    assert detect_dev_intent(text) is None


def test_the_repository_is_taken_from_the_message() -> None:
    intent = detect_dev_intent("глянь https://github.com/user/repo.git, там тесты красные")

    assert intent is not None
    assert intent.source == "https://github.com/user/repo.git"


def test_a_local_path_counts_too() -> None:
    intent = detect_dev_intent("посмотри код в ~/projects/efi, там модуль странный")

    assert intent is not None
    assert intent.source == "~/projects/efi"


def test_big_rewrites_are_not_something_you_start_silently() -> None:
    intent = detect_dev_intent("давай перепишем хранилище на async")

    assert intent is not None
    assert intent.kind.needs_discussion is True
    assert intent.is_actionable is False


# -- контекст разговора -------------------------------------------------------


def _desk(tmp_path: Path, *, available: bool = True) -> DevPartnerDesk:
    store = DevTaskStore(Database(tmp_path / "efi.db", migrations=MIGRATIONS))
    return DevPartnerDesk(store, available=available)


def test_a_follow_up_is_understood_without_repeating_the_word_code(tmp_path: Path) -> None:
    """
    После «глянь репу X» реплика «допиши туда флаг --json» очевидно про код.
    Требовать в каждой фразе слово «код» — значит понимать только первую
    фразу из беседы.
    """
    desk = _desk(tmp_path)
    desk.consider_message(_CHAT, "глянь репу https://github.com/user/repo")

    intent = desk.consider_message(_CHAT, "допиши туда флаг --json")

    assert intent is not None
    assert intent.kind is DevIntentKind.CHANGE
    assert desk.may_work(_CHAT) is True


def test_the_repository_is_remembered_for_the_next_requests(tmp_path: Path) -> None:
    """Сказанное один раз «вот моя репа» действует и для следующих просьб — так говорят люди."""
    desk = _desk(tmp_path)

    desk.consider_message(_CHAT, "глянь репу https://github.com/user/repo")
    desk.consider_message(_CHAT, "и почини там импорт")

    assert desk.resolve_source(_CHAT) == "https://github.com/user/repo"
    assert desk.may_work(_CHAT) is True


def test_without_a_repository_there_is_nothing_to_work_on(tmp_path: Path) -> None:
    desk = _desk(tmp_path)

    desk.consider_message(_CHAT, "почини там импорт")

    assert desk.may_work(_CHAT) is False, "непонятно, где чинить, — сначала надо спросить"


def test_a_big_rewrite_does_not_unlock_the_tool(tmp_path: Path) -> None:
    desk = _desk(tmp_path)

    desk.consider_message(_CHAT, "глянь https://github.com/user/repo")
    desk.consider_message(_CHAT, "давай перепишем это всё на async")

    assert desk.may_work(_CHAT) is False


def test_with_the_engine_down_she_cannot_promise_anything(tmp_path: Path) -> None:
    """Обещание, которое некому выполнить, хуже честного «не могу»: человек ждёт ветку."""
    desk = _desk(tmp_path, available=False)

    desk.consider_message(_CHAT, "глянь репу https://github.com/user/repo и почини импорт")

    assert desk.may_work(_CHAT) is False


async def test_a_request_becomes_a_task_of_its_own_kind(tmp_path: Path) -> None:
    """
    SWE-задача не должна уехать в конвейер собственных проектов: там её
    попытались бы спроектировать с нуля вместо того, чтобы починить импорт.
    """
    store = DevTaskStore(Database(tmp_path / "efi.db", migrations=MIGRATIONS))
    desk = DevPartnerDesk(store, available=True)
    desk.consider_message(_CHAT, "глянь https://github.com/user/repo, там импорт сломан")

    task = await desk.start(_CHAT, instruction="почини импорт в src/main.py")

    assert task is not None
    assert task.kind is DevTaskKind.SWE
    assert task.source == "https://github.com/user/repo"
    assert await store.next_pending() is None, "конвейер проектов её не видит"
    swe = await store.next_pending(kind=DevTaskKind.SWE)
    assert swe is not None and swe.id == task.id


async def test_a_second_request_does_not_start_a_parallel_task(tmp_path: Path) -> None:
    store = DevTaskStore(Database(tmp_path / "efi.db", migrations=MIGRATIONS))
    desk = DevPartnerDesk(store, available=True)
    desk.consider_message(_CHAT, "глянь https://github.com/user/repo, почини импорт")
    first = await desk.start(_CHAT, instruction="почини импорт")

    desk.consider_message(_CHAT, "а ещё допиши там флаг --json")
    second = await desk.start(_CHAT, instruction="допиши флаг --json")

    assert first is not None
    assert second is None, "две задачи разом в одном чате — верный способ не доделать обе"


async def test_branches_she_made_are_remembered_for_the_talk(tmp_path: Path) -> None:
    desk = _desk(tmp_path)
    desk.consider_message(_CHAT, "глянь https://github.com/user/repo")

    desk.remember_branch(_CHAT, "fix/import-bug")

    context = desk.context(_CHAT)
    assert context is not None
    assert "fix/import-bug" in context.render_for_prompt()


# -- обещание словами ---------------------------------------------------------


async def test_a_promise_in_her_own_words_becomes_a_task(tmp_path: Path) -> None:
    """
    Самый неприятный исход обсуждения: договорились, она написала «сейчас
    гляну» — и не вызвала инструмент. Для человека это неотличимо от
    согласия, но не порождает ничего.
    """
    store = DevTaskStore(Database(tmp_path / "efi.db", migrations=MIGRATIONS))
    desk = DevPartnerDesk(store, available=True)
    desk.consider_message(_CHAT, "глянь https://github.com/user/repo, там импорт сломан")

    task = await desk.consider_reply(_CHAT, "ага, сейчас гляну и напишу, что там")

    assert task is not None
    assert task.kind is DevTaskKind.SWE


async def test_an_ordinary_reply_promises_nothing(tmp_path: Path) -> None:
    store = DevTaskStore(Database(tmp_path / "efi.db", migrations=MIGRATIONS))
    desk = DevPartnerDesk(store, available=True)
    desk.consider_message(_CHAT, "глянь https://github.com/user/repo")

    assert await desk.consider_reply(_CHAT, "а что там за проект вообще?") is None
    assert await desk.consider_reply(_CHAT, "не буду я это чинить, там всё гнилое") is None
    assert await store.next_pending(kind=DevTaskKind.SWE) is None


# -- инструмент и промпт ------------------------------------------------------


def _context() -> ToolContext:
    return ToolContext(
        notification=Notification(
            type=NotificationType.USER_MESSAGE, chat_id=_CHAT, message="почини импорт"
        )
    )


def test_the_tool_appears_only_when_there_is_something_to_work_on(tmp_path: Path) -> None:
    desk = _desk(tmp_path)
    tool = WorkOnRepoTool(desk)

    assert tool.is_available(_context()) is False, "репозиторий ещё не назывался"

    desk.consider_message(_CHAT, "глянь https://github.com/user/repo и почини импорт")
    assert tool.is_available(_context()) is True


async def test_the_tool_says_what_is_missing_instead_of_failing_silently(tmp_path: Path) -> None:
    desk = _desk(tmp_path)
    desk.consider_message(_CHAT, "почини импорт в модуле")

    answer = await WorkOnRepoTool(desk).execute({"instruction": "почини импорт в модуле"}, _context())

    assert "error" in answer
    assert "репозиторием" in answer


async def test_starting_work_reports_the_task_and_the_repository(tmp_path: Path) -> None:
    store = DevTaskStore(Database(tmp_path / "efi.db", migrations=MIGRATIONS))
    desk = DevPartnerDesk(store, available=True)
    desk.consider_message(_CHAT, "глянь https://github.com/user/repo, почини импорт")

    answer = await WorkOnRepoTool(desk).execute(
        {"instruction": "почини импорт в src/main.py"}, _context()
    )

    assert "Взяла в работу" in answer
    assert "https://github.com/user/repo" in answer
    task = await store.next_pending(kind=DevTaskKind.SWE)
    assert task is not None and task.status is DevTaskStatus.PENDING


def test_the_prompt_tells_her_to_argue_about_rewrites_and_to_just_do_small_things() -> None:
    small = detect_dev_intent("почини импорт в парсере")
    big = detect_dev_intent("давай перепишем всё хранилище на async")
    assert small is not None and big is not None

    small_block = _build_dev_partner_block(small, None, engine_available=True)
    big_block = _build_dev_partner_block(big, None, engine_available=True)

    assert "work_on_repo" in small_block
    assert "не переспрашивай" in small_block
    assert "спорить" in big_block or "право спорить" in big_block
    assert "work_on_repo" not in big_block, "крупную переделку нельзя начать молча"


def test_with_the_engine_down_the_prompt_forbids_promising() -> None:
    intent = detect_dev_intent("почини импорт в парсере")
    assert intent is not None

    block = _build_dev_partner_block(intent, None, engine_available=False)

    assert "не можешь" in block
    assert "без обещаний" in block
