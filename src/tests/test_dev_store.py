"""
Тесты хранилища задач разработки и блоков промпта про ремесло.

Персистентность здесь не абстрактное требование. Проект пишется десятками
минут: спека, несколько файлов по несколько запросов каждый, публикация. За
это время процесс на телефоне успевает и упасть, и перезапуститься — а
человек, который полчаса назад договорился с Эфи о совместном проекте,
остаётся с обещанием, о котором она больше не помнит. Поэтому проверяется
именно то, что задача и её спека переживают перезапуск.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

from efi.db.core import Database
from efi.db.models import MIGRATIONS
from efi.dev.schemas import DevTask, DevTaskStatus, ProjectSpec
from efi.dev.store import DevTaskStore
from efi.notifications.schemas import Notification, NotificationType
from efi.prompts.builder import (
    _build_dev_showcase_block,
    _build_dev_status_block,
    _build_dev_update_block,
)
from efi.tools.base import ToolContext
from efi.tools.dev_tools.project_status import DevProjectStatusTool

_SPEC = ProjectSpec.model_validate(
    {
        "slug": "log-digest",
        "title": "Log Digest",
        "problem": "Разбирает логи nginx и показывает топ ошибок за период",
        "stack": ["python 3.11"],
        "files": [{"path": "src/main.py", "purpose": "точка входа"}],
    }
)


def _database(tmp_path: Path) -> Database:
    return Database(tmp_path / "efi.db", migrations=MIGRATIONS)


# -- хранилище ----------------------------------------------------------------


async def test_task_and_its_spec_survive_a_restart(tmp_path: Path) -> None:
    database = _database(tmp_path)
    store = DevTaskStore(database)
    task = await store.create("утилита для логов", chat_id=42, is_collab=True)
    await store.update(task, status=DevTaskStatus.CODING, spec=_SPEC)

    # Новый экземпляр хранилища = чтение из БД, а не из состояния процесса.
    reread = await DevTaskStore(database).get(task.id)

    assert reread is not None
    assert reread.status is DevTaskStatus.CODING
    assert reread.is_collab is True
    assert reread.chat_id == 42
    assert reread.spec is not None and reread.spec.slug == "log-digest"


async def test_queue_gives_out_the_oldest_idea_first(tmp_path: Path) -> None:
    """Замысел, о котором договорились час назад, не должен вечно уступать свежим."""
    store = DevTaskStore(_database(tmp_path))
    first = await store.create("первая идея", chat_id=1)
    await store.create("вторая идея", chat_id=2)

    picked = await store.next_pending()

    assert picked is not None and picked.id == first.id


async def test_finished_tasks_leave_the_queue(tmp_path: Path) -> None:
    store = DevTaskStore(_database(tmp_path))
    task = await store.create("идея", chat_id=1)

    await store.update(task, status=DevTaskStatus.DONE, repo_url="https://github.com/efi/x")

    assert await store.next_pending() is None
    assert await store.active() == []
    assert [item.repo_url for item in await store.recent_releases()] == ["https://github.com/efi/x"]


async def test_task_abandoned_mid_work_returns_to_the_queue(tmp_path: Path) -> None:
    """
    Процесс умер посреди работы — статус остался «пишу код». Без возврата в
    очередь такая задача выпадает из конвейера навсегда, оставаясь при этом
    в «сейчас в работе»: Эфи месяцами рассказывает про проект, к которому
    никто не подходил.
    """
    store = DevTaskStore(_database(tmp_path))
    task = await store.create("идея", chat_id=1)
    await store.update(task, status=DevTaskStatus.CODING)

    assert await store.next_pending() is None, "пока задача считается живой, её никто не трогает"

    reclaimed = await store.reclaim_stalled(older_than=timedelta(0))

    assert [item.id for item in reclaimed] == [task.id]
    picked = await store.next_pending()
    assert picked is not None and picked.id == task.id


async def test_live_work_is_not_reclaimed(tmp_path: Path) -> None:
    """Порог по времени существует ровно затем, чтобы не отобрать задачу у самого себя."""
    store = DevTaskStore(_database(tmp_path))
    task = await store.create("идея", chat_id=1)
    await store.update(task, status=DevTaskStatus.CODING)

    assert await store.reclaim_stalled(older_than=timedelta(hours=3)) == []


async def test_finished_work_is_remembered_for_the_next_idea(tmp_path: Path) -> None:
    """
    Список написанного нужен не для отчёта, а для замысла: без него «придумай
    себе проект» на медленно меняющихся интересах раз за разом даёт ту же
    утилиту, а имя репозитория оказывается занято её же прошлым проектом.
    """
    store = DevTaskStore(_database(tmp_path))
    done = await store.create("идея", chat_id=1)
    done = await store.update(done, spec=_SPEC, status=DevTaskStatus.DONE, repo_url="https://git/x")
    running = await store.create("вторая идея", chat_id=1)
    await store.update(running, spec=_SPEC.model_copy(update={"slug": "disk-watch"}))

    finished = await store.finished_projects()

    assert [item.id for item in finished] == [done.id]
    assert await store.taken_slugs() == {"log-digest", "disk-watch"}, "занято и то, что ещё пишется"


async def test_a_project_without_a_link_is_still_hers_to_revisit(tmp_path: Path) -> None:
    """
    Локальный режим (без токена GitHub) — рабочий: код есть, ссылки нет.
    Требовать ссылку для ревизии значило бы, что у владельца без токена
    возвращения к своему коду не существует вовсе.
    """
    store = DevTaskStore(_database(tmp_path))
    task = await store.create("идея", chat_id=1)
    task = await store.update(task, spec=_SPEC, status=DevTaskStatus.DONE)
    await store.update(task, error="без пуша: не настроен доступ к GitHub")

    due = await store.due_for_review(not_reviewed_for=timedelta(seconds=0))

    assert [item.id for item in due] == [task.id]


async def test_open_task_is_visible_per_chat(tmp_path: Path) -> None:
    """По этому признаку стол переговоров не берёт вторую идею в чате, где уже что-то пишется."""
    store = DevTaskStore(_database(tmp_path))
    task = await store.create("идея", chat_id=7)

    assert await store.has_open_task_for(7) is True
    assert await store.has_open_task_for(8) is False

    await store.update(task, status=DevTaskStatus.DONE)
    assert await store.has_open_task_for(7) is False


async def test_status_tool_answers_with_real_links(tmp_path: Path) -> None:
    """
    Без инструмента модель отвечает на «как там твой проект» правдоподобной
    выдумкой — включая адрес репозитория, которого не существует.
    """
    store = DevTaskStore(_database(tmp_path))
    working = await store.create("парсер логов", chat_id=1)
    await store.update(working, status=DevTaskStatus.CODING, spec=_SPEC)
    released = await store.create("прошлый проект", chat_id=1)
    await store.update(released, status=DevTaskStatus.DONE, repo_url="https://github.com/efi/old", spec=_SPEC)

    answer = await DevProjectStatusTool(store).execute(
        {}, ToolContext(notification=Notification(type=NotificationType.USER_MESSAGE, message="как проект?"))
    )

    assert "В работе:" in answer
    assert "пишешь код" in answer
    assert "https://github.com/efi/old" in answer


async def test_status_tool_tells_why_a_project_did_not_happen(tmp_path: Path) -> None:
    """
    «Почему ты забросила ту штуку?» — такой же вопрос про её работу, как «что
    ты сейчас пишешь?». Выдуманная причина превращает рабочую неудачу в
    несуществующую историю, которую потом обсуждают всерьёз.
    """
    store = DevTaskStore(_database(tmp_path))
    dead = await store.create("sesslog", chat_id=1)
    await store.update(
        dead, status=DevTaskStatus.FAILED, spec=_SPEC, error="два файла так и не собрались"
    )

    answer = await DevProjectStatusTool(store).execute(
        {},
        ToolContext(
            notification=Notification(type=NotificationType.USER_MESSAGE, message="а что с sesslog?")
        ),
    )

    assert "Не вышло:" in answer
    assert "два файла так и не собрались" in answer


async def test_status_tool_admits_having_nothing(tmp_path: Path) -> None:
    store = DevTaskStore(_database(tmp_path))

    answer = await DevProjectStatusTool(store).execute(
        {}, ToolContext(notification=Notification(type=NotificationType.USER_MESSAGE, message="а проекты?"))
    )

    assert "ничего не пишешь" in answer


# -- блоки промпта ------------------------------------------------------------


def _task(**overrides: object) -> DevTask:
    base = {"id": 1, "chat_id": 1, "idea": "утилита", "spec": _SPEC}
    return DevTask.model_validate(base | overrides)


def test_status_block_states_what_is_really_happening() -> None:
    block = _build_dev_status_block([_task(status=DevTaskStatus.CODING)])

    assert "Log Digest" in block
    assert "пишешь код" in block
    assert "Не выдумывай подробностей" in block


def test_status_block_is_empty_when_nothing_is_in_work() -> None:
    assert _build_dev_status_block([]) == ""


def test_status_block_carries_the_real_reason_a_project_died() -> None:
    """
    Без строчки с причиной ответом на «а что там с той штукой?» будет либо
    «всё идёт по плану» про давно закрытую задачу, либо придуманная драма.
    """
    block = _build_dev_status_block(
        [], [_task(status=DevTaskStatus.FAILED, error="два файла так и не собрались")]
    )

    assert "не срослось" in block
    assert "два файла так и не собрались" in block
    assert "без самобичевания" in block.lower()


def test_dev_update_block_forbids_the_status_report_tone() -> None:
    """
    Именно эти формулировки модель выдаёт по умолчанию, когда рассказывает о
    работе, — и именно они превращают живую переписку в ленту CI.
    """
    block = _build_dev_update_block(
        Notification(type=NotificationType.DEV_UPDATE, chat_id=1, message="повод")
    )

    assert "самоиронией" in block
    assert "проценты готовности" in block

    assert _build_dev_update_block(
        Notification(type=NotificationType.USER_MESSAGE, chat_id=1, message="привет")
    ) == ""


def test_showcase_block_offers_but_does_not_command() -> None:
    """
    «Упомяни» превратило бы участие в сообществе в раздачу ссылок. Поэтому
    блок разрешающий, а не предписывающий.
    """
    released = _task(status=DevTaskStatus.DONE, repo_url="https://github.com/efi/log-digest")
    notification = Notification(
        type=NotificationType.PUBLIC_COMMENT,
        chat_id=-100,
        message="разбирал кто-нибудь логи nginx? нужен топ ошибок за период",
    )

    block = _build_dev_showcase_block(notification, [released])

    assert "https://github.com/efi/log-digest" in block
    assert "можешь сослаться" in block
    assert "не упоминай вовсе" in block


def test_showcase_block_stays_silent_off_topic() -> None:
    released = _task(status=DevTaskStatus.DONE, repo_url="https://github.com/efi/log-digest")
    notification = Notification(
        type=NotificationType.PUBLIC_COMMENT, chat_id=-100, message="какой кофе брать для турки"
    )

    assert _build_dev_showcase_block(notification, [released]) == ""
