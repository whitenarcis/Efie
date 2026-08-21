"""
Тесты того, как работа над проектом становится словами: реплики о процессе,
релиз со ссылкой, внешний флекс — и фоновый цикл, который всё это связывает.

Отдельно проверяется, что репортёр НЕ пишет текст сообщений. Он ставит
повод, а словами его делает Worker с личностью и историей чата (тот же
принцип, что у всех проактивных служб). Тесты поэтому смотрят на поводы:
есть ли в них факты, ради которых реплика затевалась, и нет ли в них
готовых «✅ Проект собран успешно».
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from efi.config.schema import QuietHoursSettings
from efi.db.core import Database
from efi.db.models import MIGRATIONS
from efi.dev import worker as worker_module
from efi.dev.engine import BuildResult
from efi.dev.github_sync import GitHubSyncError, PublishResult, RepoRef
from efi.dev.reporter import DevReporter
from efi.dev.schemas import DevTask, DevTaskStatus, GeneratedFile, ProjectSpec
from efi.dev.showcase import pick_showcase
from efi.dev.store import DevTaskStore
from efi.dev.worker import DevWorker
from efi.notifications.manager import NotificationManager
from efi.notifications.schemas import Notification, NotificationType

_CHAT_ID = 777
_SPEC = ProjectSpec.model_validate(
    {
        "slug": "log-digest",
        "title": "Log Digest",
        "problem": "Разбирает логи nginx и показывает топ ошибок за период",
        "stack": ["python 3.11"],
        "files": [{"path": "src/main.py", "purpose": "точка входа"}],
    }
)
_REPO = RepoRef(
    full_name="efi/log-digest",
    html_url="https://github.com/efi/log-digest",
    ssh_url="git@github.com:efi/log-digest.git",
    clone_url="https://github.com/efi/log-digest.git",
)


def _task(**overrides: Any) -> DevTask:
    base: dict[str, Any] = {
        "id": 1,
        "chat_id": _CHAT_ID,
        "idea": "утилита для разбора логов",
        "is_collab": True,
        "spec": _SPEC,
    }
    return DevTask.model_validate(base | overrides)


class _CollectingManager(NotificationManager):
    """Настоящий менеджер, но с доступом к тому, что в него положили."""

    def __init__(self) -> None:
        super().__init__(worker_count=1)
        self.notifications: list[Notification] = []

    async def put(self, notification: Notification) -> None:
        self.notifications.append(notification)
        await super().put(notification)


class _RecordingMemory:
    def __init__(self) -> None:
        self.records: list[Any] = []

    async def record(self, interaction: Any) -> int:
        self.records.append(interaction)
        return len(self.records)


# -- реплики о процессе -------------------------------------------------------


async def test_progress_note_becomes_a_reason_not_a_ready_text() -> None:
    manager = _CollectingManager()
    reporter = DevReporter(manager, progress_probability=1.0)

    assert await reporter.report_progress(_task(), "переписывала парсер 3 раза из-за линтера") is True

    notification = manager.notifications[0]
    assert notification.type is NotificationType.DEV_UPDATE
    assert notification.chat_id == _CHAT_ID
    assert "переписывала парсер 3 раза" in notification.message
    assert "ЗАПРЕЩЕНО" in notification.message, "правила тона — часть повода"
    assert notification.payload["dev_task_id"] == 1


async def test_progress_is_throttled_to_one_per_hour() -> None:
    """Разработка идёт этапами; без кулдауна одна задача выдавала бы по бабблу на файл."""
    manager = _CollectingManager()
    reporter = DevReporter(manager, progress_probability=1.0)

    assert await reporter.report_progress(_task(), "первый этап") is True
    assert await reporter.report_progress(_task(), "второй этап") is False
    assert len(manager.notifications) == 1


async def test_quiet_hours_silence_the_progress_talk() -> None:
    manager = _CollectingManager()
    now_hour = datetime.now().hour
    quiet = QuietHoursSettings(enabled=True, start_hour=now_hour, end_hour=(now_hour + 2) % 24)
    reporter = DevReporter(manager, progress_probability=1.0, quiet_hours=quiet)

    assert await reporter.report_progress(_task(), "линтер задушил") is False
    assert manager.notifications == []


async def test_unanswered_message_stops_the_progress_talk() -> None:
    """Рассказ о своей работе — тоже инициатива: правило «написала и не ответили — жди» действует и здесь."""
    manager = _CollectingManager()

    class _ClosedGate:
        async def may_initiate(self, chat_id: int, *, now: datetime | None = None) -> bool:
            return False

    reporter = DevReporter(manager, progress_probability=1.0, initiative=_ClosedGate())  # type: ignore[arg-type]

    assert await reporter.report_progress(_task(), "что-то сделала") is False


async def test_task_without_a_chat_has_nobody_to_tell() -> None:
    manager = _CollectingManager()
    reporter = DevReporter(manager, progress_probability=1.0)

    assert await reporter.report_progress(_task(chat_id=None), "своя затея") is False


# -- релиз --------------------------------------------------------------------


async def test_release_carries_the_link_and_lands_in_memory() -> None:
    manager = _CollectingManager()
    memory = _RecordingMemory()
    reporter = DevReporter(manager, social_memory=memory)  # type: ignore[arg-type]
    build = BuildResult(files=[GeneratedFile(path="src/main.py", content="x = 1", fix_rounds=2)])

    await reporter.report_release(_task(), url=_REPO.html_url, build=build)

    notification = manager.notifications[0]
    assert _REPO.html_url in notification.message
    assert "ОБЯЗАТЕЛЬНО дай ссылку" in notification.message
    assert "2 раз" in notification.message, "борьба с линтером — материал для живой реплики"
    assert memory.records[0].kind.value == "dev_release"
    assert _REPO.html_url in memory.records[0].text


async def test_release_of_a_private_project_is_still_remembered() -> None:
    """«Я сделала эту штуку» — часть её опыта, даже если рассказать об этом некому."""
    manager = _CollectingManager()
    memory = _RecordingMemory()
    reporter = DevReporter(manager, social_memory=memory)  # type: ignore[arg-type]

    await reporter.report_release(_task(chat_id=None), url=_REPO.html_url)

    assert manager.notifications == []
    assert len(memory.records) == 1


async def test_failure_is_reported_only_for_joint_projects() -> None:
    manager = _CollectingManager()
    reporter = DevReporter(manager)

    await reporter.report_failure(_task(is_collab=False), "кодер не осилил")
    assert manager.notifications == [], "о провале собственной затеи никто не просил отчитываться"

    await reporter.report_failure(_task(is_collab=True), "кодер не осилил")
    assert "кодер не осилил" in manager.notifications[0].message


# -- память о ремесле ---------------------------------------------------------


async def test_an_abandoned_project_is_remembered_with_its_reason() -> None:
    """
    Рассказ живёт в чате один вечер, память всплывает через неделю сама.
    «Почему ты забросила ту штуку с логами?» — вопрос, на который без записи
    ответом будет вежливая выдумка: признаться «не помню» модели тяжелее, чем
    сочинить.
    """
    memory = _RecordingMemory()
    reporter = DevReporter(_CollectingManager(), social_memory=memory)  # type: ignore[arg-type]

    await reporter.report_failure(_task(is_collab=False, attempts=3), "два файла так и не собрались")

    record = memory.records[0]
    assert record.kind.value == "dev_abandoned"
    assert "два файла так и не собрались" in record.text
    assert "Log Digest" in record.text
    assert "3 захода" in record.text


async def test_taking_on_a_project_is_remembered_before_it_is_finished() -> None:
    """Между замыслом и результатом часы, и всё это время «чем ты занята?» — вопрос без ответа."""
    memory = _RecordingMemory()
    reporter = DevReporter(_CollectingManager(), social_memory=memory)  # type: ignore[arg-type]

    await reporter.remember_start(_task(is_collab=False))

    record = memory.records[0]
    assert record.kind.value == "dev_started"
    assert "Затеяла сама" in record.text
    assert "src/main.py" in record.text, "в памяти остаётся и то, как она задумала это делать"


async def test_returning_to_an_old_project_is_remembered_too() -> None:
    memory = _RecordingMemory()
    reporter = DevReporter(_CollectingManager(), social_memory=memory)  # type: ignore[arg-type]

    await reporter.remember_revision(_task(), "Поправила README: пример запуска не работал")

    assert memory.records[0].kind.value == "dev_revision"
    assert "пример запуска не работал" in memory.records[0].text


# -- внешний флекс ------------------------------------------------------------


def _released(**overrides: Any) -> DevTask:
    return _task(status=DevTaskStatus.DONE, repo_url=_REPO.html_url, **overrides)


def test_showcase_fires_only_when_the_talk_is_really_about_it() -> None:
    releases = [_released()]

    on_topic = "кто-нибудь разбирал логи nginx? хочу видеть топ ошибок за период"
    assert pick_showcase(on_topic, releases) is not None

    off_topic = "посоветуйте кофемолку, а то моя сдохла"
    assert pick_showcase(off_topic, releases) is None


def test_shared_technical_words_are_not_enough() -> None:
    """«Оба про питон» — не повод давать ссылку: так и получается спам в чужих каналах."""
    releases = [_released()]

    assert pick_showcase("пишу на python проект, код такой себе", releases) is None


# -- фоновый цикл целиком -----------------------------------------------------


class _StubEngine:
    def __init__(
        self,
        *,
        spec: ProjectSpec | None = _SPEC,
        build: BuildResult | None = None,
        design_failure: str = "модель ответила чем-то другим",
    ) -> None:
        self._spec = spec
        self._build = build or BuildResult(
            files=[GeneratedFile(path="src/main.py", content="x = 1", fix_rounds=1)]
        )
        self._design_failure = design_failure
        self.design_context = ""
        self.design_calls = 0
        self.built_seen: list[ProjectSpec] = []
        self.existing_seen: dict[str, str] = {}

    async def design(
        self, idea: str = "", *, context: str = "", built: Sequence[ProjectSpec] = ()
    ) -> tuple[ProjectSpec | None, str]:
        self.design_calls += 1
        self.design_context = context
        self.built_seen = list(built)
        return (self._spec, "") if self._spec is not None else (None, self._design_failure)

    async def build(
        self, spec: ProjectSpec, *, existing: Mapping[str, str] | None = None
    ) -> BuildResult:
        self.existing_seen = dict(existing or {})
        return self._build


class _StubGitHub:
    def __init__(self, *, pushed: bool = True, error: str = "") -> None:
        self._pushed = pushed
        self._error = error
        self.published: list[ProjectSpec] = []

    async def publish(self, spec: ProjectSpec, files: list[GeneratedFile]) -> PublishResult:
        if self._error:
            raise GitHubSyncError(self._error)
        self.published.append(spec)
        return PublishResult(
            local_path=Path("/tmp/log-digest"),
            commits=["chore: scaffolding"],
            repo=_REPO if self._pushed else None,
            pushed=self._pushed,
        )


class _StubInterests:
    async def current_interests(self) -> list[str]:
        return ["разбор логов", "cli-утилиты"]


def _worker(
    store: DevTaskStore,
    manager: _CollectingManager,
    *,
    engine: Any = None,
    github: Any = None,
    memory: Any = None,
    self_initiated: float = 0.0,
) -> DevWorker:
    return DevWorker(
        store,
        engine or _StubEngine(),  # type: ignore[arg-type]
        github or _StubGitHub(),  # type: ignore[arg-type]
        DevReporter(manager, progress_probability=1.0, social_memory=memory),
        interests=_StubInterests(),
        owner_chat_id=_CHAT_ID,
        self_initiated_probability=self_initiated,
    )


def _store(tmp_path: Path) -> DevTaskStore:
    return DevTaskStore(Database(tmp_path / "efi.db", migrations=MIGRATIONS))


async def _tick_until_settled(worker: DevWorker, *, limit: int = 5) -> None:
    """Гоняет цикл, пока задача не придёт к терминальному статусу: сбой теперь не хоронит с первого раза."""
    for _ in range(limit):
        await worker._tick()


async def test_pending_task_goes_all_the_way_to_a_link(tmp_path: Path) -> None:
    store = _store(tmp_path)
    manager = _CollectingManager()
    task = await store.create("утилита для логов", chat_id=_CHAT_ID, is_collab=True)
    worker = _worker(store, manager)

    await worker._tick()

    finished = await store.get(task.id)
    assert finished is not None
    assert finished.status is DevTaskStatus.DONE
    assert finished.repo_url == _REPO.html_url
    assert finished.spec is not None and finished.spec.slug == "log-digest"

    kinds = [notification.message for notification in manager.notifications]
    assert any("запушила" in message for message in kinds), "релиз обязан прозвучать"


async def test_the_worker_is_busy_while_coding(tmp_path: Path) -> None:
    """`is_coding` — вход для BusyEngine: занятость, которую видно в поведении."""
    store = _store(tmp_path)
    await store.create("утилита", chat_id=_CHAT_ID, is_collab=True)
    worker = _worker(store, _CollectingManager())

    assert worker.is_coding is False
    await worker._tick()
    assert worker.is_coding is False, "флаг снимается и после успешного прохода"


async def test_unbuildable_project_fails_loudly_for_the_person_who_asked(tmp_path: Path) -> None:
    """Провал заказанной задачи обязан быть слышен — но только окончательный, а не каждый заход."""
    store = _store(tmp_path)
    manager = _CollectingManager()
    task = await store.create("утилита", chat_id=_CHAT_ID, is_collab=True)
    broken = BuildResult(files=[], broken_paths=["src/main.py"])
    worker = _worker(store, manager, engine=_StubEngine(build=broken))

    await worker._tick()

    retrying = await store.get(task.id)
    assert retrying is not None
    assert retrying.status is DevTaskStatus.PENDING, "первый сбой — повод вернуться, а не хоронить"
    assert not any("не вышел" in item.message for item in manager.notifications)

    await _tick_until_settled(worker)

    failed = await store.get(task.id)
    assert failed is not None
    assert failed.status is DevTaskStatus.FAILED
    assert "src/main.py" in failed.error
    assert any("не вышел" in item.message for item in manager.notifications)


async def test_a_dead_coder_is_not_retried(tmp_path: Path) -> None:
    """
    Снятая с обслуживания модель к следующему часу не вернётся. Повторять
    такое — тратить запросы на заведомо тот же ответ, а человеку показывать
    «делаю», когда делать нечем.
    """
    store = _store(tmp_path)
    manager = _CollectingManager()
    task = await store.create("утилита", chat_id=_CHAT_ID, is_collab=True)
    dead = BuildResult(
        files=[],
        failure_reason="модель 'qwen' недоступна: HTTP 404 model_decommissioned",
        permanent=True,
    )
    worker = _worker(store, manager, engine=_StubEngine(build=dead))

    await worker._tick()

    failed = await store.get(task.id)
    assert failed is not None
    assert failed.status is DevTaskStatus.FAILED
    assert "model_decommissioned" in failed.error


async def test_github_failure_keeps_the_task_honest(tmp_path: Path) -> None:
    store = _store(tmp_path)
    manager = _CollectingManager()
    task = await store.create("утилита", chat_id=_CHAT_ID, is_collab=True)
    worker = _worker(store, manager, github=_StubGitHub(error="токен без прав repo"))

    await _tick_until_settled(worker)

    failed = await store.get(task.id)
    assert failed is not None
    assert failed.status is DevTaskStatus.FAILED
    assert "токен без прав" in failed.error


async def test_local_only_run_does_not_brag_about_a_link_it_does_not_have(tmp_path: Path) -> None:
    """Без пуша ссылки нет — и хвастаться нечем, хотя проект написан и лежит на диске."""
    store = _store(tmp_path)
    manager = _CollectingManager()
    task = await store.create("утилита", chat_id=_CHAT_ID, is_collab=True)
    worker = _worker(store, manager, github=_StubGitHub(pushed=False))

    await worker._tick()

    done = await store.get(task.id)
    assert done is not None
    assert done.status is DevTaskStatus.DONE
    assert done.repo_url == ""
    assert not any("запушила" in item.message for item in manager.notifications)


async def test_written_files_survive_a_failed_attempt(tmp_path: Path) -> None:
    """
    Проект развалился на четвёртом файле из-за лимита — переписывать первые
    три в следующий заход значит потратить те же запросы на тот же лимит и
    получить новый шанс разойтись с тем, что уже сходилось.
    """
    store = _store(tmp_path)
    task = await store.create("утилита", chat_id=_CHAT_ID, is_collab=True)
    half_done = BuildResult(
        files=[GeneratedFile(path="src/main.py", content="x = 1")], broken_paths=["src/parser.py"]
    )
    engine = _StubEngine(build=half_done)
    worker = _worker(store, _CollectingManager(), engine=engine)

    await worker._tick()

    kept = await store.get(task.id)
    assert kept is not None
    assert kept.artifacts == {"src/main.py": "x = 1"}, "написанное остаётся при задаче"

    await worker._tick()

    assert engine.existing_seen == {"src/main.py": "x = 1"}, "второй заход не переписывает готовое"


async def test_an_abandoned_project_gets_a_second_wind(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Довести до конца начатое важнее, чем затеять новое: человек, у которого
    пять брошенных проектов и шестой начатый, ничего не доводит. Возвращается
    проект не с нуля — спека и написанные файлы при нём, и заход начинается
    с того места, где в прошлый раз кончились лимиты.
    """
    store = _store(tmp_path)
    memory = _RecordingMemory()
    dead = await store.create("sesslog", chat_id=_CHAT_ID, is_collab=False)
    dead = await store.update(
        dead,
        spec=_SPEC,
        status=DevTaskStatus.FAILED,
        error="два файла так и не собрались",
        attempts=3,
        artifacts={"src/main.py": "x = 1"},
    )
    monkeypatch.setattr(worker_module, "_REVIVE_AFTER", timedelta(0))
    engine = _StubEngine()
    worker = _worker(store, _CollectingManager(), engine=engine, memory=memory, self_initiated=1.0)

    await worker._tick()

    revived = await store.get(dead.id)
    assert revived is not None
    assert revived.status is DevTaskStatus.DONE, "к брошенному вернулись и довели"
    assert revived.revivals == 1
    assert engine.existing_seen == {"src/main.py": "x = 1"}, "написанное в прошлый раз не переписывалось"
    assert any(record.kind.value == "dev_revision" for record in memory.records)


async def test_a_project_is_not_revived_forever(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Замысел, который не собрался и на третий раз, стоит квоты, за которую пишется что-то новое."""
    store = _store(tmp_path)
    dead = await store.create("sesslog", chat_id=_CHAT_ID)
    dead = await store.update(dead, spec=_SPEC, status=DevTaskStatus.FAILED, revivals=2)
    monkeypatch.setattr(worker_module, "_REVIVE_AFTER", timedelta(0))
    worker = _worker(store, _CollectingManager())

    await worker._tick()

    left = await store.get(dead.id)
    assert left is not None and left.status is DevTaskStatus.FAILED


async def test_own_project_is_started_only_when_nothing_else_is_running(tmp_path: Path) -> None:
    store = _store(tmp_path)
    manager = _CollectingManager()
    worker = _worker(store, manager, self_initiated=1.0)

    await worker._tick()

    tasks = await store.recent_releases()
    assert len(tasks) == 1
    assert tasks[0].is_collab is False
    assert tasks[0].chat_id == _CHAT_ID, "своя затея рассказывается владельцу"


async def test_own_projects_can_be_disabled(tmp_path: Path) -> None:
    store = _store(tmp_path)
    worker = _worker(store, _CollectingManager(), self_initiated=0.0)

    await worker._tick()

    assert await store.active() == []
    assert await store.recent_releases() == []


async def test_a_stillborn_idea_leaves_no_trace(tmp_path: Path) -> None:
    """
    Замысел придумывается ДО того, как заводится задача. Раньше было наоборот,
    и каждая неудачная попытка навсегда оседала в базе строчкой «замысел без
    названия — не вышло»: за сутки их набиралось больше, чем настоящих
    проектов, а полезного в них нет вообще — ни идеи, ни кода, ни причины
    возвращаться.
    """
    store = _store(tmp_path)
    manager = _CollectingManager()
    worker = _worker(store, manager, engine=_StubEngine(spec=None), self_initiated=1.0)

    await worker._tick()

    assert await store.active() == []
    assert await store.recent_failures() == []
    assert manager.notifications == [], "не придумалось — не повод писать об этом владельцу"


async def test_a_requested_project_that_fails_design_keeps_the_real_reason(tmp_path: Path) -> None:
    """
    Заказанная человеком задача — другое дело: она уже обещана, и её провал
    обязан быть виден. Но с настоящей причиной: под общим «не придумалось»
    одинаково прятались битый JSON, обрыв по лимиту и отказ от учебной идеи, а
    чинятся они по-разному.
    """
    store = _store(tmp_path)
    manager = _CollectingManager()
    task = await store.create("утилита для логов", chat_id=_CHAT_ID, is_collab=True)
    engine = _StubEngine(spec=None, design_failure="ответ модели оборвался по лимиту в 2048 токенов")
    worker = _worker(store, manager, engine=engine)

    await _tick_until_settled(worker)

    failed = await store.get(task.id)
    assert failed is not None
    assert failed.status is DevTaskStatus.FAILED
    assert "оборвался по лимиту" in failed.error


async def test_old_empty_failures_are_swept_away_once(tmp_path: Path) -> None:
    """Мусор, накопленный прошлой версией, чистится сам — иначе страница проектов так и остаётся кладбищем."""
    store = _store(tmp_path)
    stub = await store.create("", chat_id=_CHAT_ID, is_collab=False)
    await store.update(stub, status=DevTaskStatus.FAILED, error="не придумалось ничего")
    real = await store.create("утилита для логов", chat_id=_CHAT_ID, is_collab=True)
    await store.update(real, status=DevTaskStatus.FAILED, error="кодер не ответил")

    worker = _worker(store, _CollectingManager())
    await worker._tick()

    left = await store.recent_failures()
    assert [item.id for item in left] == [real.id], "чистится только пустышка, настоящая неудача остаётся"


async def test_own_project_starts_from_what_she_lives_by(tmp_path: Path) -> None:
    """Замысел из интересов, а не из воздуха: иначе проекты не имеют отношения к её жизни."""
    store = _store(tmp_path)
    engine = _StubEngine()
    worker = _worker(store, _CollectingManager(), engine=engine, self_initiated=1.0)

    await worker._tick()

    assert "разбор логов" in engine.design_context


async def test_the_next_idea_knows_what_is_already_written(tmp_path: Path) -> None:
    """Без этого списка она раз в неделю придумывает ту же утилиту — и упирается в занятое имя репозитория."""
    store = _store(tmp_path)
    done = await store.create("прошлый проект", chat_id=_CHAT_ID)
    await store.update(done, spec=_SPEC, status=DevTaskStatus.DONE, repo_url=_REPO.html_url)
    engine = _StubEngine()
    worker = _worker(store, _CollectingManager(), engine=engine, self_initiated=1.0)

    await worker._tick()

    assert [item.slug for item in engine.built_seen] == ["log-digest"]


async def test_her_own_idea_is_worth_mentioning_when_it_appears(tmp_path: Path) -> None:
    """
    «Придумала себе штуку и сажусь писать» человек говорит в начале работы, а
    не только когда всё готово. Без этой реплики собственный проект молчит
    ровно до релиза — то есть пока не станет фактом, к которому уже нечего
    добавить.
    """
    store = _store(tmp_path)
    manager = _CollectingManager()
    worker = _worker(store, manager, self_initiated=1.0)

    await worker._tick()

    assert any("придумала структуру" in item.message for item in manager.notifications)
