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

from datetime import datetime
from pathlib import Path
from typing import Any

from efi.config.schema import QuietHoursSettings
from efi.db.core import Database
from efi.db.models import MIGRATIONS
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
    def __init__(self, *, spec: ProjectSpec | None = _SPEC, build: BuildResult | None = None) -> None:
        self._spec = spec
        self._build = build or BuildResult(
            files=[GeneratedFile(path="src/main.py", content="x = 1", fix_rounds=1)]
        )
        self.design_context = ""

    async def design(self, idea: str = "", *, context: str = "") -> ProjectSpec | None:
        self.design_context = context
        return self._spec

    async def build(self, spec: ProjectSpec) -> BuildResult:
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
    self_initiated: float = 0.0,
) -> DevWorker:
    return DevWorker(
        store,
        engine or _StubEngine(),  # type: ignore[arg-type]
        github or _StubGitHub(),  # type: ignore[arg-type]
        DevReporter(manager, progress_probability=1.0),
        interests=_StubInterests(),
        owner_chat_id=_CHAT_ID,
        self_initiated_probability=self_initiated,
    )


def _store(tmp_path: Path) -> DevTaskStore:
    return DevTaskStore(Database(tmp_path / "efi.db", migrations=MIGRATIONS))


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
    store = _store(tmp_path)
    manager = _CollectingManager()
    task = await store.create("утилита", chat_id=_CHAT_ID, is_collab=True)
    broken = BuildResult(files=[], broken_paths=["src/main.py"])
    worker = _worker(store, manager, engine=_StubEngine(build=broken))

    await worker._tick()

    failed = await store.get(task.id)
    assert failed is not None
    assert failed.status is DevTaskStatus.FAILED
    assert "src/main.py" in failed.error
    assert any("не вышел" in item.message for item in manager.notifications)


async def test_github_failure_keeps_the_task_honest(tmp_path: Path) -> None:
    store = _store(tmp_path)
    manager = _CollectingManager()
    task = await store.create("утилита", chat_id=_CHAT_ID, is_collab=True)
    worker = _worker(store, manager, github=_StubGitHub(error="токен без прав repo"))

    await worker._tick()

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


async def test_own_project_starts_from_what_she_lives_by(tmp_path: Path) -> None:
    """Замысел из интересов, а не из воздуха: иначе проекты не имеют отношения к её жизни."""
    store = _store(tmp_path)
    engine = _StubEngine()
    worker = _worker(store, _CollectingManager(), engine=engine, self_initiated=1.0)

    await worker._tick()

    assert "разбор логов" in engine.design_context
