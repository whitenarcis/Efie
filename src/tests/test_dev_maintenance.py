"""
Тесты возвращения к своим проектам: перечитала — поправила — иногда спросила.

Главное, что здесь проверяется, — СДЕРЖАННОСТЬ. Модель, которую попросили
найти, что улучшить в коде, находит всегда: переименовать переменную,
добавить тайпхинт, «стоит покрыть тестами». Если пускать каждую такую
находку в коммиты и в чат, получается не человек со своими проектами, а бот,
еженедельно спрашивающий разрешения переименовать переменную. Поэтому:

  * «всё нормально» — законный и самый частый исход, и он тоже засчитывается
    как просмотр;
  * правка требует порога важности, проходит через песочницу и не имеет права
    сломать работающий проект;
  * вопрос владельцу — порог заметно выше: чужое время дороже своего.
"""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pytest

from efi.db.core import Database
from efi.db.models import MIGRATIONS
from efi.dev.github_sync import GitHubSync
from efi.dev.maintenance import ProjectMaintainer, ReviewVerdict, parse_verdict
from efi.dev.readme import README_PATH
from efi.dev.reporter import DevReporter
from efi.dev.sandbox import CodeSandbox
from efi.dev.schemas import DevTask, DevTaskStatus, GeneratedFile, ProjectSpec
from efi.dev.store import DevTaskStore
from efi.llm.schemas import Choice, LLMParams, Message, Response, Role, Session
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
_ORIGINAL_CODE = "def main() -> None:\n    print('топ ошибок')\n"
_GOOD_README = (
    "# Log Digest\n\nразбор логов\n\n## Назначение\n\nразбирает логи nginx и показывает топ ошибок "
    "за период тем, кто держит сервер и не хочет читать гигабайты руками\n\n"
    "## Установка\n\nPython 3.11, зависимостей нет\n\n"
    "## Использование\n\n```bash\npython src/main.py access.log\n```\n\n"
    "## Структура\n\n- `src/main.py` — точка входа\n"
)

requires_git = pytest.mark.skipif(
    subprocess.run(["git", "--version"], capture_output=True, check=False).returncode != 0,  # noqa: S603, S607
    reason="в системе нет git",
)


class _CollectingManager(NotificationManager):
    def __init__(self) -> None:
        super().__init__(worker_count=1)
        self.notifications: list[Notification] = []

    async def put(self, notification: Notification) -> None:
        self.notifications.append(notification)
        await super().put(notification)


class _ScriptedRouter:
    """Ревизор, отвечающий заранее заданным вердиктом."""

    def __init__(self, verdict: dict[str, Any] | str) -> None:
        self.text = verdict if isinstance(verdict, str) else json.dumps(verdict, ensure_ascii=False)
        self.calls = 0

    async def chat(self, role: Any, params: LLMParams, session: Session) -> Response:
        self.calls += 1
        return Response(choices=[Choice(message=Message(role=Role.ASSISTANT, content=self.text))])


class _ScriptedCoder:
    """Кодер, отвечающий заранее заданным содержимым исправленного файла."""

    def __init__(self, fixed: str | None) -> None:
        self._fixed = fixed
        self.calls: list[tuple[str, str]] = []

    async def fix_file(self, path: str, source: str, diagnostics: str) -> str | None:
        self.calls.append((path, diagnostics))
        return self._fixed


async def _project_on_disk(tmp_path: Path, *, code: str = _ORIGINAL_CODE) -> Path:
    """Настоящий git-клон проекта — то, что остаётся после публикации и по чему идёт правка."""
    workspace = tmp_path / "projects"
    sync = GitHubSync(workspace)
    await sync.publish(
        _SPEC,
        [
            GeneratedFile(path="src/main.py", content=code),
            GeneratedFile(path=README_PATH, content=_GOOD_README),
        ],
    )
    return workspace


def _git_log(repo: Path) -> list[str]:
    result = subprocess.run(  # noqa: S603
        ["git", "log", "--format=%s"],  # noqa: S607
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip().splitlines()


async def _released_task(store: DevTaskStore, *, reviewed_days_ago: float | None = None) -> DevTask:
    task = await store.create("утилита для логов", chat_id=_CHAT_ID, is_collab=False)
    task = await store.update(
        task, status=DevTaskStatus.DONE, spec=_SPEC, repo_url="https://github.com/efi/log-digest"
    )
    if reviewed_days_ago is not None:
        moment = datetime.now(UTC) - timedelta(days=reviewed_days_ago)
        await store._database.execute(  # доступ к БД напрямую: нужна ИСТОРИЯ, а не «сейчас»
            "UPDATE dev_tasks SET reviewed_at = ? WHERE id = ?", (moment.isoformat(), task.id)
        )
        refreshed = await store.get(task.id)
        assert refreshed is not None
        return refreshed
    return task


def _maintainer(
    store: DevTaskStore,
    workspace: Path,
    manager: _CollectingManager,
    *,
    router: Any,
    coder: Any = None,
    review_probability: float = 1.0,
) -> ProjectMaintainer:
    return ProjectMaintainer(
        store,
        router,
        coder or _ScriptedCoder(None),
        CodeSandbox(enable_linter=False),
        GitHubSync(workspace),
        DevReporter(manager, progress_probability=1.0),
        workspace,
        review_probability=review_probability,
    )


# -- разбор вердикта ----------------------------------------------------------


def test_unparseable_verdict_means_do_nothing() -> None:
    """Безопасный исход по умолчанию: проект остаётся работающим, а не правится наугад."""
    assert parse_verdict("").verdict == "nothing"
    assert parse_verdict("да вроде норм всё").verdict == "nothing"
    assert parse_verdict('{"verdict": "снести всё"}').verdict == "nothing"


def test_verdict_is_read_from_fenced_json() -> None:
    verdict = parse_verdict(
        '```json\n{"verdict": "patch", "importance": 0.7, "path": "src/main.py", '
        '"what": "обработать пустой файл", "commit": "fix: пустой лог больше не роняет"}\n```'
    )

    assert verdict.wants_patch is True
    assert verdict.importance == pytest.approx(0.7)
    assert verdict.path == "src/main.py"


def test_importance_is_clamped() -> None:
    assert parse_verdict('{"verdict": "patch", "importance": 42}').importance == 1.0
    assert parse_verdict('{"verdict": "patch", "importance": "не число"}').importance == 0.0


# -- «всё нормально» — законный исход ----------------------------------------


@requires_git
async def test_nothing_to_fix_still_counts_as_a_look(tmp_path: Path) -> None:
    """
    Без отметки о просмотре один и тот же проект пересматривался бы каждый
    тик, а остальные не дождались бы очереди никогда.
    """
    workspace = await _project_on_disk(tmp_path)
    store = DevTaskStore(Database(tmp_path / "efi.db", migrations=MIGRATIONS))
    task = await _released_task(store)
    manager = _CollectingManager()
    maintainer = _maintainer(store, workspace, manager, router=_ScriptedRouter({"verdict": "nothing"}))

    outcome = await maintainer.review(task)

    assert outcome is not None and outcome.changed_anything is False
    assert manager.notifications == [], "молчание — нормальный итог просмотра"
    reread = await store.get(task.id)
    assert reread is not None and reread.reviewed_at is not None
    assert reread.revisions == 0


@requires_git
async def test_cosmetic_finding_below_the_threshold_is_ignored(tmp_path: Path) -> None:
    """
    «Переименовать переменную» с важностью 0.2 — ровно то, что модель находит
    всегда. Порог существует, чтобы такие находки не превращались в коммиты.
    """
    workspace = await _project_on_disk(tmp_path)
    store = DevTaskStore(Database(tmp_path / "efi.db", migrations=MIGRATIONS))
    task = await _released_task(store)
    coder = _ScriptedCoder("def main() -> None:\n    pass\n")
    maintainer = _maintainer(
        store,
        workspace,
        _CollectingManager(),
        router=_ScriptedRouter(
            {
                "verdict": "patch",
                "importance": 0.2,
                "path": "src/main.py",
                "what": "переименовать переменную",
            }
        ),
        coder=coder,
    )

    outcome = await maintainer.review(task)

    assert outcome is not None and outcome.patched is False
    assert coder.calls == [], "до кодера дело даже не доходит"


# -- правка -------------------------------------------------------------------


@requires_git
async def test_real_fix_is_committed_and_told_about(tmp_path: Path) -> None:
    workspace = await _project_on_disk(tmp_path)
    store = DevTaskStore(Database(tmp_path / "efi.db", migrations=MIGRATIONS))
    task = await _released_task(store)
    manager = _CollectingManager()
    fixed = "import sys\n\n\ndef main() -> None:\n    if len(sys.argv) < 2:\n        sys.exit('нужен файл')\n"
    maintainer = _maintainer(
        store,
        workspace,
        manager,
        router=_ScriptedRouter(
            {
                "verdict": "patch",
                "importance": 0.8,
                "path": "src/main.py",
                "what": "падает без аргументов",
                "commit": "fix: понятная ошибка вместо трейсбека без аргументов",
                "note": "наткнулась на свой же скрипт без аргументов и словила трейсбек, стыдно",
            }
        ),
        coder=_ScriptedCoder(fixed),
    )

    outcome = await maintainer.review(task)

    assert outcome is not None and outcome.patched is True
    project_dir = workspace / _SPEC.slug
    assert (project_dir / "src" / "main.py").read_text(encoding="utf-8") == fixed
    assert _git_log(project_dir)[0] == "fix: понятная ошибка вместо трейсбека без аргументов"

    reread = await store.get(task.id)
    assert reread is not None and reread.revisions == 1
    assert "стыдно" in manager.notifications[0].message
    assert manager.notifications[0].type is NotificationType.DEV_UPDATE


@requires_git
async def test_a_fix_that_breaks_the_code_is_refused(tmp_path: Path) -> None:
    """До правки проект работал. Сломать его «улучшением» — худший из возможных исходов."""
    workspace = await _project_on_disk(tmp_path)
    store = DevTaskStore(Database(tmp_path / "efi.db", migrations=MIGRATIONS))
    task = await _released_task(store)
    maintainer = _maintainer(
        store,
        workspace,
        _CollectingManager(),
        router=_ScriptedRouter(
            {"verdict": "patch", "importance": 0.9, "path": "src/main.py", "what": "почистить"}
        ),
        coder=_ScriptedCoder("def main(:\n    сломано\n"),
    )

    outcome = await maintainer.review(task)

    assert outcome is not None and outcome.patched is False
    project_dir = workspace / _SPEC.slug
    assert (project_dir / "src" / "main.py").read_text(encoding="utf-8") == _ORIGINAL_CODE


@requires_git
async def test_readme_edit_may_not_make_it_worse(tmp_path: Path) -> None:
    """README — обязательный критерий проекта, и правка не имеет права его нарушить."""
    workspace = await _project_on_disk(tmp_path)
    store = DevTaskStore(Database(tmp_path / "efi.db", migrations=MIGRATIONS))
    task = await _released_task(store)
    maintainer = _maintainer(
        store,
        workspace,
        _CollectingManager(),
        router=_ScriptedRouter(
            {"verdict": "patch", "importance": 0.9, "path": README_PATH, "what": "сократить"}
        ),
        coder=_ScriptedCoder("# Log Digest\n\nкороче некуда\n"),
    )

    outcome = await maintainer.review(task)

    assert outcome is not None and outcome.patched is False
    assert (workspace / _SPEC.slug / README_PATH).read_text(encoding="utf-8") == _GOOD_README


@requires_git
async def test_edit_that_changes_nothing_makes_no_commit(tmp_path: Path) -> None:
    """Пустой коммит хуже отсутствия правки: история проекта должна показывать работу, а не активность."""
    workspace = await _project_on_disk(tmp_path)
    store = DevTaskStore(Database(tmp_path / "efi.db", migrations=MIGRATIONS))
    task = await _released_task(store)
    before = len(_git_log(workspace / _SPEC.slug))
    maintainer = _maintainer(
        store,
        workspace,
        _CollectingManager(),
        router=_ScriptedRouter(
            {"verdict": "patch", "importance": 0.9, "path": "src/main.py", "what": "поправить"}
        ),
        coder=_ScriptedCoder(_ORIGINAL_CODE),
    )

    outcome = await maintainer.review(task)

    assert outcome is not None and outcome.patched is False
    assert len(_git_log(workspace / _SPEC.slug)) == before


# -- вопрос владельцу ---------------------------------------------------------


@requires_git
async def test_important_question_reaches_the_owner(tmp_path: Path) -> None:
    workspace = await _project_on_disk(tmp_path)
    store = DevTaskStore(Database(tmp_path / "efi.db", migrations=MIGRATIONS))
    task = await _released_task(store)
    manager = _CollectingManager()
    maintainer = _maintainer(
        store,
        workspace,
        manager,
        router=_ScriptedRouter(
            {
                "verdict": "discuss",
                "importance": 0.9,
                "question": "формат вывода менять на json? тогда старые скрипты сломаются",
            }
        ),
    )

    outcome = await maintainer.review(task)

    assert outcome is not None and outcome.asked is True
    message = manager.notifications[0].message
    assert "json" in message
    assert "Спроси собеседника прямо" in message


@requires_git
async def test_a_question_that_is_not_worth_asking_stays_unasked(tmp_path: Path) -> None:
    """
    Порог для вопроса заметно выше, чем для правки: беспокоить человека
    ради «может, переименуем флаг?» — это и есть та назойливость, из-за
    которой такие механики выключают.
    """
    workspace = await _project_on_disk(tmp_path)
    store = DevTaskStore(Database(tmp_path / "efi.db", migrations=MIGRATIONS))
    task = await _released_task(store)
    manager = _CollectingManager()
    maintainer = _maintainer(
        store,
        workspace,
        manager,
        router=_ScriptedRouter(
            {"verdict": "discuss", "importance": 0.55, "question": "может, переименуем флаг?"}
        ),
    )

    outcome = await maintainer.review(task)

    assert outcome is not None and outcome.asked is False
    assert manager.notifications == []


# -- отбор проектов -----------------------------------------------------------


async def test_only_projects_untouched_for_a_while_are_reviewed(tmp_path: Path) -> None:
    """Свежий проект пересматривать незачем: он ровно такой, каким его дописали час назад."""
    store = DevTaskStore(Database(tmp_path / "efi.db", migrations=MIGRATIONS))
    await _released_task(store, reviewed_days_ago=1)

    assert await store.due_for_review(not_reviewed_for=timedelta(days=7)) == []
    assert len(await store.due_for_review(not_reviewed_for=timedelta(hours=1))) == 1


async def test_review_is_occasional_not_scheduled(tmp_path: Path) -> None:
    """
    Перечитывать свой код строго по будильнику — это cron, а не привычка
    автора. Нулевая вероятность выключает возвращение к проектам совсем.
    """
    store = DevTaskStore(Database(tmp_path / "efi.db", migrations=MIGRATIONS))
    await _released_task(store, reviewed_days_ago=30)
    router = _ScriptedRouter({"verdict": "nothing"})
    maintainer = _maintainer(
        store, tmp_path / "projects", _CollectingManager(), router=router, review_probability=0.0
    )

    assert await maintainer.maybe_review() is None
    assert router.calls == 0


async def test_missing_local_clone_is_not_a_crash(tmp_path: Path) -> None:
    """Каталог могли почистить. Это не повод ни падать, ни выкладывать проект заново под видом правки."""
    store = DevTaskStore(Database(tmp_path / "efi.db", migrations=MIGRATIONS))
    task = await _released_task(store, reviewed_days_ago=30)
    router = _ScriptedRouter({"verdict": "patch", "importance": 1.0})
    maintainer = _maintainer(store, tmp_path / "projects", _CollectingManager(), router=router)

    assert await maintainer.review(task) is None
    assert router.calls == 0
    reread = await store.get(task.id)
    assert reread is not None and reread.reviewed_at is not None, "отметка нужна, иначе он вечно в выборке"


def test_commit_messages_are_forced_into_a_readable_shape() -> None:
    from efi.dev.maintenance import _commit_message

    assert _commit_message(ReviewVerdict(commit="fix: убрал падение")) == "fix: убрал падение"
    assert _commit_message(ReviewVerdict(commit="Обновление файла")) == "fix: Обновление файла"
    assert _commit_message(ReviewVerdict(what="починила разбор дат")) == "fix: починила разбор дат"
