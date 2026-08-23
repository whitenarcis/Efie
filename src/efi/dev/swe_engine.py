"""
efi/dev/swe_engine.py

Работа с ЧУЖИМ кодом: понять репозиторий, поправить в нём две-три вещи,
проверить запуском и оставить ветку, которую можно забрать.

Чем это отличается от efi/dev/engine.py, который уже есть. Тот пишет проект
с нуля: спека → файлы → публикация, и там всё принадлежит Эфи. Здесь всё
наоборот — код уже есть, он чужой, его нельзя переписывать, и цена ошибки
другая: сломать чужой проект хуже, чем не написать свой. Отсюда три правила,
которых нет в первом конвейере:

    смотреть, а не гадать    — сначала карта репозитория (efi/dev/repo_map.py),
                               потом выбор двух-трёх файлов, и только потом
                               правка. Модель, которой не показали структуру,
                               уверенно правит файл, которого нет.
    трогать точечно          — только блоки SEARCH/REPLACE (efi/dev/edits.py).
                               Правка, не нашедшая своего места, отвергается,
                               а не применяется «примерно туда».
    проверять запуском       — импорт, линтер, тесты в изолированной копии
                               (efi/dev/auto_fix.py). Ветка создаётся ТОЛЬКО
                               когда всё зелено: ветка с падающими тестами —
                               это не помощь, а работа для того, кто её
                               откроет.

Существующий конвейер собственных проектов этот модуль не трогает вовсе:
у них общий только кодер (через efi/llm/network_router.py, который сам решает,
думать на ноутбуке или облаком).
"""

from __future__ import annotations

import logging
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from pathlib import Path

from efi.dev.auto_fix import Lookup, RepairLoop, RepairReport
from efi.dev.edits import EDIT_FORMAT_INSTRUCTIONS, apply_edits, parse_edits
from efi.dev.repo_map import DEFAULT_MAP_BUDGET_BYTES, build_repo_map
from efi.dev.workspace import Workspace, WorkspaceError, WorkspaceManager
from efi.llm.network_router import NetworkModelRouter
from efi.llm.resilience import AttemptLog, ConcurrencyGate
from efi.llm.schemas import LLMParams, Message, Role, Session

logger = logging.getLogger(__name__)

#: Сколько файлов открывать целиком. Два-три — не экономия, а качество:
#: модель, которой дали десять файлов, правит их все понемногу вместо того,
#: чтобы починить один.
MAX_FILES_IN_FOCUS = 3

#: Потолок на один файл в контексте. Файл на три тысячи строк не влезет
#: целиком ни в какой бюджет, но его начало обычно отвечает на вопрос.
_MAX_FILE_CHARS = 20_000

_ANALYSIS_MAX_TOKENS = 1024
_EDIT_MAX_TOKENS = 4096

_COMMIT_AUTHOR_NAME = "Efi"
_COMMIT_AUTHOR_EMAIL = "efi@users.noreply.github.com"

_SELECT_SYSTEM_PROMPT = (
    "Тебе дают карту репозитория (пути и объявления без тел) и задачу. Определи, в каких файлах "
    "придётся что-то менять.\n"
    "Отвечай СТРОГО так: сначала одна строка «ФАЙЛЫ: путь1, путь2» (не больше трёх, только "
    "существующие пути из карты; новый файл указывай, только если без него никак), затем с новой "
    "строки в двух-трёх предложениях — что именно там не так и что ты собираешься сделать. Без "
    "markdown-заголовков, без списков, без кода."
)

_EDIT_SYSTEM_PROMPT = (
    "Ты правишь ЧУЖОЙ работающий проект. Правило одно: минимальное вмешательство. Меняй только то, "
    "что требует задача; не переименовывай публичные имена, не переставляй импорты, не приводи чужой "
    "стиль к своему, не добавляй комментарии «здесь было исправлено».\n"
    "Стиль кода бери из файла, а не из своих привычек: как там называют переменные, так и называй.\n"
    f"{EDIT_FORMAT_INSTRUCTIONS}"
)

#: Строка выбора файлов. Модель почти всегда пишет её как просили, но
#: регулярно добавляет ``` или маркеры — разбор к этому готов.
_FILES_LINE_RE = re.compile(r"ФАЙЛЫ\s*:?\s*(?P<paths>[^\n]+)", re.IGNORECASE)

#: Как назвать ветку, если задача не подсказала. Префикс тот же, что у людей:
#: по нему в списке веток видно, что это правка, а не эксперимент.
_BRANCH_SAFE_RE = re.compile(r"[^a-z0-9]+")


@dataclass(slots=True, frozen=True)
class SweRequest:
    """Что и где надо сделать."""

    #: Ссылка на репозиторий или путь к локальному каталогу.
    source: str
    #: Задача словами человека — ровно так, как он её сказал в чате.
    instruction: str
    #: Идентификатор задачи; из него получается имя каталога рабочей копии.
    session_id: str
    chat_id: int | None = None
    #: Куда пушить результат. Пусто — ветка остаётся локальной в рабочей копии.
    push: bool = False
    branch: str = ""

    def branch_name(self) -> str:
        """Имя ветки: своё, если задано, иначе из задачи — `fix/...` или `feat/...`."""
        if self.branch:
            return self.branch
        stem = _BRANCH_SAFE_RE.sub("-", self.instruction.lower())[:40].strip("-") or "task"
        kind = "fix" if any(word in self.instruction.lower() for word in _FIX_WORDS) else "feat"
        return f"{kind}/{stem}"


#: По этим словам задача читается как починка, а не как новая возможность.
_FIX_WORDS = ("почини", "падает", "ошибк", "баг", "не работает", "fix", "сломал", "чинить")


@dataclass(slots=True)
class SweOutcome:
    """Что получилось: ветка, изменённые файлы, отчёт проверок и поводы для реплик."""

    request: SweRequest
    ok: bool = False
    summary: str = ""
    branch: str = ""
    commit: str = ""
    changed_files: list[str] = field(default_factory=list)
    plan: str = ""
    repair: RepairReport | None = None
    failure_reason: str = ""
    notes: list[str] = field(default_factory=list)
    #: Где физически считалась генерация — «ноутбук (sonnet)» или облако.
    tier: str = ""
    workspace_path: str = ""

    @property
    def green(self) -> bool:
        return self.ok and (self.repair is None or self.repair.green)


#: Короткая реплика в чат по ходу работы — тот же контракт, что у
#: efi.dev.auto_fix.Narrator.
Narrator = Callable[[str], Awaitable[None]]


class SweEngine:
    """
    Один проход по чужому репозиторию: разобраться → поправить → проверить →
    оставить ветку.

    Состояния между задачами не держит: рабочая копия своя на каждую задачу и
    удаляется, если не просили иначе.
    """

    def __init__(
        self,
        router: NetworkModelRouter,
        workspaces: WorkspaceManager,
        *,
        gate: ConcurrencyGate | None = None,
        narrator: Narrator | None = None,
        lookup: Lookup | None = None,
        max_repair_rounds: int = 4,
        map_budget_bytes: int = DEFAULT_MAP_BUDGET_BYTES,
        keep_workspace: bool = False,
    ) -> None:
        self._router = router
        self._workspaces = workspaces
        self._gate = gate or ConcurrencyGate(limit=2)
        self._narrator = narrator
        #: Поиск ответа на ошибку, пережившую первую правку — то же, что у
        #: проверки собственных проектов (efi/dev/auto_fix.py).
        self._lookup = lookup
        self._max_repair_rounds = max_repair_rounds
        self._map_budget_bytes = map_budget_bytes
        #: Оставлять ли рабочую копию после задачи. По умолчанию нет: это
        #: /tmp на телефоне, и десяток чужих репозиториев там не нужен.
        #: Включается, когда ветку ещё предстоит забрать руками.
        self._keep_workspace = keep_workspace

    async def work_on(self, request: SweRequest) -> SweOutcome:
        """
        Полный проход. Не поднимает исключений: любая беда возвращается
        объектом с причиной — этот код вызывается из фонового цикла, где
        падение означает молчание вместо ответа человеку.
        """
        outcome = SweOutcome(request=request, tier=await self._router.tier())
        try:
            workspace = await self._workspaces.prepare(request.source, session_id=request.session_id)
        except WorkspaceError as exc:
            outcome.failure_reason = str(exc)
            return outcome

        outcome.workspace_path = str(workspace.root)
        try:
            await self._run(workspace, request, outcome)
        except Exception as exc:  # noqa: BLE001 — фоновая задача не имеет права падать наверх
            logger.exception("swe_engine: задача %s сорвалась", request.session_id)
            outcome.failure_reason = f"внутренняя ошибка: {exc}"
        finally:
            if not self._keep_workspace and not outcome.ok:
                workspace.discard()
        return outcome

    async def _run(self, workspace: Workspace, request: SweRequest, outcome: SweOutcome) -> None:
        repo_map = build_repo_map(workspace.root, budget_bytes=self._map_budget_bytes)
        if not repo_map.files:
            outcome.failure_reason = "в репозитории нет ни одного файла, который я понимаю"
            return

        focus, plan = await self._choose_files(repo_map.render(), request, repo_map.paths)
        outcome.plan = plan
        if not focus:
            outcome.failure_reason = "не поняла, какие файлы тут править"
            return
        logger.info("swe_engine: работаю с %s", ", ".join(focus))

        answer = await self._ask_for_edits(workspace, request, focus, plan)
        if not answer:
            outcome.failure_reason = "модель не прислала правок"
            return

        edits = parse_edits(answer)
        if not edits:
            outcome.failure_reason = "в ответе не было ни одного блока правки"
            return

        changed, problems = apply_edits(workspace.root, edits)
        if not changed:
            outcome.failure_reason = _first_problem(problems)
            return
        outcome.changed_files = list(changed)

        repair = RepairLoop(
            self._fixer(),
            max_rounds=self._max_repair_rounds,
            narrator=self._narrator,
            lookup=self._lookup,
        )
        report = await repair.run(workspace, list(changed))
        outcome.repair = report
        outcome.notes = list(report.notes)
        for path in report.changed_files:
            if path not in outcome.changed_files:
                outcome.changed_files.append(path)

        if not report.green:
            failure = report.last_failure
            outcome.failure_reason = (
                f"проверки так и не позеленели: {failure.kind.human}" if failure else "проверки не прошли"
            )
            return

        branch, commit = await self._commit(workspace, request, outcome)
        if not branch:
            outcome.failure_reason = "изменения есть, но коммит не встал"
            return
        outcome.branch = branch
        outcome.commit = commit
        outcome.ok = True
        outcome.summary = _summarize(outcome)

    async def _choose_files(
        self, rendered_map: str, request: SweRequest, known_paths: list[str]
    ) -> tuple[list[str], str]:
        """
        Выбор двух-трёх файлов по карте. Ответ модели проверяется по карте:
        путь, которого в репозитории нет, — это не выбор, а фантазия.
        """
        answer = await self._ask(
            _SELECT_SYSTEM_PROMPT,
            f"Задача: {request.instruction}\n\nКарта репозитория:\n{rendered_map}",
            max_tokens=_ANALYSIS_MAX_TOKENS,
        )
        if not answer:
            return [], ""

        match = _FILES_LINE_RE.search(answer)
        raw_paths = match.group("paths") if match else ""
        known = set(known_paths)
        chosen: list[str] = []
        for candidate in re.split(r"[,\s]+", raw_paths):
            path = candidate.strip().strip("`'\"").lstrip("./")
            if path and path in known and path not in chosen:
                chosen.append(path)
        plan = answer[match.end() :].strip() if match else answer.strip()
        return chosen[:MAX_FILES_IN_FOCUS], plan

    async def _ask_for_edits(
        self, workspace: Workspace, request: SweRequest, focus: list[str], plan: str
    ) -> str | None:
        files = "\n\n".join(
            f"### {path}\n{workspace.read(path, max_chars=_MAX_FILE_CHARS)}" for path in focus
        )
        intent = f"\nТы уже решила так: {plan}\n" if plan else ""
        return await self._ask(
            _EDIT_SYSTEM_PROMPT,
            f"Задача: {request.instruction}\n{intent}\nФайлы целиком:\n\n{files}",
            max_tokens=_EDIT_MAX_TOKENS,
        )

    def _fixer(self) -> Callable[[str, str], Awaitable[str | None]]:
        """Тот же способ спросить модель, что и у самого движка, — для цикла починки."""

        async def call(system_prompt: str, request: str) -> str | None:
            return await self._ask(system_prompt, request, max_tokens=_EDIT_MAX_TOKENS)

        return call

    async def _ask(self, system_prompt: str, request: str, *, max_tokens: int) -> str | None:
        """
        Один запрос к модели — через сетевой роутер (ноутбук или облако) и
        под общим потолком параллелизма.
        """
        params = LLMParams(model="", system_prompt=system_prompt, max_output_tokens=max_tokens)
        session = Session(messages=[Message(role=Role.USER, content=request)])
        log = AttemptLog()

        async def call() -> str | None:
            response = await self._router.chat(params, session, log=log)
            return response.text

        try:
            return await self._gate.run(call)
        except Exception as exc:  # noqa: BLE001 — отказ модели это «правок нет», а не крах задачи
            logger.warning("swe_engine: модель не ответила (%s)", exc)
            return None

    async def _commit(
        self, workspace: Workspace, request: SweRequest, outcome: SweOutcome
    ) -> tuple[str, str]:
        """
        Ветка и коммит — только после зелёных проверок.

        Пуш пробуется, только если его просили: у чужого репозитория прав на
        запись обычно нет, и падать из-за этого нельзя — ветка всё равно
        осталась в рабочей копии, и её видно.
        """
        branch = request.branch_name()
        checkout = await workspace.run("git", "checkout", "-b", branch)
        if not checkout.ok:
            # Ветка уже есть (повторный заход по той же задаче) — переходим в неё.
            await workspace.run("git", "checkout", branch)

        add = await workspace.run("git", "add", "--", *outcome.changed_files)
        if not add.ok:
            logger.warning("swe_engine: git add не прошёл: %s", add.output[:200])
            return "", ""

        message = _commit_message(request, outcome)
        commit = await workspace.run(
            "git",
            "-c", f"user.name={_COMMIT_AUTHOR_NAME}",
            "-c", f"user.email={_COMMIT_AUTHOR_EMAIL}",
            "commit", "-m", message,
        )
        if not commit.ok:
            logger.warning("swe_engine: коммит не встал: %s", commit.output[:200])
            return "", ""

        revision = await workspace.run("git", "rev-parse", "--short", "HEAD")
        if request.push:
            pushed = await workspace.run("git", "push", "-u", "origin", branch)
            if not pushed.ok:
                logger.info("swe_engine: пуш не прошёл (%s) — ветка осталась локальной", pushed.output[:160])
                outcome.notes.append("запушить не вышло — прав на репозиторий нет, ветка лежит локально")
        return branch, revision.stdout.strip()


def _commit_message(request: SweRequest, outcome: SweOutcome) -> str:
    """Сообщение коммита: что просили, одной строкой в принятом стиле."""
    kind = "fix" if request.branch_name().startswith("fix/") else "feat"
    subject = request.instruction.strip().splitlines()[0][:68]
    body = outcome.plan.strip()[:500]
    head = f"{kind}: {subject}"
    return f"{head}\n\n{body}" if body else head


def _summarize(outcome: SweOutcome) -> str:
    """Сводка для реплики в чат — факты, а не текст сообщения."""
    files = ", ".join(outcome.changed_files[:4])
    rounds = outcome.repair.rounds if outcome.repair is not None else 0
    tail = f", по дороге чинила себя {rounds} раз(а)" if rounds else ""
    return f"ветка {outcome.branch}, правки в {files}, проверки зелёные{tail}"


def _first_problem(problems: list[str]) -> str:
    return problems[0] if problems else "ни одна правка не легла на файлы"


def local_source(path: Path) -> str:
    """Локальный путь как источник задачи — для работы над своим же кодом."""
    return str(path.resolve())


__all__ = [
    "MAX_FILES_IN_FOCUS",
    "Narrator",
    "SweEngine",
    "SweOutcome",
    "SweRequest",
    "local_source",
]
