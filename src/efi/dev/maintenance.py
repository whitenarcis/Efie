"""
efi/dev/maintenance.py

Возвращение к своим проектам: перечитать, подумать, иногда поправить, ещё
реже — спросить.

Зачем это вообще. Сгенерировать репозиторий и забыть о нём умеет скрипт;
человека от скрипта отличает как раз то, что он через неделю открывает свой
старый код, морщится и что-то в нём меняет. Пока проект после релиза
неподвижен, «своё ремесло» остаётся разовой генерацией — и любая фраза Эфи
про её проекты будет правдой ровно один день.

Три исхода одного просмотра, и они не равновероятны:

    ничего   — самый частый и совершенно нормальный. Код рабочий, трогать
               нечего. Просмотр всё равно засчитывается: она посмотрела.
    правка   — нашла конкретную мелочь (битый пример в README, необработанное
               исключение, мёртвый параметр) и молча починила: файл
               переписывается кодером, проверяется песочницей, уходит одним
               осмысленным коммитом.
    спросить — нашла то, что решать не ей: сменить формат данных, выкинуть
               половину замысла, поменять поведение, которого от неё ждут.
               Тогда она пишет владельцу — но только если вопрос правда
               стоит его времени (см. `discuss_importance`).

Порог «важности» — не украшение, а весь смысл модуля. Модель, которую
попросили найти, что улучшить, находит всегда: переименовать переменную,
добавить тайпхинт, разбить функцию. Если пускать в чат каждую такую находку,
получится бот, который еженедельно спрашивает разрешения переименовать
переменную, — то есть ровно противоположность живому человеку с проектами.
Поэтому важность приходит числом, и оба действия (правка и особенно вопрос)
включаются только выше своих порогов.
"""

from __future__ import annotations

import json
import logging
import random
import re
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

from efi.config.schema import TaskRole
from efi.dev.github_sync import GitHubSync, GitHubSyncError
from efi.dev.qwen_client import QwenCoderClient
from efi.dev.readme import README_PATH, is_acceptable
from efi.dev.reporter import DevReporter
from efi.dev.sandbox import CodeSandbox
from efi.dev.schemas import DevTask, GeneratedFile
from efi.dev.store import DevTaskStore
from efi.llm.errors import LLMError
from efi.llm.router import LLMRouter
from efi.llm.schemas import LLMParams, Message, Role, Session

logger = logging.getLogger(__name__)

#: Как давно проект должен быть не тронут, чтобы к нему возвращаться. Неделя:
#: раньше возвращаться незачем (он ровно такой, каким его дописали), а сильно
#: позже — уже не «свой проект», а археология.
DEFAULT_REVIEW_INTERVAL = timedelta(days=7)

#: Пороги важности. Правка — от «это правда стоит коммита», вопрос — заметно
#: выше: чужое время дороже своего.
DEFAULT_PATCH_THRESHOLD = 0.5
DEFAULT_DISCUSS_THRESHOLD = 0.8

#: Сколько символов файла показывать ревизору. Проекты маленькие, но контекст
#: не бесконечный, и обрезка по началу файла оставляет самое важное: импорты,
#: сигнатуры, разбор аргументов.
_MAX_FILE_PREVIEW = 2000
_MAX_FILES_REVIEWED = 6

_REVIEW_MAX_OUTPUT_TOKENS = 800

_REVIEW_SYSTEM_PROMPT = (
    "Ты перечитываешь СВОЙ старый проект — тот, который сама написала и выложила. Не ревью для "
    "заказчика, а взгляд автора: что тут стыдно, что сломается у человека при первом запуске, чего не "
    "хватает по делу.\n"
    "Главное: «всё нормально» — НОРМАЛЬНЫЙ и самый частый ответ. Не выдумывай улучшения ради "
    "улучшений. Переименования, тайпхинты, разбиение функций, «можно добавить тесты» и прочий "
    "косметический зуд — это verdict=nothing.\n"
    "Правка (verdict=patch) — когда есть конкретная поломка или прямое враньё: пример в README не "
    "работает, необработанное исключение на очевидном вводе, документация расходится с кодом, "
    "оставшаяся заглушка.\n"
    "Вопрос (verdict=discuss) — только когда решение НЕ ТВОЁ: поменять формат данных или интерфейс, "
    "выкинуть заметную часть замысла, изменить поведение, на которое человек мог рассчитывать. "
    "Пустяки в вопросы не выносят.\n"
    "Ответь ОДНИМ объектом JSON без markdown:\n"
    '{"verdict": "nothing|patch|discuss", "importance": 0.0-1.0, "path": "файл, который править", '
    '"what": "что именно сделать, одним предложением", "commit": "сообщение коммита в стиле '
    'fix:/docs:", "question": "что спросить у человека, если verdict=discuss", '
    '"note": "как ты сама об этом скажешь одной живой фразой"}'
)

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n(?P<body>.*?)(?:\n\s*```|\Z)", re.DOTALL)


@dataclass(slots=True, frozen=True)
class ReviewVerdict:
    """Что Эфи решила по итогам перечитывания своего проекта."""

    verdict: str = "nothing"
    importance: float = 0.0
    path: str = ""
    what: str = ""
    commit: str = ""
    question: str = ""
    note: str = ""

    @property
    def wants_patch(self) -> bool:
        return self.verdict == "patch" and bool(self.path) and bool(self.what)

    @property
    def wants_discussion(self) -> bool:
        return self.verdict == "discuss" and bool(self.question)


@dataclass(slots=True, frozen=True)
class ReviewOutcome:
    """Итог одного просмотра — то, что уходит в лог, в дашборд и в отметку о ревизии."""

    task_id: int
    verdict: ReviewVerdict
    patched: bool = False
    asked: bool = False

    @property
    def changed_anything(self) -> bool:
        return self.patched or self.asked


class ProjectMaintainer:
    """
    Один проект за проход: прочитать с диска, показать модели, исполнить её
    решение.

    Работает по локальному клону, оставшемуся от публикации (см.
    efi/dev/github_sync.py): это тот же код, что и в репозитории, и правка
    ложится обычным коммитом поверх истории, а не переписыванием проекта
    заново.
    """

    def __init__(
        self,
        store: DevTaskStore,
        router: LLMRouter,
        coder: QwenCoderClient,
        sandbox: CodeSandbox,
        github: GitHubSync,
        reporter: DevReporter,
        workspace: Path,
        *,
        review_role: TaskRole = TaskRole.BACKGROUND,
        review_interval: timedelta = DEFAULT_REVIEW_INTERVAL,
        patch_threshold: float = DEFAULT_PATCH_THRESHOLD,
        discuss_threshold: float = DEFAULT_DISCUSS_THRESHOLD,
        review_probability: float = 0.3,
    ) -> None:
        self._store = store
        self._router = router
        self._coder = coder
        self._sandbox = sandbox
        self._github = github
        self._reporter = reporter
        self._workspace = workspace
        self._review_role = review_role
        self._review_interval = review_interval
        self._patch_threshold = patch_threshold
        self._discuss_threshold = discuss_threshold
        self._review_probability = review_probability

    async def maybe_review(self) -> ReviewOutcome | None:
        """
        Заглянуть в один из своих проектов — иногда.

        Вероятность здесь не для «естественности», а по смыслу: перечитывать
        свой код по расписанию, как по будильнику, — это не привычка автора,
        а cron. Плюс к ней действует интервал: проект, просмотренный на
        прошлой неделе, в выборку не попадает вовсе.
        """
        if self._review_probability <= 0.0 or random.random() > self._review_probability:
            return None

        candidates = await self._store.due_for_review(not_reviewed_for=self._review_interval)
        if not candidates:
            return None
        return await self.review(candidates[0])

    async def review(self, task: DevTask) -> ReviewOutcome | None:
        """Полный проход по одному проекту. None — не удалось даже прочитать его с диска."""
        if task.spec is None:
            return None

        sources = self._read_project(task)
        if not sources:
            logger.info(
                "dev_maintenance: локального клона %s нет, пересматривать нечего", task.spec.slug
            )
            # Отметку всё равно ставим: иначе этот же недоступный проект
            # будет попадать в выборку каждый раз и вытеснять остальные.
            await self._store.mark_reviewed(task)
            return None

        verdict = await self._ask_for_verdict(task, sources)
        outcome = await self._apply(task, verdict, sources)
        await self._store.mark_reviewed(task, revised=outcome.patched)
        return outcome

    def _read_project(self, task: DevTask) -> dict[str, str]:
        """Файлы проекта с диска. Читается КЛОН, а не то, что когда-то сгенерировали: правки могли быть и раньше."""
        assert task.spec is not None  # проверено вызывающей стороной
        project_dir = (self._workspace / task.spec.slug).resolve()
        if not project_dir.is_dir():
            return {}

        sources: dict[str, str] = {}
        for relative in [*(item.path for item in task.spec.files), README_PATH]:
            path = project_dir / relative
            if not path.is_file():
                continue
            try:
                sources[relative] = path.read_text(encoding="utf-8")
            except OSError:
                logger.warning("dev_maintenance: не читается %s", path, exc_info=True)
        return sources

    async def _ask_for_verdict(self, task: DevTask, sources: dict[str, str]) -> ReviewVerdict:
        assert task.spec is not None
        listing = "\n\n".join(
            f"### {path}\n{content[:_MAX_FILE_PREVIEW]}"
            for path, content in list(sources.items())[:_MAX_FILES_REVIEWED]
        )
        user_content = (
            f"Твой проект «{task.spec.title}» ({task.spec.slug}).\n"
            f"Задумывался так: {task.spec.problem}\n"
            f"Правок после релиза: {task.revisions}\n"
            f"Ссылка: {task.repo_url}\n\n"
            f"Файлы:\n{listing}"
        )
        params = LLMParams(
            model="", system_prompt=_REVIEW_SYSTEM_PROMPT, max_output_tokens=_REVIEW_MAX_OUTPUT_TOKENS
        )
        session = Session(messages=[Message(role=Role.USER, content=user_content)])
        try:
            response = await self._router.chat(self._review_role, params, session)
        except LLMError as exc:
            logger.warning("dev_maintenance: не удалось перечитать %s: %s", task.spec.slug, exc)
            return ReviewVerdict()
        return parse_verdict(response.text)

    async def _apply(
        self, task: DevTask, verdict: ReviewVerdict, sources: dict[str, str]
    ) -> ReviewOutcome:
        assert task.spec is not None

        if verdict.wants_discussion and verdict.importance >= self._discuss_threshold:
            await self._reporter.remember_revision(
                task, f"Упёрлась в развилку, которую не решаю одна: {verdict.question}"
            )
            await self._reporter.report_question(task, verdict.question)
            logger.info(
                "dev_maintenance: %s — вопрос владельцу (важность %.2f)", task.spec.slug, verdict.importance
            )
            return ReviewOutcome(task_id=task.id, verdict=verdict, asked=True)

        if verdict.wants_patch and verdict.importance >= self._patch_threshold:
            patched = await self._patch(task, verdict, sources)
            return ReviewOutcome(task_id=task.id, verdict=verdict, patched=patched)

        logger.info(
            "dev_maintenance: %s — посмотрела, трогать нечего (%s, важность %.2f)",
            task.spec.slug, verdict.verdict, verdict.importance,
        )
        return ReviewOutcome(task_id=task.id, verdict=verdict)

    async def _patch(self, task: DevTask, verdict: ReviewVerdict, sources: dict[str, str]) -> bool:
        assert task.spec is not None
        source = sources.get(verdict.path)
        if source is None:
            logger.info("dev_maintenance: %s — файла %s нет, правка отменяется", task.spec.slug, verdict.path)
            return False

        updated = await self._coder.fix_file(verdict.path, source, verdict.what)
        if updated is None or updated.strip() == source.strip():
            return False

        report = await self._sandbox.check(verdict.path, updated)
        if report.syntax_broken:
            # Сломать работающий проект «улучшением» — худший исход правки:
            # до неё код работал.
            logger.warning(
                "dev_maintenance: правка %s ломает синтаксис, откатываю замысел", verdict.path
            )
            return False
        if verdict.path == README_PATH and not is_acceptable(updated):
            logger.info("dev_maintenance: правка README ухудшила бы его, пропускаю")
            return False

        try:
            committed = await self._github.commit_revision(
                task.spec,
                [GeneratedFile(path=verdict.path, content=updated)],
                message=_commit_message(verdict),
            )
        except GitHubSyncError as exc:
            logger.warning("dev_maintenance: правку %s не удалось выложить: %s", task.spec.slug, exc)
            return False

        if committed:
            note = verdict.note or _fallback_note(verdict)
            # В память — всегда, в чат — по настроению и кулдауну. Правка,
            # о которой она не успела рассказать, всё равно её работа: через
            # неделю «я к этой штуке возвращалась и вот что поправила» должно
            # находиться, а не сочиняться.
            await self._reporter.remember_revision(
                task, f"Поправила {verdict.path}: {verdict.what}. Коммит: {_commit_message(verdict)}"
            )
            await self._reporter.report_progress(task, note)
        return committed


def parse_verdict(raw: str) -> ReviewVerdict:
    """
    Разбирает ответ ревизора. Чистая функция; всё непонятное трактуется как
    «ничего не делать» — это безопасный исход: проект остаётся работающим.
    """
    text = (raw or "").strip()
    fenced = _JSON_FENCE_RE.search(text)
    if fenced is not None:
        text = fenced.group("body").strip()

    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return ReviewVerdict()
    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return ReviewVerdict()
    if not isinstance(payload, dict):
        return ReviewVerdict()

    verdict = str(payload.get("verdict", "nothing")).strip().lower()
    if verdict not in ("nothing", "patch", "discuss"):
        verdict = "nothing"
    try:
        importance = min(1.0, max(0.0, float(payload.get("importance", 0.0))))
    except (TypeError, ValueError):
        importance = 0.0

    return ReviewVerdict(
        verdict=verdict,
        importance=importance,
        path=str(payload.get("path", "")).strip(),
        what=str(payload.get("what", "")).strip(),
        commit=str(payload.get("commit", "")).strip(),
        question=str(payload.get("question", "")).strip(),
        note=str(payload.get("note", "")).strip(),
    )


#: Префиксы conventional commits, которые мы принимаем от модели как есть.
#: Всё остальное превращается в `fix: ...`: история проекта не должна
#: состоять из сообщений вида «Обновление файла».
_ALLOWED_COMMIT_PREFIXES = ("fix:", "docs:", "feat:", "refactor:", "chore:", "test:", "perf:")
_MAX_COMMIT_SUBJECT = 72


def _commit_message(verdict: ReviewVerdict) -> str:
    message = (verdict.commit or verdict.what).strip().replace("\n", " ")
    if not message.lower().startswith(_ALLOWED_COMMIT_PREFIXES):
        message = f"fix: {message}"
    return message[:_MAX_COMMIT_SUBJECT]


def _fallback_note(verdict: ReviewVerdict) -> str:
    """Если модель не сформулировала живую фразу — берём суть правки: факт важнее формулировки."""
    return f"перечитала свой старый код и поправила {verdict.path}: {verdict.what}"


__all__ = [
    "DEFAULT_DISCUSS_THRESHOLD",
    "DEFAULT_PATCH_THRESHOLD",
    "DEFAULT_REVIEW_INTERVAL",
    "ProjectMaintainer",
    "ReviewOutcome",
    "ReviewVerdict",
    "parse_verdict",
]
