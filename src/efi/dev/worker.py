"""
efi/dev/worker.py

Фоновый цикл ремесла: берёт задачу и ведёт её от замысла до ссылки.

Место в жизни Эфи — рядом с efi.behavior.life_engine.BackgroundLifeWorker:
такой же периодический сервис, запускаемый из efi/app.py, с тем же правилом
«сбой итерации не завершает цикл» (efi/utils/loops.py). Разница в предмете:
life_engine превращает любопытство в мысль, этот — замысел в репозиторий.

Два источника задач:

    1. Совместные — их создаёт efi.behavior.collab_coding.CollabCodingDesk,
       когда человек и Эфи договорились в чате. Такие идут первыми: их ждут.
    2. Собственные — если очередь пуста, Эфи с некоторой вероятностью
       затевает проект сама, отталкиваясь от того, чем сейчас живёт
       (интересы и семена любопытства — тот же источник, что у участия в
       сообществе, см. efi/telegram/comments.py).

`is_coding` — вход для efi.behavior.busy_engine.BusyEngine, ровно как
`is_researching` у движка жизни: пока Эфи пишет код, она не отвечает на
сообщение в ту же секунду, потому что действительно занята. Это не
косметика: занятость, которую видно в поведении, — единственная разница
между «у неё есть свои дела» и «она утверждает, что у неё есть свои дела».

Задача двигается по статусам с сохранением в БД на каждом переходе (см.
efi/dev/store.py): перезапуск посреди работы возвращает её в очередь, а не
теряет.
"""

from __future__ import annotations

import logging
import random
from datetime import timedelta
from typing import Protocol

from efi.dev.engine import BuildResult, DevEngine
from efi.dev.github_sync import GitHubSync, GitHubSyncError
from efi.dev.reporter import DevReporter
from efi.dev.schemas import DevTask, DevTaskStatus, ProjectSpec
from efi.dev.store import DevTaskStore
from efi.utils.loops import run_periodically

logger = logging.getLogger(__name__)

#: Сколько интересов подмешивать в замысел собственного проекта. Больше —
#: и модель начинает сочинять «универсальный инструмент для всего сразу».
_MAX_CONTEXT_INTERESTS = 5

#: После какого простоя задача «в работе» считается брошенной. Три часа —
#: заведомо больше самого долгого прохода (спека, несколько файлов с
#: правками, публикация) даже на медленных бесплатных лимитах, поэтому под
#: порог не попадёт живая работа, а не только мёртвая.
_STALLED_AFTER = timedelta(hours=3)


class InterestSource(Protocol):
    """
    Чем Эфи сейчас живёт. Реализация — efi.telegram.comments.CommunityInterests
    (worldview.json + семена любопытства из разговоров).
    """

    async def current_interests(self) -> list[str]: ...


class DevWorker:
    """
    Один проект за тик. Намеренно: конвейер — это десятки запросов к моделям,
    и параллельно вести два проекта на бесплатных лимитах значит не довести
    ни одного.
    """

    def __init__(
        self,
        store: DevTaskStore,
        engine: DevEngine,
        github: GitHubSync,
        reporter: DevReporter,
        *,
        interests: InterestSource | None = None,
        owner_chat_id: int | None = None,
        check_interval_seconds: float = 3600.0,
        self_initiated_probability: float = 0.25,
    ) -> None:
        self._store = store
        self._engine = engine
        self._github = github
        self._reporter = reporter
        self._interests = interests
        self._owner_chat_id = owner_chat_id
        self._check_interval_seconds = check_interval_seconds
        self._self_initiated_probability = self_initiated_probability
        self._is_coding = False

    @property
    def is_coding(self) -> bool:
        """True на всё время работы над проектом — вход для efi.behavior.busy_engine.BusyEngine."""
        return self._is_coding

    async def run(self) -> None:
        """Основной цикл. Останавливается по отмене задачи (CancelledError) — см. efi/app.py graceful shutdown."""
        logger.info(
            "dev_worker: интервал %.0fс, своя инициатива %s",
            self._check_interval_seconds,
            f"{self._self_initiated_probability:.0%}" if self._self_initiated_probability > 0 else "выключена",
        )
        await run_periodically(
            self._tick, interval_seconds=self._check_interval_seconds, name="dev_worker"
        )

    async def _tick(self) -> None:
        # Задачи, брошенные посреди работы (процесс упал, телефон убил
        # фоновую задачу), возвращаются в очередь ПЕРЕД выбором следующей:
        # иначе они навсегда остаются в статусе «пишу код», и Эфи месяцами
        # рассказывает про проект, к которому никто не подходил.
        await self._store.reclaim_stalled(older_than=_STALLED_AFTER)

        task = await self._store.next_pending()
        if task is None:
            task = await self._maybe_start_own_project()
        if task is None:
            return

        self._is_coding = True
        try:
            await self._process(task)
        finally:
            self._is_coding = False

    async def _maybe_start_own_project(self) -> DevTask | None:
        """
        Затеять что-то своё. Не каждый тик и не поверх уже идущей работы:
        человек, у которого одновременно пять начатых проектов, ничего не
        доводит до конца — и выглядит это так же.
        """
        if self._self_initiated_probability <= 0.0:
            return None
        if await self._store.active():
            return None
        if random.random() > self._self_initiated_probability:
            return None
        return await self._store.create("", chat_id=self._owner_chat_id, is_collab=False)

    async def _process(self, task: DevTask) -> None:
        spec = task.spec
        if spec is None:
            task = await self._store.update(task, status=DevTaskStatus.SPECCING)
            spec = await self._engine.design(task.idea, context=await self._render_context())
            if spec is None:
                await self._fail(task, "не придумалось ничего, что стоило бы писать")
                return
            task = await self._store.update(task, spec=spec)
            await self._reporter.report_progress(task, _design_note(spec))

        task = await self._store.update(task, status=DevTaskStatus.CODING)
        build = await self._engine.build(spec)
        if not build.is_publishable:
            await self._fail(task, _broken_reason(build))
            return

        note = _build_note(build)
        if note:
            await self._reporter.report_progress(task, note)

        task = await self._store.update(task, status=DevTaskStatus.PUBLISHING)
        try:
            published = await self._github.publish(spec, build.files)
        except GitHubSyncError as exc:
            await self._fail(task, str(exc))
            return

        if not published.pushed:
            # Локальный режим: код есть, ссылки нет. Это не провал задачи —
            # но и хвастаться нечем, поэтому в чат ничего не уходит.
            task = await self._store.update(
                task, status=DevTaskStatus.DONE, error="без пуша: не настроен доступ к GitHub"
            )
            logger.info(
                "dev_worker: проект %s собран локально в %s (%d коммитов), пуш не настроен",
                spec.slug, published.local_path, len(published.commits),
            )
            return

        task = await self._store.update(task, status=DevTaskStatus.DONE, repo_url=published.url)
        logger.info("dev_worker: проект %s готов: %s", spec.slug, published.url)
        await self._reporter.report_release(task, url=published.url, build=build)

    async def _fail(self, task: DevTask, reason: str) -> None:
        logger.warning("dev_worker: задача #%s провалилась: %s", task.id, reason)
        failed = await self._store.update(task, status=DevTaskStatus.FAILED, error=reason)
        await self._reporter.report_failure(failed, reason)

    async def _render_context(self) -> str:
        if self._interests is None:
            return ""
        try:
            interests = await self._interests.current_interests()
        except Exception:
            logger.warning("dev_worker: не удалось получить интересы", exc_info=True)
            return ""
        return ", ".join(interests[:_MAX_CONTEXT_INTERESTS])


def _design_note(spec: ProjectSpec) -> str:
    """Факт для реплики: она только что придумала, что писать."""
    return (
        f"придумала структуру проекта: {len(spec.files)} файла(ов), "
        f"стек {', '.join(spec.stack) or 'чистый python'}"
    )


def _build_note(build: BuildResult) -> str:
    """
    Факт для реплики о процессе — из того, что РЕАЛЬНО происходило.

    Ради этого BuildResult и хранит число правок и незакрытые замечания: без
    фактуры реплика про работу превращается в «всё идёт по плану», а с ней
    получается то самое «линтер задушил меня из-за аннотаций».
    """
    if build.fix_rounds:
        worst = max(build.files, key=lambda item: item.fix_rounds)
        detail = worst.unresolved_diagnostics.splitlines()[0] if worst.unresolved_diagnostics else ""
        tail = f", и он всё ещё бурчит: {detail}" if detail else ""
        return (
            f"переписывала {worst.path} {worst.fix_rounds} раз(а) из-за замечаний линтера{tail}"
        )
    if build.unresolved:
        return f"линтер докопался до {build.unresolved[0].path}, но по делу — оставила как есть"
    return f"код готов, {len(build.files)} файла(ов), и всё прошло проверку с первого раза"


def _broken_reason(build: BuildResult) -> str:
    if build.broken_paths:
        return f"файлы {', '.join(build.broken_paths)} так и не заработали — код не собирается"
    return "кодер не написал ни одного файла"


__all__ = ["DevWorker", "InterestSource"]
