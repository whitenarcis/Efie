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

import asyncio
import logging
import random
from datetime import timedelta
from typing import Protocol

from efi.dev.engine import BuildResult, DevEngine
from efi.dev.github_sync import GitHubSync, GitHubSyncError
from efi.dev.maintenance import ProjectMaintainer
from efi.dev.reporter import DevReporter
from efi.dev.schemas import DevTask, DevTaskKind, DevTaskStatus, ProjectSpec
from efi.dev.store import DevTaskStore
from efi.dev.swe_engine import SweEngine, SweRequest
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

#: Сколько раз конвейер берётся за одну задачу, прежде чем признать её
#: несбывшейся. Три — потому что провалы здесь в основном временные: 429 на
#: третьем файле из четырёх, оборванная сеть на пуше, недоступный на минуту
#: провайдер. Без повторов такой сбой хоронил проект навсегда, хотя к
#: следующему часу всё уже работает; с бесконечными повторами Эфи вечно
#: возвращалась бы к замыслу, который не выходит.
_MAX_ATTEMPTS = 3

#: Через сколько после провала имеет смысл вернуться к брошенному проекту.
#: Полсуток — потому что чаще всего провал случается из-за упёршегося лимита
#: провайдера, а он отпускает к следующему дню; плюс это просто похоже на
#: правду: к тому, что не пошло вечером, возвращаются наутро, а не через
#: минуту.
_REVIVE_AFTER = timedelta(hours=12)

#: Сколько раз возвращаться к одному и тому же брошенному замыслу. Дважды:
#: третий круг по проекту, который не собрался ни разу за два дня, стоит
#: квоты, за которую пишется что-то новое.
_MAX_REVIVALS = 2


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
        maintainer: ProjectMaintainer | None = None,
        swe: SweEngine | None = None,
        interests: InterestSource | None = None,
        owner_chat_id: int | None = None,
        check_interval_seconds: float = 3600.0,
        self_initiated_probability: float = 0.25,
    ) -> None:
        self._store = store
        self._engine = engine
        self._github = github
        self._reporter = reporter
        #: Возвращение к уже выложенным проектам (efi/dev/maintenance.py).
        #: Необязательно: без него Эфи просто пишет новое и не перечитывает
        #: старое — то есть ведёт себя как генератор репозиториев.
        self._maintainer = maintainer
        #: Работа с чужим кодом (efi/dev/swe_engine.py). Необязательна: без
        #: неё Эфи остаётся автором собственных проектов и не берётся за
        #: чужие репозитории — ровно то поведение, что было до этого модуля.
        self._swe = swe
        self._interests = interests
        self._owner_chat_id = owner_chat_id
        self._check_interval_seconds = check_interval_seconds
        self._self_initiated_probability = self_initiated_probability
        self._is_coding = False
        #: Сигнал «появилась работа, не жди следующего тика». Ставится, когда
        #: человек договорился о проекте в чате: ждать час после «ок, берусь»
        #: — это ровно то, из-за чего непонятно, взялась она вообще или
        #: просто поддакнула (см. request_tick).
        self._wake = asyncio.Event()
        #: Разовая чистка пустых провалов при первом тике (см. _tick).
        self._purged_stubs = False

    @property
    def is_coding(self) -> bool:
        """True на всё время работы над проектом — вход для efi.behavior.busy_engine.BusyEngine."""
        return self._is_coding

    def request_tick(self) -> None:
        """
        Разбудить цикл сейчас, не дожидаясь расписания.

        Вызывается, когда задача появилась не из таймера, а из разговора
        (efi.behavior.collab_coding.CollabCodingDesk.start). Синхронный и
        дешёвый: ставит событие, которое ждёт `run`.
        """
        self._wake.set()

    async def run(self) -> None:
        """Основной цикл. Останавливается по отмене задачи (CancelledError) — см. efi/app.py graceful shutdown."""
        logger.info(
            "dev_worker: интервал %.0fс, своя инициатива %s",
            self._check_interval_seconds,
            f"{self._self_initiated_probability:.0%}" if self._self_initiated_probability > 0 else "выключена",
        )
        await run_periodically(
            self._tick,
            interval_seconds=self._check_interval_seconds,
            name="dev_worker",
            wake_event=self._wake,
        )

    async def _tick(self) -> None:
        # Мусор от прошлых версий: задачи, которые падали на проектировании и
        # оседали в базе пустыми строчками «замысел без названия». Чистится
        # один раз, а не каждый тик — см. purge_empty_failures.
        if not self._purged_stubs:
            self._purged_stubs = True
            await self._store.purge_empty_failures()

        # Задачи, брошенные посреди работы (процесс упал, телефон убил
        # фоновую задачу), возвращаются в очередь ПЕРЕД выбором следующей:
        # иначе они навсегда остаются в статусе «пишу код», и Эфи месяцами
        # рассказывает про проект, к которому никто не подходил.
        await self._store.reclaim_stalled(older_than=_STALLED_AFTER)

        # Просьбы по коду идут первыми: их ждёт живой человек в чате, а свой
        # проект подождёт следующего тика — он никого не держит.
        if await self._take_swe_task():
            return

        task = await self._store.next_pending()
        if task is None:
            # Довести до конца начатое важнее, чем затеять новое: человек, у
            # которого пять брошенных проектов и шестой начатый, ничего не
            # доводит — и выглядит это именно так.
            task = await self._revive_abandoned()
        if task is None:
            # Собственная затея тоже начинается с обращения к модели, поэтому
            # занятость поднимается до неё, а не только на сборке.
            self._is_coding = True
            try:
                task = await self._maybe_start_own_project()
            finally:
                self._is_coding = False
        if task is None:
            # Работы нет — самое время перечитать что-нибудь своё. Именно в
            # этом порядке: новый проект и чужая просьба важнее ревизии
            # старого, а ревизия — не «занятие на безрыбье», а то, чем автор
            # и занимается между проектами.
            await self._maybe_review_old_work()
            return

        self._is_coding = True
        try:
            await self._process(task)
        finally:
            self._is_coding = False

    async def _take_swe_task(self) -> bool:
        """
        Разбирает очередь просьб по коду. True — задача взята и обработана.

        Отдельная очередь и отдельный конвейер: у собственного проекта нет
        исходного кода, а у чужой правки нет замысла — общего между ними
        только то, что и там и там работает она.
        """
        if self._swe is None:
            return False
        task = await self._store.next_pending(kind=DevTaskKind.SWE)
        if task is None:
            return False

        self._is_coding = True
        try:
            await self._process_swe(task)
        finally:
            self._is_coding = False
        return True

    async def _process_swe(self, task: DevTask) -> None:
        """Один проход по чужому репозиторию — от просьбы до ветки."""
        assert self._swe is not None  # проверено вызывающей стороной

        if task.attempts >= _MAX_ATTEMPTS:
            await self._fail(task, f"не пережила {task.attempts} заходов конвейера", retriable=False)
            return
        task = await self._store.update(task, attempts=task.attempts + 1, status=DevTaskStatus.CODING)
        await self._reporter.remember_start(task)

        outcome = await self._swe.work_on(
            SweRequest(
                source=task.source,
                instruction=task.idea,
                session_id=f"task-{task.id}",
                chat_id=task.chat_id,
            )
        )
        if not outcome.ok:
            await self._fail(task, outcome.failure_reason or "не справилась с этой правкой")
            return

        task = await self._store.update(task, status=DevTaskStatus.DONE, branch=outcome.branch)
        logger.info(
            "dev_worker: правка по %s готова: ветка %s (%s)", task.source, outcome.branch, outcome.tier
        )
        await self._reporter.report_handover(task, outcome=outcome)

    async def _revive_abandoned(self) -> DevTask | None:
        """
        Второе дыхание для брошенного проекта.

        Возвращается он не с нуля: спека уже есть, написанные файлы лежат в
        задаче (`artifacts`), и заход начнётся ровно с того места, где в
        прошлый раз кончились силы или лимиты. Счётчик заходов обнуляется —
        это новый подход к снаряду, а не продолжение старого.
        """
        candidate = await self._store.abandoned_worth_another_try(
            not_touched_for=_REVIVE_AFTER, max_revivals=_MAX_REVIVALS
        )
        if candidate is None:
            return None

        logger.info(
            "dev_worker: возвращаюсь к брошенной задаче #%s (%s), заход %d",
            candidate.id, candidate.error, candidate.revivals + 1,
        )
        revived = await self._store.update(
            candidate,
            status=DevTaskStatus.PENDING,
            attempts=0,
            revivals=candidate.revivals + 1,
        )
        await self._reporter.remember_revision(
            revived,
            f"Вернулась к брошенному проекту и попробовала снова. В прошлый раз встало на: "
            f"{candidate.error or 'непонятно чём'}",
        )
        return revived

    async def _maybe_review_old_work(self) -> None:
        if self._maintainer is None:
            return
        self._is_coding = True
        try:
            await self._maintainer.maybe_review()
        finally:
            self._is_coding = False

    async def _maybe_start_own_project(self) -> DevTask | None:
        """
        Затеять что-то своё. Не каждый тик и не поверх уже идущей работы:
        человек, у которого одновременно пять начатых проектов, ничего не
        доводит до конца — и выглядит это так же.

        Замысел придумывается ДО того, как заводится задача. Раньше было
        наоборот: задача создавалась пустой, потом падала на проектировании, и
        каждая неудачная попытка навсегда оседала в базе строчкой «замысел без
        названия — не вышло». За сутки таких строчек набиралось больше, чем
        настоящих проектов, а полезного в них нет вообще: ни идеи, ни кода, ни
        причины возвращаться. Не придумалось — просто не придумалось, следов
        оставаться не должно.
        """
        if self._self_initiated_probability <= 0.0:
            return None
        if await self._store.active():
            return None
        if random.random() > self._self_initiated_probability:
            return None

        spec, reason = await self._engine.design(
            "", context=await self._render_context(), built=await self._built_specs()
        )
        if spec is None:
            logger.info("dev_worker: своя затея не сложилась (%s) — задачу не завожу", reason)
            return None

        task = await self._store.create(spec.title, chat_id=self._owner_chat_id, is_collab=False)
        task = await self._store.update(task, spec=spec)
        # Замысел собственного проекта — такой же повод для реплики, как и
        # заказанного: «придумала себе штуку и сажусь писать» человек говорит
        # в начале работы, а не только когда всё готово. И такой же повод для
        # записи в память: между замыслом и результатом часы, и всё это время
        # «чем ты занята?» — вопрос без ответа.
        await self._reporter.remember_start(task)
        await self._reporter.report_progress(task, _design_note(spec))
        return task

    async def _process(self, task: DevTask) -> None:
        if task.attempts >= _MAX_ATTEMPTS:
            # Сюда попадает задача, которая не доживает даже до отказа: процесс
            # падает на ней раз за разом, reclaim_stalled возвращает её в
            # очередь, и так по кругу. Счётчик заходов закрывает и этот случай.
            await self._fail(task, f"не пережила {task.attempts} заходов конвейера", retriable=False)
            return
        task = await self._store.update(task, attempts=task.attempts + 1)

        spec = task.spec
        if spec is None:
            task = await self._store.update(task, status=DevTaskStatus.SPECCING)
            spec, reason = await self._engine.design(
                task.idea, context=await self._render_context(), built=await self._built_specs()
            )
            if spec is None:
                # Причина — дословно от движка: под общим «не придумалось»
                # одинаково прятались битый JSON, обрыв по лимиту и настоящий
                # отказ от учебной идеи, а чинятся они по-разному.
                await self._fail(task, f"замысел не сложился: {reason}")
                return
            task = await self._store.update(task, spec=spec)
            await self._reporter.remember_start(task)
            await self._reporter.report_progress(task, _design_note(spec))

        task = await self._store.update(task, status=DevTaskStatus.CODING)
        build = await self._engine.build(spec, existing=task.artifacts)
        if not build.is_publishable:
            # То, что успело написаться, остаётся при задаче: следующий заход
            # не будет переписывать три готовых файла ради четвёртого — это и
            # лишние запросы к тому же лимиту, и новый шанс разойтись с тем,
            # что уже сходилось.
            task = await self._store.update(task, artifacts=_keep_written(spec, build))
            # Отказ кодера как таковой (снятая модель, отвергнутый ключ) к
            # следующему заходу не исправится — повторять его незачем. А вот
            # разошедшиеся между собой файлы со второй генерации часто
            # сходятся: это неудача захода, а не приговор замыслу.
            await self._fail(task, _broken_reason(build), retriable=not build.permanent)
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

    async def _fail(self, task: DevTask, reason: str, *, retriable: bool = True) -> None:
        """
        Провал одного захода. Временный — возвращает задачу в очередь, и в чат
        не уходит ничего: «не смогла, попробую позже» — это не новость, а шум.

        Разница между временным и окончательным здесь и есть разница между
        «проект не вышел» и «в тот час лежал провайдер». Без неё 429 на
        третьем файле из четырёх хоронил замысел навсегда — а это самый
        частый конец работы на бесплатных лимитах.
        """
        if retriable and task.attempts < _MAX_ATTEMPTS:
            logger.info(
                "dev_worker: задача #%s не задалась с %d-й попытки (%s) — вернусь к ней",
                task.id, task.attempts, reason,
            )
            await self._store.update(task, status=DevTaskStatus.PENDING, error=reason)
            return

        logger.warning("dev_worker: задача #%s провалилась: %s", task.id, reason)
        failed = await self._store.update(task, status=DevTaskStatus.FAILED, error=reason)
        await self._reporter.report_failure(failed, reason)

    async def _built_specs(self) -> list[ProjectSpec]:
        """
        Что она уже написала — материал для замысла, а не для отчёта.

        Без этого списка «придумай себе проект» на медленно меняющихся
        интересах раз за разом даёт одну и ту же утилиту: тот же разбор
        логов под новым именем (а иногда и под тем же — тогда пуш ещё и
        отклоняется, см. efi/dev/engine.py::_find_repeat).
        """
        return [task.spec for task in await self._store.finished_projects() if task.spec is not None]

    async def _render_context(self) -> str:
        if self._interests is None:
            return ""
        try:
            interests = await self._interests.current_interests()
        except Exception:
            logger.warning("dev_worker: не удалось получить интересы", exc_info=True)
            return ""
        return ", ".join(interests[:_MAX_CONTEXT_INTERESTS])


def _keep_written(spec: ProjectSpec, build: BuildResult) -> dict[str, str]:
    """
    Файлы спеки, которые в этом заходе получились. README и обвязка не
    сохраняются: они собираются по готовому коду и должны пересобираться.
    """
    wanted = {item.path for item in spec.files}
    return {item.path: item.content for item in build.files if item.path in wanted}


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
    # Причина отказа кодера важнее списка непрошедших файлов: «нет такой
    # модели» и «файл не парсится после трёх правок» чинятся совершенно
    # по-разному, а выглядели бы одинаково.
    if build.failure_reason:
        return build.failure_reason
    if build.broken_paths:
        return f"файлы {', '.join(build.broken_paths)} так и не заработали — код не собирается"
    return "кодер не написал ни одного файла"


__all__ = ["DevWorker", "InterestSource"]
