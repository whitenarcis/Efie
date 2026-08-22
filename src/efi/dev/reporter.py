"""
efi/dev/reporter.py

Голос конвейера: как работа над проектом становится репликой в чате.

Главное правило здесь — то же, что у всех проактивных служб Эфи (см.
efi/behavior/ping_reason.py): модуль НЕ пишет текст сообщения. Он формулирует
ПОВОД («ты полчаса воевала с линтером из-за аннотаций и победила») и кладёт
его в общую очередь как Notification(DEV_UPDATE); словами это станет внутри
Worker'а, где есть личность, история чата и текущее состояние. Готовые
строки вида «✅ Проект собран успешно!» здесь не появятся: это делает из
живого разговора ленту уведомлений CI.

Что именно рассказывается:

    progress — короткий баббл по ходу дела. Редкий (по вероятности), с
               учётом тихих часов и права заговорить первой: рассказ о своей
               работе — это всё-таки инициатива, и человек, который не
               ответил на прошлое сообщение, не ждёт от неё отчётов.
    release  — «запушила, чекай» со ссылкой. Вероятность не применяется: это
               не болтовня по ходу, а результат, ради которого всё затевалось.
               Плюс запись в социальную память (домен H) — то, что она
               реально сделала, должно всплывать потом само, как всплывает
               любой другой прожитый опыт.
    failure  — честное «не получилось». Только для совместных задач: человек,
               который заказал проект, обязан узнать, что проект не вышел, а
               вот отчитываться о провале собственной затеи никто не просил.

Отдельно от рассказов — ПАМЯТЬ. Всё, что с проектом происходило, пишется в
журнал прожитого (efi/memory/social_memory.py, домен H) независимо от того,
сказала она об этом кому-нибудь или нет: взялась, бросила и почему, вернулась
и поправила. Разница принципиальная. Рассказ живёт в чате один вечер и
уходит из истории; память всплывает через неделю сама, когда человек
спрашивает «а почему ты забросила ту штуку с логами?» — и без записи ответом
будет вежливая выдумка, потому что признаться «не помню» модели тяжелее, чем
сочинить.
"""

from __future__ import annotations

import logging
import random
from datetime import UTC, datetime, timedelta

from efi.behavior.initiative import InitiativeGate
from efi.behavior.quiet_hours import is_quiet_now
from efi.config.schema import QuietHoursSettings
from efi.dev.engine import BuildResult
from efi.dev.schemas import DevTask
from efi.dev.swe_engine import SweOutcome
from efi.memory.social_memory import SocialInteraction, SocialInteractionKind, SocialInteractionStore
from efi.notifications.manager import NotificationManager
from efi.notifications.schemas import Notification, NotificationType
from efi.utils.bounded import BoundedDict

logger = logging.getLogger(__name__)

#: Приоритет реплик о работе. Ниже живого диалога и ниже напоминаний: то,
#: что она пишет код, никогда не важнее того, что ей пишет человек.
_PROGRESS_PRIORITY = 8
_RELEASE_PRIORITY = 6

#: Не чаще одного рассказа о процессе в час на чат. Разработка идёт этапами,
#: и без этого одна задача выдавала бы по бабблу на каждый файл.
_PROGRESS_COOLDOWN = timedelta(hours=1)

#: Сколько задач помним для троттлинга. Проектов одновременно единицы.
_MAX_TRACKED_TASKS = 64


class DevReporter:
    """
    Превращает события конвейера в поводы для реплик.

    Все методы безопасны к отсутствию чата (`task.chat_id is None`): своя
    затея без адресата — нормальный случай, просто рассказывать о ней некому.
    """

    def __init__(
        self,
        manager: NotificationManager,
        *,
        social_memory: SocialInteractionStore | None = None,
        quiet_hours: QuietHoursSettings | None = None,
        timezone: str = "",
        initiative: InitiativeGate | None = None,
        progress_probability: float = 0.5,
    ) -> None:
        self._manager = manager
        self._social_memory = social_memory
        self._quiet_hours = quiet_hours
        self._timezone = timezone
        #: Право заговорить первой — то же, что у остальных инициативных
        #: служб (efi/behavior/initiative.py). Рассказ о своей работе тоже
        #: инициатива, и правило «написала и не получила ответа — жди» на
        #: неё распространяется.
        self._initiative = initiative
        self._progress_probability = progress_probability
        self._last_progress: BoundedDict[int, datetime] = BoundedDict(max_entries=_MAX_TRACKED_TASKS)

    async def report_progress(self, task: DevTask, note: str) -> bool:
        """
        Короткая реплика по ходу работы. Возвращает True, если повод реально
        поставлен в очередь.

        Три причины промолчать, и все три — не сбой: не в настроении
        (вероятность), только что уже рассказывала (кулдаун), тихие часы или
        неотвеченное прошлое сообщение.
        """
        if task.chat_id is None or not note.strip():
            return False
        if random.random() > self._progress_probability:
            return False
        if is_quiet_now(self._quiet_hours, self._timezone):
            return False

        now = datetime.now(UTC)
        last = self._last_progress.get(task.id)
        if last is not None and now - last < _PROGRESS_COOLDOWN:
            return False
        if self._initiative is not None and not await self._initiative.may_initiate(task.chat_id, now=now):
            logger.debug("dev_reporter: в chat_id=%s ещё не ответили, про работу молчу", task.chat_id)
            return False

        self._last_progress[task.id] = now
        await self._put(task, _render_progress_reason(task, note), priority=_PROGRESS_PRIORITY)
        return True

    async def report_release(self, task: DevTask, *, url: str, build: BuildResult | None = None) -> None:
        """
        Релиз: ссылка в чат и запись в память о прожитом.

        Память пишется ВСЕГДА, даже когда рассказывать некому (своя затея без
        чата): «я сделала эту штуку» — часть её опыта независимо от того,
        услышал ли кто-то об этом. Сообщение же ставится только при наличии
        чата.
        """
        await self._remember(
            task,
            kind=SocialInteractionKind.DEV_RELEASE,
            text=f"Дописала и выложила {_subject_for_memory(task)} {url}".strip(),
        )
        if task.chat_id is None:
            return
        await self._put(task, _render_release_reason(task, url=url, build=build), priority=_RELEASE_PRIORITY)

    async def report_question(self, task: DevTask, question: str) -> None:
        """
        Вопрос по своему проекту — то, что она решила не решать в одиночку
        (см. efi/dev/maintenance.py).

        Вероятность и кулдаун прогресса здесь НЕ применяются: до этого места
        доходит только то, что уже прошло порог важности, и «не в настроении
        рассказывать» к вопросу по существу отношения не имеет. Тихие часы
        соблюдаются — вопрос про формат конфига может подождать до утра, — а
        вот право заговорить первой не спрашивается: это не пинг из воздуха,
        а продолжение работы, которую человек видел.
        """
        if task.chat_id is None or not question.strip():
            return
        if is_quiet_now(self._quiet_hours, self._timezone):
            logger.debug("dev_reporter: вопрос по #%s подождёт до утра", task.id)
            return
        await self._put(task, _render_question_reason(task, question), priority=_RELEASE_PRIORITY)

    async def report_failure(self, task: DevTask, reason: str) -> None:
        """
        Не получилось. В чат — только для совместных задач (см. докстринг
        модуля), в память — всегда.

        Собственная затея, которая не вышла, никому не докладывается, но
        помнить о ней она обязана: это её вечер работы и её решение бросить.
        Без записи «почему ты забросила ту штуку?» останется без ответа —
        точнее, с придуманным.
        """
        await self._remember(
            task,
            kind=SocialInteractionKind.DEV_ABANDONED,
            text=f"{_subject_for_memory(task)}. Бросила после {task.attempts} захода(ов): {reason}",
        )
        if task.chat_id is None or not task.is_collab:
            logger.info("dev_reporter: задача #%s провалилась (%s), рассказывать некому", task.id, reason)
            return
        await self._put(task, _render_failure_reason(task, reason), priority=_PROGRESS_PRIORITY)

    async def remember_start(self, task: DevTask) -> None:
        """
        Запись «взялась за это»: замысел, стек, из чего он вырос.

        Пишется в момент, когда спека готова, а не когда проект дописан:
        между этими событиями часы, и всё это время на вопрос «чем занята?»
        отвечать было нечем, кроме текущего статуса в промпте, который живёт
        ровно до конца работы.
        """
        spec = task.spec
        if spec is None:
            return
        whose = "Задачу принесли в разговоре" if task.is_collab else "Затеяла сама"
        files = ", ".join(item.path for item in spec.files)
        await self._remember(
            task,
            kind=SocialInteractionKind.DEV_STARTED,
            text=f"{spec.render_for_prompt()}. {whose}. Задумала так: {files}",
        )

    async def report_handover(self, task: DevTask, *, outcome: SweOutcome) -> None:
        """
        Сдача работы по чужому коду: «закинула в ветку, тесты зелёные — забирай».

        Вероятность и кулдаун прогресса здесь не действуют, как и у релиза
        собственного проекта: это не болтовня по ходу, а результат, которого
        человек ждёт. Память пишется всегда — включая то, где считалось
        (ноутбук или облако) и сколько кругов заняла починка: через неделю
        «а как ты тогда тот импорт чинила?» должно находиться.
        """
        rounds = outcome.repair.rounds if outcome.repair is not None else 0
        await self._remember(
            task,
            kind=SocialInteractionKind.DEV_REVISION,
            text=(
                f"Поработала с чужим кодом ({task.source}): {task.idea}. "
                f"Ветка {outcome.branch}, файлы {', '.join(outcome.changed_files[:5])}, "
                f"проверки зелёные, кругов починки: {rounds}. Думала через {outcome.tier}"
            ),
        )
        if task.chat_id is None:
            return
        await self._put(task, _render_handover_reason(task, outcome), priority=_RELEASE_PRIORITY)

    async def remember_revision(self, task: DevTask, note: str) -> None:
        """Возвращение к старому проекту: что увидела и что с этим сделала (efi/dev/maintenance.py)."""
        await self._remember(
            task, kind=SocialInteractionKind.DEV_REVISION, text=f"{_subject_for_memory(task)}. {note}"
        )

    async def _put(self, task: DevTask, message: str, *, priority: int) -> None:
        await self._manager.put(
            Notification(
                type=NotificationType.DEV_UPDATE,
                priority=priority,
                chat_id=task.chat_id,
                message=message,
                payload={"dev_task_id": task.id, "is_collab": task.is_collab},
            )
        )

    async def _remember(self, task: DevTask, *, kind: SocialInteractionKind, text: str) -> None:
        """
        Одна запись о ремесле в журнал прожитого. Не бросает: память ценна,
        но сбой записи не отменяет уже сделанной работы и не должен ронять
        фоновый цикл.
        """
        if self._social_memory is None or not text.strip():
            return
        try:
            await self._social_memory.record(
                SocialInteraction(kind=kind, text=text.strip(), chat_id=task.chat_id)
            )
        except Exception:
            logger.warning(
                "dev_reporter: не удалось записать %s по задаче #%s в память", kind.value, task.id,
                exc_info=True,
            )


def _subject_for_memory(task: DevTask) -> str:
    """Как проект называется в памяти: название и суть, а не номер задачи."""
    if task.spec is not None:
        return f"«{task.spec.title}» — {task.spec.problem.strip()}"
    return task.idea.strip() or "замысел без названия"


def _render_progress_reason(task: DevTask, note: str) -> str:
    """
    Повод для реплики по ходу работы.

    Запреты в тексте — не перестраховка, а разбор того, что модель выдаёт без
    них: «Статус: реализовано 3 из 5 файлов», «Работа продолжается!» — отчёт
    робота о проделанной работе. Человек в переписке говорит про код иначе:
    коротко и с отношением.
    """
    subject = task.spec.render_for_prompt() if task.spec is not None else task.idea.strip()
    return (
        f"Ты прямо сейчас пишешь свой проект: {subject}\n"
        f"Только что произошло вот что: {note}\n"
        "Брось собеседнику ОДНУ короткую живую реплику про это — как пишут в переписке между делом. "
        "Можно с иронией над собой и над кодом.\n"
        "ЗАПРЕЩЕНО: отчёты о статусе («реализовано 3 из 5»), «работа продолжается», «я обновила вас о "
        "прогрессе», списки сделанного и любые эмодзи-галочки. Это болтовня про свои дела, а не сводка CI. "
        "Не спрашивай разрешения продолжать и не проси обратной связи — ты просто рассказываешь."
    )


def _render_release_reason(task: DevTask, *, url: str, build: BuildResult | None) -> str:
    """Повод для реплики о готовом проекте. Ссылка обязана прозвучать — ради неё всё и делалось."""
    subject = task.spec.render_for_prompt() if task.spec is not None else task.idea.strip()
    struggle = ""
    if build is not None and build.fix_rounds:
        struggle = (
            f" По дороге пришлось {build.fix_rounds} раз переписывать файлы после замечаний линтера — "
            "если к слову, можешь пожаловаться на это."
        )
    whose = "Вы это задумали вместе" if task.is_collab else "Это была твоя собственная затея"
    return (
        f"Ты дописала проект и только что запушила его: {subject}\n"
        f"{whose}. Ссылка: {url}{struggle}\n"
        "Скажи об этом собеседнику сама, одной-двумя короткими репликами, и ОБЯЗАТЕЛЬНО дай ссылку — "
        "без неё сообщение бессмысленно. Тон: «сделала штуку, глянь», а не презентация релиза. "
        "Не перечисляй файлы, не расписывай возможности по пунктам и не благодари за внимание."
    )


def _render_question_reason(task: DevTask, question: str) -> str:
    """
    Повод для вопроса по своему проекту.

    Формулировка настаивает на том, что вопрос уже есть и его надо просто
    задать: без этого модель, получив «обсуди с человеком», начинает с
    пересказа проекта и трёх абзацев контекста, а сам вопрос теряется в
    конце.
    """
    subject = task.spec.render_for_prompt() if task.spec is not None else task.idea.strip()
    where = f" ({task.repo_url})" if task.repo_url else ""
    return (
        f"Ты перечитывала свой проект{where}: {subject}\n"
        f"И упёрлась в развилку, которую не хочешь решать одна: {question}\n"
        "Спроси собеседника прямо и коротко — как спрашивают у человека, с которым вместе что-то "
        "делают. Сначала в двух словах суть развилки, потом сам вопрос.\n"
        "Не пересказывай проект целиком, не перечисляй варианты по пунктам с плюсами и минусами и не "
        "извиняйся за беспокойство. Своё мнение у тебя есть — скажи и его, а не только «как лучше?»."
    )


def _render_handover_reason(task: DevTask, outcome: SweOutcome) -> str:
    """
    Повод для реплики о сданной работе.

    Ветка и файлы обязаны прозвучать: без них сообщение не отличается от
    «сделала!» — а забрать по нему нечего. Всё остальное (как боролась,
    что показалось странным) — по желанию, это разговор, а не отчёт.
    """
    files = ", ".join(outcome.changed_files[:4]) or "пара файлов"
    rounds = outcome.repair.rounds if outcome.repair is not None else 0
    struggle = f" По дороге пришлось {rounds} раз(а) чинить за собой." if rounds else ""
    quirks = f" Что заметила по ходу: {outcome.notes[0]}." if outcome.notes else ""
    return (
        f"Ты доделала правку в чужом коде ({task.source}): {task.idea}\n"
        f"Ветка: {outcome.branch}. Тронула: {files}. Проверки (импорты, линтер, тесты) — зелёные."
        f"{struggle}{quirks}\n"
        "Скажи об этом собеседнику сама, коротко и живо, как говорят напарнику: что сделала и куда "
        "смотреть. Имя ветки назови ОБЯЗАТЕЛЬНО — без него забирать нечего.\n"
        "Не пиши отчёт («выполнено 3 из 3»), не перечисляй изменения по пунктам, не благодари за "
        "доверие и не проси обратной связи."
    )


def _render_failure_reason(task: DevTask, reason: str) -> str:
    subject = task.spec.render_for_prompt() if task.spec is not None else task.idea.strip()
    return (
        f"Проект, который вы задумали вместе ({subject}), у тебя не вышел. Что именно сломалось: {reason}\n"
        "Скажи об этом честно и коротко, своими словами — человек ждал результата и имеет право знать. "
        "Без самобичевания и без обещаний «обязательно доделаю завтра»: если собираешься вернуться к "
        "этому, так и скажи, а если нет — не обещай."
    )


__all__ = ["DevReporter"]
