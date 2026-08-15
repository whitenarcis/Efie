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
        await self._remember(task, url=url)
        if task.chat_id is None:
            return
        await self._put(task, _render_release_reason(task, url=url, build=build), priority=_RELEASE_PRIORITY)

    async def report_failure(self, task: DevTask, reason: str) -> None:
        """Не получилось. Только для совместных задач — см. докстринг модуля."""
        if task.chat_id is None or not task.is_collab:
            logger.info("dev_reporter: задача #%s провалилась (%s), рассказывать некому", task.id, reason)
            return
        await self._put(task, _render_failure_reason(task, reason), priority=_PROGRESS_PRIORITY)

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

    async def _remember(self, task: DevTask, *, url: str) -> None:
        if self._social_memory is None:
            return
        title = task.spec.title if task.spec is not None else task.idea
        problem = task.spec.problem if task.spec is not None else ""
        text = f"Дописала и выложила {title}. {problem}".strip()
        try:
            await self._social_memory.record(
                SocialInteraction(
                    kind=SocialInteractionKind.DEV_RELEASE,
                    text=f"{text} {url}".strip(),
                    chat_id=task.chat_id,
                )
            )
        except Exception:
            # Память о релизе ценна, но уже отправленную ссылку она не
            # отменяет — сбой записи не должен ронять фоновый цикл.
            logger.warning("dev_reporter: не удалось записать релиз #%s в память", task.id, exc_info=True)


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


def _render_failure_reason(task: DevTask, reason: str) -> str:
    subject = task.spec.render_for_prompt() if task.spec is not None else task.idea.strip()
    return (
        f"Проект, который вы задумали вместе ({subject}), у тебя не вышел. Что именно сломалось: {reason}\n"
        "Скажи об этом честно и коротко, своими словами — человек ждал результата и имеет право знать. "
        "Без самобичевания и без обещаний «обязательно доделаю завтра»: если собираешься вернуться к "
        "этому, так и скажи, а если нет — не обещай."
    )


__all__ = ["DevReporter"]
