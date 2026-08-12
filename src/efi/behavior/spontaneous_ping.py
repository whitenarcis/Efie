"""
efi/behavior/spontaneous_ping.py

Спонтанные пинги — Эфи сама решает написать первой. Перенос
try_spontaneous_ping из текущей реализации: периодическая проверка списка
чатов-кандидатов с вероятностным решением "написать / не написать сейчас".

Сама формулировка того, ЧТО написать, здесь не решается — это дело личности
(EfiSystemPromptBuilder + LLM внутри Worker'а). Этот модуль лишь порождает
факт "пора бы написать" как Notification(SPONTANEOUS_PING) и передаёт его в
общую очередь — дальше событие обрабатывается точно так же, как обычное
сообщение пользователя.

Единственное исключение — текст самого события: если к моменту пинга уже
есть инкубированная мысль от efi.behavior.researcher.BackgroundResearcher
(Эфи сама что-то нагуглила и обдумала в затишье), именно она становится
поводом для пинга вместо дежурного "как дела" — см. `incubated_thought_provider`.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable

from efi.behavior.initiative import InitiativeGate
from efi.behavior.ping_reason import PingReasonBuilder, render_incubated_reason
from efi.behavior.quiet_hours import is_quiet_now
from efi.config.schema import QuietHoursSettings
from efi.notifications.manager import NotificationManager
from efi.notifications.schemas import Notification, NotificationType

logger = logging.getLogger(__name__)

#: Функция, возвращающая список chat_id — кандидатов на спонтанный пинг прямо сейчас.
#: Конкретный источник (например, "приватные чаты, где Эфи писала хотя бы раз") —
#: забота вызывающей стороны (efi/app.py); этот модуль не хранит список чатов сам.
CandidateChatsProvider = Callable[[], Awaitable[list[int]]]

class SpontaneousPingScheduler:
    """
    Раз в `check_interval_seconds` спрашивает `candidate_chats_provider` за
    списком чатов-кандидатов и с вероятностью `ping_probability` — независимо
    для каждого кандидата — кладёт для него Notification(SPONTANEOUS_PING).
    """

    def __init__(
        self,
        manager: NotificationManager,
        candidate_chats_provider: CandidateChatsProvider,
        *,
        check_interval_seconds: float = 1800.0,  # 30 минут
        ping_probability: float = 0.15,
        reasons: PingReasonBuilder | None = None,
        quiet_hours: QuietHoursSettings | None = None,
        timezone: str = "",
        initiative: InitiativeGate | None = None,
    ) -> None:
        self._manager = manager
        self._candidate_chats_provider = candidate_chats_provider
        self._check_interval_seconds = check_interval_seconds
        self._ping_probability = ping_probability
        #: Откуда берётся повод написать. Без него служба молчит всегда —
        #: и это правильный дефолт, см. efi/behavior/ping_reason.py.
        self._reasons = reasons if reasons is not None else PingReasonBuilder()
        self._quiet_hours = quiet_hours
        self._timezone = timezone
        #: Право заговорить первой — общее на все инициативные службы, см.
        #: efi/behavior/initiative.py.
        self._initiative = initiative

    async def run(self) -> None:
        """Основной цикл. Останавливается по отмене задачи (CancelledError) — см. efi/app.py graceful shutdown."""
        logger.info(
            "spontaneous_ping: started (interval=%.0fs, p=%.2f)",
            self._check_interval_seconds, self._ping_probability,
        )
        try:
            while True:
                await asyncio.sleep(self._check_interval_seconds)
                await self._maybe_ping_candidates()
        except asyncio.CancelledError:
            logger.info("spontaneous_ping: stopped")
            raise

    async def _maybe_ping_candidates(self) -> None:
        if is_quiet_now(self._quiet_hours, self._timezone):
            logger.debug("spontaneous_ping: skipping check — quiet hours")
            return

        try:
            candidates = await self._candidate_chats_provider()
        except Exception:
            logger.exception("spontaneous_ping: candidate_chats_provider failed")
            return

        # Мысль (если есть) достаётся ОДИН раз за проход и уходит первому же
        # запинговавшему кандидату — она одноразовая по своей природе (см.
        # BackgroundResearcher.consume_incubated_thought), делить одну и ту же
        # мысль между несколькими чатами в одном проходе не имеет смысла.
        incubated_thought = await self._reasons.consume_incubated_thought()

        for chat_id in candidates:
            if random.random() > self._ping_probability:
                continue
            if self._initiative is not None and not await self._initiative.may_initiate(chat_id):
                # На прошлое сообщение так и не ответили. Второе «эй» не
                # увеличивает шанс ответа — оно только показывает, что
                # пишущий не заметил молчания.
                logger.debug("spontaneous_ping: chat_id=%s ещё не ответил, пропускаю", chat_id)
                continue

            if incubated_thought is not None:
                message = render_incubated_reason(incubated_thought)
                incubated_thought = None
            else:
                message = await self._reasons.reason_for(chat_id) or ""

            if not message:
                # Повода нет — и сообщения нет. Раньше здесь стояло «просто
                # напомнить о себе», и из этого получалось единственное
                # возможное «эй, ты там живой?». Молчание читается как
                # «занята своими делами», пустой пинг — как навязчивость.
                logger.debug("spontaneous_ping: для chat_id=%s нет повода, молчу", chat_id)
                continue

            notification = Notification(
                type=NotificationType.SPONTANEOUS_PING,
                priority=6,
                chat_id=chat_id,
                message=message,
                payload={},
            )
            await self._manager.put(notification)
            logger.info("spontaneous_ping: queued for chat_id=%s", chat_id)



__all__ = ["CandidateChatsProvider", "SpontaneousPingScheduler"]
