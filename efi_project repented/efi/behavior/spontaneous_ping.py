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
from datetime import datetime

from efi.behavior.quiet_hours import is_quiet_hours
from efi.config.schema import QuietHoursSettings
from efi.notifications.manager import NotificationManager
from efi.notifications.schemas import Notification, NotificationType

logger = logging.getLogger(__name__)

#: Функция, возвращающая список chat_id — кандидатов на спонтанный пинг прямо сейчас.
#: Конкретный источник (например, "приватные чаты, где Эфи писала хотя бы раз") —
#: забота вызывающей стороны (efi/app.py); этот модуль не хранит список чатов сам.
CandidateChatsProvider = Callable[[], Awaitable[list[int]]]

#: Функция, возвращающая текст инкубированной мысли (и СБРАСЫВАЮЩАЯ её после
#: возврата — см. BackgroundResearcher.consume_incubated_thought), либо None,
#: если фоновое исследование ничего не подготовило к этому моменту.
IncubatedThoughtProvider = Callable[[], Awaitable[str | None]]

_DEFAULT_PING_MESSAGE = "У тебя есть желание написать первой, без особого повода — просто чтобы напомнить о себе."


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
        incubated_thought_provider: IncubatedThoughtProvider | None = None,
        quiet_hours: QuietHoursSettings | None = None,
    ) -> None:
        self._manager = manager
        self._candidate_chats_provider = candidate_chats_provider
        self._check_interval_seconds = check_interval_seconds
        self._ping_probability = ping_probability
        self._incubated_thought_provider = incubated_thought_provider
        self._quiet_hours = quiet_hours

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
        if (
            self._quiet_hours is not None
            and self._quiet_hours.enabled
            and is_quiet_hours(
                datetime.now(), start_hour=self._quiet_hours.start_hour, end_hour=self._quiet_hours.end_hour
            )
        ):
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
        incubated_thought = await self._consume_incubated_thought()

        for chat_id in candidates:
            if random.random() > self._ping_probability:
                continue

            if incubated_thought is not None:
                message = _render_incubated_ping(incubated_thought)
                incubated_thought = None
            else:
                message = _DEFAULT_PING_MESSAGE

            notification = Notification(
                type=NotificationType.SPONTANEOUS_PING,
                priority=6,
                chat_id=chat_id,
                message=message,
                payload={},
            )
            await self._manager.put(notification)
            logger.info(
                "spontaneous_ping: queued for chat_id=%s (incubated_thought=%s)",
                chat_id, message != _DEFAULT_PING_MESSAGE,
            )

    async def _consume_incubated_thought(self) -> str | None:
        if self._incubated_thought_provider is None:
            return None
        try:
            return await self._incubated_thought_provider()
        except Exception:
            logger.exception("spontaneous_ping: incubated_thought_provider failed")
            return None


def _render_incubated_ping(thought: str) -> str:
    return (
        f"Пока было тихо, тебе самой пришла в голову мысль (сама погуглила и подумала): {thought} "
        "Поделись этим с собеседником как своей спонтанной идеей — а не дежурным 'как дела'."
    )


__all__ = ["CandidateChatsProvider", "IncubatedThoughtProvider", "SpontaneousPingScheduler"]
