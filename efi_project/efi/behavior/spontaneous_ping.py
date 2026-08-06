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
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable

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
    ) -> None:
        self._manager = manager
        self._candidate_chats_provider = candidate_chats_provider
        self._check_interval_seconds = check_interval_seconds
        self._ping_probability = ping_probability

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
        try:
            candidates = await self._candidate_chats_provider()
        except Exception:
            logger.exception("spontaneous_ping: candidate_chats_provider failed")
            return

        for chat_id in candidates:
            if random.random() > self._ping_probability:
                continue
            notification = Notification(
                type=NotificationType.SPONTANEOUS_PING,
                priority=6,
                chat_id=chat_id,
                message="У тебя есть желание написать первой, без особого повода — просто чтобы напомнить о себе.",
                payload={},
            )
            await self._manager.put(notification)
            logger.info("spontaneous_ping: queued for chat_id=%s", chat_id)


__all__ = ["CandidateChatsProvider", "SpontaneousPingScheduler"]
