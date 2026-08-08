"""
efi/behavior/silence_monitor.py

Два родственных проактивных механизма, объединённых в одном модуле, как и в
текущей реализации Эфи:
    - SILENCE_PING: реакция на длительное затишье в чате (аналог
      silence_monitor_lifecycle) — если ни от пользователя, ни от Эфи давно
      не было сообщений, возможно, стоит написать первой.
    - FOLLOW_UP: отложенное возвращение к теме, которую Эфи сама решила
      отложить ("вернусь к этому позже") — аналог resume-callback.

Оба варианта в конечном счёте просто кладут Notification в
NotificationManager; что конкретно будет сказано — решает личность внутри
Worker'а, не этот модуль.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from efi.behavior.quiet_hours import is_quiet_hours
from efi.config.schema import QuietHoursSettings
from efi.notifications.manager import NotificationManager
from efi.notifications.schemas import Notification, NotificationType

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _PendingFollowUp:
    topic: str
    resume_at: datetime


class SilenceMonitor:
    """
    Отслеживает последнюю активность (в любую сторону) по чату и очередь
    отложенных follow-up'ов.

    `record_activity()` должна вызываться на КАЖДОЕ сообщение — и входящее
    (efi.telegram.handlers.TelegramEventHandlers), и исходящее
    (efi.tools.telegram_actions.send_message.SendMessageTool) — иначе Эфи
    будет считать чат тихим сразу после того, как сама в нём написала.
    """

    def __init__(
        self,
        manager: NotificationManager,
        *,
        check_interval_seconds: float = 900.0,  # 15 минут
        silence_threshold: timedelta = timedelta(hours=6),
        quiet_hours: QuietHoursSettings | None = None,
    ) -> None:
        self._manager = manager
        self._check_interval_seconds = check_interval_seconds
        self._silence_threshold = silence_threshold
        self._quiet_hours = quiet_hours
        self._last_activity: dict[int, datetime] = {}
        self._last_silence_ping: dict[int, datetime] = {}
        self._pending_follow_ups: dict[int, list[_PendingFollowUp]] = {}

    def record_activity(self, chat_id: int) -> None:
        """Отмечает, что в чате только что что-то произошло (сообщение в любую сторону). Синхронный, дешёвый вызов."""
        self._last_activity[chat_id] = datetime.now(timezone.utc)

    def schedule_follow_up(self, chat_id: int, topic: str, resume_at: datetime) -> None:
        """
        Регистрирует тему, к которой Эфи должна вернуться в чате `chat_id` не
        раньше `resume_at`. Предназначено для вызова инструментом, которым
        модель сама помечает "вернусь к этому позже" (такой инструмент — из
        числа тех, что перечислены как "перенести остальное" в предыдущих
        шагах, ещё не реализован; сам механизм готов принять такие
        регистрации уже сейчас).
        """
        self._pending_follow_ups.setdefault(chat_id, []).append(_PendingFollowUp(topic=topic, resume_at=resume_at))

    async def run(self) -> None:
        """Основной цикл. Останавливается по отмене задачи (CancelledError) — см. efi/app.py graceful shutdown."""
        logger.info(
            "silence_monitor: started (interval=%.0fs, threshold=%s)",
            self._check_interval_seconds, self._silence_threshold,
        )
        try:
            while True:
                await asyncio.sleep(self._check_interval_seconds)
                await self._check_silence()
                await self._check_follow_ups()
        except asyncio.CancelledError:
            logger.info("silence_monitor: stopped")
            raise

    async def _check_silence(self) -> None:
        if (
            self._quiet_hours is not None
            and self._quiet_hours.enabled
            and is_quiet_hours(
                datetime.now(), start_hour=self._quiet_hours.start_hour, end_hour=self._quiet_hours.end_hour
            )
        ):
            # Не трогаем _last_silence_ping — намеренно: как только тихие часы
            # закончатся, следующий проход снова увидит то же затишье и
            # запингует нормально, а не будет молчать ещё один полный
            # silence_threshold из-за пропущенного окна.
            logger.debug("silence_monitor: skipping silence check — quiet hours")
            return

        now = datetime.now(timezone.utc)
        for chat_id, last_activity in list(self._last_activity.items()):
            if now - last_activity < self._silence_threshold:
                continue
            last_ping = self._last_silence_ping.get(chat_id)
            if last_ping is not None and now - last_ping < self._silence_threshold:
                continue  # уже пинговали по этому периоду тишины — не спамим повторно каждые check_interval_seconds
            await self._manager.put(
                Notification(
                    type=NotificationType.SILENCE_PING,
                    priority=7,
                    chat_id=chat_id,
                    message=f"В этом чате тихо уже больше {self._silence_threshold}. Возможно, стоит написать первой.",
                    payload={},
                )
            )
            self._last_silence_ping[chat_id] = now
            logger.info("silence_monitor: queued SILENCE_PING for chat_id=%s", chat_id)

    async def _check_follow_ups(self) -> None:
        now = datetime.now(timezone.utc)
        for chat_id, items in list(self._pending_follow_ups.items()):
            due = [item for item in items if item.resume_at <= now]
            if not due:
                continue
            self._pending_follow_ups[chat_id] = [item for item in items if item.resume_at > now]
            for item in due:
                await self._manager.put(
                    Notification(
                        type=NotificationType.FOLLOW_UP,
                        priority=4,
                        chat_id=chat_id,
                        message=f"Ты обещала себе вернуться к теме: {item.topic}",
                        payload={"topic": item.topic},
                    )
                )
                logger.info("silence_monitor: queued FOLLOW_UP for chat_id=%s (%s)", chat_id, item.topic)


__all__ = ["SilenceMonitor"]
