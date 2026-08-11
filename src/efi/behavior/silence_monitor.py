"""
efi/behavior/silence_monitor.py

SILENCE_PING: реакция на длительное затишье в чате (аналог
silence_monitor_lifecycle) — если ни от пользователя, ни от Эфи давно не
было сообщений, возможно, стоит написать первой. Модуль просто кладёт
Notification в NotificationManager; что конкретно будет сказано — решает
личность внутри Worker'а.

Здесь же раньше жила вторая, in-memory очередь отложенных FOLLOW_UP'ов
(`schedule_follow_up`). Она удалена: её никто никогда не вызывал, а
пережить перезапуск она не могла по построению — обещание «напиши через 10
минут» исчезало вместе с процессом. Отложенные возвращения к теме теперь
целиком в efi/behavior/reminders.py, поверх SQLite.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta

from efi.behavior.quiet_hours import is_quiet_hours
from efi.config.schema import QuietHoursSettings
from efi.notifications.manager import NotificationManager
from efi.notifications.schemas import Notification, NotificationType

logger = logging.getLogger(__name__)


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

    def record_activity(self, chat_id: int) -> None:
        """Отмечает, что в чате только что что-то произошло (сообщение в любую сторону). Синхронный, дешёвый вызов."""
        self._last_activity[chat_id] = datetime.now(UTC)

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

        now = datetime.now(UTC)
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


__all__ = ["SilenceMonitor"]
