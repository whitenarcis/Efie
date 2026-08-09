"""
efi/behavior/scheduler.py

Фоновые планировщики: ночная консолидация памяти, утреннее пробуждение и
прочие задачи, привязанные к времени суток, а не к событиям в чатах. Каждая
задача триггерит Notification (обычно NIGHTLY_TASK) в NotificationManager —
вся дальнейшая обработка идёт через тот же Worker/tool-calling путь, что и
обычные сообщения (тот самый принцип: все проактивные выводы проходят через
полную модель личности, а не через отдельный облегчённый путь).

Лёгкий self-contained планировщик без внешних зависимостей — APScheduler и
подобные избыточны для нескольких ежедневных задач: каждое задание описано
временем суток (локальное время сервера) и асинхронно спит до следующего
срабатывания.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, time, timedelta

from efi.notifications.manager import NotificationManager
from efi.notifications.schemas import Notification, NotificationType

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class ScheduledJob:
    """Одна задача планировщика: имя, время суток срабатывания и текст уведомления."""

    name: str
    trigger_at: time
    notification_message: str
    notification_type: NotificationType = NotificationType.NIGHTLY_TASK
    priority: int = 8


class Scheduler:
    """
    Держит список ScheduledJob и крутит по одной asyncio-задаче на каждую:
    "спать до следующего срабатывания -> положить Notification -> спать до
    следующего срабатывания (уже завтра)". Останавливается по отмене задачи
    (CancelledError) — см. efi/app.py graceful shutdown.
    """

    def __init__(self, manager: NotificationManager, jobs: list[ScheduledJob]) -> None:
        self._manager = manager
        self._jobs = jobs

    async def run(self) -> None:
        """Запускает все задания параллельно; завершается, когда завершены (отменены) все дочерние задачи."""
        tasks = [asyncio.create_task(self._run_job(job), name=f"scheduler:{job.name}") for job in self._jobs]
        try:
            await asyncio.gather(*tasks)
        except asyncio.CancelledError:
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def _run_job(self, job: ScheduledJob) -> None:
        logger.info("scheduler: job %r scheduled daily at %s", job.name, job.trigger_at)
        try:
            while True:
                delay = seconds_until_next(job.trigger_at)
                await asyncio.sleep(delay)
                await self._fire(job)
        except asyncio.CancelledError:
            logger.info("scheduler: job %r stopped", job.name)
            raise

    async def _fire(self, job: ScheduledJob) -> None:
        notification = Notification(
            type=job.notification_type,
            priority=job.priority,
            chat_id=None,
            message=job.notification_message,
            payload={"job_name": job.name},
        )
        await self._manager.put(notification)
        logger.info("scheduler: fired job %r", job.name)


def seconds_until_next(trigger_at: time) -> float:
    """
    Секунд до следующего наступления времени суток `trigger_at`. Публичная
    (переиспользуется за пределами Scheduler — см. efi/app.py, отдельный цикл
    программной консолидации памяти на своём daily-расписании).
    """
    now = datetime.now()
    next_run = datetime.combine(now.date(), trigger_at)
    if next_run <= now:
        next_run += timedelta(days=1)
    return (next_run - now).total_seconds()


__all__ = ["ScheduledJob", "Scheduler", "seconds_until_next"]
