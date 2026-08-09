"""
efi/notifications/manager.py

NotificationManager — очередь событий с закреплением чатов за воркерами
(перенос идеи pinned workers из Kuni), реализованная поверх N экземпляров
asyncio.PriorityQueue — по одному на воркер.

Почему не одна общая очередь с динамическим "захватом" пина воркером (как у
референса, где nextNotification сканирует накопленные уведомления в поисках
подходящего по пину, попутно оставляя остальные в очереди): asyncio.PriorityQueue
не даёт заглянуть внутрь и выбрать конкретный элемент, минуя более
приоритетные — только .get(), возвращающий глобальный минимум. Реализовать
"сканирование в поисках своего пина" пришлось бы через приватное состояние
очереди (`_queue`), что и хрупко, и не идиоматично для asyncio.

Вместо этого маршрутизация детерминирована и решается один раз на этапе
put(): routing_key события хэшируется в номер воркера. Гарантия та же самая,
что и у референса (события одного чата обрабатываются строго
последовательно одним и тем же воркером), но без сканирования чужих
уведомлений и без гонок за "захват" пина между несколькими воркерами.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import logging
from dataclasses import dataclass, field

from efi.notifications.schemas import Notification

logger = logging.getLogger(__name__)


@dataclass(order=True)
class _QueueItem:
    """
    Обёртка для PriorityQueue: сравнение по (priority, sequence).
    `notification` в сравнении не участвует (`compare=False`) — иначе
    asyncio.PriorityQueue попыталась бы сравнивать сами Notification при
    равенстве priority и sequence (что, впрочем, невозможно — sequence
    строго монотонный и уникальный, но это всё равно лишняя связанность).
    """

    priority: int
    sequence: int
    notification: Notification = field(compare=False)


class NotificationManager:
    """
    Очередь событий на N экземплярах asyncio.PriorityQueue (по одной на
    воркера) с детерминированной маршрутизацией по `Notification.routing_key`.

    Публичный интерфейс сознательно зеркалирует asyncio.Queue: `put()`,
    `get()`, `task_done()` — с той разницей, что `get()`/`task_done()`
    принимают `worker_index`, поскольку у каждого воркера — своя подочередь
    со своим собственным порядком приоритетов среди закреплённых за ним чатов.
    """

    def __init__(self, worker_count: int) -> None:
        if worker_count < 1:
            raise ValueError("worker_count должен быть >= 1")
        self._worker_count = worker_count
        self._queues: list[asyncio.PriorityQueue[_QueueItem]] = [asyncio.PriorityQueue() for _ in range(worker_count)]
        self._sequence_counter = itertools.count()

    @property
    def worker_count(self) -> int:
        return self._worker_count

    def worker_index_for(self, routing_key: str) -> int:
        """
        Детерминированно определяет номер воркера для данного routing_key.

        Используется blake2b, а не встроенный hash(): начиная с Python 3.3
        hash() для строк рандомизирован между запусками процесса
        (PYTHONHASHSEED) ради защиты от DoS через коллизии — это ломает
        воспроизводимость маршрутизации внутри одного и того же запуска
        приложения не было бы, но исключает её предсказуемость между
        запусками (что важно, например, для тестов и отладки).
        """
        digest = hashlib.blake2b(routing_key.encode("utf-8"), digest_size=8).digest()
        return int.from_bytes(digest, "big") % self._worker_count

    async def put(self, notification: Notification) -> None:
        """Кладёт уведомление в подочередь воркера, детерминированно закреплённого за его routing_key."""
        index = self.worker_index_for(notification.routing_key)
        item = _QueueItem(
            priority=notification.priority,
            sequence=next(self._sequence_counter),
            notification=notification,
        )
        await self._queues[index].put(item)
        logger.debug(
            "notifications: queued %s id=%s (priority=%d, chat_id=%s) -> worker %d",
            notification.type.value, notification.id, notification.priority, notification.chat_id, index,
        )

    async def get(self, worker_index: int) -> Notification:
        """
        Забирает следующее (по приоритету, затем по времени постановки в
        очередь) уведомление подочереди конкретного воркера. Блокируется,
        пока подочередь пуста.
        """
        self._check_worker_index(worker_index)
        item = await self._queues[worker_index].get()
        return item.notification

    def task_done(self, worker_index: int) -> None:
        """Сигнализирует о завершении обработки элемента, ранее полученного через get(worker_index)."""
        self._check_worker_index(worker_index)
        self._queues[worker_index].task_done()

    async def join(self) -> None:
        """Ждёт, пока все подочереди опустеют и обработка всех элементов будет подтверждена task_done()."""
        await asyncio.gather(*(queue.join() for queue in self._queues))

    def qsize(self, worker_index: int | None = None) -> int:
        """Размер подочереди конкретного воркера, либо суммарный размер всех подочередей (для дашборда/метрик)."""
        if worker_index is not None:
            self._check_worker_index(worker_index)
            return self._queues[worker_index].qsize()
        return sum(queue.qsize() for queue in self._queues)

    def _check_worker_index(self, worker_index: int) -> None:
        if not (0 <= worker_index < self._worker_count):
            raise ValueError(f"worker_index {worker_index} вне диапазона [0, {self._worker_count})")


__all__ = ["NotificationManager"]
