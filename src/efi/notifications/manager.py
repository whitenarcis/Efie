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

Здесь же живёт отложенный повтор (`retry_later`). Проактивное событие — это
намерение Эфи что-то сказать, а не запрос, который можно молча потерять:
если LLM не ответил (таймаут на бесплатном тире — обычное дело), намерение
должно пережить неудачу и повториться. Таймеры повторов живут в самом
менеджере, а не в воркере: воркер обрабатывает уведомления по одному и не
может ждать минуту, не блокируя свою подочередь, а незарегистрированная
`create_task` потерялась бы при остановке приложения.
"""

from __future__ import annotations

import asyncio
import hashlib
import itertools
import logging
from dataclasses import dataclass, field

from efi.notifications.schemas import Notification

logger = logging.getLogger(__name__)

#: Сколько всего раз пытаться доставить одно проактивное уведомление, считая
#: первую попытку. Три — это «пережить единичный таймаут провайдера и ещё
#: один», но не «долбиться час»: если модель недоступна десять минут подряд,
#: повод («напиши мне через 10 минут») уже протух сам по себе.
MAX_DELIVERY_ATTEMPTS = 3


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
        self._retry_timers: set[asyncio.Task[None]] = set()

    @property
    def worker_count(self) -> int:
        return self._worker_count

    @property
    def pending_retries(self) -> int:
        """Сколько повторов сейчас ждёт своего часа — для дашборда и тестов."""
        return len(self._retry_timers)

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

    def retry_later(self, notification: Notification, *, delay: float) -> bool:
        """
        Ставит уведомление в очередь заново через `delay` секунд, увеличив
        счётчик попыток. Возвращает False, если попытки исчерпаны и повтора
        не будет, — вызывающая сторона по этому решает, пора ли сдаваться
        вслух (пометить обещание, написать в лог как о потере).

        Синхронный по умыслу: вызывающий воркер не должен ждать ни секунды
        из `delay` — он обязан немедленно взять следующее уведомление.
        """
        if notification.attempt + 1 >= MAX_DELIVERY_ATTEMPTS:
            logger.warning(
                "notifications: %s id=%s исчерпало %d попыток, повтора не будет",
                notification.type.value, notification.id, MAX_DELIVERY_ATTEMPTS,
            )
            return False

        # Копия, а не мутация: исходное уведомление ещё живёт в обработчике,
        # который его уронил, и менять его под ним нехорошо.
        retry = notification.model_copy(update={"attempt": notification.attempt + 1})

        async def _sleep_and_put() -> None:
            try:
                await asyncio.sleep(delay)
                await self.put(retry)
                logger.info(
                    "notifications: повторная попытка %d/%d для %s id=%s (chat_id=%s)",
                    retry.attempt + 1, MAX_DELIVERY_ATTEMPTS, retry.type.value, retry.id, retry.chat_id,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                logger.exception("notifications: не удалось поставить повтор для id=%s", retry.id)

        task = asyncio.create_task(_sleep_and_put(), name=f"notification_retry:{retry.id}")
        self._retry_timers.add(task)
        task.add_done_callback(self._retry_timers.discard)
        logger.info(
            "notifications: %s id=%s не доставлено, повтор через %.0fs (попытка %d/%d)",
            notification.type.value, notification.id, delay, retry.attempt + 1, MAX_DELIVERY_ATTEMPTS,
        )
        return True

    async def cancel_retries(self) -> None:
        """
        Снимает все ждущие повторы. Вызывается при остановке приложения:
        без этого несработавшие таймеры остались бы висеть как незавершённые
        задачи и мешали бы чистому выключению.
        """
        timers = list(self._retry_timers)
        for task in timers:
            task.cancel()
        if timers:
            await asyncio.gather(*timers, return_exceptions=True)
        self._retry_timers.clear()

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


__all__ = ["MAX_DELIVERY_ATTEMPTS", "NotificationManager"]
