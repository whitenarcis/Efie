"""
efi/telegram/buffer.py

InboundMessageBuffer — сборка быстрых сообщений собеседника в одну пачку
перед тем, как они уйдут в NotificationManager.

Пришёл на смену efi/telegram/debounce.py::MessageDebouncer и отличается от
него двумя вещами, обе — про «не выглядеть запоздалым ботом»:

1. ПЛАВАЮЩЕЕ ОКНО СБОРКИ. Раньше пауза после того, как собеседник перестал
   печатать, была почти нулевой (0.1–1с), а само окно не имело собственной
   длительности: буфер держался ровно столько, сколько горел статус
   "печатает". Между двумя короткими репликами ("найду романтику" / "и пох")
   статус успевает погаснуть — и Эфи запускала генерацию на первую строчку,
   а вторая прилетала уже в ответ. Теперь у окна есть своя длительность
   (`window_range`, ~1.5–2.5с), и КАЖДОЕ новое сообщение сдвигает его вперёд:
   пока человек досыпает мысль короткими репликами, пачка растёт.

   Живой статус "печатает" никуда не делся — он работает поверх окна: пока
   собеседник печатает, ждём дальше даже за пределами окна (но не дольше
   `max_wait_seconds` от первого сообщения пачки).

2. МГНОВЕННЫЙ СИГНАЛ ПРЕРЫВАНИЯ. `on_interrupt` вызывается в момент ПРИЁМА
   сообщения, до всякого ожидания, — чтобы устаревшая генерация (или пауза
   между бабблами уже неактуального ответа) была снята сразу, а не через
   полторы секунды, когда окно закроется. См. efi/telegram/chat_orchestrator.py.

Не привязан к Pyrogram: работает с произвольным типом T, статус печатания
получает через Protocol.
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import Generic, Protocol, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")
OnFlush = Callable[[int, list[T]], Awaitable[None]]
OnInterrupt = Callable[[int], Awaitable[None]]

#: Плавающее окно сборки. Нижняя граница — чтобы вторая реплика из быстрой
#: пачки успела долететь; верхняя — чтобы одиночное сообщение не ждало
#: заметно дольше, чем человек терпит перед началом ответа.
_DEFAULT_WINDOW_RANGE = (1.5, 2.5)
_DEFAULT_TYPING_POLL_INTERVAL_SECONDS = 0.3
_DEFAULT_MAX_WAIT_SECONDS = 15.0


class TypingStatusProvider(Protocol):
    """Абстракция живого статуса 'печатает'. Конкретная реализация — efi.telegram.typing_tracker.TypingTracker."""

    def is_typing(self, chat_id: int) -> bool: ...


@dataclass(slots=True)
class _ChatBuffer(Generic[T]):
    items: list[T] = field(default_factory=list)
    flush_task: asyncio.Task[None] | None = None
    #: Момент первого сообщения пачки — точка отсчёта для max_wait_seconds.
    started_at: float = 0.0


class InboundMessageBuffer(Generic[T]):
    """
    Копит сообщения по chat_id и отдаёт их одной пачкой через `on_flush`.

    Окно сборки сдвигается на каждом новом сообщении; статус "печатает"
    продлевает ожидание сверх окна; `max_wait_seconds` ограничивает всё
    сверху, чтобы собеседник, печатающий без остановки, не откладывал ответ
    бесконечно.
    """

    def __init__(
        self,
        on_flush: OnFlush[T],
        *,
        typing_tracker: TypingStatusProvider | None = None,
        window_range: tuple[float, float] = _DEFAULT_WINDOW_RANGE,
        typing_poll_interval_seconds: float = _DEFAULT_TYPING_POLL_INTERVAL_SECONDS,
        max_wait_seconds: float = _DEFAULT_MAX_WAIT_SECONDS,
        on_interrupt: OnInterrupt | None = None,
    ) -> None:
        self._on_flush = on_flush
        self._typing_tracker = typing_tracker
        self._window_range = window_range
        self._typing_poll_interval_seconds = typing_poll_interval_seconds
        self._max_wait_seconds = max_wait_seconds
        self._on_interrupt = on_interrupt
        self._buffers: dict[int, _ChatBuffer[T]] = {}
        self._lock = asyncio.Lock()

    async def add(self, chat_id: int, item: T) -> None:
        """
        Кладёт сообщение в пачку и сдвигает окно сборки вперёд.

        Прерывание устаревшей генерации идёт ПЕРВЫМ действием, до захвата
        лока и до любого ожидания: смысл прерывания в том, что оно
        мгновенное. Если Эфи прямо сейчас печатает ответ на предыдущую
        реплику, она должна замолчать в ту же секунду, когда прилетела
        новая, — а не досказать неактуальное.
        """
        if self._on_interrupt is not None:
            await self._on_interrupt(chat_id)

        async with self._lock:
            buffer = self._buffers.setdefault(chat_id, _ChatBuffer())
            if not buffer.items:
                buffer.started_at = asyncio.get_running_loop().time()
            buffer.items.append(item)

            if buffer.flush_task is not None:
                buffer.flush_task.cancel()
            buffer.flush_task = asyncio.create_task(
                self._wait_and_flush(chat_id, buffer.started_at), name=f"inbound-window-{chat_id}"
            )

    async def _wait_and_flush(self, chat_id: int, started_at: float) -> None:
        """
        Ждёт окно сборки, затем — пока собеседник печатает, и флашит пачку.

        Отмена означает только одно: пришло ещё одно сообщение, и `add()`
        уже запустил новое ожидание с более поздним окном. Ничего чистить
        не нужно — пачка живёт в буфере, а не в этой корутине.
        """
        try:
            await asyncio.sleep(self._remaining(started_at, random.uniform(*self._window_range)))
            while self._typing_tracker is not None and self._typing_tracker.is_typing(chat_id):
                remaining = self._remaining(started_at, self._typing_poll_interval_seconds)
                if remaining <= 0.0:
                    break  # потолок исчерпан — отвечаем, что бы ни происходило с typing
                await asyncio.sleep(remaining)
        except asyncio.CancelledError:
            return

        await self._flush(chat_id)

    def _remaining(self, started_at: float, desired: float) -> float:
        """Сколько реально можно ждать: `desired`, но не выходя за общий потолок пачки."""
        elapsed = asyncio.get_running_loop().time() - started_at
        return max(min(desired, self._max_wait_seconds - elapsed), 0.0)

    async def _flush(self, chat_id: int) -> None:
        async with self._lock:
            buffer = self._buffers.pop(chat_id, None)
        if buffer is None or not buffer.items:
            return
        if len(buffer.items) > 1:
            logger.info("inbound_buffer: flushing a batch of %d messages for chat_id=%s", len(buffer.items), chat_id)
        try:
            await self._on_flush(chat_id, buffer.items)
        except Exception:
            logger.exception("inbound_buffer: on_flush raised for chat_id=%s", chat_id)

    async def flush_all(self) -> None:
        """Принудительно сбрасывает все буферы — graceful shutdown не должен терять недописанные сообщения."""
        async with self._lock:
            chat_ids = list(self._buffers.keys())
        for chat_id in chat_ids:
            buffer = self._buffers.get(chat_id)
            if buffer is not None and buffer.flush_task is not None:
                buffer.flush_task.cancel()
            await self._flush(chat_id)


__all__ = ["InboundMessageBuffer", "TypingStatusProvider"]
