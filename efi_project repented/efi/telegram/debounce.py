"""
efi/telegram/debounce.py

Группировка быстрых последовательных сообщений от одного собеседника в одно
событие перед тем, как оно уйдёт в NotificationManager — "anti-interrupt"
поведение: без него Worker реагировал бы на КАЖДОЕ сообщение по отдельности
(включая полную пересборку контекста на каждое), даже если человек просто
дописывает мысль через несколько быстрых реплик подряд.

Механизм — не фиксированный таймер тишины, а отслеживание ЖИВОГО статуса
"печатает" (через TypingStatusProvider, обычно efi.telegram.typing_tracker.
TypingTracker): пока собеседник печатает — ждём; как только статус пропал —
короткая случайная пауза (`post_typing_delay_range`, по умолчанию 0.1–1с) и
сразу флаш. Это заметно быстрее старого подхода "жди N секунд тишины после
последнего сообщения" — там задержка не знала, реально ли человек ещё
печатает, или просто выдержала пауза.

`fallback_delay_seconds` — обычный таймер тишины, если типинг вообще не
отслеживается (typing_tracker не передан или ни разу не сработал для этого
чата) — не оставляем чат совсем без дебаунса в этом случае.

`max_wait_seconds` — жёсткий потолок от момента ПЕРВОГО сообщения в пачке:
даже если собеседник печатает без остановки дольше этого времени, всё равно
отвечаем, не ждём вечно.

Не привязан к Pyrogram напрямую — работает с произвольным типом T через
Generic, а статус печатания получает через Protocol, а не конкретный класс.
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

_DEFAULT_POST_TYPING_DELAY_RANGE = (0.1, 1.0)
_DEFAULT_TYPING_POLL_INTERVAL_SECONDS = 0.3
_DEFAULT_FALLBACK_DELAY_SECONDS = 2.0
_DEFAULT_MAX_WAIT_SECONDS = 15.0


class TypingStatusProvider(Protocol):
    """Абстракция живого статуса 'печатает'. Конкретная реализация — efi.telegram.typing_tracker.TypingTracker."""

    def is_typing(self, chat_id: int) -> bool: ...


@dataclass(slots=True)
class _ChatBuffer(Generic[T]):
    items: list[T] = field(default_factory=list)
    flush_task: asyncio.Task | None = None


class MessageDebouncer(Generic[T]):
    """Копит элементы по chat_id и отдаёт их пачкой через `on_flush`, ориентируясь на живой статус 'печатает'."""

    def __init__(
        self,
        on_flush: OnFlush[T],
        *,
        typing_tracker: TypingStatusProvider | None = None,
        post_typing_delay_range: tuple[float, float] = _DEFAULT_POST_TYPING_DELAY_RANGE,
        typing_poll_interval_seconds: float = _DEFAULT_TYPING_POLL_INTERVAL_SECONDS,
        fallback_delay_seconds: float = _DEFAULT_FALLBACK_DELAY_SECONDS,
        max_wait_seconds: float = _DEFAULT_MAX_WAIT_SECONDS,
    ) -> None:
        self._on_flush = on_flush
        self._typing_tracker = typing_tracker
        self._post_typing_delay_range = post_typing_delay_range
        self._typing_poll_interval_seconds = typing_poll_interval_seconds
        self._fallback_delay_seconds = fallback_delay_seconds
        self._max_wait_seconds = max_wait_seconds
        self._buffers: dict[int, _ChatBuffer[T]] = {}
        self._first_item_at: dict[int, float] = {}
        self._lock = asyncio.Lock()

    async def add(self, chat_id: int, item: T) -> None:
        """Добавляет элемент в буфер чата и (пере)запускает ожидание — отменяя предыдущее, если оно уже шло."""
        async with self._lock:
            buffer = self._buffers.setdefault(chat_id, _ChatBuffer())
            if not buffer.items:
                self._first_item_at[chat_id] = asyncio.get_running_loop().time()
            buffer.items.append(item)

            if buffer.flush_task is not None:
                buffer.flush_task.cancel()

            buffer.flush_task = asyncio.create_task(self._wait_and_flush(chat_id, self._first_item_at[chat_id]))

    async def _wait_and_flush(self, chat_id: int, first_item_at: float) -> None:
        """
        Пока собеседник печатает — ждём короткими интервалами (перепроверяя
        статус), не дольше общего потолка max_wait_seconds от первого
        сообщения пачки. Как только статус пропал (или трекера нет вовсе) —
        короткая динамическая пауза и флаш.
        """
        try:
            while True:
                elapsed = asyncio.get_running_loop().time() - first_item_at
                remaining_to_cap = self._max_wait_seconds - elapsed
                if remaining_to_cap <= 0:
                    break  # потолок исчерпан — отвечаем, что бы ни происходило с typing

                if self._typing_tracker is not None and self._typing_tracker.is_typing(chat_id):
                    await asyncio.sleep(min(self._typing_poll_interval_seconds, remaining_to_cap))
                    continue

                post_delay = (
                    random.uniform(*self._post_typing_delay_range)
                    if self._typing_tracker is not None
                    else self._fallback_delay_seconds
                )
                await asyncio.sleep(min(post_delay, max(remaining_to_cap, 0.0)))
                break
        except asyncio.CancelledError:
            return  # добавилось новое сообщение — новый _wait_and_flush уже запущен в add()

        await self._flush(chat_id)

    async def _flush(self, chat_id: int) -> None:
        async with self._lock:
            buffer = self._buffers.pop(chat_id, None)
            self._first_item_at.pop(chat_id, None)
        if buffer is None or not buffer.items:
            return
        try:
            await self._on_flush(chat_id, buffer.items)
        except Exception:
            logger.exception("debounce: on_flush raised for chat_id=%s", chat_id)

    async def flush_all(self) -> None:
        """Принудительно сбрасывает все текущие буферы — используется при graceful shutdown, чтобы не потерять недописанные сообщения."""
        async with self._lock:
            chat_ids = list(self._buffers.keys())
        for chat_id in chat_ids:
            buffer = self._buffers.get(chat_id)
            if buffer is not None and buffer.flush_task is not None:
                buffer.flush_task.cancel()
            await self._flush(chat_id)


__all__ = ["MessageDebouncer", "TypingStatusProvider"]
