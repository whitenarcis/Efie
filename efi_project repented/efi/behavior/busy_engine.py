"""
efi/behavior/busy_engine.py

Симуляция занятости — прежде чем Worker вообще "заметит" уведомление (в
первую очередь входящее сообщение собеседника), он не должен реагировать
мгновенно. BusyEngine считает `ignore_delay`: паузу перед тем, как
efi.notifications.worker.Worker сделает хоть одно видимое телеграм-действие
(зайдёт в чат/включит typing/позовёт LLM) — тот самый человеческий эффект
"телефон лежал, не сразу увидела".

"Появление в сети строго в момент активации действия в Telegram" (см. задачу)
обеспечивается не здесь, а порядком вызовов в Worker._handle(): пока идёт
ignore_delay, Worker не делает НИ ОДНОГО обращения к Pyrogram (ни
read_history, ни send_chat_action) — Telegram сам не покажет клиента "в
сети", если тот совсем ничего не делает. BusyEngine отвечает только за то,
сколько ждать, а не за то, как скрыть присутствие — это два разных слоя
ответственности, а не один и тот же расчёт.

Вход в расчёт:
    - занята ли Эфи фоновым исследованием прямо сейчас
      (efi.behavior.life_engine.BackgroundLifeWorker.is_researching) — если
      да, пауза длиннее: она "делом занята";
    - energy (efi.memory.working_memory.WorkingMemorySnapshot.energy, 0..1) —
      чем ниже, тем больше пауза (устала, не сразу берёт телефон);
    - affinity/respect_level чата (efi.behavior.affinity.AffinitySnapshot) —
      близкому человеку отвечает быстрее, чем случайному знакомому.

Сама арифметика (`_calculate_ignore_delay`) — чистая синхронная функция без
I/O, специально вынесена отдельно от `BusyEngine.compute_ignore_delay`
(которая уже читает WorkingMemory/AffinityTracker), чтобы формулу можно было
тестировать без БД/event loop, как и остальные подобные пары "чистая функция
+ асинхронная обёртка" в проекте (см. efi/humanizer/typing_simulation.py).
"""

from __future__ import annotations

import asyncio
import random
from typing import Protocol

from efi.behavior.affinity import AffinitySnapshot, AffinityTracker
from efi.config.schema import BusyEngineSettings
from efi.memory.working_memory import WorkingMemory


class BusyState(Protocol):
    """Абстракция 'занята ли Эфи фоновым делом'. Реализация — efi.behavior.life_engine.BackgroundLifeWorker."""

    @property
    def is_researching(self) -> bool: ...


class BusyEngine:
    """
    Асинхронная точка входа для Worker: `compute_ignore_delay(chat_id)` сама
    забирает всё нужное (WorkingMemory/AffinityTracker читаются конкурентно)
    и возвращает готовое число секунд — Worker не обязан знать, из каких
    источников оно собрано, тот же принцип разделения ответственности, что и
    у efi.prompts.builder.EfiSystemPromptBuilder.
    """

    def __init__(
        self,
        working_memory: WorkingMemory,
        affinity: AffinityTracker,
        life_engine: BusyState,
        settings: BusyEngineSettings,
    ) -> None:
        self._working_memory = working_memory
        self._affinity = affinity
        self._life_engine = life_engine
        self._settings = settings

    async def compute_ignore_delay(self, chat_id: int | None) -> float:
        """
        Критический путь (в начале обработки КАЖДОГО уведомления — см.
        efi.notifications.worker.Worker._handle): WorkingMemory и
        AffinityTracker читаются конкурентно; `is_researching` — синхронное
        свойство, без I/O. События без chat_id (например, NIGHTLY_TASK) не
        имеют своей близости — используется нейтральный дефолт.
        """
        working_memory_task = self._working_memory.load()
        affinity_task = self._affinity.get_snapshot(chat_id) if chat_id is not None else _default_affinity()
        memory_snapshot, affinity_snapshot = await asyncio.gather(working_memory_task, affinity_task)

        return _calculate_ignore_delay(
            is_researching=self._life_engine.is_researching,
            energy=memory_snapshot.energy,
            affinity=affinity_snapshot,
            settings=self._settings,
        )


async def _default_affinity() -> AffinitySnapshot:
    """Тривиальная async-обёртка ради единообразного `asyncio.gather()` выше — без похода в БД."""
    return AffinitySnapshot()


def _calculate_ignore_delay(
    *, is_researching: bool, energy: float, affinity: AffinitySnapshot, settings: BusyEngineSettings
) -> float:
    """
    Чистая функция: базовая случайная задержка (растянутая вдвое-втрое, если
    Эфи занята исследованием) + штраф за низкую энергию - скидка за высокую
    близость/уважение, зажатые в [min_delay_seconds, max_delay_seconds].
    """
    low, high = settings.base_delay_min_seconds, settings.base_delay_max_seconds
    if is_researching:
        high *= settings.research_busy_multiplier
    base = random.uniform(low, max(low, high))

    clamped_energy = max(0.0, min(energy, 1.0))
    energy_penalty = (1.0 - clamped_energy) * settings.low_energy_extra_seconds

    closeness = max(0.0, min((affinity.affinity + affinity.respect_level) / 2.0, 1.0))
    affinity_discount = closeness * settings.high_affinity_discount_seconds

    delay = base + energy_penalty - affinity_discount
    return max(settings.min_delay_seconds, min(delay, settings.max_delay_seconds))


__all__ = ["BusyEngine", "BusyState"]
