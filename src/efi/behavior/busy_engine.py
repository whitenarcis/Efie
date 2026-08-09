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
      близкому человеку отвечает быстрее, чем случайному знакомому;
    - идёт ли уже АКТИВНЫЙ разговор в этом чате (efi.db.history_repository.
      SqliteHistoryRepository.get_last_message_at) — см. ниже.

ВАЖНО про активный разговор: раньше полноценная ignore_delay (1-8с и больше)
считалась на КАЖДОЕ сообщение без исключения — если собеседник уже был в
диалоге и переписывался быстро, Эфи всё равно каждый раз выглядела так,
будто только что "взяла телефон", хотя она явно уже была на связи секунду
назад. Реальный человек посреди быстрой переписки не перечитывает статус
"занята" перед каждой репликой — задержка "не сразу увидела" уместна только
для ПЕРВОГО сообщения после паузы, не для продолжения диалога. Если с
последнего сообщения в чате (в любую сторону) прошло меньше
`active_conversation_window_seconds` — разговор считается активным, и вместо
полной ignore_delay применяется только маленький "живой" джиттер
(`active_conversation_delay_min/max_seconds`), кроме случая, когда Эфи
ДЕЙСТВИТЕЛЬНО занята прямо сейчас (`is_researching`) — тогда полная задержка
всё равно применяется, потому что занятость в этом случае настоящая, а не
формальность "с момента последнего сообщения".

Сама арифметика (`_calculate_ignore_delay`) — чистая синхронная функция без
I/O, специально вынесена отдельно от `BusyEngine.compute_ignore_delay`
(которая уже читает WorkingMemory/AffinityTracker/историю), чтобы формулу
можно было тестировать без БД/event loop, как и остальные подобные пары
"чистая функция + асинхронная обёртка" в проекте (см.
efi/humanizer/typing_simulation.py).
"""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Protocol

from efi.behavior.affinity import AffinitySnapshot, AffinityTracker
from efi.config.schema import BusyEngineSettings
from efi.memory.working_memory import WorkingMemory


@dataclass(slots=True, frozen=True)
class BusyDecision:
    """
    Результат одного расчёта занятости: сколько ждать И почему.

    `is_active_conversation` нужен вызывающей стороне отдельно от задержки:
    efi.notifications.worker.Worker по нему решает, отмечать ли сообщение
    прочитанным СРАЗУ. Если Эфи уже в контексте активного чата, держать
    сообщение непрочитанным незачем — она физически "смотрит в этот чат"
    прямо сейчас, и задержка перед read_history выглядела бы как
    искусственное удержание в непрочитанных, а не как живое поведение.
    """

    delay_seconds: float
    is_active_conversation: bool


class BusyState(Protocol):
    """Абстракция 'занята ли Эфи фоновым делом'. Реализация — efi.behavior.life_engine.BackgroundLifeWorker."""

    @property
    def is_researching(self) -> bool: ...


class LastMessageSource(Protocol):
    """
    Абстракция 'когда было последнее сообщение чата'. Реализация —
    efi.db.history_repository.SqliteHistoryRepository.
    """

    async def get_last_message_at(self, chat_id: int) -> datetime | None: ...


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
        *,
        last_message_source: LastMessageSource | None = None,
    ) -> None:
        self._working_memory = working_memory
        self._affinity = affinity
        self._life_engine = life_engine
        self._settings = settings
        self._last_message_source = last_message_source

    async def decide(self, chat_id: int | None) -> BusyDecision:
        """
        Критический путь (в начале обработки КАЖДОГО уведомления — см.
        efi.notifications.worker.Worker._handle): WorkingMemory,
        AffinityTracker и момент последнего сообщения читаются конкурентно;
        `is_researching` — синхронное свойство, без I/O. События без chat_id
        (например, NIGHTLY_TASK) не имеют своей близости/разговора —
        используются нейтральные дефолты.
        """
        working_memory_task = self._working_memory.load()
        affinity_task = self._affinity.get_snapshot(chat_id) if chat_id is not None else _default_affinity()
        last_message_task = self._get_last_message_at(chat_id)
        memory_snapshot, affinity_snapshot, last_message_at = await asyncio.gather(
            working_memory_task, affinity_task, last_message_task
        )

        is_active_conversation = _is_active_conversation(last_message_at, self._settings)
        delay = _calculate_ignore_delay(
            is_researching=self._life_engine.is_researching,
            energy=memory_snapshot.energy,
            affinity=affinity_snapshot,
            is_active_conversation=is_active_conversation,
            settings=self._settings,
        )
        return BusyDecision(delay_seconds=delay, is_active_conversation=is_active_conversation)

    async def compute_ignore_delay(self, chat_id: int | None) -> float:
        """Только задержка, без остального контекста решения — тонкая обёртка над `decide()`."""
        return (await self.decide(chat_id)).delay_seconds

    async def _get_last_message_at(self, chat_id: int | None) -> datetime | None:
        if chat_id is None or self._last_message_source is None:
            return None
        return await self._last_message_source.get_last_message_at(chat_id)


async def _default_affinity() -> AffinitySnapshot:
    """Тривиальная async-обёртка ради единообразного `asyncio.gather()` выше — без похода в БД."""
    return AffinitySnapshot()


def _is_active_conversation(last_message_at: datetime | None, settings: BusyEngineSettings) -> bool:
    """Чат без истории (last_message_at is None) — по определению не 'уже идущий' разговор."""
    if last_message_at is None:
        return False
    elapsed = (datetime.now(UTC) - last_message_at).total_seconds()
    return elapsed <= settings.active_conversation_window_seconds


def _calculate_ignore_delay(
    *,
    is_researching: bool,
    energy: float,
    affinity: AffinitySnapshot,
    is_active_conversation: bool,
    settings: BusyEngineSettings,
) -> float:
    """
    Чистая функция. Внутри уже идущего разговора (is_active_conversation),
    пока Эфи не занята чем-то реальным (is_researching), полноценный расчёт
    занятости пропускается — только маленький "живой" джиттер, чтобы не
    выглядело мгновенной автоматической реакцией. Иначе (первое сообщение
    после паузы, ИЛИ разговор активен, но Эфи реально занята) — базовая
    случайная задержка (растянутая вдвое-втрое, если занята исследованием)
    + штраф за низкую энергию - скидка за высокую близость/уважение,
    зажатые в [min_delay_seconds, max_delay_seconds].
    """
    if is_active_conversation and not is_researching:
        low = settings.active_conversation_delay_min_seconds
        high = max(low, settings.active_conversation_delay_max_seconds)
        return random.uniform(low, high)

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


__all__ = ["BusyDecision", "BusyEngine", "BusyState", "LastMessageSource"]
