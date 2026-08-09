"""
efi/memory/pulse.py

Пульс памяти — непрерывное превращение прожитого в воспоминания, вместо
одного ночного прохода.

ЗАЧЕМ. До появления этого модуля единственным источником памяти о разговорах
была ночная новеллизация в 03:30 (efi/app.py::_run_consolidation_loop). Это
давало три плохих эффекта сразу:

    1. Всё, что случилось за день, существовало только как строки в таблице
       `messages` и не было памятью вообще. Перезапуск или падение до 03:30 —
       и день не осмыслен; отметка last_novelized_at не сдвинута, но окно
       уже перекрывается следующими сутками и разбирается одним запросом с
       общим лимитом вывода, то есть часть дня всё равно теряется.
    2. Эфи не могла сослаться днём на то, что было утром: RAG-поиск ищет по
       дневнику, а в дневнике этого ещё не было.
    3. Опыт распадался на несвязные потоки: разговор новеллизировался ночью,
       комментарий в сообществе писался в журнал мгновенно отдельной строкой,
       фоновая находка — третьим путём. Ни в одной точке не было "меня
       сегодня", было три лога разных подсистем.

КАК. Раз в `check_interval_seconds` проверяются все чаты с активностью и
новеллизируется тот, чей эпизод ЗАВЕРШИЛСЯ. Завершение определяется не
таймером, а формой самого разговора:

    - разговор остыл: с последнего сообщения прошло больше
      `episode_idle_seconds` (человек так и запоминает — не по часам, а
      когда общение закончилось и можно отложить);
    - либо разговор длинный и всё ещё идёт: накопилось
      `max_messages_before_flush` сообщений с прошлого разбора — не ждём
      конца, иначе марафонская переписка снова превратится в один
      обрубленный кусок.

Внутрь каждого эпизода подмешивается внешний опыт того же чата за тот же
период (ExperienceSource — что гуглила, где комментировала), поэтому
эпизод осмысляется как один прожитый кусок жизни. Это и есть техническая
форма требования "Эфи должна быть единой личностью": не отдельные логи
подсистем, а один поток, разобранный одним запросом.

Пульс НЕ отменяет ночной проход: тот остаётся как подбор хвостов (чаты,
где эпизод так и не закрылся) плюс обслуживание корпуса — dedup и сжатие
старых записей в мемуары.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from typing import Protocol

from efi.memory.consolidation import DiaryConsolidator, ExperienceSource, HistorySource
from efi.memory.facts import FactStore

logger = logging.getLogger(__name__)

#: Насколько далеко назад вообще смотреть в поиске активных чатов. Пульс
#: интересуется недавним; всё, что старше, — забота ночного прохода.
_ACTIVE_CHAT_WINDOW = timedelta(hours=12)


class PulseHistorySource(HistorySource, Protocol):
    """
    То, что пульсу нужно от истории сверх обычного HistorySource: момент
    последнего сообщения чата — по нему определяется, остыл ли эпизод.
    Конкретная реализация — efi.db.history_repository.SqliteHistoryRepository
    (метод уже есть, его использует BusyEngine).
    """

    async def get_last_message_at(self, chat_id: int) -> datetime | None: ...


class MemoryPulse:
    """
    Фоновый цикл частой новеллизации. Предназначен для запуска через
    `asyncio.create_task(pulse.run())` при старте приложения, как и
    остальные фоновые сервисы.

    Сбой разбора ОДНОГО чата не должен ронять цикл и не должен мешать
    остальным чатам: отметка last_novelized_at по упавшему чату не
    сдвигается, значит следующий тик просто попробует снова.
    """

    def __init__(
        self,
        consolidator: DiaryConsolidator,
        history: PulseHistorySource,
        facts: FactStore,
        *,
        experience: ExperienceSource | None = None,
        check_interval_seconds: float = 600.0,
        episode_idle_seconds: float = 900.0,
        max_messages_before_flush: int = 30,
        min_messages: int = 3,
        lookback: timedelta = timedelta(hours=12),
    ) -> None:
        self._consolidator = consolidator
        self._history = history
        self._facts = facts
        self._experience = experience
        self._check_interval_seconds = check_interval_seconds
        self._episode_idle_seconds = episode_idle_seconds
        self._max_messages_before_flush = max_messages_before_flush
        self._min_messages = min_messages
        self._lookback = lookback
        self._lock = asyncio.Lock()

    @property
    def novelization_lock(self) -> asyncio.Lock:
        """
        Взаимное исключение с ночным проходом (efi/app.py). Оба пути двигают
        одну и ту же отметку last_novelized_at, и без лока они могут прочитать
        её одновременно и разобрать одно окно дважды — в дневник ушли бы два
        разных пересказа одного эпизода. Дедупликация по эмбеддингам от этого
        не спасает: формулировки будут разные, а порог там намеренно высокий.
        """
        return self._lock

    async def run(self) -> None:
        """Основной цикл. Останавливается по отмене задачи (CancelledError) — см. efi/app.py graceful shutdown."""
        logger.info(
            "memory_pulse: started (interval=%.0fs, idle=%.0fs, flush_at=%d messages)",
            self._check_interval_seconds, self._episode_idle_seconds, self._max_messages_before_flush,
        )
        try:
            while True:
                await asyncio.sleep(self._check_interval_seconds)
                try:
                    await self.tick()
                except Exception:
                    logger.exception("memory_pulse: tick failed, will retry on the next interval")
        except asyncio.CancelledError:
            logger.info("memory_pulse: stopped")
            raise

    async def tick(self) -> int:
        """
        Один проход по активным чатам. Возвращает число новых записей
        дневника — удобно и для логов, и для тестов.
        """
        now = datetime.now(UTC)
        async with self._lock:
            chat_ids = await self._history.get_active_chat_ids(since=now - _ACTIVE_CHAT_WINDOW)
            created = 0
            for chat_id in chat_ids:
                created += await self._tick_chat(chat_id, now)
        if created:
            logger.info("memory_pulse: wrote %d new diary entries", created)
        return created

    async def _tick_chat(self, chat_id: int, now: datetime) -> int:
        since = await self._consolidator.resolve_last_novelized_at(self._facts, chat_id, self._lookback)
        session = await self._history.get_since(chat_id, since=since)

        # Внешний опыт считается наравне с репликами: в канале сообщества Эфи
        # может за период не написать ни одного сообщения в привычном смысле,
        # а оставить пару комментариев и прочитать тред — по счёту сообщений
        # это "пусто", хотя прожито там больше, чем в ином разговоре.
        pending = len(session.messages)
        if self._experience is not None:
            pending += len(await self._experience.context_lines_for_chat(chat_id, since=since))
        if pending < self._min_messages:
            return 0

        if not await self._episode_is_ready(chat_id, pending, now):
            return 0

        # Разбор идёт повторным чтением истории внутри novelize_chat (тот же
        # запрос, что и выше) — сознательная плата за то, что единица работы
        # остаётся общей с ночным проходом и вызывается одинаково из обоих
        # мест. Это локальный SQLite-запрос вне критического пути ответа.
        return await self._consolidator.novelize_chat(
            chat_id,
            history=self._history,
            facts=self._facts,
            since=since,
            min_messages=self._min_messages,
            experience=self._experience,
        )

    async def _episode_is_ready(self, chat_id: int, pending: int, now: datetime) -> bool:
        """
        Эпизод пора запоминать, если разговор остыл ИЛИ уже слишком длинный.

        Порядок проверок именно такой: длинный, но всё ещё идущий разговор
        мы разбираем частями, а короткий — целиком, дождавшись паузы. Иначе
        либо марафон снова схлопнулся бы в один обрубок, либо каждая реплика
        живого диалога тянула бы за собой LLM-вызов.
        """
        if pending >= self._max_messages_before_flush:
            return True

        last_message_at = await self._history.get_last_message_at(chat_id)
        if last_message_at is None:
            return False
        return (now - last_message_at).total_seconds() >= self._episode_idle_seconds


__all__ = ["MemoryPulse", "PulseHistorySource"]
