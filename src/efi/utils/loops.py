"""
efi/utils/loops.py

Фоновый цикл, который переживает собственные ошибки.

Проблема, которую этот модуль закрывает, выглядела так. У Эфи шесть
периодических служб — спонтанный пинг, монитор тишины, фоновый
исследователь, движок жизни, напоминания, пульс памяти, — и каждая была
написана одинаково:

    while True:
        await asyncio.sleep(interval)
        await self._tick()

Ни в одной, кроме пульса памяти, `_tick()` не был обёрнут ничем. Любое
исключение — секундная недоступность SQLite, ошибка Pyrogram, баг в новой
ветке кода — навсегда завершало цикл. Не «пропустила одну итерацию», а
именно навсегда: `while True` выходит, задача заканчивается, служба больше не
работает до перезапуска процесса.

Заметить это со стороны почти невозможно. Эфи не падает, отвечает на
сообщения как обычно — просто перестаёт, например, писать первой. Ровно тот
сорт поломки, который списывают на «настроение» и обнаруживают через неделю.

Отсюда правило, которое здесь и живёт: **сбой одной итерации не завершает
цикл**. Следующая попытка придёт по расписанию, а ошибка уходит в лог с
трассировкой. Отмена (`CancelledError`) — единственное, что цикл
останавливает: это не сбой, а штатная остановка приложения.

Отдельный модуль, а не по копии в каждой службе: шесть копий одного правила —
это шесть мест, где его можно забыть, и седьмая служба забудет его наверняка.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

#: Верхняя граница паузы, которую цикл выдерживает после ПОДРЯД идущих сбоев.
#: Нужна на случай, когда сломано надолго (кончилось место на диске, отвалилась
#: БД): молотить в полную силу в такой ситуации значит только забивать лог и
#: греть телефон. Штатный интервал служб и так измеряется минутами, поэтому
#: потолок сознательно небольшой — он про «не частить», а не про «отступить».
_MAX_BACKOFF_SECONDS = 300.0

#: После скольких подряд неудачных итераций начинать растягивать паузу.
#: Первая ошибка почти всегда единичная (моргнула сеть), и наказывать за неё
#: задержкой незачем.
_BACKOFF_AFTER_FAILURES = 3


async def run_periodically(
    tick: Callable[[], Awaitable[object]],
    *,
    interval_seconds: float,
    name: str,
    wake_event: asyncio.Event | None = None,
) -> None:
    """
    Вызывает `tick()` раз в `interval_seconds`, пока задачу не отменят.

    Исключение из `tick()` логируется и НЕ прерывает цикл. Несколько сбоев
    подряд растягивают паузу (см. _MAX_BACKOFF_SECONDS): если сломано
    надолго, стучаться каждую минуту бессмысленно, а лог за ночь становится
    нечитаемым.

    Первая пауза — ДО первого вызова, а не после. Так было во всех службах до
    появления этого модуля, и на это опирается старт: одновременный залп из
    шести служб в первую же секунду после запуска — не то, чего ждёшь от
    «фоновых» задач, особенно на телефоне.

    `wake_event` — способ разбудить службу раньше срока. Нужен там, где
    появилась работа, которую бессмысленно откладывать до следующего тика:
    человек договорился о проекте и ждёт, что за него возьмутся сейчас, а не
    через час (см. efi.dev.worker.DevWorker.request_tick). Событие
    сбрасывается перед вызовом `tick()`, поэтому один сигнал даёт ровно один
    внеочередной проход.
    """
    logger.info("%s: started (interval=%.0fs)", name, interval_seconds)
    failures = 0
    try:
        while True:
            await _sleep_until(_next_delay(interval_seconds, failures), wake_event, name=name)
            try:
                await tick()
            except asyncio.CancelledError:
                raise
            except Exception:
                failures += 1
                logger.exception(
                    "%s: итерация упала (подряд: %d), цикл продолжается", name, failures
                )
            else:
                failures = 0
    except asyncio.CancelledError:
        logger.info("%s: stopped", name)
        raise


async def _sleep_until(delay: float, wake_event: asyncio.Event | None, *, name: str) -> None:
    """Пауза до следующего тика — или до внеочередного сигнала, если он пришёл раньше."""
    if wake_event is None:
        await asyncio.sleep(delay)
        return
    try:
        await asyncio.wait_for(wake_event.wait(), timeout=delay)
    except TimeoutError:
        return
    wake_event.clear()
    logger.info("%s: внеочередной проход по сигналу", name)


def _next_delay(interval_seconds: float, failures: int) -> float:
    if failures < _BACKOFF_AFTER_FAILURES:
        return interval_seconds
    # Удвоение за каждый сбой сверх порога, но не выше потолка.
    overshoot = failures - _BACKOFF_AFTER_FAILURES + 1
    return float(min(interval_seconds * (2**overshoot), _MAX_BACKOFF_SECONDS))


__all__ = ["run_periodically"]
