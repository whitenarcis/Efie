"""
Тесты устойчивости фоновых служб (efi.utils.loops + EfiApp._spawn_supervised).

Аудит перед выпуском. У Эфи шесть периодических служб, и каждая была
написана так:

    while True:
        await asyncio.sleep(interval)
        await self._tick()

Ни в одной, кроме пульса памяти, тик не был обёрнут ничем. Любое исключение —
секундная недоступность SQLite, ошибка Pyrogram, баг в новой ветке кода —
навсегда завершало цикл. Не «пропустила итерацию», а именно навсегда:
`while True` выходит, задача заканчивается, служба не работает до
перезапуска процесса.

Заметить это со стороны почти невозможно: Эфи не падает и на сообщения
отвечает — просто перестаёт, например, писать первой. Ровно тот сорт
поломки, который списывают на «настроение» и обнаруживают через неделю.

Здесь же проверяется и вторая линия: супервизор приложения, который раньше
только ЛОГИРОВАЛ смерть службы. Лог делает поломку видимой, но Эфи от этого
писать первой не начинает, а в лог смотрят уже после того, как заметили
странность поведения.
"""

from __future__ import annotations

import asyncio

import pytest

from efi.utils.loops import run_periodically


async def _run_briefly(coro_factory, *, ticks: int, interval: float = 0.01) -> None:  # type: ignore[no-untyped-def]
    """Даёт циклу отработать несколько итераций и снимает его, как это делает shutdown."""
    task = asyncio.create_task(coro_factory())
    await asyncio.sleep(interval * (ticks + 0.5))
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task


# -- сбой итерации не убивает цикл -------------------------------------------


async def test_a_failing_tick_does_not_end_the_loop() -> None:
    """Главная регрессия аудита: одна ошибка забирала службу навсегда."""
    calls = 0

    async def tick() -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("БД моргнула")

    await _run_briefly(
        lambda: run_periodically(tick, interval_seconds=0.01, name="test"), ticks=3
    )

    assert calls >= 3, "цикл обязан продолжаться после сбоя"


async def test_the_loop_recovers_after_a_transient_failure() -> None:
    """Одна неудача не должна ничего портить в дальнейшей работе."""
    outcomes: list[str] = []

    async def tick() -> None:
        if len(outcomes) == 0:
            outcomes.append("fail")
            raise RuntimeError("разово")
        outcomes.append("ok")

    await _run_briefly(
        lambda: run_periodically(tick, interval_seconds=0.01, name="test"), ticks=3
    )

    assert outcomes[0] == "fail"
    assert "ok" in outcomes[1:]


async def test_cancellation_still_stops_the_loop() -> None:
    """
    Отмена — не сбой, а штатная остановка приложения. Проглоти цикл
    CancelledError вместе с остальным, и graceful shutdown повис бы навсегда.
    """
    async def tick() -> None:
        await asyncio.sleep(10)

    task = asyncio.create_task(run_periodically(tick, interval_seconds=0.001, name="test"))
    await asyncio.sleep(0.02)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task


async def test_repeated_failures_slow_the_loop_down() -> None:
    """
    Если сломано надолго (кончилось место, отвалилась БД), молотить в полную
    силу значит забить лог и посадить батарею.
    """
    from efi.utils.loops import _BACKOFF_AFTER_FAILURES, _MAX_BACKOFF_SECONDS, _next_delay

    assert _next_delay(60.0, 0) == 60.0
    assert _next_delay(60.0, _BACKOFF_AFTER_FAILURES - 1) == 60.0
    assert _next_delay(60.0, _BACKOFF_AFTER_FAILURES) > 60.0
    assert _next_delay(60.0, 99) == _MAX_BACKOFF_SECONDS


async def test_the_first_tick_waits_out_the_interval() -> None:
    """
    Иначе старт приложения превращался бы в одновременный залп из шести
    служб — не то, чего ждёшь от «фоновых» задач, особенно на телефоне.
    """
    calls = 0

    async def tick() -> None:
        nonlocal calls
        calls += 1

    task = asyncio.create_task(run_periodically(tick, interval_seconds=5.0, name="test"))
    await asyncio.sleep(0.05)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert calls == 0


# -- супервизор перезапускает упавшую службу ----------------------------------


async def test_a_dead_service_is_restarted() -> None:
    """
    Вторая линия обороны. Логирования, которое здесь стояло раньше, мало:
    оно делает поломку видимой, но Эфи от этого снова писать первой не
    начинает.
    """
    import efi.app as app_module

    starts = 0

    async def flaky() -> None:
        nonlocal starts
        starts += 1
        raise RuntimeError("упала на старте")

    # Приложение целиком тут не нужно — проверяется ровно метод супервизии.
    monkey = app_module.EfiApp.__new__(app_module.EfiApp)
    original_delay = app_module._SERVICE_RESTART_BASE_DELAY
    app_module._SERVICE_RESTART_BASE_DELAY = 0.001
    try:
        task = monkey._spawn_supervised(flaky, name="flaky")
        await asyncio.sleep(0.05)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    finally:
        app_module._SERVICE_RESTART_BASE_DELAY = original_delay

    assert starts > 1, "служба должна перезапускаться, а не только логироваться"


async def test_restarts_are_not_endless() -> None:
    """Безнадёжно сломанная служба (нет файла, нет прав) не должна крутиться вечно."""
    import efi.app as app_module

    starts = 0

    async def hopeless() -> None:
        nonlocal starts
        starts += 1
        raise RuntimeError("сломано насовсем")

    monkey = app_module.EfiApp.__new__(app_module.EfiApp)
    original_delay = app_module._SERVICE_RESTART_BASE_DELAY
    app_module._SERVICE_RESTART_BASE_DELAY = 0.0001
    try:
        await asyncio.wait_for(monkey._spawn_supervised(hopeless, name="hopeless"), timeout=5)
    finally:
        app_module._SERVICE_RESTART_BASE_DELAY = original_delay

    assert starts == app_module._MAX_SERVICE_RESTARTS + 1


async def test_a_service_that_finishes_cleanly_is_not_restarted() -> None:
    """Завершившаяся сама служба — это не сбой, перезапускать её незачем."""
    import efi.app as app_module

    starts = 0

    async def once() -> None:
        nonlocal starts
        starts += 1

    monkey = app_module.EfiApp.__new__(app_module.EfiApp)
    await asyncio.wait_for(monkey._spawn_supervised(once, name="once"), timeout=5)

    assert starts == 1
