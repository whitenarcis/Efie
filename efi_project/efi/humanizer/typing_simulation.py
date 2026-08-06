"""
efi/humanizer/typing_simulation.py

Симуляция набора текста человеком — расчёт и (опционально) фактическое
ожидание задержки перед отправкой сообщения, на основе длины текста и
WPM-диапазона из HumanizerSettings. Перенос calculate_typing_delay из текущей
реализации Эфи, переведённый на конфигурируемый WPM (HumanizerSettings.
characters_per_second_range, см. efi/config/schema.py) вместо захардкоженных
cps-констант, и обёрнутый в async def.
"""

from __future__ import annotations

import asyncio
import random

from efi.config.schema import HumanizerSettings


def calculate_typing_delay(text: str, settings: HumanizerSettings) -> float:
    """
    Считает задержку в секундах: случайная пауза "на подумать" + время набора
    текста по случайной скорости из диапазона WPM, всё вместе зажатое в
    [typing_delay_min_seconds, typing_delay_max_seconds].

    Чистая функция без I/O — специально отделена от фактического ожидания
    (см. simulate_typing_delay), чтобы расчёт можно было тестировать и
    использовать (например, для оценки "когда примерно придёт ответ" в
    дашборде) без побочных эффектов.
    """
    cps_min, cps_max = settings.characters_per_second_range()
    chars_per_second = random.uniform(cps_min, cps_max)
    typing_time = len(text) / chars_per_second if chars_per_second > 0 else 0.0

    thinking_pause = random.uniform(
        settings.typing_thinking_pause_min_seconds,
        settings.typing_thinking_pause_max_seconds,
    )

    total_delay = thinking_pause + typing_time
    return min(max(total_delay, settings.typing_delay_min_seconds), settings.typing_delay_max_seconds)


async def simulate_typing_delay(text: str, settings: HumanizerSettings) -> float:
    """
    Считает задержку (calculate_typing_delay) и асинхронно её выжидает —
    asyncio.sleep не блокирует event loop, остальные Worker'ы продолжают
    работать параллельно. Возвращает фактически выжданную задержку (удобно
    для логов/метрик дашборда).
    """
    delay = calculate_typing_delay(text, settings)
    await asyncio.sleep(delay)
    return delay


__all__ = ["calculate_typing_delay", "simulate_typing_delay"]
