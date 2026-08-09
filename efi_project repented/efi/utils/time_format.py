"""
efi/utils/time_format.py

Форматирование времени для промптов и логов — "N часов назад" вместо сырых
timestamp'ов, которые сложнее интерпретировать и модели, и человеку,
читающему логи.
"""

from __future__ import annotations

from datetime import UTC, datetime


def time_ago(moment: datetime, *, now: datetime | None = None) -> str:
    """
    Человекочитаемая разница во времени: "только что", "5 мин. назад",
    "3 ч. назад", "2 дн. назад" и т.п.

    `moment` должен быть timezone-aware. Naive datetime намеренно не
    поддерживается молча (трактовать его как UTC можно было бы, но это легко
    маскирует реальную ошибку в вызывающем коде) — на несогласованность
    лучше упасть явно, чем тихо посчитать неправильно.
    """
    if moment.tzinfo is None:
        raise ValueError("time_ago() требует timezone-aware datetime")

    reference = now or datetime.now(UTC)
    delta_seconds = (reference - moment).total_seconds()

    if delta_seconds < 0:
        return "в будущем"
    if delta_seconds < 60:
        return "только что"
    if delta_seconds < 3600:
        return f"{int(delta_seconds // 60)} мин. назад"
    if delta_seconds < 86400:
        return f"{int(delta_seconds // 3600)} ч. назад"

    days = int(delta_seconds // 86400)
    if days < 30:
        return f"{days} дн. назад"
    months = days // 30
    if months < 12:
        return f"{months} мес. назад"
    return f"{days // 365} г. назад"


def format_past_hours(hours: float) -> str:
    """Форматирует продолжительность в часах человекочитаемо: "меньше часа", "3 ч.", "1.5 дн." и т.п."""
    if hours < 1:
        return "меньше часа"
    if hours < 24:
        return f"{int(hours)} ч."
    return f"{hours / 24:.1f} дн."


__all__ = ["time_ago", "format_past_hours"]
