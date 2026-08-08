"""Тесты для efi.behavior.quiet_hours.is_quiet_hours."""

from __future__ import annotations

from datetime import datetime

from efi.behavior.quiet_hours import is_quiet_hours


def _at(hour: int) -> datetime:
    return datetime(2026, 8, 8, hour, 0, 0)


def test_wraps_past_midnight() -> None:
    # 23..8 — типичный ночной интервал, оборачивающийся через полночь.
    assert is_quiet_hours(_at(23), start_hour=23, end_hour=8) is True
    assert is_quiet_hours(_at(5), start_hour=23, end_hour=8) is True
    assert is_quiet_hours(_at(7), start_hour=23, end_hour=8) is True


def test_edges_of_wrapping_interval() -> None:
    assert is_quiet_hours(_at(8), start_hour=23, end_hour=8) is False  # конец интервала не включён
    assert is_quiet_hours(_at(22), start_hour=23, end_hour=8) is False  # ещё до начала


def test_non_wrapping_interval() -> None:
    assert is_quiet_hours(_at(13), start_hour=12, end_hour=14) is True
    assert is_quiet_hours(_at(11), start_hour=12, end_hour=14) is False
    assert is_quiet_hours(_at(14), start_hour=12, end_hour=14) is False


def test_equal_start_and_end_means_never_quiet() -> None:
    for hour in range(24):
        assert is_quiet_hours(_at(hour), start_hour=5, end_hour=5) is False
