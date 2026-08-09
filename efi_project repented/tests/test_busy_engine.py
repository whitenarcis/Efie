"""Тесты для efi.behavior.busy_engine: чистая арифметика ignore_delay + асинхронная обёртка."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from efi.behavior.affinity import AffinitySnapshot, AffinityTracker
from efi.behavior.busy_engine import BusyEngine, _calculate_ignore_delay, _is_active_conversation
from efi.config.schema import BusyEngineSettings
from efi.db.core import Database
from efi.db.models import MIGRATIONS
from efi.memory.working_memory import WorkingMemory, WorkingMemorySnapshot

_SETTINGS = BusyEngineSettings(
    base_delay_min_seconds=2.0,
    base_delay_max_seconds=20.0,
    research_busy_multiplier=2.5,
    low_energy_extra_seconds=25.0,
    high_affinity_discount_seconds=8.0,
    min_delay_seconds=0.5,
    max_delay_seconds=90.0,
)


class _FakeLifeEngine:
    def __init__(self, is_researching: bool) -> None:
        self.is_researching = is_researching


def _make_busy_engine(tmp_path: Path, *, is_researching: bool = False) -> BusyEngine:
    database = Database(tmp_path / "test.db", migrations=MIGRATIONS)
    working_memory = WorkingMemory(tmp_path / "working_memory.json")
    affinity = AffinityTracker(database)
    return BusyEngine(working_memory, affinity, _FakeLifeEngine(is_researching), _SETTINGS)


def test_calculate_ignore_delay_stays_within_bounds() -> None:
    zero_affinity = AffinitySnapshot(affinity=0.0, respect_level=0.0)
    for _ in range(200):
        delay = _calculate_ignore_delay(
            is_researching=True, energy=0.0, affinity=zero_affinity, is_active_conversation=False, settings=_SETTINGS
        )
        assert _SETTINGS.min_delay_seconds <= delay <= _SETTINGS.max_delay_seconds


def test_calculate_ignore_delay_low_energy_increases_delay() -> None:
    neutral_affinity = AffinitySnapshot(affinity=0.5, respect_level=0.5)
    low_energy_delays = [
        _calculate_ignore_delay(
            is_researching=False, energy=0.0, affinity=neutral_affinity, is_active_conversation=False,
            settings=_SETTINGS,
        )
        for _ in range(50)
    ]
    high_energy_delays = [
        _calculate_ignore_delay(
            is_researching=False, energy=1.0, affinity=neutral_affinity, is_active_conversation=False,
            settings=_SETTINGS,
        )
        for _ in range(50)
    ]
    assert min(low_energy_delays) > max(high_energy_delays) - _SETTINGS.low_energy_extra_seconds
    assert sum(low_energy_delays) / len(low_energy_delays) > sum(high_energy_delays) / len(high_energy_delays)


def test_calculate_ignore_delay_high_affinity_decreases_delay() -> None:
    close = AffinitySnapshot(affinity=1.0, respect_level=1.0)
    distant = AffinitySnapshot(affinity=0.0, respect_level=0.0)
    close_delays = [
        _calculate_ignore_delay(
            is_researching=False, energy=0.7, affinity=close, is_active_conversation=False, settings=_SETTINGS
        )
        for _ in range(50)
    ]
    distant_delays = [
        _calculate_ignore_delay(
            is_researching=False, energy=0.7, affinity=distant, is_active_conversation=False, settings=_SETTINGS
        )
        for _ in range(50)
    ]
    assert sum(close_delays) / len(close_delays) < sum(distant_delays) / len(distant_delays)


def test_calculate_ignore_delay_researching_increases_upper_bound() -> None:
    neutral_affinity = AffinitySnapshot(affinity=0.5, respect_level=0.5)
    idle_delays = [
        _calculate_ignore_delay(
            is_researching=False, energy=0.7, affinity=neutral_affinity, is_active_conversation=False,
            settings=_SETTINGS,
        )
        for _ in range(200)
    ]
    researching_delays = [
        _calculate_ignore_delay(
            is_researching=True, energy=0.7, affinity=neutral_affinity, is_active_conversation=False,
            settings=_SETTINGS,
        )
        for _ in range(200)
    ]
    assert max(researching_delays) > max(idle_delays)


async def test_compute_ignore_delay_reads_energy_and_affinity(tmp_path: Path) -> None:
    engine = _make_busy_engine(tmp_path)
    delay = await engine.compute_ignore_delay(42)
    assert _SETTINGS.min_delay_seconds <= delay <= _SETTINGS.max_delay_seconds


async def test_compute_ignore_delay_without_chat_id_uses_default_affinity(tmp_path: Path) -> None:
    engine = _make_busy_engine(tmp_path)
    delay = await engine.compute_ignore_delay(None)
    assert _SETTINGS.min_delay_seconds <= delay <= _SETTINGS.max_delay_seconds


async def test_compute_ignore_delay_uses_working_memory_energy(tmp_path: Path) -> None:
    database = Database(tmp_path / "test.db", migrations=MIGRATIONS)
    working_memory = WorkingMemory(tmp_path / "working_memory.json")
    await working_memory.save(WorkingMemorySnapshot(energy=0.0))
    affinity = AffinityTracker(database)
    engine = BusyEngine(working_memory, affinity, _FakeLifeEngine(False), _SETTINGS)

    delays = [await engine.compute_ignore_delay(1) for _ in range(20)]
    assert min(delays) >= _SETTINGS.base_delay_min_seconds


# -- регрессия: раньше полная ignore_delay считалась на КАЖДОЕ сообщение, -----
# -- даже посреди уже идущего быстрого диалога ("выходит из чата после  -----
# -- каждого сообщения") -----------------------------------------------------


def test_is_active_conversation_true_within_window() -> None:
    recent = datetime.now(timezone.utc) - timedelta(seconds=10)
    assert _is_active_conversation(recent, _SETTINGS) is True


def test_is_active_conversation_false_outside_window() -> None:
    old = datetime.now(timezone.utc) - timedelta(hours=6)
    assert _is_active_conversation(old, _SETTINGS) is False


def test_is_active_conversation_false_without_history() -> None:
    assert _is_active_conversation(None, _SETTINGS) is False


def test_active_conversation_uses_tiny_jitter_not_full_delay() -> None:
    neutral_affinity = AffinitySnapshot(affinity=0.5, respect_level=0.5)
    for _ in range(200):
        delay = _calculate_ignore_delay(
            is_researching=False, energy=0.0, affinity=neutral_affinity, is_active_conversation=True,
            settings=_SETTINGS,
        )
        low, high = _SETTINGS.active_conversation_delay_min_seconds, _SETTINGS.active_conversation_delay_max_seconds
        assert low <= delay <= high


def test_active_conversation_still_gets_full_delay_when_genuinely_researching() -> None:
    neutral_affinity = AffinitySnapshot(affinity=0.5, respect_level=0.5)
    delays = [
        _calculate_ignore_delay(
            is_researching=True, energy=0.7, affinity=neutral_affinity, is_active_conversation=True,
            settings=_SETTINGS,
        )
        for _ in range(200)
    ]
    # Реально занята — полный расчёт применяется, даже если разговор активен;
    # хотя бы один замер должен выйти за пределы крошечного джиттера.
    assert max(delays) > _SETTINGS.active_conversation_delay_max_seconds


class _FakeLastMessageSource:
    def __init__(self, last_message_at: datetime | None) -> None:
        self._last_message_at = last_message_at

    async def get_last_message_at(self, chat_id: int) -> datetime | None:
        return self._last_message_at


async def test_compute_ignore_delay_is_near_instant_mid_active_conversation(tmp_path: Path) -> None:
    database = Database(tmp_path / "test.db", migrations=MIGRATIONS)
    working_memory = WorkingMemory(tmp_path / "working_memory.json")
    affinity = AffinityTracker(database)
    recent = datetime.now(timezone.utc) - timedelta(seconds=5)
    engine = BusyEngine(
        working_memory, affinity, _FakeLifeEngine(False), _SETTINGS, last_message_source=_FakeLastMessageSource(recent)
    )

    delays = [await engine.compute_ignore_delay(1) for _ in range(20)]
    assert max(delays) <= _SETTINGS.active_conversation_delay_max_seconds


async def test_compute_ignore_delay_full_delay_for_first_message_after_a_gap(tmp_path: Path) -> None:
    database = Database(tmp_path / "test.db", migrations=MIGRATIONS)
    working_memory = WorkingMemory(tmp_path / "working_memory.json")
    affinity = AffinityTracker(database)
    long_ago = datetime.now(timezone.utc) - timedelta(hours=6)
    engine = BusyEngine(
        working_memory, affinity, _FakeLifeEngine(False), _SETTINGS,
        last_message_source=_FakeLastMessageSource(long_ago),
    )

    delays = [await engine.compute_ignore_delay(1) for _ in range(20)]
    assert min(delays) >= _SETTINGS.base_delay_min_seconds
