"""Тесты для efi.behavior.busy_engine: чистая арифметика ignore_delay + асинхронная обёртка."""

from __future__ import annotations

from pathlib import Path

from efi.behavior.affinity import AffinitySnapshot, AffinityTracker
from efi.behavior.busy_engine import BusyEngine, _calculate_ignore_delay
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
        delay = _calculate_ignore_delay(is_researching=True, energy=0.0, affinity=zero_affinity, settings=_SETTINGS)
        assert _SETTINGS.min_delay_seconds <= delay <= _SETTINGS.max_delay_seconds


def test_calculate_ignore_delay_low_energy_increases_delay() -> None:
    neutral_affinity = AffinitySnapshot(affinity=0.5, respect_level=0.5)
    low_energy_delays = [
        _calculate_ignore_delay(is_researching=False, energy=0.0, affinity=neutral_affinity, settings=_SETTINGS)
        for _ in range(50)
    ]
    high_energy_delays = [
        _calculate_ignore_delay(is_researching=False, energy=1.0, affinity=neutral_affinity, settings=_SETTINGS)
        for _ in range(50)
    ]
    assert min(low_energy_delays) > max(high_energy_delays) - _SETTINGS.low_energy_extra_seconds
    assert sum(low_energy_delays) / len(low_energy_delays) > sum(high_energy_delays) / len(high_energy_delays)


def test_calculate_ignore_delay_high_affinity_decreases_delay() -> None:
    close = AffinitySnapshot(affinity=1.0, respect_level=1.0)
    distant = AffinitySnapshot(affinity=0.0, respect_level=0.0)
    close_delays = [
        _calculate_ignore_delay(is_researching=False, energy=0.7, affinity=close, settings=_SETTINGS) for _ in range(50)
    ]
    distant_delays = [
        _calculate_ignore_delay(is_researching=False, energy=0.7, affinity=distant, settings=_SETTINGS)
        for _ in range(50)
    ]
    assert sum(close_delays) / len(close_delays) < sum(distant_delays) / len(distant_delays)


def test_calculate_ignore_delay_researching_increases_upper_bound() -> None:
    neutral_affinity = AffinitySnapshot(affinity=0.5, respect_level=0.5)
    idle_delays = [
        _calculate_ignore_delay(is_researching=False, energy=0.7, affinity=neutral_affinity, settings=_SETTINGS)
        for _ in range(200)
    ]
    researching_delays = [
        _calculate_ignore_delay(is_researching=True, energy=0.7, affinity=neutral_affinity, settings=_SETTINGS)
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
