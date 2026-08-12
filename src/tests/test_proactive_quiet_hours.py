"""
Тесты на то, что тихие часы (efi.behavior.quiet_hours) реально подавляют все
три проактивных пути: SpontaneousPingScheduler, OrganicPingGenerator,
SilenceMonitor. Регрессия на дефект: раньше ни один из них вообще не смотрел
на время суток, из-за чего Эфи писала первой в 5 и 7 утра наравне с днём.

Интервал тихих часов строится вокруг текущего часа теста (а не захардкожен),
чтобы тест был детерминирован независимо от времени запуска CI.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from efi.behavior.affinity import AffinityTracker
from efi.behavior.life_engine import InformedThought
from efi.behavior.organic_ping import OrganicPingGenerator
from efi.behavior.ping_reason import PingReasonBuilder
from efi.behavior.silence_monitor import SilenceMonitor
from efi.behavior.spontaneous_ping import SpontaneousPingScheduler
from efi.config.schema import QuietHoursSettings
from efi.db.core import Database
from efi.db.models import MIGRATIONS
from efi.notifications.manager import NotificationManager


def _quiet_hours_covering_now() -> QuietHoursSettings:
    """Часовое окно, которое заведомо включает текущий час — не важно, когда запущен тест."""
    hour = datetime.now().hour
    return QuietHoursSettings(enabled=True, start_hour=hour, end_hour=(hour + 1) % 24)


def _quiet_hours_never() -> QuietHoursSettings:
    """start_hour == end_hour — тихих часов нет никогда, вне зависимости от текущего времени."""
    return QuietHoursSettings(enabled=True, start_hour=5, end_hour=5)


async def test_spontaneous_ping_skips_candidates_during_quiet_hours() -> None:
    manager = NotificationManager(worker_count=1)

    async def candidates() -> list[int]:
        return [1]

    scheduler = SpontaneousPingScheduler(
        manager, candidates, ping_probability=1.0, quiet_hours=_quiet_hours_covering_now()
    )
    await scheduler._maybe_ping_candidates()
    assert manager.qsize() == 0


async def test_spontaneous_ping_fires_outside_quiet_hours() -> None:
    """
    Вне тихих часов пинг проходит — но только когда есть с чем прийти:
    инициатива без повода больше не отправляется вовсе
    (см. tests/test_initiative.py).
    """
    manager = NotificationManager(worker_count=1)

    async def candidates() -> list[int]:
        return [1]

    async def thought() -> str | None:
        return "плёночные сканеры до сих пор быстрее половины современных"

    scheduler = SpontaneousPingScheduler(
        manager,
        candidates,
        ping_probability=1.0,
        quiet_hours=_quiet_hours_never(),
        reasons=PingReasonBuilder(incubated_thought_provider=thought),
    )
    await scheduler._maybe_ping_candidates()
    assert manager.qsize() == 1


async def test_organic_ping_skips_during_quiet_hours(tmp_path: Path) -> None:
    manager = NotificationManager(worker_count=1)
    database = Database(tmp_path / "test.db", migrations=MIGRATIONS)
    affinity = AffinityTracker(database)
    generator = OrganicPingGenerator(manager, affinity, quiet_hours=_quiet_hours_covering_now())

    thought = InformedThought(seed_id=1, topic="x", source_chat_id=42, finding="y", weight=0.9)
    await generator.notify(thought)
    assert manager.qsize() == 0


async def test_silence_monitor_skips_ping_during_quiet_hours() -> None:
    manager = NotificationManager(worker_count=1)
    monitor = SilenceMonitor(
        manager, silence_threshold=timedelta(seconds=0), quiet_hours=_quiet_hours_covering_now()
    )
    monitor._last_activity[1] = datetime.now(UTC) - timedelta(hours=1)

    await monitor._check_silence()
    assert manager.qsize() == 0


async def test_silence_monitor_pings_outside_quiet_hours() -> None:
    manager = NotificationManager(worker_count=1)
    monitor = SilenceMonitor(manager, silence_threshold=timedelta(seconds=0), quiet_hours=_quiet_hours_never())
    monitor._last_activity[1] = datetime.now(UTC) - timedelta(hours=1)

    await monitor._check_silence()
    assert manager.qsize() == 1
