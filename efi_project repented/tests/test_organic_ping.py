"""Тесты для efi.behavior.organic_ping: пинг по важной находке и буст affinity на ответ собеседника."""

from __future__ import annotations

from pathlib import Path

from efi.behavior.affinity import AffinitySnapshot, AffinityTracker
from efi.behavior.life_engine import InformedThought
from efi.behavior.organic_ping import OrganicPingGenerator
from efi.db.core import Database
from efi.db.models import MIGRATIONS
from efi.notifications.manager import NotificationManager
from efi.notifications.schemas import NotificationType


def _make_generator(
    tmp_path: Path, *, importance_threshold: float = 0.6
) -> tuple[OrganicPingGenerator, NotificationManager, AffinityTracker]:
    database = Database(tmp_path / "test.db", migrations=MIGRATIONS)
    manager = NotificationManager(worker_count=1)
    affinity = AffinityTracker(database)
    generator = OrganicPingGenerator(manager, affinity, importance_threshold=importance_threshold)
    return generator, manager, affinity


def _thought(**overrides: object) -> InformedThought:
    defaults: dict[str, object] = dict(
        seed_id=1, topic="eBPF", source_chat_id=42, finding="это база для трассировки ядра", weight=0.7
    )
    defaults.update(overrides)
    return InformedThought(**defaults)  # type: ignore[arg-type]


async def test_notify_queues_ping_for_important_thought(tmp_path: Path) -> None:
    generator, manager, _affinity = _make_generator(tmp_path)
    await generator.notify(_thought())

    assert manager.qsize() == 1
    notification = await manager.get(manager.worker_index_for("chat:42"))
    assert notification.type == NotificationType.SPONTANEOUS_PING
    assert notification.chat_id == 42
    assert notification.payload["seed_id"] == 1
    assert notification.payload["topic"] == "eBPF"
    assert "eBPF" in notification.payload["reason"]


async def test_notify_skips_thought_below_importance_threshold(tmp_path: Path) -> None:
    generator, manager, _affinity = _make_generator(tmp_path, importance_threshold=0.6)
    await generator.notify(_thought(weight=0.4))
    assert manager.qsize() == 0


async def test_notify_skips_thought_without_source_chat(tmp_path: Path) -> None:
    generator, manager, _affinity = _make_generator(tmp_path)
    await generator.notify(_thought(source_chat_id=None))
    assert manager.qsize() == 0


async def test_handle_reply_boosts_affinity_after_ping(tmp_path: Path) -> None:
    generator, _manager, affinity = _make_generator(tmp_path)
    await generator.notify(_thought(source_chat_id=42))

    before = await affinity.get_snapshot(42)
    await generator.handle_reply(42)
    after = await affinity.get_snapshot(42)

    assert after.affinity > before.affinity
    assert after.respect_level > before.respect_level


async def test_handle_reply_is_noop_without_pending_ping(tmp_path: Path) -> None:
    generator, _manager, affinity = _make_generator(tmp_path)
    await generator.handle_reply(999)
    assert await affinity.get_snapshot(999) == AffinitySnapshot()


async def test_handle_reply_consumes_pending_ping_once(tmp_path: Path) -> None:
    generator, _manager, affinity = _make_generator(tmp_path)
    await generator.notify(_thought(source_chat_id=42))

    await generator.handle_reply(42)
    snapshot_after_first_reply = await affinity.get_snapshot(42)
    await generator.handle_reply(42)  # второй ответ — буст уже не должен применяться повторно
    snapshot_after_second_reply = await affinity.get_snapshot(42)

    assert snapshot_after_first_reply == snapshot_after_second_reply
