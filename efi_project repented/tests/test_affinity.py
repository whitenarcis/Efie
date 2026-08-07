"""Тесты для efi.behavior.affinity: эвристика классификации реплик и трекер близости/уважения."""

from __future__ import annotations

from pathlib import Path

from efi.behavior.affinity import (
    AffinitySnapshot,
    AffinityTracker,
    MessageKind,
    classify_message,
)
from efi.db.core import Database
from efi.db.models import MIGRATIONS


def _make_tracker(tmp_path: Path) -> AffinityTracker:
    database = Database(tmp_path / "test.db", migrations=MIGRATIONS)
    return AffinityTracker(database)


def test_classify_message_detects_trolling_by_keyword() -> None:
    assert classify_message("ты тупая, отстань") == MessageKind.TROLLING


def test_classify_message_detects_shouting_as_trolling() -> None:
    assert classify_message("ДА ЗАМОЛЧИ УЖЕ НАКОНЕЦ") == MessageKind.TROLLING


def test_classify_message_detects_deep_tech() -> None:
    text = (
        "смотри вот код: ```async def handler(): await something()``` тут гонка потоков и "
        "непонятная архитектура, надо рефакторить этот кусок бэкенда полностью"
    )
    assert classify_message(text) == MessageKind.DEEP_TECH


def test_classify_message_short_technical_mention_is_not_deep_tech() -> None:
    # Короткое упоминание бага само по себе — не "глубокий тех-дискусс".
    assert classify_message("баг нашёл") == MessageKind.NEUTRAL


def test_classify_message_detects_phatic_greeting() -> None:
    assert classify_message("привет") == MessageKind.PHATIC
    assert classify_message("как дела") == MessageKind.PHATIC


def test_classify_message_defaults_to_neutral() -> None:
    text = "расскажи, что думаешь про новый фильм, который вчера смотрели вместе"
    assert classify_message(text) == MessageKind.NEUTRAL


def test_snapshot_social_distance_label() -> None:
    close = AffinitySnapshot(affinity=0.8, respect_level=0.7)
    distant = AffinitySnapshot(affinity=0.3, respect_level=0.2)
    assert close.social_distance_label == "close_peer"
    assert distant.social_distance_label == "acquaintance"


async def test_get_snapshot_defaults_when_unknown_chat(tmp_path: Path) -> None:
    tracker = _make_tracker(tmp_path)
    snapshot = await tracker.get_snapshot(12345)
    assert snapshot == AffinitySnapshot()


async def test_record_message_trolling_lowers_respect(tmp_path: Path) -> None:
    tracker = _make_tracker(tmp_path)
    before = await tracker.get_snapshot(1)
    after = await tracker.record_message(1, "ты тупая нейронка")
    assert after.respect_level < before.respect_level
    assert after.affinity < before.affinity


async def test_record_message_deep_tech_raises_respect(tmp_path: Path) -> None:
    tracker = _make_tracker(tmp_path)
    before = await tracker.get_snapshot(1)
    text = (
        "давай обсудим архитектуру этого сервиса, тут ``` async def foo(): ...``` и явная гонка "
        "потоков в бэкенде, надо рефакторить runtime поведение"
    )
    after = await tracker.record_message(1, text)
    assert after.respect_level > before.respect_level


async def test_record_message_persists_across_tracker_instances(tmp_path: Path) -> None:
    db_path = tmp_path / "shared.db"
    database = Database(db_path, migrations=MIGRATIONS)
    tracker = AffinityTracker(database)
    await tracker.record_message(7, "ты тупая нейронка")

    fresh_tracker = AffinityTracker(Database(db_path, migrations=MIGRATIONS))
    snapshot = await fresh_tracker.get_snapshot(7)
    assert snapshot.respect_level < AffinitySnapshot().respect_level


async def test_snapshot_values_stay_within_bounds(tmp_path: Path) -> None:
    tracker = _make_tracker(tmp_path)
    for _ in range(50):
        await tracker.record_message(1, "ты тупая нейронка заткнись")
    snapshot = await tracker.get_snapshot(1)
    assert 0.0 <= snapshot.affinity <= 1.0
    assert 0.0 <= snapshot.respect_level <= 1.0
