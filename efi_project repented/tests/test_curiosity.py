"""Тесты для efi.behavior.curiosity: эвристика извлечения тем и жизненный цикл семян любопытства."""

from __future__ import annotations

from pathlib import Path

from efi.behavior.curiosity import CuriosityTracker, SeedStatus, extract_topic
from efi.db.core import Database
from efi.db.models import MIGRATIONS


def _make_tracker(tmp_path: Path) -> CuriosityTracker:
    database = Database(tmp_path / "test.db", migrations=MIGRATIONS)
    return CuriosityTracker(database)


def test_extract_topic_detects_what_is_x() -> None:
    assert extract_topic("а что такое квантовый компьютер, если по-простому?") == "квантовый компьютер"


def test_extract_topic_detects_have_you_heard_about() -> None:
    assert extract_topic("слышала про новый релиз генту?") == "новый релиз генту"


def test_extract_topic_detects_why_question() -> None:
    assert extract_topic("почему кот всегда падает на лапы") == "кот всегда падает на лапы"


def test_extract_topic_returns_none_without_trigger() -> None:
    assert extract_topic("сегодня было скучно на работе") is None


def test_extract_topic_trims_trailing_punctuation() -> None:
    topic = extract_topic("кстати, что такое протокол MTProto, читал недавно.")
    assert topic == "протокол MTProto"


async def test_consider_message_stores_pending_seed(tmp_path: Path) -> None:
    tracker = _make_tracker(tmp_path)
    seed = await tracker.consider_message(42, "что такое race condition?")
    assert seed is not None
    assert seed.topic == "race condition"
    assert seed.source_chat_id == 42
    assert seed.status == SeedStatus.PENDING
    assert seed.weight > 0.5  # вопросительный знак даёт бонус к весу


async def test_consider_message_returns_none_without_trigger(tmp_path: Path) -> None:
    tracker = _make_tracker(tmp_path)
    seed = await tracker.consider_message(42, "ну норм, погнали дальше")
    assert seed is None


async def test_pick_top_pending_prefers_higher_weight(tmp_path: Path) -> None:
    tracker = _make_tracker(tmp_path)
    await tracker.consider_message(1, "слышал про новый форк ядра линукс")  # без "?", вес ниже
    await tracker.consider_message(2, "что такое eBPF?")  # с "?", вес выше

    top = await tracker.pick_top_pending()
    assert top is not None
    assert top.topic == "eBPF"


async def test_pick_top_pending_ignores_researched_seeds(tmp_path: Path) -> None:
    tracker = _make_tracker(tmp_path)
    seed = await tracker.consider_message(1, "что такое eBPF?")
    assert seed is not None
    await tracker.mark_researched(seed.id)

    assert await tracker.pick_top_pending() is None


async def test_pick_top_pending_returns_none_when_empty(tmp_path: Path) -> None:
    tracker = _make_tracker(tmp_path)
    assert await tracker.pick_top_pending() is None
