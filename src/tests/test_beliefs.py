"""Тесты для efi.memory.beliefs: хранилище убеждений и дешёвый поиск релевантности без LLM."""

from __future__ import annotations

from pathlib import Path

from efi.db.core import Database
from efi.db.models import MIGRATIONS
from efi.memory.beliefs import STRONG_BELIEF_THRESHOLD, BeliefStore


def _make_store(tmp_path: Path) -> BeliefStore:
    database = Database(tmp_path / "test.db", migrations=MIGRATIONS)
    return BeliefStore(database)


async def test_upsert_and_get_roundtrip(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    await store.upsert("vim vs vscode", "vim лучше для быстрого редактирования", confidence_score=0.9)

    belief = await store.get("vim vs vscode")
    assert belief is not None
    assert belief.stance == "vim лучше для быстрого редактирования"
    assert belief.confidence_score == 0.9


async def test_upsert_clamps_confidence(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    await store.upsert("тема", "позиция", confidence_score=5.0)
    belief = await store.get("тема")
    assert belief is not None
    assert belief.confidence_score == 1.0


async def test_upsert_overwrites_existing_topic(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    await store.upsert("линукс vs windows", "линукс удобнее", confidence_score=0.6)
    await store.upsert("линукс vs windows", "линукс однозначно лучше", confidence_score=0.95)

    belief = await store.get("линукс vs windows")
    assert belief is not None
    assert belief.stance == "линукс однозначно лучше"
    assert belief.confidence_score == 0.95
    assert len(await store.all_beliefs()) == 1


async def test_find_relevant_matches_by_word_overlap(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    await store.upsert("gran turismo 5 физика", "аркадная, но всё равно кайфовая", confidence_score=0.8)
    await store.upsert("аниме про роботов", "переоценённый жанр", confidence_score=0.5)

    results = await store.find_relevant("расскажи, что думаешь про физику в gran turismo")
    assert [belief.topic for belief in results] == ["gran turismo 5 физика"]


async def test_find_relevant_ranks_by_overlap_then_confidence(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    await store.upsert("python async", "асинхронность это база", confidence_score=0.4)
    await store.upsert("python async паттерны", "лучше явные корутины", confidence_score=0.9)

    results = await store.find_relevant("объясни python async паттерны в деталях", limit=2)
    assert results[0].topic == "python async паттерны"


async def test_find_relevant_returns_empty_without_overlap(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    await store.upsert("phonk музыка", "лучший жанр для гонок", confidence_score=0.7)

    results = await store.find_relevant("какая сегодня погода")
    assert results == []


async def test_delete_removes_belief(tmp_path: Path) -> None:
    store = _make_store(tmp_path)
    await store.upsert("тема", "позиция", confidence_score=0.5)
    await store.delete("тема")
    assert await store.get("тема") is None


def test_strong_belief_threshold_is_high_bar() -> None:
    # Эпистемическая инерция должна включаться только для реально укоренившихся
    # убеждений, не для лёгких склонностей — регрессия на этот порог сломала
    # бы поведение из efi.prompts.builder._resolve_mood незаметно.
    assert STRONG_BELIEF_THRESHOLD == 0.7
