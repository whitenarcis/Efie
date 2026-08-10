"""
Тесты режима уточнения (efi.behavior.ambiguity) и конвейера приёма знаний
(efi.memory.ingest).

Сценарий, ради которого всё это писалось: «скинь это в Феникс», а «Феникс» —
и рабочий проект, и кот. Пока код молча брал лучшего по счёту, ошибка не
проявлялась сразу — она оседала в памяти как факт и всплывала через неделю,
когда Эфи начинала уверенно говорить чушь. Испорченную запись потом не
отличить от настоящей: у неё та же структура и та же уверенность.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from efi.behavior.ambiguity import (
    AmbiguityDetector,
    EntityCandidate,
    PendingClarifications,
    render_clarification,
)
from efi.db.core import Database
from efi.db.models import MIGRATIONS
from efi.memory.dedup import KnowledgeStore
from efi.memory.ingest import MemoryIngestor
from efi.memory.parser import FactCandidate, PerceptionBatch
from efi.memory.validator import FactValidator

_OWNER_ID = 625207005

_PROJECT = EntityCandidate(entity_id="project:phoenix", label="Феникс", score=0.82, hint="рабочий проект")
_CAT = EntityCandidate(entity_id="pet:phoenix", label="Феникс", score=0.80, hint="кот")
_CHAT = EntityCandidate(entity_id="chat:-100", label="Феникс", score=0.41, hint="чат")


# -- обнаружение неоднозначности ---------------------------------------------


def test_single_candidate_resolves() -> None:
    resolution = AmbiguityDetector().resolve("Феникс", [_PROJECT])

    assert resolution.is_resolved
    assert resolution.entity_id == "project:phoenix"
    assert not resolution.needs_clarification


def test_clear_leader_resolves_without_asking() -> None:
    """Разрыв больше погрешности — спрашивать не о чем, вопрос ради вопроса раздражает."""
    weak = EntityCandidate(entity_id="pet:phoenix", label="Феникс", score=0.42, hint="кот")
    resolution = AmbiguityDetector().resolve("Феникс", [_PROJECT, weak])

    assert resolution.entity_id == "project:phoenix"


def test_two_close_candidates_trigger_clarification_instead_of_a_guess() -> None:
    resolution = AmbiguityDetector().resolve("Феникс", [_PROJECT, _CAT])

    assert not resolution.is_resolved, "угадывать здесь нельзя — это и портит память"
    assert resolution.needs_clarification
    assert "Феникс" in (resolution.clarification or "")


def test_clarification_mentions_the_distinguishing_hints() -> None:
    """Без подсказок вопрос выродится в «ты про Феникс или про Феникс?»."""
    resolution = AmbiguityDetector().resolve("Феникс", [_PROJECT, _CAT])

    assert "рабочий проект" in (resolution.clarification or "")
    assert "кот" in (resolution.clarification or "")


def test_weak_candidates_are_ignored_entirely() -> None:
    """Совпадение по одной букве — не кандидат; спрашивать про заведомо неподходящее хуже молчания."""
    resolution = AmbiguityDetector().resolve("Феникс", [EntityCandidate("x", "Ф", 0.05)])

    assert resolution.is_unknown
    assert not resolution.needs_clarification


def test_clarification_lists_at_most_three_options() -> None:
    """Список из пяти пунктов в чате читается как анкета, а не как вопрос живого человека."""
    candidates = tuple(
        EntityCandidate(f"e{index}", f"Вариант{index}", 0.8, f"подсказка{index}") for index in range(5)
    )
    question = render_clarification("Феникс", candidates)

    assert question.count("подсказка") == 3


def test_same_mention_always_asks_the_same_way() -> None:
    """Иначе повторный вопрос читается как заедающая пластинка, а не как уточнение."""
    first = render_clarification("Феникс", (_PROJECT, _CAT))
    second = render_clarification("Феникс", (_PROJECT, _CAT))

    assert first == second


# -- реестр незакрытых уточнений ---------------------------------------------


def test_pending_registry_stores_and_returns_the_question() -> None:
    registry = PendingClarifications()
    resolution = AmbiguityDetector().resolve("Феникс", [_PROJECT, _CAT])
    registry.remember(42, resolution)

    pending = registry.peek(42)
    assert pending is not None
    assert pending.mention == "Феникс"
    assert len(registry) == 1


def test_answer_closes_the_clarification() -> None:
    registry = PendingClarifications()
    registry.remember(42, AmbiguityDetector().resolve("Феникс", [_PROJECT, _CAT]))

    resolved = registry.resolve_with_answer(42, "да про кота же")

    assert resolved is not None
    assert resolved.entity_id == "pet:phoenix"
    assert registry.peek(42) is None


def test_ambiguous_answer_keeps_the_question_open() -> None:
    """Неопознанный ответ ничем не лучше первоначальной неоднозначности — записывать по нему нельзя."""
    registry = PendingClarifications()
    registry.remember(42, AmbiguityDetector().resolve("Феникс", [_PROJECT, _CAT]))

    assert registry.resolve_with_answer(42, "ну ты поняла") is None
    assert registry.peek(42) is not None


def test_expired_clarification_is_dropped() -> None:
    """Ответ на вчерашний вопрос сопоставится не с тем, что человек имел в виду."""
    registry = PendingClarifications(ttl=timedelta(seconds=0))
    registry.remember(42, AmbiguityDetector().resolve("Феникс", [_PROJECT, _CAT]))

    assert registry.peek(42) is None


def test_registry_is_per_chat() -> None:
    registry = PendingClarifications()
    registry.remember(1, AmbiguityDetector().resolve("Феникс", [_PROJECT, _CAT]))

    assert registry.peek(2) is None


# -- конвейер приёма знаний --------------------------------------------------


class _StubParser:
    """Подставная модель восприятия: отдаёт заранее заданную пачку кандидатов."""

    def __init__(self, batch: PerceptionBatch) -> None:
        self._batch = batch

    async def extract(self, conversation_text: str, *, source: str = "") -> PerceptionBatch:
        return self._batch


def _ingestor(tmp_path: Path, batch: PerceptionBatch) -> tuple[MemoryIngestor, KnowledgeStore]:
    database = Database(tmp_path / "efi.db", migrations=MIGRATIONS)
    store = KnowledgeStore(database)
    ingestor = MemoryIngestor(
        _StubParser(batch),  # type: ignore[arg-type]
        FactValidator(owner_id=_OWNER_ID, now=datetime(2026, 8, 9, tzinfo=UTC)),
        store,
    )
    return ingestor, store


@pytest.fixture
def batch() -> PerceptionBatch:
    return PerceptionBatch(
        candidates=[
            FactCandidate(domain="P", entity="Феникс", attribute="дедлайн", value="в пятницу"),
        ],
        source="chat:1",
    )


async def test_ambiguous_entity_is_asked_about_and_not_written(tmp_path: Path, batch: PerceptionBatch) -> None:
    """Главный сценарий: вместо искажённой записи — один короткий вопрос."""
    ingestor, store = _ingestor(tmp_path, batch)

    result = await ingestor.ingest_conversation(
        "скинь это в Феникс", source="chat:1", chat_id=1, catalog={"феникс": [_PROJECT, _CAT]}
    )

    assert result.needs_clarification
    assert result.stored_count == 0, "неоднозначная сущность не должна попадать в память"
    assert await store.count() == 0
    assert ingestor.pending_clarifications.peek(1) is not None


async def test_unambiguous_entity_is_resolved_and_written(tmp_path: Path, batch: PerceptionBatch) -> None:
    ingestor, store = _ingestor(tmp_path, batch)

    result = await ingestor.ingest_conversation(
        "скинь это в Феникс", source="chat:1", chat_id=1, catalog={"феникс": [_PROJECT, _CHAT]}
    )

    assert not result.needs_clarification
    assert len(result.created) == 1
    assert result.created[0].fact.entity_id == "project:phoenix"


async def test_resolved_entity_gets_a_matching_hash(tmp_path: Path, batch: PerceptionBatch) -> None:
    """
    Хэш включает entity_id: оставить прежний после разрешения значило бы
    поселить в базе запись, чей хэш не соответствует содержимому, — и
    дедупликация перестала бы работать именно на таких фактах.
    """
    ingestor, store = _ingestor(tmp_path, batch)
    await ingestor.ingest_conversation(
        "скинь это в Феникс", source="chat:1", chat_id=1, catalog={"феникс": [_PROJECT, _CHAT]}
    )
    await ingestor.ingest_conversation(
        "скинь это в Феникс", source="chat:1", chat_id=1, catalog={"феникс": [_PROJECT, _CHAT]}
    )

    facts = await store.recall(limit=5)
    assert len(facts) == 1
    assert facts[0].occurrence_count == 2


async def test_without_a_catalog_nothing_is_ambiguous(tmp_path: Path, batch: PerceptionBatch) -> None:
    """Неоднозначность бывает только там, где есть несколько известных претендентов."""
    ingestor, _store = _ingestor(tmp_path, batch)

    result = await ingestor.ingest_conversation("скинь это в Феникс", source="chat:1", chat_id=1)

    assert not result.needs_clarification
    assert len(result.created) == 1


async def test_rejected_candidates_are_reported_and_journalled(tmp_path: Path) -> None:
    bad = PerceptionBatch(
        candidates=[FactCandidate(domain="P", entity="Рома", attribute="last_novelized_at", value="вчера")],
        source="chat:1",
    )
    ingestor, store = _ingestor(tmp_path, bad)

    result = await ingestor.ingest_conversation("что-то", source="chat:1")

    assert result.stored_count == 0
    assert len(result.rejected) == 1
    assert await store.count() == 0


async def test_parse_failure_is_reported_without_touching_memory(tmp_path: Path) -> None:
    failed = PerceptionBatch(parse_error="невалидный JSON", source="chat:1")
    ingestor, store = _ingestor(tmp_path, failed)

    result = await ingestor.ingest_conversation("что-то", source="chat:1")

    assert result.parse_error == "невалидный JSON"
    assert await store.count() == 0
