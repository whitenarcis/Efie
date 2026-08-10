"""
Тесты семантической дедупликации и счётчика подтверждений
(efi.memory.dedup.KnowledgeStore).

Смысл: человек повторяет одно и то же разными словами, и до дедупликации
память копила синонимы, теряя при этом самое ценное — систематичность.
Двенадцать записей «ложится поздно» и одна запись «ложится поздно, упомянуто
12 раз» состоят из одних и тех же слов, но вторая говорит то, чего первая не
говорит вовсе.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from efi.db.core import Database
from efi.db.models import MIGRATIONS
from efi.llm.schemas import EmbeddingVector
from efi.memory.dedup import KnowledgeStore, StoreAction, render_facts_block
from efi.memory.parser import FactCandidate
from efi.memory.router import MemoryDomain
from efi.memory.validator import FactValidator, ValidatedFact, compute_hash

_OWNER_ID = 625207005


class _ScriptedEmbedder:
    """
    Подставной источник эмбеддингов: заранее заданный вектор на текст.

    Настоящую модель эмбеддингов в тесты тянуть нельзя (сеть, вес, время), а
    проверять нужно именно поведение порога — поэтому векторы задаются руками
    так, чтобы косинус между ними был известен.
    """

    def __init__(self, vectors: dict[str, list[float]], *, default: list[float] | None = None) -> None:
        self._vectors = vectors
        self._default = default
        self.calls: list[str] = []

    async def embed(self, text: str) -> EmbeddingVector | None:
        self.calls.append(text)
        for marker, vector in self._vectors.items():
            if marker in text:
                return vector
        return self._default


def _store(tmp_path: Path, embedder: object | None = None) -> KnowledgeStore:
    database = Database(tmp_path / "efi.db", migrations=MIGRATIONS)
    return KnowledgeStore(database, embedder=embedder)  # type: ignore[arg-type]


def _fact(
    value: str = "монтажёр",
    *,
    attribute: str = "работа",
    entity_id: str = "person:рома",
    domain: MemoryDomain = MemoryDomain.PERSONAL,
) -> ValidatedFact:
    return ValidatedFact(
        domain=domain,
        entity_id=entity_id,
        attribute=attribute,
        value=value,
        confidence=0.9,
        observed_at=datetime.now(UTC),
        source="chat:1",
        normalized_hash=compute_hash(domain, entity_id, attribute, value),
    )


# -- первая ступень: точный хэш ---------------------------------------------


async def test_first_write_creates_a_fact(tmp_path: Path) -> None:
    outcome = await _store(tmp_path).remember(_fact())

    assert outcome.action is StoreAction.CREATED
    assert outcome.fact.occurrence_count == 1


async def test_exact_repeat_increments_the_counter_instead_of_duplicating(tmp_path: Path) -> None:
    store = _store(tmp_path)
    await store.remember(_fact())
    second = await store.remember(_fact())

    assert second.action is StoreAction.REINFORCED
    assert second.fact.occurrence_count == 2
    assert await store.count() == 1


async def test_repeat_is_caught_without_asking_the_embedder(tmp_path: Path) -> None:
    """Буквальный повтор — большинство случаев; платить за него эмбеддингом незачем."""
    embedder = _ScriptedEmbedder({}, default=[1.0, 0.0])
    store = _store(tmp_path, embedder)
    await store.remember(_fact())
    calls_after_first = len(embedder.calls)

    await store.remember(_fact())

    assert len(embedder.calls) == calls_after_first


async def test_case_and_punctuation_do_not_create_a_second_record(tmp_path: Path) -> None:
    store = _store(tmp_path)
    await store.remember(_fact("монтажёр"))
    await store.remember(_fact("Монтажёр."))

    assert await store.count() == 1


async def test_last_seen_at_moves_forward_on_reinforcement(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = await store.remember(_fact())
    second = await store.remember(_fact())

    assert second.fact.last_seen_at >= first.fact.last_seen_at
    assert second.fact.first_seen_at == first.fact.first_seen_at


# -- вторая ступень: косинусное сходство -------------------------------------


async def test_paraphrase_above_threshold_reinforces(tmp_path: Path) -> None:
    """«ложится под утро» и «часто не спит ночами» — один факт, сказанный дважды."""
    embedder = _ScriptedEmbedder(
        {"под утро": [1.0, 0.0, 0.0], "не спит ночами": [0.95, 0.31, 0.0]}
    )
    store = _store(tmp_path, embedder)
    await store.remember(_fact("ложится под утро", attribute="режим_сна"))
    outcome = await store.remember(_fact("часто не спит ночами", attribute="режим_сна"))

    assert outcome.action is StoreAction.REINFORCED
    assert outcome.similarity > 0.88
    assert await store.count() == 1


async def test_different_meaning_below_threshold_creates_a_second_fact(tmp_path: Path) -> None:
    """«любит кофе» и «любит чай» близки по вектору, но склеить их было бы хуже, чем продублировать."""
    embedder = _ScriptedEmbedder({"кофе": [1.0, 0.0, 0.0], "чай": [0.5, 0.86, 0.0]})
    store = _store(tmp_path, embedder)
    await store.remember(_fact("любит кофе", attribute="напитки"))
    outcome = await store.remember(_fact("любит чай", attribute="напитки"))

    assert outcome.action is StoreAction.CREATED
    assert await store.count() == 2


async def test_same_value_for_different_people_is_not_merged(tmp_path: Path) -> None:
    """
    «любит кофе» про Рому и «любит кофе» про Костю семантически неразличимы —
    сравнение по всей базе склеило бы двух людей в одного.
    """
    embedder = _ScriptedEmbedder({}, default=[1.0, 0.0, 0.0])
    store = _store(tmp_path, embedder)
    await store.remember(_fact("любит кофе", attribute="напитки", entity_id="person:рома"))
    outcome = await store.remember(_fact("любит кофе", attribute="напитки", entity_id="person:костя"))

    assert outcome.action is StoreAction.CREATED
    assert await store.count() == 2


async def test_missing_embedder_still_stores_the_fact(tmp_path: Path) -> None:
    """Дедупликация — улучшение, а не условие записи: без эмбеддингов факт всё равно должен сохраниться."""
    store = _store(tmp_path, None)
    outcome = await store.remember(_fact())

    assert outcome.action is StoreAction.CREATED
    assert await store.count() == 1


async def test_broken_embedder_does_not_lose_the_fact(tmp_path: Path) -> None:
    class _BrokenEmbedder:
        async def embed(self, text: str) -> EmbeddingVector | None:
            raise RuntimeError("модель эмбеддингов недоступна")

    store = _store(tmp_path, _BrokenEmbedder())
    outcome = await store.remember(_fact())

    assert outcome.action is StoreAction.CREATED


async def test_dimension_mismatch_is_skipped_not_crashed(tmp_path: Path) -> None:
    """Смена модели эмбеддингов не должна ронять запись — векторы просто несравнимы."""
    store = _store(tmp_path, _ScriptedEmbedder({"первый": [1.0, 0.0, 0.0], "второй": [1.0, 0.0]}))
    await store.remember(_fact("первый факт", attribute="a"))
    outcome = await store.remember(_fact("второй факт", attribute="a"))

    assert outcome.action is StoreAction.CREATED


# -- чтение и подача в промпт ------------------------------------------------


async def test_recall_filters_by_domain_and_entity(tmp_path: Path) -> None:
    store = _store(tmp_path)
    await store.remember(_fact("монтажёр", entity_id="person:рома"))
    await store.remember(_fact("встраиваемая", attribute="тип", entity_id="topic:sqlite", domain=MemoryDomain.COMMON))

    personal = await store.recall(domains=[MemoryDomain.PERSONAL])
    common = await store.recall(domains=[MemoryDomain.COMMON])
    by_entity = await store.recall(entity_ids=["person:рома"])

    assert [fact.value for fact in personal] == ["монтажёр"]
    assert [fact.value for fact in common] == ["встраиваемая"]
    assert [fact.value for fact in by_entity] == ["монтажёр"]


async def test_recall_puts_the_most_confirmed_first(tmp_path: Path) -> None:
    """Когда места в промпте на десяток строк, систематическое важнее случайно упомянутого вчера."""
    store = _store(tmp_path)
    await store.remember(_fact("редко", attribute="разовое"))
    for _ in range(5):
        await store.remember(_fact("постоянно", attribute="систематическое"))

    facts = await store.recall(limit=10)

    assert facts[0].attribute == "систематическое"
    assert facts[0].occurrence_count == 5


async def test_prompt_line_carries_the_occurrence_count(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for _ in range(12):
        await store.remember(_fact("частый полуночник", attribute="режим_сна"))

    facts = await store.recall(limit=5)

    assert facts[0].render_for_prompt() == "[ФАКТ: режим сна: частый полуночник (упомянуто 12 раз)]"


async def test_single_observation_is_not_labelled_with_a_count(tmp_path: Path) -> None:
    """«(упомянуто 1 раз)» сообщает шум вместо смысла."""
    store = _store(tmp_path)
    await store.remember(_fact("монтажёр"))

    facts = await store.recall(limit=5)

    assert facts[0].render_for_prompt() == "[ФАКТ: работа: монтажёр]"


async def test_facts_block_is_empty_without_facts() -> None:
    assert render_facts_block([]) == ""


async def test_facts_block_explains_what_the_numbers_mean(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for _ in range(3):
        await store.remember(_fact())

    block = render_facts_block(await store.recall(limit=5))

    assert "упомянуто 3 раз" in block
    assert "сколько раз это подтверждалось" in block


# -- журнал отбраковки -------------------------------------------------------


async def test_rejections_are_journalled(tmp_path: Path) -> None:
    """Без журнала «не запомнила» и «запомнила чушь» выглядят одинаково: тишиной."""
    database = Database(tmp_path / "efi.db", migrations=MIGRATIONS)
    store = KnowledgeStore(database)
    validator = FactValidator(owner_id=_OWNER_ID)

    outcome = validator.validate(
        FactCandidate(domain="P", entity="Рома", attribute="last_novelized_at", value="вчера")
    )
    await store.record_rejections([outcome], source="chat:1")  # type: ignore[list-item]

    rows = await database.fetch_all("SELECT attribute, reason FROM knowledge_rejections")
    assert len(rows) == 1
    assert rows[0]["attribute"] == "last_novelized_at"
    assert "служебн" in rows[0]["reason"]
