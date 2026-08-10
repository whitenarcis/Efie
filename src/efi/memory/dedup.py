"""
efi/memory/dedup.py

Семантическая дедупликация и счётчик подтверждений.

Задача, которую это решает. Человек повторяет одно и то же разными словами:
«опять до трёх сидел», «я сова», «ночью нормально работается». Пока каждая
формулировка становилась отдельной записью, память распухала синонимами, а
главное — теряла самое ценное: СИСТЕМАТИЧНОСТЬ. Двенадцать записей «ложится
поздно» и одна запись «ложится поздно, упомянуто 12 раз» содержат одни и те
же слова, но вторая говорит то, чего первая не говорит вовсе.

Две ступени, дешёвая перед дорогой:
    1. Точный хэш нормализованного факта (efi.memory.validator.compute_hash) —
       ловит буквальный повтор без единого вектора. Обеспечен UNIQUE-индексом
       в схеме, то есть работает даже при гонке двух воркеров.
    2. Косинусное сходство эмбеддингов внутри той же пары (домен, сущность).
       Порог 0.88 подобран как «перефразировка того же самого»; ниже начинают
       склеиваться соседние по теме, но разные факты («любит кофе» / «любит
       чай» дают около 0.85 на многоязычных моделях, и склеить их было бы
       хуже, чем продублировать).

При совпадении новая запись НЕ создаётся: у существующей растёт
`occurrence_count` и обновляется `last_seen_at`. Значение при этом не
перезаписывается — первая формулировка обычно ближе к тому, как человек сам
это сказал, а последующие пересказы моделью дрейфуют.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol

import numpy as np

from efi.db.core import Database
from efi.llm.schemas import EmbeddingVector
from efi.memory.router import MemoryDomain
from efi.memory.validator import Rejection, ValidatedFact

logger = logging.getLogger(__name__)

#: Порог косинусного сходства, выше которого факты считаются одним и тем же.
DEFAULT_SIMILARITY_THRESHOLD = 0.88

#: Сколько существующих фактов сущности сравнивать с новым. Человек не
#: накапливает сотни атрибутов одного вида; ограничение защищает от
#: вырожденного случая, когда сравнение превращается в полный проход по базе.
_MAX_COMPARISON_CANDIDATES = 200


class Embedder(Protocol):
    """
    Источник эмбеддингов. Реализация — efi.memory.rag.RAGMemory.embed.

    Протокол, а не прямая зависимость на RAGMemory: дедупликации нужен ровно
    один метод, и связывать её со всей подсистемой поиска (дневник, TF-IDF,
    роутер LLM) ради него — лишнее. Заодно тесты обходятся без сети.
    """

    async def embed(self, text: str) -> EmbeddingVector | None: ...


class StoreAction(StrEnum):
    """Что произошло с фактом при попытке записи."""

    CREATED = "created"        # новый факт
    REINFORCED = "reinforced"  # совпал с существующим, счётчик подтверждений вырос


@dataclass(slots=True, frozen=True)
class StoredFact:
    """Факт как он лежит в базе."""

    id: int
    domain: MemoryDomain
    entity_id: str
    attribute: str
    value: str
    confidence: float
    occurrence_count: int
    source: str
    first_seen_at: datetime
    last_seen_at: datetime

    def render_for_prompt(self) -> str:
        """
        Строка для системного промпта.

        Систематичность передаётся ЯВНО, числом: модель не умеет вывести «это
        повторяется» из самого текста факта, а разница между «однажды сказал»
        и «говорит постоянно» меняет и уместность упоминания, и тон.
        Однократное наблюдение счётчиком не помечается — «(упомянуто 1 раз)»
        сообщает шум вместо смысла.
        """
        base = f"{self.attribute.replace('_', ' ')}: {self.value}"
        if self.occurrence_count > 1:
            return f"[ФАКТ: {base} (упомянуто {self.occurrence_count} раз)]"
        return f"[ФАКТ: {base}]"


@dataclass(slots=True, frozen=True)
class StoreOutcome:
    """Итог одной записи — какой факт получился и что с ним стало."""

    action: StoreAction
    fact: StoredFact
    similarity: float = 1.0

    @property
    def created(self) -> bool:
        return self.action is StoreAction.CREATED


class KnowledgeStore:
    """
    Хранилище проверенных фактов с дедупликацией.

    Принимает ТОЛЬКО `ValidatedFact` (см. efi/memory/validator.py) — на
    уровне типов невозможно передать сюда то, что пришло от модели напрямую.
    Это и есть техническая форма границы доверия: не «мы договорились не
    писать сырое», а «сырое сюда не типизируется».
    """

    def __init__(
        self,
        database: Database,
        *,
        embedder: Embedder | None = None,
        similarity_threshold: float = DEFAULT_SIMILARITY_THRESHOLD,
    ) -> None:
        self._database = database
        self._embedder = embedder
        self._similarity_threshold = similarity_threshold

    # -- запись ------------------------------------------------------------

    async def remember(self, fact: ValidatedFact) -> StoreOutcome:
        """
        Записывает факт либо подтверждает уже существующий.

        Порядок ступеней важен: хэш проверяется первым и не требует ни
        эмбеддинга, ни сети — на буквальных повторах (а их большинство)
        дедупликация не стоит вообще ничего.
        """
        existing = await self._find_by_hash(fact)
        if existing is not None:
            return await self._reinforce(existing, similarity=1.0)

        embedding = await self._embed(fact)
        if embedding is not None:
            similar, similarity = await self._find_similar(fact, embedding)
            if similar is not None:
                logger.info(
                    "dedup: %s/%s совпал с записью #%d (сходство %.3f), подтверждаю вместо новой записи",
                    fact.entity_id, fact.attribute, similar.id, similarity,
                )
                return await self._reinforce(similar, similarity=similarity)

        return await self._insert(fact, embedding)

    async def remember_all(self, facts: Sequence[ValidatedFact]) -> list[StoreOutcome]:
        """
        Последовательно, а не через gather: два кандидата одной пачки часто
        оказываются перефразировкой друг друга, и параллельная запись
        разошлась бы мимо дедупликации — оба не увидели бы друг друга.
        """
        return [await self.remember(fact) for fact in facts]

    async def record_rejections(self, rejections: Sequence[Rejection], *, source: str = "") -> None:
        """Журнал отбраковки. Сбой записи журнала не должен ломать приём остальных фактов."""
        if not rejections:
            return
        now = datetime.now(UTC).isoformat()
        try:
            async with self._database.connection() as conn:
                await conn.executemany(
                    """
                    INSERT INTO knowledge_rejections
                        (domain, entity_id, attribute, value, reason, source, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    [
                        (
                            rejection.candidate.domain,
                            rejection.candidate.entity,
                            rejection.candidate.attribute,
                            rejection.candidate.value,
                            rejection.reason,
                            source,
                            now,
                        )
                        for rejection in rejections
                    ],
                )
                await conn.commit()
        except Exception:
            logger.warning("dedup: не удалось записать журнал отбраковки (%d шт.)", len(rejections), exc_info=True)

    # -- чтение ------------------------------------------------------------

    async def recall(
        self,
        *,
        entity_ids: Sequence[str] | None = None,
        domains: Sequence[MemoryDomain] | None = None,
        limit: int = 12,
    ) -> list[StoredFact]:
        """
        Факты для подмешивания в промпт.

        Сортировка по (occurrence_count, last_seen_at), а не по свежести:
        когда места в промпте на десяток строк, систематически повторяющееся
        важнее случайно упомянутого вчера.
        """
        conditions: list[str] = []
        params: list[object] = []
        if entity_ids:
            placeholders = ",".join("?" for _ in entity_ids)
            conditions.append(f"entity_id IN ({placeholders})")
            params.extend(entity_ids)
        if domains:
            placeholders = ",".join("?" for _ in domains)
            conditions.append(f"domain IN ({placeholders})")
            params.extend(domain.value for domain in domains)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        rows = await self._database.fetch_all(
            f"""
            SELECT id, domain, entity_id, attribute, value, confidence, occurrence_count,
                   source, first_seen_at, last_seen_at
            FROM knowledge_facts
            {where}
            ORDER BY occurrence_count DESC, last_seen_at DESC
            LIMIT ?
            """,  # noqa: S608 — условия собраны из плейсхолдеров, значения идут параметрами
            (*params, max(1, limit)),
        )
        return [_row_to_fact(row) for row in rows]

    async def count(self) -> int:
        row = await self._database.fetch_one("SELECT COUNT(*) FROM knowledge_facts")
        return int(row[0]) if row is not None else 0

    # -- внутреннее --------------------------------------------------------

    async def _find_by_hash(self, fact: ValidatedFact) -> StoredFact | None:
        row = await self._database.fetch_one(
            """
            SELECT id, domain, entity_id, attribute, value, confidence, occurrence_count,
                   source, first_seen_at, last_seen_at
            FROM knowledge_facts
            WHERE domain = ? AND normalized_hash = ?
            """,
            (fact.domain.value, fact.normalized_hash),
        )
        return _row_to_fact(row) if row is not None else None

    async def _find_similar(
        self, fact: ValidatedFact, embedding: EmbeddingVector
    ) -> tuple[StoredFact | None, float]:
        """
        Ищет семантический дубль СРЕДИ ФАКТОВ ТОЙ ЖЕ СУЩНОСТИ И ДОМЕНА.

        Сужение принципиальное, а не оптимизация: «любит кофе» про Рому и
        «любит кофе» про Костю семантически почти неразличимы, и сравнение
        по всей базе склеило бы двух разных людей в одного.
        """
        rows = await self._database.fetch_all(
            """
            SELECT id, domain, entity_id, attribute, value, confidence, occurrence_count,
                   source, first_seen_at, last_seen_at, embedding, embedding_dim
            FROM knowledge_facts
            WHERE domain = ? AND entity_id = ? AND embedding IS NOT NULL
            ORDER BY last_seen_at DESC
            LIMIT ?
            """,
            (fact.domain.value, fact.entity_id, _MAX_COMPARISON_CANDIDATES),
        )
        if not rows:
            return None, 0.0

        query_vector = np.asarray(embedding, dtype=np.float32)
        best_row = None
        best_similarity = 0.0
        for row in rows:
            stored_dim = int(row["embedding_dim"])
            if stored_dim != query_vector.size:
                # Модель эмбеддингов сменилась — сравнивать несравнимое
                # нельзя. Пропускаем молча: следующая запись просто ляжет
                # рядом, а старая со временем уйдёт консолидацией.
                continue
            candidate_vector = np.frombuffer(row["embedding"], dtype=np.float32)
            similarity = _cosine(query_vector, candidate_vector)
            if similarity > best_similarity:
                best_similarity = similarity
                best_row = row

        if best_row is None or best_similarity < self._similarity_threshold:
            return None, best_similarity
        return _row_to_fact(best_row), best_similarity

    async def _reinforce(self, existing: StoredFact, *, similarity: float) -> StoreOutcome:
        """
        Подтверждение существующего факта: счётчик +1, last_seen_at = сейчас.

        Инкремент делается ВЫРАЖЕНИЕМ в SQL (`occurrence_count + 1`), а не
        «прочитали, прибавили, записали»: второй способ теряет подтверждения
        при одновременной работе нескольких воркеров, а именно они и пишут
        память параллельно.
        """
        now = datetime.now(UTC)
        await self._database.execute(
            "UPDATE knowledge_facts SET occurrence_count = occurrence_count + 1, last_seen_at = ? WHERE id = ?",
            (now.isoformat(), existing.id),
        )
        updated = StoredFact(
            id=existing.id,
            domain=existing.domain,
            entity_id=existing.entity_id,
            attribute=existing.attribute,
            value=existing.value,
            confidence=existing.confidence,
            occurrence_count=existing.occurrence_count + 1,
            source=existing.source,
            first_seen_at=existing.first_seen_at,
            last_seen_at=now,
        )
        return StoreOutcome(action=StoreAction.REINFORCED, fact=updated, similarity=similarity)

    async def _insert(self, fact: ValidatedFact, embedding: EmbeddingVector | None) -> StoreOutcome:
        now = datetime.now(UTC)
        blob = _pack_embedding(embedding)
        async with self._database.connection() as conn:
            cursor = await conn.execute(
                """
                INSERT INTO knowledge_facts
                    (domain, entity_id, attribute, value, normalized_hash, embedding, embedding_dim,
                     confidence, occurrence_count, source, first_seen_at, last_seen_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, ?)
                ON CONFLICT (domain, normalized_hash) DO UPDATE SET
                    occurrence_count = occurrence_count + 1,
                    last_seen_at = excluded.last_seen_at
                """,
                (
                    fact.domain.value,
                    fact.entity_id,
                    fact.attribute,
                    fact.value,
                    fact.normalized_hash,
                    blob,
                    len(embedding) if embedding else 0,
                    fact.confidence,
                    fact.source,
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
            await conn.commit()
            row_id = cursor.lastrowid or 0

        stored = await self._find_by_hash(fact)
        if stored is None:  # pragma: no cover — запись только что вставлена
            stored = StoredFact(
                id=row_id,
                domain=fact.domain,
                entity_id=fact.entity_id,
                attribute=fact.attribute,
                value=fact.value,
                confidence=fact.confidence,
                occurrence_count=1,
                source=fact.source,
                first_seen_at=now,
                last_seen_at=now,
            )
        # ON CONFLICT выше отработал, если параллельный воркер успел вставить
        # тот же факт между нашей проверкой хэша и вставкой: тогда это не
        # создание, а подтверждение — и отчитаться надо честно.
        action = StoreAction.CREATED if stored.occurrence_count == 1 else StoreAction.REINFORCED
        return StoreOutcome(action=action, fact=stored)

    async def _embed(self, fact: ValidatedFact) -> EmbeddingVector | None:
        if self._embedder is None:
            return None
        try:
            return await self._embedder.embed(fact.canonical_text)
        except Exception:
            # Дедупликация — улучшение, а не условие записи: без эмбеддинга
            # факт всё равно сохранится (точный повтор поймает хэш).
            logger.warning("dedup: не удалось получить эмбеддинг для %s", fact.entity_id, exc_info=True)
            return None


def render_facts_block(facts: Sequence[StoredFact]) -> str:
    """Готовый блок для системного промпта. Пустой список — пустая строка, а не заголовок без содержимого."""
    if not facts:
        return ""
    lines = "\n".join(fact.render_for_prompt() for fact in facts)
    return (
        "[Что ты про это знаешь наверняка]\n"
        f"{lines}\n"
        "Числа в скобках — сколько раз это подтверждалось. Часто повторяющееся можно считать "
        "устойчивой чертой, единичное — просто известным тебе обстоятельством."
    )


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    denominator = float(np.linalg.norm(first) * np.linalg.norm(second))
    if denominator <= 0.0:
        return 0.0
    return float(np.dot(first, second) / denominator)


def _pack_embedding(embedding: EmbeddingVector | None) -> bytes | None:
    if not embedding:
        return None
    return np.asarray(embedding, dtype=np.float32).tobytes()


def _row_to_fact(row: object) -> StoredFact:
    mapping = dict(row)  # type: ignore[call-overload]  # aiosqlite.Row поддерживает dict()
    return StoredFact(
        id=int(mapping["id"]),
        domain=MemoryDomain.parse(mapping["domain"], default=MemoryDomain.HISTORY),
        entity_id=str(mapping["entity_id"]),
        attribute=str(mapping["attribute"]),
        value=str(mapping["value"]),
        confidence=float(mapping["confidence"]),
        occurrence_count=int(mapping["occurrence_count"]),
        source=str(mapping["source"]),
        first_seen_at=datetime.fromisoformat(str(mapping["first_seen_at"])),
        last_seen_at=datetime.fromisoformat(str(mapping["last_seen_at"])),
    )


__all__ = [
    "DEFAULT_SIMILARITY_THRESHOLD",
    "Embedder",
    "KnowledgeStore",
    "StoreAction",
    "StoreOutcome",
    "StoredFact",
    "render_facts_block",
]
