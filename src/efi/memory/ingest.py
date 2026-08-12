"""
efi/memory/ingest.py

Конвейер приёма знаний: восприятие -> валидация -> разрешение сущностей ->
дедупликация -> запись.

Зачем отдельный модуль, а не метод в одном из слоёв. parser/validator/dedup —
это слои с однонаправленной зависимостью (каждый следующий знает про
предыдущий, но не наоборот), и сборка их в цепочку не принадлежит ни одному:
поставь её в parser — и он узнает про БД, поставь в dedup — и он узнает про
LLM. Ровно та связанность, ради разрыва которой границу доверия и вводили.

Единственная точка, через которую знание попадает в память. Всё остальное
приложение (пульс памяти, консолидация, инструменты модели) обращается
сюда — и физически не может обойти валидацию, потому что `KnowledgeStore`
принимает только `ValidatedFact`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from efi.behavior.ambiguity import AmbiguityDetector, EntityCandidate, PendingClarifications, Resolution
from efi.memory.dedup import KnowledgeStore, StoreAction, StoreOutcome
from efi.memory.parser import FactCandidate, PerceptionBatch, PerceptionParser
from efi.memory.validator import FactValidator, Rejection, ValidatedFact, compute_hash

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class IngestResult:
    """Что произошло с одним эпизодом на входе в память."""

    created: list[StoreOutcome] = field(default_factory=list)
    reinforced: list[StoreOutcome] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)
    #: Вопросы, которые Эфи должна задать вместо того, чтобы записать
    #: искажённые данные (см. efi/behavior/ambiguity.py).
    clarifications: list[str] = field(default_factory=list)
    parse_error: str = ""

    @property
    def stored_count(self) -> int:
        return len(self.created) + len(self.reinforced)

    @property
    def needs_clarification(self) -> bool:
        return bool(self.clarifications)

    def summary(self) -> str:
        return (
            f"новых {len(self.created)}, подтверждено {len(self.reinforced)}, "
            f"отклонено {len(self.rejected)}, уточнений {len(self.clarifications)}"
        )


#: Как приложение сообщает конвейеру, какие сущности вообще существуют.
#: Список кандидатов на упоминание приходит извне (люди из PeopleStore, темы
#: из семян любопытства) — конвейер сам их не ищет: это не его знание.
EntityCatalog = dict[str, list[EntityCandidate]]


class MemoryIngestor:
    """
    Полный путь знания в память. Порядок шагов зафиксирован и не подлежит
    перестановке: разрешение сущностей идёт ПОСЛЕ валидации (незачем
    разбираться, о ком речь, если факт всё равно негоден) и ДО дедупликации
    (сравнивать факты о разных людях бессмысленно, а о неразрешённом
    упоминании — вредно).
    """

    def __init__(
        self,
        parser: PerceptionParser,
        validator: FactValidator,
        store: KnowledgeStore,
        *,
        detector: AmbiguityDetector | None = None,
        pending: PendingClarifications | None = None,
    ) -> None:
        self._parser = parser
        self._validator = validator
        self._store = store
        self._detector = detector if detector is not None else AmbiguityDetector()
        # Именно `is not None`, а не `or`. У PendingClarifications есть
        # __len__, поэтому ПУСТОЙ реестр — а он всегда пуст при старте —
        # ложен по значению, и `pending or PendingClarifications()` молча
        # подменял переданный извне реестр своим. Внешне это выглядело как
        # «уточняющие вопросы не работают»: конвейер исправно их формулировал
        # и складывал в объект, которого не видел больше никто.
        self._pending = pending if pending is not None else PendingClarifications()

    async def ingest_conversation(
        self,
        conversation_text: str,
        *,
        source: str = "",
        chat_id: int | None = None,
        catalog: EntityCatalog | None = None,
    ) -> IngestResult:
        """Полный проход: спросить модель восприятия и провести кандидатов через все проверки."""
        batch = await self._parser.extract(conversation_text, source=source)
        if batch.failed:
            return IngestResult(parse_error=batch.parse_error)
        return await self.ingest_batch(batch, chat_id=chat_id, catalog=catalog)

    async def ingest_batch(
        self,
        batch: PerceptionBatch,
        *,
        chat_id: int | None = None,
        catalog: EntityCatalog | None = None,
    ) -> IngestResult:
        """
        Та же обработка, но для уже полученной пачки кандидатов — точка
        входа для тестов и для случаев, когда кандидатов принесли не из LLM.
        """
        report = self._validator.validate_batch(batch)
        result = IngestResult(rejected=list(report.rejected))

        accepted: list[ValidatedFact] = []
        for fact in report.accepted:
            resolved, resolution = self._resolve_entity(fact, catalog)
            if resolution is not None and resolution.needs_clarification:
                assert resolution.clarification is not None  # гарантировано needs_clarification
                result.clarifications.append(resolution.clarification)
                result.rejected.append(
                    Rejection(
                        _as_candidate(fact),
                        f"неоднозначная сущность {fact.entity_id!r}, задан уточняющий вопрос",
                    )
                )
                if chat_id is not None:
                    self._pending.remember(chat_id, resolution)
                continue
            accepted.append(resolved)

        for outcome in await self._store.remember_all(accepted):
            if outcome.action is StoreAction.CREATED:
                result.created.append(outcome)
            else:
                result.reinforced.append(outcome)

        await self._store.record_rejections(result.rejected, source=batch.source)
        if result.stored_count or result.rejected:
            logger.info("ingest: %s (источник %s)", result.summary(), batch.source or "—")
        return result

    def _resolve_entity(
        self, fact: ValidatedFact, catalog: EntityCatalog | None
    ) -> tuple[ValidatedFact, Resolution | None]:
        """
        Сопоставляет сущность факта с известными приложению.

        Без каталога разрешать нечего — сущность остаётся такой, какой её
        нормализовал валидатор. Это не дыра: валидатор уже привёл её к
        каноническому виду, а неоднозначность бывает только там, где есть
        несколько известных претендентов, то есть только при наличии каталога.
        """
        if not catalog:
            return fact, None

        mention = fact.entity_id.split(":", 1)[-1]
        candidates = catalog.get(mention)
        if not candidates:
            return fact, None

        resolution = self._detector.resolve(mention, candidates)
        if resolution.needs_clarification:
            return fact, resolution
        if resolution.is_resolved and resolution.entity_id is not None:
            return _with_entity(fact, resolution.entity_id), resolution
        return fact, resolution

    @property
    def pending_clarifications(self) -> PendingClarifications:
        """Реестр незакрытых уточнений — читает диалоговый оркестратор, чтобы задать вопрос."""
        return self._pending


def _with_entity(fact: ValidatedFact, entity_id: str) -> ValidatedFact:
    """
    Пересобирает факт с разрешённой сущностью.

    Хэш обязан пересчитаться: он включает entity_id, и оставить прежний
    значило бы поселить в базе запись, чей хэш не соответствует содержимому —
    дедупликация после этого перестала бы работать именно на тех фактах,
    которые прошли разрешение сущности.
    """
    if entity_id == fact.entity_id:
        return fact
    return ValidatedFact(
        domain=fact.domain,
        entity_id=entity_id,
        attribute=fact.attribute,
        value=fact.value,
        confidence=fact.confidence,
        observed_at=fact.observed_at,
        source=fact.source,
        normalized_hash=compute_hash(fact.domain, entity_id, fact.attribute, fact.value),
    )


def _as_candidate(fact: ValidatedFact) -> FactCandidate:
    """Обратное превращение для журнала отбраковки — он ведётся в терминах кандидатов."""
    return FactCandidate(
        domain=fact.domain.value,
        entity=fact.entity_id,
        attribute=fact.attribute,
        value=fact.value,
        confidence=fact.confidence,
    )


__all__ = ["EntityCatalog", "IngestResult", "MemoryIngestor"]
