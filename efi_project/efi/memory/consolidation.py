"""
efi/memory/consolidation.py

Ночная консолидация памяти: убирает дубли/почти-дубли записей дневника
(dedup — тот же принцип, что и diaryPlagiarismThreshold при сохранении новой
записи, но применяется сплошным проходом по всему уже накопленному корпусу) и
сворачивает старые записи в более общую "мемуарную" запись через LLM (аналог
sleepingConsolidation из референса) — снижает объём дневника, не теряя смысла.

Это программная, в основной части (dedup) детерминированная операция, а не
диалоговый ответ модели — поэтому она НЕ идёт через NotificationManager/Worker.
LLM используется только для одной узкой задачи: сжатия пачки записей в
связный текст. Планировщик (efi/behavior/scheduler.py) отдельно шлёт
NIGHTLY_TASK-уведомление, чтобы личность могла отрефлексировать день в
разговорном формате — это дополняет, а не заменяет DiaryConsolidator.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone

from efi.config.schema import TaskRole
from efi.llm.errors import LLMError
from efi.llm.router import LLMRouter
from efi.llm.schemas import DiaryEntry, DiaryEntryMetadata, LLMParams, Message, Role, Session
from efi.memory.diary import Diary

logger = logging.getLogger(__name__)

#: Для каждой записи при dedup-проходе проверяем несколько ближайших
#: соседей, а не только первого — иначе легко пропустить дубль, который
#: оказался вторым по релевантности из-за шума эмбеддинга.
_DEDUP_NEIGHBOR_CHECK_COUNT = 3

_CONSOLIDATION_SYSTEM_PROMPT = (
    "Ты помогаешь сжать несколько отдельных дневниковых записей в одну связную "
    "заметку, сохраняя все важные факты и детали, но убирая повторы и лишние слова. "
    "Пиши от первого лица, коротко и по существу. Не придумывай ничего нового."
)


class DiaryConsolidator:
    """Программная консолидация дневника: dedup существующего корпуса + сжатие старых записей через LLM."""

    def __init__(self, diary: Diary, router: LLMRouter, *, summarization_role: TaskRole = TaskRole.FAST) -> None:
        self._diary = diary
        self._router = router
        self._summarization_role = summarization_role

    async def deduplicate(self, *, plagiarism_threshold: float) -> int:
        """
        Проходит по всем записям с эмбеддингами и убирает почти-дубли
        (relatedness выше `plagiarism_threshold`), оставляя в каждой паре
        запись с более высоким confidence (при равенстве — с большим
        usage_count). Возвращает число удалённых записей.

        Заметка про сложность: наивный обход подходит для масштаба одного
        персонажа (десятки-сотни записей, не миллионы) — то же допущение,
        что и у остального RAG-слоя проекта (см. memory/diary.py::_score_entries).
        """
        entries = await self._diary.all_entries()
        embedded_entries = [entry for entry in entries if entry.metadata.embedding]
        removed_ids: set[str] = set()

        for entry in embedded_entries:
            if entry.id in removed_ids:
                continue

            def _not_self(candidate: DiaryEntry, current_id: str = entry.id) -> bool:
                return candidate.id != current_id

            duplicates = await self._diary.query(entry.metadata.embedding, filter_fn=_not_self)
            for duplicate in duplicates[:_DEDUP_NEIGHBOR_CHECK_COUNT]:
                if duplicate.relatedness < plagiarism_threshold:
                    continue
                if duplicate.entry.id in removed_ids or entry.id in removed_ids:
                    continue
                loser_id = _pick_duplicate_to_remove(entry, duplicate.entry)
                removed_ids.add(loser_id)
                logger.info(
                    "consolidation: dropping duplicate entry %s (relatedness=%.3f with %s)",
                    loser_id, duplicate.relatedness,
                    duplicate.entry.id if loser_id == entry.id else entry.id,
                )

        for entry_id in removed_ids:
            await self._diary.delete(entry_id)

        return len(removed_ids)

    async def summarize_stale_entries(
        self,
        *,
        older_than: timedelta = timedelta(days=30),
        batch_size: int = 10,
    ) -> DiaryEntry | None:
        """
        Берёт до `batch_size` самых старых записей, которым больше
        `older_than` и которые ещё не являются подтверждённым фактом
        (confidence < 1.0 — ground truth не трогаем: она не должна
        "размываться" пересказом), просит LLM сжать их в одну заметку и
        сохраняет результат как новую запись с confidence, усреднённым по
        исходным. Исходные записи после этого удаляются — их смысл теперь
        живёт в сводной записи.

        Возвращает None, если подходящих записей меньше двух (сжимать нечего)
        или если LLM-запрос не удался (в этом случае ничего не удаляется —
        лучше оставить дневник как есть, чем потерять записи без сводки).

        Примечание: у новой сводной записи эмбеддинг не проставляется —
        Diary сама эмбеддинги не считает (см. её докстринг), а эта операция
        не обращается к LLMRouter.embedding() намеренно, чтобы не дублировать
        логику remember() из memory/rag.py. Если сводная запись должна сразу
        участвовать в семантическом поиске, следующим шагом стоит прогнать её
        через RAGMemory.remember() вместо прямого Diary.save() здесь.
        """
        entries = await self._diary.all_entries()
        cutoff = datetime.now(timezone.utc) - older_than
        candidates = [
            entry
            for entry in entries
            if not entry.metadata.is_ground_truth and (entry.metadata.last_used or _min_datetime()) < cutoff
        ]
        if len(candidates) < 2:
            return None

        batch = sorted(candidates, key=lambda entry: entry.metadata.last_used or _min_datetime())[:batch_size]
        summary_text = await self._summarize_via_llm(batch)
        if summary_text is None:
            return None

        average_confidence = sum(entry.metadata.confidence for entry in batch) / len(batch)
        merged_entry = DiaryEntry(
            id=f"memoir_{int(datetime.now(timezone.utc).timestamp())}",
            metadata=DiaryEntryMetadata(confidence=average_confidence),
            body=summary_text,
        )
        await self._diary.save(merged_entry)

        for entry in batch:
            await self._diary.delete(entry.id)

        logger.info("consolidation: merged %d stale entries into %s", len(batch), merged_entry.id)
        return merged_entry

    async def _summarize_via_llm(self, entries: list[DiaryEntry]) -> str | None:
        bodies = "\n\n".join(f"- {entry.body.strip()}" for entry in entries)
        session = Session(messages=[Message(role=Role.USER, content=bodies)])
        params = LLMParams(model="", system_prompt=_CONSOLIDATION_SYSTEM_PROMPT, max_output_tokens=512)
        try:
            response = await self._router.chat(self._summarization_role, params, session)
        except LLMError as exc:
            logger.warning("consolidation: summarization request failed: %s", exc)
            return None
        return response.text.strip() or None


def _pick_duplicate_to_remove(a: DiaryEntry, b: DiaryEntry) -> str:
    """Из пары почти-дублей выбирает, какую запись убрать: ниже confidence, при равенстве — меньше usage_count."""
    if a.metadata.confidence != b.metadata.confidence:
        return a.id if a.metadata.confidence < b.metadata.confidence else b.id
    return a.id if a.metadata.usage_count <= b.metadata.usage_count else b.id


def _min_datetime() -> datetime:
    """Для записей без last_used (никогда не использовались) — «максимально старые», кандидаты на консолидацию в первую очередь."""
    return datetime.min.replace(tzinfo=timezone.utc)


__all__ = ["DiaryConsolidator"]
