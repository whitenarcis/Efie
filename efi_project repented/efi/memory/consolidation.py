"""
efi/memory/consolidation.py

Ночная консолидация памяти — три независимые операции:

1. `deduplicate()` — убирает дубли/почти-дубли уже существующих записей
   дневника (тот же принцип, что и diaryPlagiarismThreshold при сохранении
   новой записи, но применяется сплошным проходом по всему корпусу).
2. `summarize_stale_entries()` — сворачивает старые записи в более общую
   "мемуарную" запись через LLM, снижая объём дневника без потери смысла.
3. `novelize_recent_history()` — САМАЯ ВАЖНАЯ: без неё дневник никогда не
   пополняется сам по себе. Разбирает недавнюю переписку по всем активным
   чатам и просит LLM выделить то, что реально стоит запомнить надолго
   (факты, события, договорённости), сохраняя каждое как отдельную запись
   через RAGMemory.remember(). Прямой аналог sleepingConsolidation из
   референса — единственный источник ДОЛГОСРОЧНОЙ памяти помимо явного
   вызова remember_diary_entry самой моделью посреди разговора
   (efi/tools/memory_tools/remember_diary_entry.py — для того, что стоит
   запомнить прямо сейчас, не дожидаясь ночи).

Все три операции программные, не диалоговые — идут НЕ через
NotificationManager/Worker. LLM используется точечно (сжатие/извлечение
текста), а не для полноценного разговорного ответа.
"""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from typing import Protocol

from efi.config.schema import TaskRole
from efi.llm.errors import LLMError
from efi.llm.router import LLMRouter
from efi.llm.schemas import DiaryEntry, DiaryEntryMetadata, LLMParams, Message, Role, Session
from efi.memory.diary import Diary
from efi.memory.facts import FactStore
from efi.memory.rag import RAGMemory

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

_NOVELIZATION_SYSTEM_PROMPT = (
    "Тебе показан фрагмент недавней переписки. Определи, есть ли там что-то, что реально "
    "стоит запомнить надолго: новые факты о собеседнике, важные события, договорённости, "
    "эмоционально значимые моменты, повторяющиеся темы разговора. Игнорируй пустую болтовню, "
    "техническую рутину и то, что не имеет значения через неделю.\n"
    "КАК ПИСАТЬ КАЖДОЕ ВОСПОМИНАНИЕ (это не протокол собрания, а твой личный дневник):\n"
    "- Строго от первого лица, как будто это ТЫ сейчас вспоминаешь, а не сторонний наблюдатель.\n"
    "- С твоей субъективной, эмоциональной оценкой произошедшего — не 'обсудили X', а то, что ты "
    "реально почувствовала по этому поводу (бесит, смешно, зацепило, обидно, гордишься).\n"
    "- Зафиксируй личное отношение, даже если оно неоднозначное: можно одновременно раздражаться "
    "и признавать правоту, это нормально для живой памяти.\n"
    "Пример ПЛОХОГО воспоминания (протокольное, безличное): 'Обсудили баг в коде, договорились "
    "исправить позже.'\n"
    "Пример ХОРОШЕГО воспоминания (субъективное, живое): 'Опять спорили из-за этого бага — бесит, "
    "когда занудствуют по мелочи, но чёрт, доля правды в его придирках была.'\n"
    "Каждое отдельное воспоминание пиши коротко (1-3 предложения), и раздели их строкой из трёх "
    "дефисов (---) на отдельной строке. "
    "Если ничего запоминающегося не было — ответь ровно одним словом: ПУСТО."
)

_NOVELIZATION_EMPTY_MARKER = "ПУСТО"
_ENTRY_SPLIT_RE = re.compile(r"\n\s*-{3,}\s*\n")
_MESSAGE_PREVIEW_MAX_LENGTH = 2000


class HistorySource(Protocol):
    """
    Абстракция источника истории для новеллизации — минимум, который нужен
    отсюда (не полный efi.notifications.worker.HistoryRepository, чтобы не
    тянуть зависимость memory -> notifications ради одного протокола).
    Конкретная реализация — efi.db.history_repository.SqliteHistoryRepository.
    """

    async def get_active_chat_ids(self, *, since: datetime) -> list[int]: ...

    async def get_since(self, chat_id: int, *, since: datetime) -> Session: ...


class DiaryConsolidator:
    """Программная консолидация и пополнение дневника: dedup, сжатие старых записей, автоматическое извлечение новых."""

    def __init__(
        self,
        diary: Diary,
        router: LLMRouter,
        rag: RAGMemory,
        *,
        summarization_role: TaskRole = TaskRole.FAST,
    ) -> None:
        self._diary = diary
        self._router = router
        self._rag = rag
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

    async def novelize_recent_history(
        self,
        *,
        history: HistorySource,
        facts: FactStore,
        lookback: timedelta = timedelta(days=1),
        min_messages: int = 6,
        chat_lookback_ceiling: timedelta = timedelta(days=30),
    ) -> int:
        """
        Автоматическое пополнение дневника из недавней переписки — без этого
        шага Diary остаётся пустым до тех пор, пока модель сама явно не
        воспользуется remember_diary_entry, а в быстрой переписке это
        происходит редко (см. докстринг модуля). Прямой аналог
        sleepingConsolidation у референса.

        Отслеживает "докуда уже новеллизировано" по каждому чату через
        FactStore (entity_id=f"chat:{chat_id}", key="last_novelized_at") —
        чтобы не пересказывать одно и то же на каждый ночной проход. Для
        чата без отметки (первый раз) смотрит на последние `lookback`
        (по умолчанию сутки).

        `chat_lookback_ceiling` ограничивает, какие чаты вообще считаются
        "активными" для обхода — не пытаемся новеллизировать чат, где
        последнее сообщение было полгода назад.

        Возвращает число новых записей, реально сохранённых в дневник
        (дубли, отбракованные RAGMemory.remember(), в счёт не идут).
        """
        chat_ids = await history.get_active_chat_ids(since=datetime.now(timezone.utc) - chat_lookback_ceiling)
        created_count = 0

        for chat_id in chat_ids:
            since = await self._resolve_last_novelized_at(facts, chat_id, lookback)
            session = await history.get_since(chat_id, since=since)
            if len(session.messages) < min_messages:
                continue

            memories = await self._extract_memories(session)
            saved_in_chat = 0
            for memory_text in memories:
                entry = await self._rag.remember(memory_text, confidence=0.5)
                if entry is not None:
                    created_count += 1
                    saved_in_chat += 1

            await facts.upsert(f"chat:{chat_id}", "last_novelized_at", datetime.now(timezone.utc).isoformat())
            if memories:
                logger.info(
                    "consolidation: novelized chat_id=%s -> %d candidate memories (%d actually saved)",
                    chat_id, len(memories), saved_in_chat,
                )

        return created_count

    async def _resolve_last_novelized_at(self, facts: FactStore, chat_id: int, lookback: timedelta) -> datetime:
        raw = await facts.get(f"chat:{chat_id}", "last_novelized_at")
        if raw is None:
            return datetime.now(timezone.utc) - lookback
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            logger.warning(
                "consolidation: unparseable last_novelized_at for chat_id=%s (%r), falling back to lookback",
                chat_id, raw,
            )
            return datetime.now(timezone.utc) - lookback

    async def _extract_memories(self, session: Session) -> list[str]:
        conversation_text = _render_conversation(session)
        if not conversation_text:
            return []

        params = LLMParams(model="", system_prompt=_NOVELIZATION_SYSTEM_PROMPT, max_output_tokens=768)
        prompt_session = Session(messages=[Message(role=Role.USER, content=conversation_text)])
        try:
            response = await self._router.chat(self._summarization_role, params, prompt_session)
        except LLMError as exc:
            logger.warning("consolidation: novelization request failed: %s", exc)
            return []

        text = response.text.strip()
        if not text or text.strip().upper() == _NOVELIZATION_EMPTY_MARKER:
            return []

        pieces = [piece.strip() for piece in _ENTRY_SPLIT_RE.split(text)]
        return [piece for piece in pieces if piece and piece.upper() != _NOVELIZATION_EMPTY_MARKER]

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


def _render_conversation(session: Session) -> str:
    """Плоский текст переписки для промпта новеллизации — только реплики с содержимым (не голые tool-calls)."""
    lines = [f"{message.role.value}: {message.content}" for message in session if message.content.strip()]
    text = "\n".join(lines)
    return text[:_MESSAGE_PREVIEW_MAX_LENGTH] if text else ""


def _pick_duplicate_to_remove(a: DiaryEntry, b: DiaryEntry) -> str:
    """Из пары почти-дублей выбирает, какую запись убрать: ниже confidence, при равенстве — меньше usage_count."""
    if a.metadata.confidence != b.metadata.confidence:
        return a.id if a.metadata.confidence < b.metadata.confidence else b.id
    return a.id if a.metadata.usage_count <= b.metadata.usage_count else b.id


def _min_datetime() -> datetime:
    """Для записей без last_used (никогда не использовались) — «максимально старые», кандидаты на консолидацию в первую очередь."""
    return datetime.min.replace(tzinfo=timezone.utc)


__all__ = ["DiaryConsolidator", "HistorySource"]
