"""
efi/memory/rag.py

Поиск по долгосрочной памяти: оркестрирует Diary (memory/diary.py) и
TfidfFallbackIndex (memory/tfidf_fallback.py), получая эмбеддинги асинхронно
через LLMRouter — удалённый API (OmniRoute/Groq), без локальной embedding-модели.

Критический путь — `search()`: вызывается прямо перед генерацией ответа,
поэтому короткие/фатические сообщения уходят в дешёвый TF-IDF-путь напрямую
(minus сетевой round-trip), а при сбое embedding-запроса RAGMemory деградирует
до TF-IDF вместо того, чтобы уронить формирование ответа целиком.

Запись (`remember()`) — фоновая операция по духу (embedding + дубль-чек + I/O
на диск не бесплатны), но сама по себе остаётся `async def`: решение, вызывать
её через `await` или через `asyncio.create_task()`, принимает вызывающая
сторона (notifications/worker.py, memory/consolidation.py).
"""

from __future__ import annotations

import logging
import time
import uuid

from efi.config.schema import TaskRole
from efi.llm.errors import LLMError
from efi.llm.router import LLMRouter
from efi.llm.schemas import DiaryEntry, DiaryEntryMetadata, DiaryQueryOptions, DiaryQueryResult
from efi.memory.diary import Diary
from efi.memory.tfidf_fallback import TfidfFallbackIndex, should_use_tfidf_shortcut

logger = logging.getLogger(__name__)


class RAGMemory:
    """
    Единая точка входа для семантического поиска и записи в долгосрочную память.

    Роль LLM для эмбеддингов настраивается (`embedding_role`) — по умолчанию
    FAST, поскольку получение эмбеддинга типично дешёвая/быстрая операция и не
    должна занимать тяжёлый MAIN-канал, которым бот параллельно генерирует ответ.
    """

    def __init__(
        self,
        diary: Diary,
        router: LLMRouter,
        tfidf: TfidfFallbackIndex,
        *,
        embedding_role: TaskRole = TaskRole.FAST,
    ) -> None:
        self._diary = diary
        self._router = router
        self._tfidf = tfidf
        self._embedding_role = embedding_role

    async def search(self, query_text: str, options: DiaryQueryOptions | None = None) -> list[DiaryQueryResult]:
        """
        Критический путь: подбирает быстрый TF-IDF или полноценный
        семантический поиск в зависимости от текста запроса, и деградирует до
        TF-IDF, если embedding-запрос по сети завершился ошибкой (сеть/лимиты/
        авторизация — см. efi.llm.errors), вместо того чтобы пробрасывать
        исключение выше и срывать ответ пользователю.
        """
        if should_use_tfidf_shortcut(query_text):
            return await self._tfidf.search(query_text, options)

        try:
            query_embedding = await self._router.embedding(self._embedding_role, query_text)
        except LLMError as exc:
            logger.warning("rag: embedding request failed (%s), falling back to TF-IDF search", exc)
            return await self._tfidf.search(query_text, options)

        return await self._diary.query(query_embedding, options)

    async def remember(self, body: str, *, confidence: float = 0.0) -> DiaryEntry | None:
        """
        Добавляет новую запись в долгосрочную память: считает эмбеддинг,
        проверяет на дубль/плагиат (Diary.is_duplicate_of, аналог
        diaryPlagiarismThreshold), и при отсутствии дубля сохраняет запись
        (в т.ч. индексирует её в TF-IDF fallback-индексе, чтобы он не отставал
        от Diary).

        Возвращает сохранённую запись, либо None, если она отбракована как
        дубль существующей.
        """
        query_embedding = await self._router.embedding(self._embedding_role, body)

        duplicate = await self._diary.is_duplicate_of(query_embedding)
        if duplicate is not None:
            logger.info(
                "rag: skipping duplicate diary entry (relatedness=%.3f, existing id=%s)",
                duplicate.relatedness,
                duplicate.entry.id,
            )
            return None

        entry = DiaryEntry(
            id=_generate_entry_id(),
            metadata=DiaryEntryMetadata(confidence=confidence, embedding=query_embedding),
            body=body,
        )
        await self._diary.save(entry)
        await self._tfidf.add(entry)
        return entry


def _generate_entry_id() -> str:
    """Unix-таймстамп + короткий случайный суффикс — уникально даже при параллельных вызовах в одну секунду."""
    return f"{int(time.time())}_{uuid.uuid4().hex[:6]}"


__all__ = ["RAGMemory"]
