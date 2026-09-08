"""
efi/memory/rag.py

Поиск по долгосрочной памяти: оркестрирует Diary (memory/diary.py) и
TfidfFallbackIndex (memory/tfidf_fallback.py).

Источник эмбеддингов — трёхуровневый, от быстрого/бесплатного к самому
примитивному:
    1. Локальный движок (LocalEmbeddingEngine, fastembed/ONNX) — без сети,
       без зависимости от того, какие модели держит текущий LLM-провайдер.
       Основной путь с тех пор, как выяснилось, что у Groq (роль FAST) нет
       embedding-моделей вообще.
    2. LLMRouter.embedding() — облачный фолбэк. Оставлен на случай, если
       локальный движок не установлен/не загрузился, или если позже
       появится провайдер, у которого embeddings действительно есть.
    3. TF-IDF (TfidfFallbackIndex) — совсем без эмбеддингов, если оба пути
       выше недоступны или для заведомо коротких/фатических сообщений
       (see should_use_tfidf_shortcut) — туда сетевой/модельный round-trip
       вообще не имеет смысла запускать.

Критический путь — `search()`: вызывается прямо перед генерацией ответа, и
ни на одном из трёх уровней сбой не должен ронять формирование ответа целиком.

Запись (`remember()`) — фоновая операция по духу (эмбеддинг + дубль-чек + I/O
на диск не бесплатны), но сама по себе остаётся `async def`: решение, вызывать
её через `await` или через `asyncio.create_task()`, принимает вызывающая
сторона (notifications/worker.py, memory/consolidation.py).
"""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Iterable

from efi.config.schema import TaskRole
from efi.llm.errors import LLMError
from efi.llm.router import LLMRouter
from efi.llm.schemas import DiaryEntry, DiaryEntryMetadata, DiaryQueryOptions, DiaryQueryResult, EmbeddingVector
from efi.memory.diary import Diary
from efi.memory.local_embeddings import LocalEmbeddingEngine
from efi.memory.router import MemoryDomain, MemoryRouter
from efi.memory.tfidf_fallback import TfidfFallbackIndex, should_use_tfidf_shortcut

logger = logging.getLogger(__name__)

#: Роутер без состояния — один на модуль, чтобы не плодить объекты на каждый поиск.
_MEMORY_ROUTER = MemoryRouter()


class RAGMemory:
    """
    Единая точка входа для семантического поиска и записи в долгосрочную память.

    `local_embeddings` — необязательный (можно совсем не передавать, тогда
    остаётся только LLMRouter.embedding()/TF-IDF); `embedding_role`
    настраивает, какая роль LLMRouter используется как облачный фолбэк —
    по умолчанию FAST, чтобы получение эмбеддинга не занимало тяжёлый
    MAIN-канал, которым бот параллельно генерирует ответ.
    """

    def __init__(
        self,
        diary: Diary,
        router: LLMRouter,
        tfidf: TfidfFallbackIndex,
        *,
        local_embeddings: LocalEmbeddingEngine | None = None,
        embedding_role: TaskRole = TaskRole.FAST,
    ) -> None:
        self._diary = diary
        self._router = router
        self._tfidf = tfidf
        self._local_embeddings = local_embeddings
        self._embedding_role = embedding_role
        #: Уже сказанные вслух жалобы на эмбеддинги. Эмбеддинг считается на
        #: КАЖДУЮ сборку промпта, то есть на каждое сообщение, и обе причины
        #: отказа стабильны: «fastembed не установлен» не изменится никогда, а
        #: недоступный провайдер держится в cooldown минутами. Без этого лог
        #: за вечер состоял из одной и той же пары строк, и настоящая ошибка
        #: в нём просто не была видна — а лог здесь читают именно тогда, когда
        #: что-то сломалось.
        self._reported_embedding_troubles: set[str] = set()

    async def search(
        self,
        query_text: str,
        options: DiaryQueryOptions | None = None,
        *,
        domains: Iterable[MemoryDomain] | None = None,
    ) -> list[DiaryQueryResult]:
        """
        Критический путь: короткие/фатические сообщения уходят в дешёвый
        TF-IDF напрямую (см. should_use_tfidf_shortcut), для остальных
        считается embedding запроса (локально или облачно — см. докстринг
        модуля) и делается семантический поиск по Diary; если эмбеддинг
        получить не удалось никаким путём — деградирует до TF-IDF вместо
        того, чтобы пробрасывать исключение и срывать ответ пользователю.

        `domains` сужает выборку до нужных доменов памяти (см.
        efi/memory/router.py). Фильтр применяется ДО ранжирования, а не
        после: иначе десять релевантных записей чужого домена вытеснили бы
        своим сходством единственную нужную, и до неё бы просто не дошло.
        None — без фильтрации (обратная совместимость и явный «ищи везде»).
        """
        filter_fn = _MEMORY_ROUTER.diary_filter(domains) if domains is not None else None

        if should_use_tfidf_shortcut(query_text):
            return await self._tfidf.search(query_text, options, filter_fn=filter_fn)

        query_embedding = await self._compute_embedding(query_text, is_query=True)
        if query_embedding is None:
            return await self._tfidf.search(query_text, options, filter_fn=filter_fn)

        return await self._diary.query(query_embedding, options, filter_fn=filter_fn)

    async def embed(self, text: str) -> EmbeddingVector | None:
        """
        Публичный доступ к тому же источнику эмбеддингов, которым пользуется
        сам RAG. Нужен семантической дедупликации (efi/memory/dedup.py):
        сравнивать факты вектором, полученным другим движком, чем дневник, —
        значит сравнивать несравнимое.
        """
        return await self._compute_embedding(text, is_query=False)

    async def remember(
        self, body: str, *, confidence: float = 0.0, domain: MemoryDomain = MemoryDomain.HISTORY
    ) -> DiaryEntry | None:
        """
        Добавляет новую запись в долгосрочную память: считает эмбеддинг,
        проверяет на дубль/плагиат (Diary.is_duplicate_of, аналог
        diaryPlagiarismThreshold), и при отсутствии дубля сохраняет запись
        (в т.ч. индексирует её в TF-IDF fallback-индексе, чтобы он не отставал
        от Diary).

        Если эмбеддинг получить не удалось никаким путём (локальный движок
        не установлен И облачный запрос упал) — запись всё равно сохраняется,
        но без эмбеддинга и без дубль-чека: лучше запомнить текстом и найти
        потом через TF-IDF, чем потерять запись целиком.

        Возвращает сохранённую запись, либо None, если она отбракована как
        дубль существующей.
        """
        embedding = await self._compute_embedding(body, is_query=False)

        if embedding is None:
            logger.warning(
                "rag: no embedding available for new entry (local and cloud both failed), storing without one"
            )
            entry = DiaryEntry(
                id=_generate_entry_id(),
                metadata=DiaryEntryMetadata(confidence=confidence, domain=domain.value),
                body=body,
            )
            await self._diary.save(entry)
            await self._tfidf.add(entry)
            return entry

        duplicate = await self._diary.is_duplicate_of(embedding)
        if duplicate is not None:
            logger.info(
                "rag: skipping duplicate diary entry (relatedness=%.3f, existing id=%s)",
                duplicate.relatedness,
                duplicate.entry.id,
            )
            return None

        entry = DiaryEntry(
            id=_generate_entry_id(),
            metadata=DiaryEntryMetadata(confidence=confidence, embedding=embedding, domain=domain.value),
            body=body,
        )
        await self._diary.save(entry)
        await self._tfidf.add(entry)
        return entry

    async def _compute_embedding(self, text: str, *, is_query: bool) -> EmbeddingVector | None:
        """
        Единая точка получения эмбеддинга. Локальный движок пробуется первым
        (если настроен); LLMRouter — фолбэк. Возвращает None, если оба пути
        недоступны — вызывающая сторона (search/remember) сама решает, что
        делать дальше (TF-IDF или сохранение без эмбеддинга).
        """
        if self._local_embeddings is not None:
            try:
                if is_query:
                    return await self._local_embeddings.embed_query(text)
                return await self._local_embeddings.embed_document(text)
            except RuntimeError as exc:
                self._complain_once(
                    "rag: local embedding failed (%s), falling back to LLMRouter", str(exc)
                )

        try:
            return await self._router.embedding(self._embedding_role, text)
        except LLMError as exc:
            self._complain_once("rag: cloud embedding request failed (%s)", str(exc))
            return None

    def _complain_once(self, template: str, detail: str) -> None:
        """
        Жалуется вслух один раз на каждую отдельную причину, дальше — в debug.

        Эмбеддинг считается на каждую сборку промпта, то есть на каждое
        сообщение, а причины отказа стабильны: «fastembed не установлен» не
        изменится никогда, недоступный провайдер держится в cooldown минутами.
        Пока каждая попытка писала WARNING, лог за вечер состоял из одной и
        той же пары строк — и настоящая ошибка в нём терялась.
        """
        key = f"{template}|{detail}"
        if key in self._reported_embedding_troubles:
            logger.debug(template, detail)
            return
        self._reported_embedding_troubles.add(key)
        logger.warning(template, detail)


def _generate_entry_id() -> str:
    """Unix-таймстамп + короткий случайный суффикс — уникально даже при параллельных вызовах в одну секунду."""
    return f"{int(time.time())}_{uuid.uuid4().hex[:6]}"


__all__ = ["RAGMemory"]
