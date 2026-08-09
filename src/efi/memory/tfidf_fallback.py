"""
efi/memory/tfidf_fallback.py

Быстрый TF-IDF поиск по памяти — short-circuit оптимизация критического пути.

Не каждое сообщение пользователя заслуживает сетевого embedding-запроса:
короткие реплики и фатические фразы ("привет", "ок", "спасибо") дешевле и
достаточно точно обслужить локальным TF-IDF-поиском без похода в сеть.
`RAGMemory.search` (memory/rag.py) вызывает `should_use_tfidf_shortcut()`,
чтобы решить, идти ли по этому пути, и делегирует сюда же, если удалённый
embedding-запрос завершился ошибкой (graceful degradation).

Собственная реализация TF-IDF без внешних ML-зависимостей (scikit-learn и
подобные плохо совместимы с Termux/ARM) — та же логика, что уже проверена в
текущей реализации (rag_memory.py), здесь переписана асинхронно и с чётким
разделением CPU-bound части (вынесена в asyncio.to_thread).
"""

from __future__ import annotations

import asyncio
import logging
import math
import re
from collections import Counter
from dataclasses import dataclass

from efi.llm.schemas import DiaryEntry, DiaryQueryOptions, DiaryQueryResult

logger = logging.getLogger(__name__)

# Буквы любого алфавита (юникод), без цифр и подчёркиваний.
_TOKEN_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

# Частотные фатические реплики (RU) — сообщения из них одних не заслуживают
# семантического поиска. Список сознательно небольшой и приблизительный:
# ложноотрицательный результат здесь не страшен (просто уйдём в полный RAG).
_PHATIC_WORDS = frozenset(
    {
        "привет", "прив", "хай", "ку", "здравствуй", "здравствуйте", "йо", "ало",
        "ок", "окей", "ладно", "понял", "поняла", "ясно", "спасибо", "благодарю", "спс",
        "пока", "бай", "чмок", "ага", "угу", "да", "нет", "неа", "лол", "хах", "ахах", "кек",
    }
)
_SHORT_TEXT_LENGTH_THRESHOLD = 12
_MAX_PHATIC_TOKEN_COUNT = 3


def tokenize(text: str) -> list[str]:
    """Простая токенизация: слова любого алфавита в нижнем регистре, без цифр/пунктуации."""
    return [match.group(0).lower() for match in _TOKEN_RE.finditer(text)]


def should_use_tfidf_shortcut(text: str) -> bool:
    """
    Эвристика короткого замыкания критического пути.

    True, если текст либо короче порога, либо состоит целиком из нескольких
    фатических слов — семантический поиск через сетевой embedding для такого
    сообщения избыточен.
    """
    stripped = text.strip()
    if len(stripped) <= _SHORT_TEXT_LENGTH_THRESHOLD:
        return True
    tokens = tokenize(stripped)
    if not tokens:
        return True
    return len(tokens) <= _MAX_PHATIC_TOKEN_COUNT and all(token in _PHATIC_WORDS for token in tokens)


@dataclass(slots=True)
class _IndexedDocument:
    entry: DiaryEntry
    term_counts: Counter[str]
    length: int


class TfidfFallbackIndex:
    """
    Инкрементальный TF-IDF индекс поверх текстов записей (дневник, рабочая
    память и т.п.) — держится в памяти процесса, не персистится сам по себе.

    `index()`/`add()` — фоновые операции (CPU-bound на больших корпусах,
    выполняются через asyncio.to_thread). `search()` — критический путь,
    оптимизирован под минимальную задержку: без сетевых вызовов вовсе.
    """

    def __init__(self) -> None:
        self._documents: dict[str, _IndexedDocument] = {}
        self._document_frequency: Counter[str] = Counter()
        self._lock = asyncio.Lock()

    async def index(self, entries: list[DiaryEntry]) -> None:
        """Полная перестройка индекса по переданному набору записей."""
        documents, document_frequency = await asyncio.to_thread(_build_index, entries)
        async with self._lock:
            self._documents = documents
            self._document_frequency = document_frequency

    async def add(self, entry: DiaryEntry) -> None:
        """Инкрементально добавляет одну запись в уже построенный индекс."""
        document = await asyncio.to_thread(_index_document, entry)
        async with self._lock:
            self._documents[entry.id] = document
            for term in document.term_counts:
                self._document_frequency[term] += 1

    async def search(self, query_text: str, options: DiaryQueryOptions | None = None) -> list[DiaryQueryResult]:
        """Критический путь: без сети, только локальная CPU-bound работа (в потоке)."""
        options = options or DiaryQueryOptions()
        async with self._lock:
            if not self._documents:
                return []
            documents = dict(self._documents)
            document_frequency = Counter(self._document_frequency)
        return await asyncio.to_thread(_score_query, query_text, documents, document_frequency, options.max_entry_count)


def _build_index(entries: list[DiaryEntry]) -> tuple[dict[str, _IndexedDocument], Counter[str]]:
    documents: dict[str, _IndexedDocument] = {}
    document_frequency: Counter[str] = Counter()
    for entry in entries:
        document = _index_document(entry)
        documents[entry.id] = document
        for term in document.term_counts:
            document_frequency[term] += 1
    return documents, document_frequency


def _index_document(entry: DiaryEntry) -> _IndexedDocument:
    tokens = tokenize(entry.body)
    return _IndexedDocument(entry=entry, term_counts=Counter(tokens), length=len(tokens))


def _score_query(
    query_text: str,
    documents: dict[str, _IndexedDocument],
    document_frequency: Counter[str],
    max_entry_count: int,
) -> list[DiaryQueryResult]:
    """Косинусное сходство TF-IDF векторов запроса и документов (чистый Python, без numpy — векторы разреженные)."""
    query_tokens = tokenize(query_text)
    if not query_tokens:
        return []

    query_counts = Counter(query_tokens)
    corpus_size = max(len(documents), 1)

    def idf(term: str) -> float:
        # Сглаженный IDF (+1 к числителю/знаменателю и к результату) — всегда > 0,
        # не даёт неизвестным вне корпуса терминам обнулять вклад запроса.
        document_frequency_for_term = document_frequency.get(term, 0)
        return math.log((corpus_size + 1) / (document_frequency_for_term + 1)) + 1.0

    query_vector = {term: (count / len(query_tokens)) * idf(term) for term, count in query_counts.items()}
    query_norm = math.sqrt(sum(weight * weight for weight in query_vector.values())) or 1.0

    scored: list[DiaryQueryResult] = []
    for document in documents.values():
        if document.length == 0:
            continue
        dot_product = 0.0
        doc_norm_squared = 0.0
        for term, count in document.term_counts.items():
            weight = (count / document.length) * idf(term)
            doc_norm_squared += weight * weight
            if term in query_vector:
                dot_product += weight * query_vector[term]
        if dot_product <= 0.0:
            continue
        doc_norm = math.sqrt(doc_norm_squared) or 1.0
        similarity = dot_product / (query_norm * doc_norm)
        scored.append(DiaryQueryResult(entry=document.entry, relatedness=min(max(similarity, 0.0), 1.0)))

    scored.sort(key=lambda result: result.relatedness, reverse=True)
    return scored[:max_entry_count]


__all__ = ["TfidfFallbackIndex", "tokenize", "should_use_tfidf_shortcut"]
