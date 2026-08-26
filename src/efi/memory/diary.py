"""
efi/memory/diary.py

Дневник — долгосрочная память Эфи, хранящаяся как markdown-файлы с JSON
front-matter метаданными (score/confidence/last_used/usage_count/embedding).
Каждая запись — отдельный файл `<id>.md` в `paths.diary_dir`.

Прямой аналог класса Diary из C++-референса: ленивый кэш в памяти процесса,
асинхронный поиск по косинусному сходству эмбеддингов.

Разделение чтения и записи (важно для критического пути):
    - `get`, `all_entries`, `query` — только чтение, минимальная задержка,
      предназначены для `await` прямо в пути формирования ответа.
    - `save`, `touch` — выполняют I/O на диск; сами по себе не блокируют
      дольше одной записи файла, но если вызывающей стороне (например,
      notifications/worker.py) не нужно дожидаться результата, ей следует
      обернуть вызов в `asyncio.create_task(...)` самостоятельно — Diary
      не форсирует fire-and-forget неявно, чтобы поведение было предсказуемым.

Диary НИЧЕГО не знает про LLM/эмбеддинги-по-сети: эмбеддинг для новой записи
должен быть посчитан заранее вызывающей стороной (memory/rag.py, через
LLMRouter) и передан в готовом виде. Это разделение ответственности позволяет
тестировать Diary без сети и без мока LLM.
"""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Callable
from pathlib import Path

import aiofiles
import aiofiles.os
import numpy as np

from efi.llm.schemas import (
    DiaryEntry,
    DiaryEntryMetadata,
    DiaryQueryOptions,
    DiaryQueryResult,
    EmbeddingVector,
)
from efi.utils.atomic import write_text_atomic

logger = logging.getLogger(__name__)

_FRONT_MATTER_DELIMITER = "---"


class Diary:
    """
    Асинхронное хранилище записей долгосрочной памяти в виде markdown-файлов
    с front-matter метаданными.

    Кэш ленивый: каталог не читается, пока не понадобятся данные (`get`,
    `all_entries`, `query`). `reload()` инвалидирует кэш, следующий доступ
    перечитает каталог с диска — полезно, если файлы дневника были изменены
    в обход этого объекта (например, вручную или другим процессом).
    """

    def __init__(self, diary_dir: Path, *, plagiarism_threshold: float = 0.97) -> None:
        self._diary_dir = diary_dir
        self._plagiarism_threshold = plagiarism_threshold
        self._cache: dict[str, DiaryEntry] | None = None
        self._cache_lock = asyncio.Lock()

    # -- загрузка / кэш ---------------------------------------------------

    async def _ensure_loaded(self) -> dict[str, DiaryEntry]:
        if self._cache is not None:
            return self._cache
        async with self._cache_lock:
            if self._cache is not None:  # кто-то успел загрузить, пока мы ждали лок
                return self._cache
            self._cache = await self._load_all()
            return self._cache

    async def _load_all(self) -> dict[str, DiaryEntry]:
        await aiofiles.os.makedirs(self._diary_dir, exist_ok=True)
        # Листинг каталога — блокирующий синхронный вызов; уводим в поток,
        # чтобы не подвесить event loop при большом числе файлов.
        paths = await asyncio.to_thread(lambda: sorted(self._diary_dir.glob("*.md")))
        entries = await asyncio.gather(*(self._read_entry(path) for path in paths))
        return {entry.id: entry for entry in entries if entry is not None}

    async def _read_entry(self, path: Path) -> DiaryEntry | None:
        try:
            async with aiofiles.open(path, encoding="utf-8") as f:
                raw = await f.read()
        except OSError as exc:
            logger.warning("diary: failed to read %s: %s", path, exc)
            return None
        try:
            return _parse_entry(entry_id=path.stem, raw=raw)
        except ValueError as exc:
            logger.warning("diary: failed to parse %s: %s", path, exc)
            return None

    def reload(self) -> None:
        """Инвалидирует кэш. Следующий async-вызов перечитает каталог с диска."""
        self._cache = None

    # -- чтение (критический путь) -----------------------------------------

    async def get(self, entry_id: str) -> DiaryEntry | None:
        """Возвращает запись по id, либо None, если такой записи нет."""
        cache = await self._ensure_loaded()
        return cache.get(entry_id)

    async def all_entries(self) -> list[DiaryEntry]:
        """Все записи дневника (используется, например, консолидацией)."""
        cache = await self._ensure_loaded()
        return list(cache.values())

    async def query(
        self,
        query_embedding: EmbeddingVector,
        options: DiaryQueryOptions | None = None,
        *,
        filter_fn: Callable[[DiaryEntry], bool] | None = None,
    ) -> list[DiaryQueryResult]:
        """
        Асинхронный поиск по дневнику: косинусное сходство `query_embedding`
        с эмбеддингом каждой записи, слегка скорректированное на `confidence`
        (аналог Diary::query у референса).

        Записи без эмбеддинга в выборку не попадают — подразумевается, что
        эмбеддинг проставляется при сохранении (memory/rag.py), а не
        досчитывается на лету в критическом пути чтения.
        """
        options = options or DiaryQueryOptions()
        cache = await self._ensure_loaded()
        candidates = [entry for entry in cache.values() if entry.metadata.embedding]
        if filter_fn is not None:
            candidates = [entry for entry in candidates if filter_fn(entry)]
        if not candidates:
            return []

        # Векторная математика — CPU-bound, уводим в поток. На небольших
        # дневниках (десятки-сотни записей) это доли миллисекунды, но принцип
        # "не блокировать event loop numpy-вычислениями" соблюдаем всегда.
        results = await asyncio.to_thread(_score_entries, query_embedding, candidates, options.confidence_factor)
        results.sort(key=lambda r: r.relatedness, reverse=True)
        if options.min_relatedness > 0.0:
            results = [r for r in results if r.relatedness >= options.min_relatedness]
        return results[: options.max_entry_count]

    async def is_duplicate_of(self, candidate_embedding: EmbeddingVector) -> DiaryQueryResult | None:
        """
        Проверка на дубль/плагиат перед сохранением новой записи (аналог
        diaryPlagiarismThreshold из референса). Возвращает лучший найденный
        дубль, если его relatedness выше порога, иначе — None.
        """
        best = await self.query(candidate_embedding, DiaryQueryOptions(max_entry_count=1, confidence_factor=0.0))
        if best and best[0].relatedness > self._plagiarism_threshold:
            return best[0]
        return None

    # -- запись (для фоновых задач) -----------------------------------------

    async def save(self, entry: DiaryEntry) -> None:
        """
        Персистит запись на диск и обновляет кэш.

        Не проверяет на дубли сама — эта проверка (`is_duplicate_of`) требует
        эмбеддинга запроса и явно вызывается на уровне memory/rag.py ДО save().
        """
        path = self._diary_dir / entry.filename
        # Атомарно: запись дневника — это прожитый вечер, и терять его из-за
        # того, что телефон выключился посреди сохранения, нельзя
        # (см. efi/utils/atomic.py).
        await write_text_atomic(path, _serialize_entry(entry))
        cache = await self._ensure_loaded()
        cache[entry.id] = entry

    async def touch(self, entry_id: str) -> DiaryEntry | None:
        """
        Отмечает использование записи (usage_count/last_used) БЕЗ пересчёта
        эмбеддинга — сам инкремент in-memory мгновенный, но результат всё
        равно персистится на диск через save(). Если вызывающей стороне не
        нужно дожидаться этой записи, оборачивайте вызов в asyncio.create_task().
        """
        cache = await self._ensure_loaded()
        entry = cache.get(entry_id)
        if entry is None:
            return None
        entry.metadata.touch()
        await self.save(entry)
        return entry

    async def delete(self, entry_id: str) -> None:
        """
        Удаляет запись с диска и из кэша. Используется программной
        консолидацией (memory/consolidation.py) при dedup/сжатии старых
        записей в мемуары — обычный диалоговый путь записи в дневник
        (memory/rag.py) удаление не вызывает никогда.
        """
        cache = await self._ensure_loaded()
        entry = cache.pop(entry_id, None)
        if entry is None:
            return
        try:
            await aiofiles.os.remove(self._diary_dir / entry.filename)
        except FileNotFoundError:
            pass


def _score_entries(
    query_embedding: EmbeddingVector,
    entries: list[DiaryEntry],
    confidence_factor: float,
) -> list[DiaryQueryResult]:
    """
    Синхронная CPU-bound часть запроса: векторизованное косинусное сходство
    через numpy. Вызывается исключительно через asyncio.to_thread.
    """
    dimension = len(query_embedding)
    same_dimension_entries = [entry for entry in entries if len(entry.metadata.embedding) == dimension]
    skipped = len(entries) - len(same_dimension_entries)
    if skipped:
        logger.warning(
            "diary: skipping %d entries with embedding dimensionality mismatch (expected %d)", skipped, dimension
        )
    if not same_dimension_entries:
        return []

    query_vector = np.asarray(query_embedding, dtype=np.float64)
    matrix = np.asarray([entry.metadata.embedding for entry in same_dimension_entries], dtype=np.float64)

    query_norm = np.linalg.norm(query_vector)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    denom = matrix_norms * query_norm
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = np.where(denom > 0, matrix @ query_vector / denom, 0.0)
    normalized = (cosine + 1.0) / 2.0  # cosine ∈ [-1, 1] -> [0, 1]

    results: list[DiaryQueryResult] = []
    for entry, relatedness in zip(same_dimension_entries, normalized, strict=True):
        adjusted = float(np.clip(relatedness + confidence_factor * entry.metadata.confidence, 0.0, 1.0))
        results.append(DiaryQueryResult(entry=entry, relatedness=adjusted))
    return results


def _parse_entry(*, entry_id: str, raw: str) -> DiaryEntry:
    """
    Разбирает markdown-файл записи: необязательный front-matter блок между
    `---`-разделителями (JSON), затем свободный текст.

    Формат специально совместим по виду с YAML front-matter (та же
    разделительная нотация), но парсится как JSON — JSON является валидным
    подмножеством YAML, а собственный парсер избавляет от зависимости на PyYAML.
    """
    metadata = DiaryEntryMetadata()
    body = raw
    stripped = raw.lstrip()
    if stripped.startswith(_FRONT_MATTER_DELIMITER):
        remainder = stripped[len(_FRONT_MATTER_DELIMITER) :]
        closing_index = remainder.find(_FRONT_MATTER_DELIMITER)
        if closing_index != -1:
            front_matter_raw = remainder[:closing_index].strip()
            body = remainder[closing_index + len(_FRONT_MATTER_DELIMITER) :].lstrip("\n")
            if front_matter_raw:
                metadata = DiaryEntryMetadata.model_validate(json.loads(front_matter_raw))
    return DiaryEntry(id=entry_id, metadata=metadata, body=body)


def _serialize_entry(entry: DiaryEntry) -> str:
    """Сериализует запись обратно в markdown: JSON front-matter + свободный текст."""
    front_matter = entry.metadata.model_dump_json(indent=2)
    return f"{_FRONT_MATTER_DELIMITER}\n{front_matter}\n{_FRONT_MATTER_DELIMITER}\n{entry.body}"


__all__ = ["Diary"]
