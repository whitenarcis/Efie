"""
efi/memory/diary.py

Дневник — долгосрочная память Эфи, хранящаяся как markdown-файлы.
Каждая запись — отдельный файл `<id>.md` в `paths.diary_dir` с компактным
JSON front-matter (score/confidence/last_used/usage_count/domain/created_at)
и свободным текстом-телом.

Эмбеддинг НЕ лежит в `.md`: он хранится рядом в sidecar-файле `<id>.vec.json`.
Причина — именно та, что видна глазами: вектор из сотен чисел раздувал
каждый файл до ~25 КБ при теле в ~1 КБ, файл становился нечитаемым, а
`git diff` — шумом. Чтение умеет и старый формат (эмбеддинг внутри
front-matter) — такие файлы подхватываются как есть и сжимаются при первом
же `save()` или явной миграцией `migrate_embeddings_to_sidecars()`.

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
import contextlib
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
        # Sidecar-файлы эмбеддингов (`*.vec.json`) — не записи, их пропускаем.
        paths = await asyncio.to_thread(lambda: sorted(self._diary_dir.glob("*.md")))
        entries = await asyncio.gather(*(self._read_entry(path) for path in paths))
        return {entry.id: entry for entry in entries if entry is not None}

    def _sidecar_path(self, entry_id: str) -> Path:
        # Валидация id уже гарантирует отсутствие слешей (см. DiaryEntry),
        # так что склейка имени здесь безопасна.
        return self._diary_dir / f"{entry_id}.vec.json"

    async def _read_entry(self, path: Path) -> DiaryEntry | None:
        try:
            async with aiofiles.open(path, encoding="utf-8") as f:
                raw = await f.read()
        except OSError as exc:
            logger.warning("diary: failed to read %s: %s", path, exc)
            return None
        try:
            entry = _parse_entry(entry_id=path.stem, raw=raw)
        except ValueError as exc:
            logger.warning("diary: failed to parse %s: %s", path, exc)
            return None
        # Старый формат хранил эмбеддинг прямо в front-matter нового
        # формата его там уже нет — добираем из sidecar-файла рядом.
        if not entry.metadata.embedding:
            sidecar = self._sidecar_path(entry.id)
            try:
                async with aiofiles.open(sidecar, encoding="utf-8") as f:
                    payload = json.loads(await f.read())
            except FileNotFoundError:
                payload = None
            except (OSError, ValueError) as exc:
                logger.warning("diary: failed to read sidecar %s: %s", sidecar, exc)
                payload = None
            if isinstance(payload, dict):
                vector = payload.get("embedding", [])
                if isinstance(vector, list) and all(isinstance(x, (int, float)) for x in vector):
                    entry.metadata.embedding = [float(x) for x in vector]
        return entry

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
        chat_id: int | None = None,
    ) -> list[DiaryQueryResult]:
        """
        Асинхронный поиск по дневнику: косинусное сходство `query_embedding`
        с эмбеддингом каждой записи, слегка скорректированное на `confidence`
        (аналог Diary::query у референса).

        Записи без эмбеддинга в выборку не попадают — подразумевается, что
        эмбеддинг проставляется при сохранении (memory/rag.py), а не
        досчитывается на лету в критическом пути чтения.

        Если передан `chat_id`, записи с этим chat_id в метаданных получают
        буст — Эфи лучше помнит то, что обсуждалось с конкретным собеседником.
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
        results = await asyncio.to_thread(
            _score_entries, query_embedding, candidates, options.confidence_factor, 0.1, chat_id
        )
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

        `.md`-файл — компактный и читаемый глазами (без вектора), вектор
        уходит в sidecar `<id>.vec.json`. Старый `.md` с вектором внутри
        при пересохранении тоже сжимается: `_serialize_entry` вектор не пишет.
        """
        path = self._diary_dir / entry.filename
        # Атомарно: запись дневника — это прожитый вечер, и терять его из-за
        # того, что телефон выключился посреди сохранения, нельзя
        # (см. efi/utils/atomic.py).
        await write_text_atomic(path, _serialize_entry(entry))
        if entry.metadata.embedding:
            await write_text_atomic(
                self._sidecar_path(entry.id),
                json.dumps({"embedding": entry.metadata.embedding}),
            )
        else:
            # Вектора нет (запись без эмбеддинга) — залежавшийся sidecar от
            # прошлой версии записи не должен врать, что вектор есть.
            with contextlib.suppress(OSError):
                await aiofiles.os.remove(self._sidecar_path(entry.id))
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
        with contextlib.suppress(OSError):
            await aiofiles.os.remove(self._sidecar_path(entry_id))

    async def migrate_embeddings_to_sidecars(self) -> int:
        """
        Разовая чистка корпуса: выносит векторы из старых `.md` в sidecar-файлы.

        Чтению миграция не нужна (старый формат подхватывается сам), но
        файлам — да: без неё каждый старый файл так и останется нечитаемой
        простынёй на ~25 КБ, пока его не перезапишет `touch()`/`save()`.
        Возвращает число ужатых файлов.
        """
        cache = await self._ensure_loaded()
        migrated = 0
        for entry in list(cache.values()):
            raw_path = self._diary_dir / entry.filename
            try:
                async with aiofiles.open(raw_path, encoding="utf-8") as f:
                    raw = await f.read()
            except OSError:
                continue
            if '"embedding"' not in raw and '"embedding":' not in raw:
                continue
            await self.save(entry)
            migrated += 1
        if migrated:
            logger.info("diary: migrated %d entries to sidecar embeddings", migrated)
        return migrated


def _score_entries(
    query_embedding: EmbeddingVector,
    entries: list[DiaryEntry],
    confidence_factor: float,
    importance_factor: float = 0.1,
    chat_id: int | None = None,
) -> list[DiaryQueryResult]:
    """
    Синхронная CPU-bound часть запроса: векторизованное косинусное сходство
    через numpy. Вызывается исключительно через asyncio.to_thread.

    Ранжирование учитывает четыре фактора:
    1. Семантическое сходство (cosine similarity) — основной критерий
    2. Confidence (confidence_factor * confidence) — проверенные факты выше
    3. Importance (importance_factor * importance) — важные записи выше
    4. Chat context (+0.15 для записей с текущим chat_id) — контекстная актуальность
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
        # Буст важности: важные записи поднимаются выше при равном сходстве
        importance_boost = importance_factor * entry.metadata.importance
        # Буст для записей из текущего чата — контекстная актуальность
        chat_boost = 0.15 if (chat_id is not None and chat_id in entry.metadata.chat_ids) else 0.0
        adjusted = float(np.clip(
            relatedness + confidence_factor * entry.metadata.confidence + importance_boost + chat_boost,
            0.0, 1.0
        ))
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
    """
    Сериализует запись в компактный читаемый markdown: JSON front-matter
    БЕЗ эмбеддинга + свободный текст.

    Вектор уходит в sidecar `<id>.vec.json` (см. `Diary.save`): иначе файл
    раздувается до ~25 КБ при теле в ~1 КБ и перестаёт читаться ни глазами,
    ни `git diff`. Старые файлы с вектором внутри читаются как раньше
    (`_parse_entry` их понимает), а при пересохранении сжимаются сами.
    """
    slim = entry.metadata.model_copy(update={"embedding": []})
    front_matter = slim.model_dump_json(indent=2)
    return f"{_FRONT_MATTER_DELIMITER}\n{front_matter}\n{_FRONT_MATTER_DELIMITER}\n{entry.body}"


async def migrate_diary_corpus(diary: Diary) -> int:
    """
    Ужимает весь корпус: выносит векторы из старых `.md` в sidecar-файлы.

    Тонкая обёртка над `Diary.migrate_embeddings_to_sidecars` — чтобы
    скриптам обслуживания не нужно было знать про внутренности класса.
    """
    return await diary.migrate_embeddings_to_sidecars()


async def prune_unfinished_entries(diary: Diary, *, dry_run: bool = True) -> list[str]:
    """
    Находит в корпусе записи-обрубки (тело кончается на полуслове) и — если
    `dry_run=False` — удаляет их.

    Проверяется и честный `finish_reason` провайдера, и вид текста:
    `looks_unfinished` ловит старые записи, сохранённые до появления
    `salvage_truncated` (они так и остались вида «...вернулся с»,
    «...может отличаться от реа (https://...»). Половина мысли хуже, чем её
    отсутствие: обрубок притворяется целым и потом подмешивается RAG-поиском
    в промпт как воспоминание (см. efi/utils/text.py).

    Возвращает id затронутых записей (в dry-run — кандидатов, иначе —
    удалённых). Удаление — только явное: за спиной у владельца память
    не трогаем.
    """
    from efi.utils.text import looks_unfinished

    doomed = [entry.id for entry in await diary.all_entries() if looks_unfinished(entry.body)]
    if not dry_run:
        for entry_id in doomed:
            await diary.delete(entry_id)
        if doomed:
            logger.info("diary: pruned %d unfinished entries", len(doomed))
    return doomed


async def dedup_diary_corpus(diary: Diary, *, dry_run: bool = True, threshold: float = 0.95) -> list[tuple[str, str]]:
    """
    Находит и (при dry_run=False) удаляет семантические дубли в корпусе.

    Сравнивает попарно ВСЕ записи с эмбеддингами; если косинус >= threshold —
    пара считается дублем. Из пары удаляется более короткая (при равенстве —
    более старая), а у оставшейся растёт usage_count. Это аналог
    Deduplicate из efi/memory/consolidation.py, но для ручного запуска из
    скрипта и только по семантике (без слияния в мемуары).

    Возвращает список (удалённый_id, оставшийся_id). При dry_run=True записи
    не удаляются, только отчёт.
    """
    entries = [entry for entry in await diary.all_entries() if entry.metadata.embedding]
    pairs: list[tuple[str, str]] = []

    # Попарное сравнение O(n^2) — приемлемо для дневника в сотни записей
    # (десятки миллисекунд). Для тысяч записей стоит перейти на FAISS/Annoy.
    for index, candidate in enumerate(entries):
        candidate_vector = np.asarray(candidate.metadata.embedding, dtype=np.float64)
        candidate_norm = float(np.linalg.norm(candidate_vector))
        if candidate_norm <= 0.0:
            continue
        for other in entries[index + 1 :]:
            other_vector = np.asarray(other.metadata.embedding, dtype=np.float64)
            other_norm = float(np.linalg.norm(other_vector))
            if other_norm <= 0.0:
                continue
            cosine = float(np.dot(candidate_vector, other_vector) / (candidate_norm * other_norm))
            if cosine >= threshold:
                # Удаляем более короткую, при равенстве — более старую.
                if len(candidate.body) > len(other.body):
                    keep, remove = candidate, other
                elif len(other.body) > len(candidate.body):
                    keep, remove = other, candidate
                else:
                    if candidate.metadata.created_at >= other.metadata.created_at:
                        keep, remove = candidate, other
                    else:
                        keep, remove = other, candidate
                if keep.id != remove.id:
                    pairs.append((remove.id, keep.id))

    if dry_run:
        if pairs:
            logger.info("diary: dedup candidates %d pairs (dry-run)", len(pairs))
        return pairs

    seen: set[str] = set()
    removed = 0
    for remove_id, keep_id in pairs:
        if remove_id in seen or keep_id in seen:
            continue
        seen.add(remove_id)
        seen.add(keep_id)
        keep_entry = await diary.get(keep_id)
        if keep_entry is not None:
            keep_entry.metadata.usage_count += 1
            await diary.save(keep_entry)
        await diary.delete(remove_id)
        removed += 1
    if removed:
        logger.info("diary: dedup removed %d entries", removed)
    return pairs


__all__ = [
    "Diary",
    "dedup_diary_corpus",
    "migrate_diary_corpus",
    "prune_unfinished_entries",
]
