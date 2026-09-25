"""
internal/parity/bench_python.py — зеркальный Python-бенч подсистемы памяти.

Точная копия того, что делает `cmd/efie-memory -bench` в Go: тот же генератор
корпуса (слова, LCG-векторы, id), те же запросы и опции — чтобы сравнение RAM
и CPU было честным, а не «по ощущениям».

Запуск (из корня репозитория, в venv с numpy/pydantic/aiofiles):

    python internal/parity/bench_python.py gen  /tmp/efie-bench-corpus
    python internal/parity/bench_python.py run  /tmp/efie-bench-corpus

Генерация и запуск разделены намеренно: Python не возвращает память ОС, и при
замере в одном процессе RSS включал бы остатки аллокатора от генерации.
Свежий процесс, который только читает готовый корпус, — эквивалентное
условие тому, что Go-бенч меряет после сборки GC.

Сравнение:

    go run ./cmd/efie-memory -bench 5000 -bench-dims 1024 -bench-queries 200
"""

from __future__ import annotations

import asyncio
import math
import sys
import tempfile
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from efi.llm.schemas import DiaryEntry, DiaryEntryMetadata, DiaryQueryOptions  # noqa: E402
from efi.memory.diary import Diary  # noqa: E402
from efi.memory.tfidf_fallback import TfidfFallbackIndex  # noqa: E402

N = 5000
DIMS = 1024
QUERIES = 200

# Ровно то, что в cmd/efie-memory/bench.go.
BODY_WORDS = [
    "сегодня", "он", "сказал", "что", "монтаж", "звука", "оказался", "сложнее",
    "чем", "я", "думала", "и", "весь", "вечер", "перебирала", "дорожки",
    "в", "редакторе", "смешно", "но", "мне", "это", "понравилось", "утром",
    "мы", "говорили", "про", "базы", "данных", "sqlite", "её",
    "промпты", "стали", "короче", "после", "правки", "памяти",
]

MASK64 = (1 << 64) - 1


def rss_bytes() -> int:
    with open("/proc/self/status", encoding="ascii") as status:
        for line in status:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return 0


def mb(value: int) -> float:
    return value / (1 << 20)


def synthetic_body(seed: int) -> str:
    length = 20 + seed % 40
    parts = [BODY_WORDS[(seed * 7 + index * 13) % len(BODY_WORDS)] for index in range(length)]
    # Go bench добавляет пробел после КАЖДОГО слова (включая последнее) —
    # сохраняем ровно ту же длину тела, чтобы корпус был байт в байт одинаковым.
    return " ".join(parts) + " "


def synthetic_query(seed: int) -> str:
    return "%s %s %s %s" % (
        BODY_WORDS[seed % len(BODY_WORDS)],
        BODY_WORDS[(seed + 5) % len(BODY_WORDS)],
        BODY_WORDS[(seed + 11) % len(BODY_WORDS)],
        "памяти",
    )


def synthetic_vector(seed: int, dims: int) -> list[float]:
    """Тот же LCG, что в Go-бенче (64-бит переполнение, seed → единичный вектор)."""
    state = (seed * 6364136223846793005 + 1442695040888963407) & MASK64
    vector = [0.0] * dims
    norm = 0.0
    for index in range(dims):
        state = (state * 6364136223846793005 + 1442695040888963407) & MASK64
        value = float(state >> 11) / float(1 << 53) * 2 - 1
        vector[index] = value
        norm += value * value
    norm = math.sqrt(norm)
    if norm == 0:
        return vector
    return [value / norm for value in vector]


def synthetic_entry(seed: int) -> DiaryEntry:
    from datetime import UTC, datetime

    metadata = DiaryEntryMetadata(
        confidence=0.7,
        domain="H",
        created_at=datetime.fromtimestamp(1_700_000_000 + seed, tz=UTC),
        embedding=synthetic_vector(seed, DIMS),
    )
    return DiaryEntry(
        id=f"{1_700_000_000 + seed}_bench{seed:06d}",
        metadata=metadata,
        body=synthetic_body(seed),
    )


async def main() -> None:
    # Два режима, чтобы RSS был честным: Python не возвращает память ОС, и
    # если генерация и загрузка идут в одном процессе, замер включает остатки
    # аллокатора от генерации. Go-бенч грузит в том же процессе, но GC там
    # отдаёт освобождённое — эквивалентные условия даёт только свежий
    # процесс, который ТОЛЬКО читает уже готовый корпус.
    #   python bench_python.py gen  <dir>  — написать корпус
    #   python bench_python.py run  <dir>  — холодная загрузка + поиск
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    directory = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
        tempfile.mkdtemp(prefix="efie-memory-bench-py")
    )

    if mode in ("gen", "all"):
        generated = time.perf_counter()
        diary = Diary(directory)
        for seed in range(N):
            await diary.save(synthetic_entry(seed))
        generation = time.perf_counter() - generated
        if mode == "gen":
            print(f"генерация корпуса: {generation:.2f} с ({generation * 1000 / N:.2f} мс на запись)")
            return

    baseline = rss_bytes()
    diary = Diary(directory)

    loading = time.perf_counter()
    entries = await diary.all_entries()
    load_duration = time.perf_counter() - loading
    after_load = rss_bytes()

    index = TfidfFallbackIndex()
    building = time.perf_counter()
    await index.index(entries)
    index_duration = time.perf_counter() - building
    after_index = rss_bytes()

    options = DiaryQueryOptions(max_entry_count=10, min_relatedness=0.80)
    options_vector = DiaryQueryOptions(
        confidence_factor=0.01, max_entry_count=10, min_relatedness=0.80
    )

    search_duration = 0.0
    vector_duration = 0.0
    for seed in range(QUERIES):
        started = time.perf_counter()
        await index.search(synthetic_query(seed), options)
        search_duration += time.perf_counter() - started

        started = time.perf_counter()
        await diary.query(synthetic_vector(seed, DIMS), options_vector)
        vector_duration += time.perf_counter() - started

    final = rss_bytes()

    print("efie-memory benchmark (Python)")
    print(f"  corpus:            {len(entries)} записей, размерность {DIMS}, проиндексировано {len(index._documents)}")
    print(f"  холодная загрузка: {load_duration:.2f} с ({load_duration * 1000 / N:.3f} мс на запись: чтение .md + sidecar + разбор)")
    print(f"  сборка TF-IDF:     {index_duration:.2f} с")
    print(f"  поиск (TF-IDF, {QUERIES} запросов): {search_duration / QUERIES * 1000:.3f} мс на запрос")
    print(f"  поиск (векторный, без TF-IDF и дедупликации): {vector_duration / QUERIES * 1000:.3f} мс на запрос")
    print(
        f"  RSS: старт {mb(baseline):.1f} МБ -> после загрузки {mb(after_load):.1f} МБ -> "
        f"после индекса {mb(after_index):.1f} МБ (итого {mb(final):.1f} МБ)"
    )
    print(
        f"  прирост RSS на корпус: {mb(final - baseline):.1f} МБ "
        f"({(final - baseline) / N / 1024:.1f} КБ на запись, включая вектора в памяти)"
    )


if __name__ == "__main__":
    asyncio.run(main())
