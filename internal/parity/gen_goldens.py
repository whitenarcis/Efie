"""
internal/parity/gen_goldens.py

Генератор «золотых значений» для Go-тестов паритета.

Зачем. Go-версия памяти обязана быть взаимозаменяемой с Python-версией на одном
и том же каталоге дневника и на одних и тех же настройках. Проверять это
«похожими числами, которые я сам придумал» бессмысленно: важно совпадение с
тем, что РЕАЛЬНО делает Python. Поэтому здесь вызывается настоящий код
(`efi.memory.tfidf_fallback`, `efi.memory.diary`, `efi.llm.schemas`) и
результат складывается в `testdata/goldens.json`, с которым сравнивается Go.

Запуск (из корня репозитория, в venv с numpy/pydantic/aiofiles):

    PYTHONPATH=src python internal/parity/gen_goldens.py

Файл перегенерируется при изменении Python-стороны: если тест паритета упал
после правки Python-кода, это не «шумный тест», а сообщение о том, что Go-версию
нужно править вместе с ней, иначе поиск начнёт расходиться между реализациями.
"""

from __future__ import annotations

import asyncio
import json
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))

from efi.llm.schemas import DiaryEntry, DiaryEntryMetadata, DiaryQueryOptions  # noqa: E402
from efi.memory.diary import Diary, _parse_entry, _serialize_entry  # noqa: E402
from efi.memory.tfidf_fallback import (  # noqa: E402
    TfidfFallbackIndex,
    should_use_tfidf_shortcut,
    tokenize,
)

TOKENIZE_CASES = [
    "Привет, как дела? Мама_мыла раму 123 test",
    "naïve café",
    "² ³ ½ ⅓ ٣ Ⅷ ① 5",
    "ЁЖИК ёжик",
    "привет\u0301 测试 emoji👍",
    "«Вот» — сказала она…",
    "",
    "   ",
    "test123и  foo_bar",
]

SHORTCUT_CASES = [
    "привет", "ок да", "привет как дела", "Привет!", "", "   ", "?!!", "12", "спасибо большое за вчерашнее",
]

CORPUS = [
    ("2026-01-01_aaaaaa", "Сегодня мы говорили про монтаж звука, и он оказался сложнее, чем я думала."),
    ("2026-01-02_bbbbbb", "Утром обсуждали SQLite и WAL: почему база данных не теряет данные при выключении."),
    ("2026-01-03_cccccc", "Вечером перебирала дорожки в редакторе, звук стал ровнее, мне понравилось."),
    ("2026-01-04_dddddd", "Он спросил про промпты, я объяснила, что память важнее размера контекста."),
    ("2026-01-05_eeeeee", "Гуляли под дождём, промокли, но было смешно."),
]
TFIDF_QUERIES = ["монтаж звука", "sqlite база данных", "память контекст промпты", "дождь", "привет"]

VECTOR_QUERY = [0.5, 0.5, 0.5, 0.5]
VECTOR_ENTRIES = [
    ("vec-1", 0.7, [1.0, 0.0, 0.0, 0.0], "Первая запись."),
    ("vec-2", -0.5, [0.9, 0.1, 0.0, 0.0], "Почти дубль первой."),
    ("vec-3", 1.0, [0.0, 0.0, 1.0, 0.0], "Перпендикулярная запись."),
    ("vec-4", 0.0, [0.0, 0.0, 0.0, 0.0], "Нулевой вектор."),
    ("vec-5", 0.2, [0.5, 0.5, 0.5, 0.5], "Полное совпадение с запросом."),
]

SERIALIZE_CASES = [
    {
        "id": "2026-01-02_abc123",
        "score": 0.0,
        "confidence": 0.7,
        "usage_count": 3,
        "domain": "H",
        "created_at": "2026-01-02T03:04:05.123456Z",
        "last_used": None,
        "body": "Прожитый вечер.\n\nВторая строка.",
    },
    {
        "id": "2026-02-03_def456",
        "score": 0.42,
        "confidence": 1.0,
        "usage_count": 0,
        "domain": "C",
        "created_at": "2026-02-03T04:05:06Z",
        "last_used": "2026-02-04T05:06:07Z",
        "body": "Тело без перевода строки в конце",
    },
]

PARSE_CASES = [
    {"id": "plain", "raw": "no front matter\nbody"},
    {
        "id": "with-front-matter",
        "raw": '---\n{"confidence": 0.5, "domain": "P", "usage_count": 2}\n---\n\n\nТело записи ',
    },
    {"id": "empty-front-matter", "raw": '---\n{}\n---\nТело'},
    {"id": "leading-whitespace", "raw": '  \n\n---\n{"confidence": -1.0}\n---\nТело после пробелов'},
    {"id": "unterminated", "raw": '---\n{"confidence": 0.5}\nбез закрывающего разделителя'},
    {
        "id": "old-format-embedding",
        "raw": '---\n{"confidence": 0.5, "embedding": [0.1, 0.2, 0.3]}\n---\nСтарый формат: вектор внутри front-matter',
    },
    {
        "id": "explicit-created-at",
        "raw": '---\n{"confidence": 0.25, "created_at": "2026-05-06T07:08:09.123456Z"}\n---\nЗапись с явной датой',
    },
]


def timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def build_tfidf_goldens() -> dict:
    entries = [
        DiaryEntry(id=entry_id, metadata=DiaryEntryMetadata(confidence=0.5), body=body)
        for entry_id, body in CORPUS
    ]

    async def score() -> list[dict]:
        index = TfidfFallbackIndex()
        await index.index(entries)
        queries = []
        for text in TFIDF_QUERIES:
            results = await index.search(text, DiaryQueryOptions(max_entry_count=4))
            queries.append(
                {
                    "text": text,
                    "max_entry_count": 4,
                    "results": [{"id": result.entry.id, "relatedness": result.relatedness} for result in results],
                }
            )
        return queries

    return {
        "corpus": [{"id": entry_id, "body": body} for entry_id, body in CORPUS],
        "queries": asyncio.run(score()),
    }


def build_diary_query_goldens() -> dict:
    """Прогон НАСТОЯЩЕГО Diary: запись файлов на диск + query с порогами."""

    async def run(root: Path) -> dict:
        diary = Diary(root / "diary", plagiarism_threshold=0.97)
        for entry_id, confidence, embedding, body in VECTOR_ENTRIES:
            await diary.save(
                DiaryEntry(
                    id=entry_id,
                    metadata=DiaryEntryMetadata(
                        confidence=confidence,
                        embedding=embedding,
                        domain="H",
                        created_at=timestamp("2026-03-01T00:00:00Z"),
                    ),
                    body=body,
                )
            )
        options = DiaryQueryOptions(confidence_factor=0.01, max_entry_count=3, min_relatedness=0.5)
        results = await diary.query(VECTOR_QUERY, options)
        return {
            "entries": [
                {"id": entry_id, "confidence": confidence, "embedding": embedding, "body": body}
                for entry_id, confidence, embedding, body in VECTOR_ENTRIES
            ],
            "query": VECTOR_QUERY,
            "options": {"confidence_factor": 0.01, "max_entry_count": 3, "min_relatedness": 0.5},
            "results": [{"id": result.entry.id, "relatedness": result.relatedness} for result in results],
            "files": sorted(path.name for path in (root / "diary").iterdir()),
        }

    with tempfile.TemporaryDirectory() as tmp:
        return asyncio.run(run(Path(tmp)))


def build_serialize_goldens() -> list[dict]:
    goldens = []
    for case in SERIALIZE_CASES:
        entry = DiaryEntry(
            id=case["id"],
            metadata=DiaryEntryMetadata(
                score=case["score"],
                confidence=case["confidence"],
                usage_count=case["usage_count"],
                domain=case["domain"],
                created_at=timestamp(case["created_at"]),
                last_used=timestamp(case["last_used"]) if case["last_used"] else None,
            ),
            body=case["body"],
        )
        goldens.append({"case": case, "expected": _serialize_entry(entry)})
    return goldens


def build_parse_goldens() -> list[dict]:
    goldens = []
    for case in PARSE_CASES:
        entry = _parse_entry(entry_id=case["id"], raw=case["raw"])
        # created_at в Python-версии — default_factory, то есть «момент чтения»:
        # у файла без этого поля дату сравнить побайтово невозможно (она разная
        # в момент генерации золотых значений и в момент прогона Go-теста).
        # Поэтому для таких случаев фиксируется не значение, а САМ ФАКТ дефолта:
        # Go-тест проверяет, что дата проставлена только что, а не осталась нулевой.
        defaulted_created_at = '"created_at"' not in case["raw"]
        goldens.append(
            {
                "case": case,
                "expected": {
                    "body": entry.body,
                    "confidence": entry.metadata.confidence,
                    "domain": entry.metadata.domain,
                    "usage_count": entry.metadata.usage_count,
                    "score": entry.metadata.score,
                    "created_at_defaulted": defaulted_created_at,
                    "created_at": None if defaulted_created_at else entry.metadata.created_at.isoformat(),
                    "embedding": entry.metadata.embedding,
                },
            }
        )
    return goldens


def main() -> None:
    sidecar_embedding = [1.0, -0.5, 0.0, 0.125]
    payload = {
        "generated_by": "internal/parity/gen_goldens.py",
        "python_version": sys.version.split()[0],
        "tokenize": [{"text": text, "tokens": tokenize(text)} for text in TOKENIZE_CASES],
        "shortcut": [{"text": text, "shortcut": should_use_tfidf_shortcut(text)} for text in SHORTCUT_CASES],
        "tfidf": build_tfidf_goldens(),
        "diary_query": build_diary_query_goldens(),
        "serialize": build_serialize_goldens(),
        "parse": build_parse_goldens(),
        "sidecar": {
            "embedding": sidecar_embedding,
            "expected": json.dumps({"embedding": sidecar_embedding}),
        },
    }
    target = Path(__file__).resolve().parent / "testdata" / "goldens.json"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"written {target}")


if __name__ == "__main__":
    main()
