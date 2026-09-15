"""
Тесты компактного формата дневника: эмбеддинг в sidecar, а не в .md.

Регрессия: вектор из сотен чисел лежал прямо в front-matter `.md` и раздувал
каждый файл до ~25 КБ при теле в ~1 КБ — файл не читался ни глазами, ни
`git diff`. Теперь `.md` — только метаданные без вектора + тело, а вектор
живёт рядом в `<id>.vec.json`. Старые файлы с вектором внутри читаются как
раньше и сжимаются при первом `save()` или явной миграцией.
"""

from __future__ import annotations

from pathlib import Path

from efi.llm.schemas import DiaryEntry, DiaryEntryMetadata
from efi.memory.diary import Diary, dedup_diary_corpus, migrate_diary_corpus, prune_unfinished_entries


async def test_saved_file_has_no_embedding_in_front_matter(tmp_path: Path) -> None:
    diary = Diary(tmp_path / "diary")
    await diary.save(
        DiaryEntry(id="e1", metadata=DiaryEntryMetadata(confidence=0.5, embedding=[0.1, 0.2]), body="Прожитый вечер.")
    )

    raw = (tmp_path / "diary" / "e1.md").read_text(encoding="utf-8")
    assert "Прожитый вечер." in raw
    assert "[0.1, 0.2]" not in raw and "0.1" not in raw.split("---")[1]

    sidecar = (tmp_path / "diary" / "e1.vec.json").read_text(encoding="utf-8")
    assert "0.1" in sidecar


async def test_embedding_survives_a_reload_from_sidecar(tmp_path: Path) -> None:
    await Diary(tmp_path / "diary").save(
        DiaryEntry(id="e1", metadata=DiaryEntryMetadata(confidence=0.5, embedding=[0.1, 0.2]), body="Прожитый вечер.")
    )

    reloaded = await Diary(tmp_path / "diary").get("e1")

    assert reloaded is not None and reloaded.metadata.embedding == [0.1, 0.2]


async def test_old_format_with_inline_embedding_still_reads(tmp_path: Path) -> None:
    (tmp_path / "diary").mkdir(parents=True)
    (tmp_path / "diary" / "old.md").write_text(
        '---\n{"confidence": 0.5, "embedding": [0.2, 0.3]}\n---\nСтарая запись. Было дело.',
        encoding="utf-8",
    )

    reloaded = await Diary(tmp_path / "diary").get("old")

    assert reloaded is not None and reloaded.metadata.embedding == [0.2, 0.3]


async def test_migration_compacts_old_files(tmp_path: Path) -> None:
    (tmp_path / "diary").mkdir(parents=True)
    big_vector = ", ".join(["0.123456789"] * 300)
    (tmp_path / "diary" / "old.md").write_text(
        '{"confidence": 0.5, "embedding": [' + big_vector + "]}\n---\nСтарая запись. Было дело.",
        encoding="utf-8",
    )
    # Рукописный front-matter без открывающего разделителя парсером не
    # считается — дописываем его, как пишет _serialize_entry.
    raw = (tmp_path / "diary" / "old.md").read_text(encoding="utf-8")
    (tmp_path / "diary" / "old.md").write_text("---\n" + raw, encoding="utf-8")
    before = (tmp_path / "diary" / "old.md").stat().st_size

    migrated = await migrate_diary_corpus(Diary(tmp_path / "diary"))

    assert migrated == 1
    after = (tmp_path / "diary" / "old.md").read_text(encoding="utf-8")
    assert "0.123456789" not in after.split("---")[1]
    assert len(after.encode()) < before
    assert (tmp_path / "diary" / "old.vec.json").exists()


async def test_deleting_an_entry_removes_its_sidecar(tmp_path: Path) -> None:
    diary = Diary(tmp_path / "diary")
    await diary.save(
        DiaryEntry(id="e1", metadata=DiaryEntryMetadata(confidence=0.5, embedding=[0.1]), body="Прожитый вечер.")
    )
    assert (tmp_path / "diary" / "e1.vec.json").exists()

    await diary.delete("e1")

    assert not (tmp_path / "diary" / "e1.vec.json").exists()


async def test_prune_reports_but_does_not_delete_by_default(tmp_path: Path) -> None:
    diary = Diary(tmp_path / "diary")
    await diary.save(DiaryEntry(id="ok", metadata=DiaryEntryMetadata(confidence=0.5), body="Целая мысль. Было хорошо."))
    await diary.save(
        DiaryEntry(id="cut", metadata=DiaryEntryMetadata(confidence=0.5), body="...может отличаться от реа (https://x")
    )

    candidates = await prune_unfinished_entries(diary, dry_run=True)

    assert candidates == ["cut"]
    assert await diary.get("cut") is not None


async def test_prune_apply_deletes_only_the_broken_ones(tmp_path: Path) -> None:
    diary = Diary(tmp_path / "diary")
    await diary.save(DiaryEntry(id="ok", metadata=DiaryEntryMetadata(confidence=0.5), body="Целая мысль. Было хорошо."))
    await diary.save(
        DiaryEntry(id="cut", metadata=DiaryEntryMetadata(confidence=0.5), body="...может отличаться от реа (https://x")
    )

    deleted = await prune_unfinished_entries(diary, dry_run=False)

    assert deleted == ["cut"]
    assert await diary.get("ok") is not None


async def test_dedup_reports_but_does_not_delete_by_default(tmp_path: Path) -> None:
    diary = Diary(tmp_path / "diary")
    await diary.save(
        DiaryEntry(
            id="duplicate_1",
            metadata=DiaryEntryMetadata(confidence=0.5, embedding=[0.1, 0.2, 0.3]),
            body="Первая запись о гуглёже. Это было интересно.",
        )
    )
    await diary.save(
        DiaryEntry(
            id="duplicate_2",
            metadata=DiaryEntryMetadata(confidence=0.5, embedding=[0.11, 0.19, 0.31]),
            body="Тоже про гуглёж, другими словами. Тоже интересно.",
        )
    )
    await diary.save(
        DiaryEntry(
            id="unique",
            metadata=DiaryEntryMetadata(confidence=0.5, embedding=[0.9, 0.1, 0.0]),
            body="Совсем другая тема — про музыку.",
        )
    )

    pairs = await dedup_diary_corpus(diary, dry_run=True, threshold=0.90)

    assert len(pairs) == 1
    assert pairs[0] == ("duplicate_1", "duplicate_2")
    assert await diary.get("duplicate_1") is not None
    assert await diary.get("duplicate_2") is not None


async def test_dedup_apply_keeps_the_longer_entry(tmp_path: Path) -> None:
    diary = Diary(tmp_path / "diary")
    await diary.save(
        DiaryEntry(
            id="short",
            metadata=DiaryEntryMetadata(confidence=0.5, embedding=[0.1, 0.2, 0.3]),
            body="Короткая.",
        )
    )
    await diary.save(
        DiaryEntry(
            id="long",
            metadata=DiaryEntryMetadata(confidence=0.5, embedding=[0.11, 0.19, 0.31]),
            body="Более длинная версия той же самой мысли, с деталями.",
        )
    )

    pairs = await dedup_diary_corpus(diary, dry_run=False, threshold=0.90)

    assert pairs == [("short", "long")]
    assert await diary.get("short") is None
    remaining = await diary.get("long")
    assert remaining is not None and remaining.metadata.usage_count == 1
    assert await diary.get("cut") is None
