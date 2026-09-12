"""Тесты для efi.db.sticker_descriptions: кэш «file_unique_id -> описание от vision»."""

from __future__ import annotations

from pathlib import Path

from efi.db.core import Database
from efi.db.models import MIGRATIONS
from efi.db.sticker_descriptions import StickerDescriptionStore


def _store(tmp_path: Path, *, db_name: str = "test.db") -> StickerDescriptionStore:
    database = Database(tmp_path / db_name, migrations=MIGRATIONS)
    return StickerDescriptionStore(database)


async def test_unknown_sticker_is_a_cache_miss(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert await store.find("never_seen") is None


async def test_remember_assigns_a_short_id_and_find_hits(tmp_path: Path) -> None:
    store = _store(tmp_path)

    stored = await store.remember("unique_1", "FILE_1", "кот смеётся")

    assert stored is not None
    assert stored.sticker_id >= 1
    assert stored.description == "кот смеётся"
    assert stored.seen_count == 1

    found = await store.find("unique_1")
    assert found is not None
    assert found.sticker_id == stored.sticker_id
    assert found.file_id == "FILE_1"


async def test_remember_is_idempotent_on_repeat_of_same_unique_id(tmp_path: Path) -> None:
    """Тот же стикер не должен получать второй короткий id — иначе модель
    стала бы видеть дубликаты описаний с разными id."""
    store = _store(tmp_path)

    first = await store.remember("unique_1", "FILE_1", "кот смеётся")
    second = await store.remember("unique_1", "FILE_2", "что-то другое")

    assert first is not None and second is not None
    assert second.sticker_id == first.sticker_id
    # Повторный приход одного и того же стикера не переписывает описание:
    # именно поэтому кэш и существует, а новый file_id — записывается.
    assert second.file_id == "FILE_2"
    assert second.description == "кот смеётся"
    assert second.seen_count == 2


async def test_remember_refuses_to_cache_empty_description(tmp_path: Path) -> None:
    store = _store(tmp_path)
    assert await store.remember("unique_1", "FILE_1", "   ") is None
    assert await store.find("unique_1") is None


async def test_touch_refreshes_file_id_and_counts_without_touching_description(
    tmp_path: Path,
) -> None:
    store = _store(tmp_path)
    await store.remember("unique_1", "OLD_FILE", "кот")

    touched = await store.touch("unique_1", "NEW_FILE")

    assert touched is not None
    assert touched.file_id == "NEW_FILE"
    assert touched.description == "кот"
    assert touched.seen_count == 2
    assert (await store.find("unique_1")).file_id == "NEW_FILE"  # type: ignore[union-attr]


async def test_by_id_resolves_short_id_to_file_id(tmp_path: Path) -> None:
    store = _store(tmp_path)
    stored = await store.remember("unique_1", "FILE_1", "кот")

    assert stored is not None
    resolved = await store.by_id(stored.sticker_id)
    assert resolved is not None
    assert resolved.file_unique_id == "unique_1"
    assert resolved.file_id == "FILE_1"

    assert await store.by_id(999_999) is None


async def test_recent_lists_most_recently_seen_first_and_respects_limit(tmp_path: Path) -> None:
    store = _store(tmp_path)
    for index in range(3):
        await store.remember(f"unique_{index}", f"FILE_{index}", f"стикер {index}")

    recent = await store.recent(limit=2)

    assert [item.file_unique_id for item in recent] == ["unique_2", "unique_1"]


async def test_everything_persists_across_store_instances(tmp_path: Path) -> None:
    """Перезапуск процесса не должен стоить нового vision-запроса: суть кэша."""
    db_path = tmp_path / "shared.db"
    first = StickerDescriptionStore(Database(db_path, migrations=MIGRATIONS))
    stored = await first.remember("unique_1", "FILE_1", "кот смеётся")
    assert stored is not None

    fresh = StickerDescriptionStore(Database(db_path, migrations=MIGRATIONS))
    found = await fresh.find("unique_1")

    assert found is not None
    assert found.sticker_id == stored.sticker_id
    assert found.description == "кот смеётся"
    assert await fresh.by_id(stored.sticker_id) is not None
