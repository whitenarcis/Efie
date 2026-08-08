"""
Тесты для efi.memory.consolidation.DiaryConsolidator.summarize_stale_entries.

Регрессия: критерий устаревания раньше читал last_used (когда запись
последний раз НАШЛИ поиском) с фолбэком на "максимально старую дату" для
записей, которые ещё ни разу не искали (last_used is None) — а это ЛЮБАЯ
только что созданная запись. На практике это значило, что дневник за целый
день переписки мог в ту же ночь схлопнуться в один сжатый "мемуар": свежие
записи выглядели как САМЫЕ старые кандидаты на сжатие. Критерий должен
смотреть на created_at (когда запись реально появилась), а не last_used.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from efi.llm.schemas import Choice, DiaryEntry, DiaryEntryMetadata, LLMParams, Message, Response, Role, Session
from efi.memory.consolidation import DiaryConsolidator
from efi.memory.diary import Diary


class _FakeRouter:
    def __init__(self) -> None:
        self.calls = 0

    async def chat(self, role: object, params: LLMParams, session: Session) -> Response:
        self.calls += 1
        return Response(choices=[Choice(message=Message(role=Role.ASSISTANT, content="сжатая сводка старых записей"))])


async def test_freshly_created_entry_is_not_swept_up_as_stale(tmp_path: Path) -> None:
    diary = Diary(tmp_path / "diary")
    now = datetime.now(timezone.utc)

    fresh = DiaryEntry(
        id="fresh_1",
        metadata=DiaryEntryMetadata(confidence=0.5),  # created_at=now по умолчанию, last_used=None
        body="только что записанное воспоминание за сегодня",
    )
    old_one = DiaryEntry(
        id="old_1",
        metadata=DiaryEntryMetadata(confidence=0.5, created_at=now - timedelta(days=60)),
        body="старое воспоминание номер один",
    )
    old_two = DiaryEntry(
        id="old_2",
        metadata=DiaryEntryMetadata(confidence=0.5, created_at=now - timedelta(days=61)),
        body="старое воспоминание номер два",
    )
    await diary.save(fresh)
    await diary.save(old_one)
    await diary.save(old_two)

    router = _FakeRouter()
    consolidator = DiaryConsolidator(diary, router, rag=None)  # type: ignore[arg-type]
    merged = await consolidator.summarize_stale_entries(older_than=timedelta(days=30))

    assert merged is not None
    remaining_ids = {entry.id for entry in await diary.all_entries()}
    assert "fresh_1" in remaining_ids, "свежая запись без last_used не должна была уйти в сжатие"
    assert "old_1" not in remaining_ids
    assert "old_2" not in remaining_ids


async def test_returns_none_when_fewer_than_two_stale_candidates(tmp_path: Path) -> None:
    diary = Diary(tmp_path / "diary")
    fresh = DiaryEntry(id="fresh_1", metadata=DiaryEntryMetadata(confidence=0.5), body="сегодняшняя запись")
    await diary.save(fresh)

    router = _FakeRouter()
    consolidator = DiaryConsolidator(diary, router, rag=None)  # type: ignore[arg-type]
    merged = await consolidator.summarize_stale_entries(older_than=timedelta(days=30))

    assert merged is None
    assert router.calls == 0
