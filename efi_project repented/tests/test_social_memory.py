"""
Тесты для efi.memory.social_memory: гарантия сохранения внешнего опыта,
тегирование контекста и доступность этих событий обычному RAG-поиску
(Cross-Context Recall).
"""

from __future__ import annotations

from pathlib import Path

from efi.db.core import Database
from efi.db.models import MIGRATIONS
from efi.llm.schemas import DiaryEntry
from efi.memory.social_memory import (
    TAG_DISCUSSION_THREAD,
    TAG_PUBLIC_COMMENT,
    SocialInteraction,
    SocialInteractionKind,
    SocialInteractionStore,
)


class _RecordingRAG:
    """Подменяет RAGMemory: запоминает, что именно ушло в векторную память."""

    def __init__(self, *, fail: bool = False) -> None:
        self.remembered: list[str] = []
        self._fail = fail

    async def remember(self, body: str, *, confidence: float = 0.0) -> DiaryEntry | None:
        if self._fail:
            raise RuntimeError("vector store is down")
        self.remembered.append(body)
        return DiaryEntry(id="entry_1", body=body)


def _store(tmp_path: Path, *, rag: _RecordingRAG | None = None) -> tuple[SocialInteractionStore, _RecordingRAG]:
    database = Database(tmp_path / "test.db", migrations=MIGRATIONS)
    recorder = rag or _RecordingRAG()
    return SocialInteractionStore(database, rag=recorder), recorder  # type: ignore[arg-type]


def _comment(**overrides: object) -> SocialInteraction:
    defaults: dict[str, object] = dict(
        kind=SocialInteractionKind.PUBLIC_COMMENT,
        text="да там же ядро течёт, это давно известно",
        chat_id=-1001,
        thread_id=55,
        peer_user_id=777,
        peer_name="Рихтер",
        chat_title="Линуксовый канал",
    )
    defaults.update(overrides)
    return SocialInteraction(**defaults)  # type: ignore[arg-type]


# -- тегирование контекста -----------------------------------------------------


def test_public_comment_tags() -> None:
    tags = _comment().build_tags()
    assert TAG_PUBLIC_COMMENT in tags
    assert TAG_DISCUSSION_THREAD in tags
    assert "#channel_-1001" in tags
    assert "#secondary_user_777" in tags


def test_stranger_dm_is_not_tagged_as_public() -> None:
    tags = _comment(kind=SocialInteractionKind.STRANGER_DM, thread_id=None).build_tags()
    assert TAG_PUBLIC_COMMENT not in tags
    assert TAG_DISCUSSION_THREAD not in tags
    assert "#secondary_user_777" in tags


def test_diary_body_reads_as_a_first_person_episode() -> None:
    """Не протокол, а живая фраза — именно её потом достаёт RAG и превращает в отсылку в разговоре."""
    body = _comment().render_for_diary()
    assert "Написала комментарий" in body
    assert "Линуксовый канал" in body
    assert TAG_PUBLIC_COMMENT in body


def test_received_reply_names_the_person() -> None:
    body = _comment(kind=SocialInteractionKind.RECEIVED_REPLY).render_for_diary()
    assert "Рихтер ответил мне" in body


# -- гарантия сохранения --------------------------------------------------------


async def test_record_writes_both_layers(tmp_path: Path) -> None:
    store, rag = _store(tmp_path)
    interaction_id = await store.record(_comment())

    assert interaction_id > 0
    assert len(rag.remembered) == 1
    assert TAG_PUBLIC_COMMENT in rag.remembered[0]

    recorded = await store.recent(limit=5)
    assert len(recorded) == 1
    assert recorded[0].kind is SocialInteractionKind.PUBLIC_COMMENT


async def test_journal_survives_a_broken_vector_layer(tmp_path: Path) -> None:
    """
    Ключевая гарантия: векторный слой отвечает за извлекаемость, журнал —
    за сам факт. Падение первого не должно терять второй.
    """
    store, _rag = _store(tmp_path, rag=_RecordingRAG(fail=True))
    interaction_id = await store.record(_comment())

    assert interaction_id > 0
    assert len(await store.recent(limit=5)) == 1


async def test_recent_with_peer_filters_by_person(tmp_path: Path) -> None:
    store, _rag = _store(tmp_path)
    await store.record(_comment(peer_user_id=777, text="первый"))
    await store.record(_comment(peer_user_id=888, text="второй"))

    theirs = await store.recent_with_peer(777)
    assert len(theirs) == 1
    assert theirs[0].text == "первый"


async def test_count_public_comments_per_channel(tmp_path: Path) -> None:
    store, _rag = _store(tmp_path)
    await store.record(_comment(chat_id=-1001))
    await store.record(_comment(chat_id=-1001, text="ещё один"))
    await store.record(_comment(chat_id=-2002, text="в другом канале"))

    assert await store.count_public_comments(-1001) == 2
    assert await store.count_public_comments(-2002) == 1


async def test_thread_read_is_recorded_even_without_a_peer(tmp_path: Path) -> None:
    """Прочитанный тред — тоже опыт, даже если комментировать Эфи не стала."""
    store, rag = _store(tmp_path)
    await store.record(
        _comment(kind=SocialInteractionKind.THREAD_READ, peer_user_id=None, peer_name="")
    )
    assert len(await store.recent()) == 1
    assert "Читала обсуждение" in rag.remembered[0]
