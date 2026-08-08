"""
efi/memory/social_memory.py

Сквозное сохранение ВНЕШНЕГО социального опыта Эфи: комментарии, которые она
оставила публично, реплаи, которые получила, разговоры с посторонними в ЛС и
треды, которые она читала.

Двухслойное хранение — намеренно, у слоёв разные гарантии:

    1. SQLite (`social_interactions`) — ГАРАНТИЯ. Пишется всегда и без
       дедупликации: это журнал того, что реально произошло. По нему можно
       восстановить, где Эфи была и что говорила, даже если векторный слой
       отверг запись или эмбеддинги вообще недоступны.
    2. Векторная память (RAGMemory.remember -> Diary + fastembed/TF-IDF) —
       ИЗВЛЕКАЕМОСТЬ. Здесь дубли отбраковываются по релевантности (это
       желаемое поведение: десять похожих комментариев не должны забивать
       выдачу), а сама запись становится доступна обычному RAG-поиску.

Именно из-за второго слоя внешний опыт всплывает в обычном разговоре с
владельцем без какой-либо отдельной логики: efi.prompts.builder ищет по
дневнику перед каждым ответом, и туда попадают в том числе публичные
комментарии — отсюда органичные отсылки вида "я тут в комментариях у X
наткнулась на...". Чтобы это работало, тело записи пишется НЕ протоколом
("user 123 commented"), а живой фразой от первого лица с метатегами.

Метатеги (`#public_comment`, `#channel_{id}`, `#secondary_user_{id}`,
`#discussion_thread`) кладутся и в колонку `tags`, и прямо в тело
векторной записи: в SQLite по ним можно фильтровать SQL-запросом, а в
векторном слое они участвуют в тексте и помогают семантическому поиску
попадать в нужный контекст.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum

from efi.db.core import Database
from efi.memory.rag import RAGMemory

logger = logging.getLogger(__name__)

#: Confidence внешнего опыта: это достоверно произошедшее событие (Эфи там
#: была), но не «подтверждённый факт о мире» — ground truth (1.0) резервируем
#: за проверенными фактами, чтобы ночная консолидация не считала эти записи
#: неприкосновенными (см. efi.memory.consolidation.summarize_stale_entries).
_SOCIAL_CONFIDENCE = 0.6

_MAX_TEXT_PREVIEW = 400


class SocialInteractionKind(str, Enum):
    """Что именно произошло во внешнем мире."""

    #: Эфи оставила публичный комментарий (канал/тред обсуждения).
    PUBLIC_COMMENT = "public_comment"
    #: Кто-то ответил на её сообщение/комментарий.
    RECEIVED_REPLY = "received_reply"
    #: Переписка в ЛС с посторонним (не владельцем).
    STRANGER_DM = "stranger_dm"
    #: Эфи прочитала тред обсуждения (зашла посмотреть, что пишут).
    THREAD_READ = "thread_read"


#: Как каждый вид опыта звучит в дневнике от первого лица. Ключ — вид,
#: значение — шаблон с {where}/{who}/{text}. Живая формулировка здесь важна
#: не для красоты: именно этот текст потом всплывает в RAG-выдаче и
#: становится основой фразы в разговоре.
_DIARY_TEMPLATES: dict[SocialInteractionKind, str] = {
    SocialInteractionKind.PUBLIC_COMMENT: "Написала комментарий {where}: «{text}»",
    SocialInteractionKind.RECEIVED_REPLY: "{who} ответил мне {where}: «{text}»",
    SocialInteractionKind.STRANGER_DM: "Мне в ЛС писал {who}: «{text}»",
    SocialInteractionKind.THREAD_READ: "Читала обсуждение {where}. Там: «{text}»",
}

TAG_PUBLIC_COMMENT = "#public_comment"
TAG_DISCUSSION_THREAD = "#discussion_thread"


@dataclass(slots=True, frozen=True)
class SocialInteraction:
    """Одно внешнее взаимодействие — то, что попадает и в журнал, и в векторную память."""

    kind: SocialInteractionKind
    text: str
    chat_id: int | None = None
    thread_id: int | None = None
    peer_user_id: int | None = None
    peer_name: str = ""
    chat_title: str = ""
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def build_tags(self) -> list[str]:
        """
        Метатеги события. `#public_comment` ставится только реально
        публичному комментарию, `#discussion_thread` — всему, что произошло
        внутри треда обсуждения, `#channel_{id}`/`#secondary_user_{id}` —
        адресные, по ним можно найти всё, связанное с конкретным каналом или
        конкретным посторонним человеком.
        """
        tags: list[str] = [f"#{self.kind.value}"]
        if self.kind is SocialInteractionKind.PUBLIC_COMMENT:
            tags.append(TAG_PUBLIC_COMMENT)
        if self.thread_id is not None:
            tags.append(TAG_DISCUSSION_THREAD)
        if self.chat_id is not None:
            tags.append(f"#channel_{self.chat_id}")
        if self.peer_user_id is not None:
            tags.append(f"#secondary_user_{self.peer_user_id}")
        return tags

    def render_for_diary(self) -> str:
        """
        Текст записи для векторной памяти — живой фразой от первого лица,
        а не протоколом (см. докстринг модуля). Метатеги идут отдельной
        строкой в конце, чтобы не ломать читаемость самой фразы.
        """
        where = f"в «{self.chat_title}»" if self.chat_title else "в чужом чате"
        who = self.peer_name or "какой-то тип"
        text = self.text.strip()
        if len(text) > _MAX_TEXT_PREVIEW:
            text = text[:_MAX_TEXT_PREVIEW].rstrip() + "…"

        template = _DIARY_TEMPLATES[self.kind]
        body = template.format(where=where, who=who, text=text)
        return f"{body}\n{' '.join(self.build_tags())}"


class SocialInteractionStore:
    """
    Журнал внешних взаимодействий + мост в векторную память.

    `rag` необязателен: без него класс остаётся чистым журналом (SQLite), с
    ним каждое событие ЕЩЁ И становится доступно обычному RAG-поиску. Сбой
    векторного слоя не отменяет запись в журнал — гарантия сохранения важнее
    извлекаемости, и терять сам факт события из-за недоступных эмбеддингов
    недопустимо (см. докстринг модуля).
    """

    def __init__(self, database: Database, *, rag: RAGMemory | None = None) -> None:
        self._database = database
        self._rag = rag

    async def record(self, interaction: SocialInteraction) -> int:
        """
        Сохраняет взаимодействие НЕМЕДЛЕННО в оба слоя и возвращает id
        журнальной записи. Вызывается сразу в момент события (отправила
        комментарий / получила реплай / прочитала тред), а не отложенно
        ночной консолидацией: внешний опыт, потерянный до ночи, не
        восстановится ниоткуда.
        """
        tags = " ".join(interaction.build_tags())
        async with self._database.connection() as conn:
            cursor = await conn.execute(
                """
                INSERT INTO social_interactions
                    (kind, chat_id, thread_id, peer_user_id, peer_name, text, tags, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    interaction.kind.value,
                    interaction.chat_id,
                    interaction.thread_id,
                    interaction.peer_user_id,
                    interaction.peer_name,
                    interaction.text,
                    tags,
                    interaction.created_at.isoformat(),
                ),
            )
            await conn.commit()
            interaction_id = cursor.lastrowid
            assert interaction_id is not None  # AUTOINCREMENT PK — lastrowid всегда есть после INSERT

        await self._index_in_vector_memory(interaction)
        logger.info(
            "social_memory: recorded %s (chat_id=%s, thread_id=%s, peer=%s) as #%s",
            interaction.kind.value, interaction.chat_id, interaction.thread_id,
            interaction.peer_user_id, interaction_id,
        )
        return interaction_id

    async def _index_in_vector_memory(self, interaction: SocialInteraction) -> None:
        if self._rag is None:
            return
        try:
            await self._rag.remember(interaction.render_for_diary(), confidence=_SOCIAL_CONFIDENCE)
        except Exception:
            # Журнал уже записан — это главное. Векторный слой отвечает лишь
            # за то, всплывёт ли эпизод в разговоре сам собой.
            logger.warning("social_memory: failed to index interaction in vector memory", exc_info=True)

    async def recent(self, *, limit: int = 10) -> list[SocialInteraction]:
        """Последние внешние взаимодействия — от самых свежих."""
        rows = await self._database.fetch_all(
            """
            SELECT kind, chat_id, thread_id, peer_user_id, peer_name, text, created_at
            FROM social_interactions ORDER BY created_at DESC LIMIT ?
            """,
            (limit,),
        )
        return [_row_to_interaction(row) for row in rows]

    async def recent_with_peer(self, peer_user_id: int, *, limit: int = 5) -> list[SocialInteraction]:
        """История внешних пересечений с конкретным посторонним — от самых свежих."""
        rows = await self._database.fetch_all(
            """
            SELECT kind, chat_id, thread_id, peer_user_id, peer_name, text, created_at
            FROM social_interactions WHERE peer_user_id = ? ORDER BY created_at DESC LIMIT ?
            """,
            (peer_user_id, limit),
        )
        return [_row_to_interaction(row) for row in rows]

    async def count_public_comments(self, chat_id: int) -> int:
        """Сколько раз Эфи уже комментировала в этом канале — вход для «не частить» в RandomCommentEngager."""
        row = await self._database.fetch_one(
            "SELECT COUNT(*) AS total FROM social_interactions WHERE chat_id = ? AND kind = ?",
            (chat_id, SocialInteractionKind.PUBLIC_COMMENT.value),
        )
        return int(row["total"]) if row is not None else 0


def _row_to_interaction(row: object) -> SocialInteraction:
    # aiosqlite.Row поддерживает доступ по имени колонки; тип оставлен широким,
    # чтобы не тянуть aiosqlite в сигнатуры этого модуля.
    return SocialInteraction(
        kind=SocialInteractionKind(row["kind"]),  # type: ignore[index]
        chat_id=row["chat_id"],  # type: ignore[index]
        thread_id=row["thread_id"],  # type: ignore[index]
        peer_user_id=row["peer_user_id"],  # type: ignore[index]
        peer_name=row["peer_name"],  # type: ignore[index]
        text=row["text"],  # type: ignore[index]
        created_at=datetime.fromisoformat(row["created_at"]),  # type: ignore[index]
    )


__all__ = [
    "SocialInteraction",
    "SocialInteractionKind",
    "SocialInteractionStore",
    "TAG_DISCUSSION_THREAD",
    "TAG_PUBLIC_COMMENT",
]
