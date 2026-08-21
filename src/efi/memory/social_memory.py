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
from datetime import UTC, datetime
from enum import StrEnum

from efi.db.core import Database
from efi.memory.rag import RAGMemory

logger = logging.getLogger(__name__)

#: Confidence внешнего опыта: это достоверно произошедшее событие (Эфи там
#: была), но не «подтверждённый факт о мире» — ground truth (1.0) резервируем
#: за проверенными фактами, чтобы ночная консолидация не считала эти записи
#: неприкосновенными (см. efi.memory.consolidation.summarize_stale_entries).
_SOCIAL_CONFIDENCE = 0.6

_MAX_TEXT_PREVIEW = 400


class SocialInteractionKind(StrEnum):
    """Что именно произошло во внешнем мире."""

    #: Эфи оставила публичный комментарий (канал/тред обсуждения).
    PUBLIC_COMMENT = "public_comment"
    #: Кто-то ответил на её сообщение/комментарий.
    RECEIVED_REPLY = "received_reply"
    #: Переписка в ЛС с посторонним (не владельцем).
    STRANGER_DM = "stranger_dm"
    #: Эфи прочитала тред обсуждения (зашла посмотреть, что пишут).
    THREAD_READ = "thread_read"
    #: Эфи довела свой проект до репозитория и выложила его
    #: (efi.dev.reporter.DevReporter). Не «внешний контакт» в узком смысле,
    #: но по природе то же самое: сделанное во внешнем мире, о чём она потом
    #: может сослаться в разговоре («я эту штуку сама писала»). Домен H —
    #: как и у остального прожитого опыта.
    DEV_RELEASE = "dev_release"
    #: Эфи взялась за проект: придумала, что писать, и села писать. Начало
    #: работы — такая же часть прожитого, как и её конец: без этой записи на
    #: вопрос «а что ты вчера делала?» ей нечего ответить, пока проект не
    #: доведён, а доводится он часами.
    DEV_STARTED = "dev_started"
    #: Эфи бросила проект и почему. Самая нужная из трёх записей: провал без
    #: причины — это «не вышло» на карточке дашборда, а с причиной это то, что
    #: можно обсудить («почему ты забросила ту штуку с логами?»).
    DEV_ABANDONED = "dev_abandoned"
    #: Эфи вернулась к своему старому проекту и что-то в нём сделала
    #: (efi/dev/maintenance.py) — или посмотрела и решила не трогать.
    DEV_REVISION = "dev_revision"
    #: Эфи полезла в интернет прямо по ходу разговора (web_search).
    #: Раньше этот опыт не сохранялся НИГДЕ: результаты поиска приходят
    #: модели TOOL-сообщением, а в таблицу `messages` пишутся только реплика
    #: собеседника и итоговый ответ Эфи (см. efi/notifications/worker.py) —
    #: значит, ни история, ни новеллизация этого следа не видели. Через день
    #: Эфи не помнила, что вообще что-то гуглила, хотя разговор строился
    #: вокруг найденного.
    WEB_LOOKUP = "web_lookup"


#: Как каждый вид опыта звучит в дневнике от первого лица. Ключ — вид,
#: значение — шаблон с {where}/{who}/{text}. Живая формулировка здесь важна
#: не для красоты: именно этот текст потом всплывает в RAG-выдаче и
#: становится основой фразы в разговоре.
_DIARY_TEMPLATES: dict[SocialInteractionKind, str] = {
    SocialInteractionKind.PUBLIC_COMMENT: "Написала комментарий {where}: «{text}»",
    SocialInteractionKind.RECEIVED_REPLY: "{who} ответил мне {where}: «{text}»",
    SocialInteractionKind.STRANGER_DM: "Мне в ЛС писал {who}: «{text}»",
    SocialInteractionKind.THREAD_READ: "Читала обсуждение {where}. Там: «{text}»",
    SocialInteractionKind.WEB_LOOKUP: "Полезла гуглить и вычитала: «{text}»",
    SocialInteractionKind.DEV_RELEASE: "Дописала и выложила свой проект: {text}",
    SocialInteractionKind.DEV_STARTED: "Взялась за свой проект: {text}",
    SocialInteractionKind.DEV_ABANDONED: "Бросила свой проект: {text}",
    SocialInteractionKind.DEV_REVISION: "Вернулась к своему старому проекту: {text}",
}

TAG_PUBLIC_COMMENT = "#public_comment"
TAG_DISCUSSION_THREAD = "#discussion_thread"
TAG_WEB_LOOKUP = "#web_lookup"

#: Сколько символов результата поиска сохранять в журнале. Больше, чем
#: _MAX_TEXT_PREVIEW: сниппеты — это фактура (числа, названия, ссылки),
#: ради которой запись и делается, а обрезанный до пары фраз результат
#: поиска в дневнике бесполезен.
_MAX_LOOKUP_DIGEST = 1200


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
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))

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

    def render_for_prompt(self) -> str:
        """
        Живая фраза от первого лица БЕЗ метатегов — то, что подмешивается в
        промпт новеллизации как «а ещё за это время со мной было вот что»
        (см. efi/memory/pulse.py). Теги там только зашумляли бы контекст:
        они нужны для поиска по уже сохранённому, а не для осмысления.
        """
        where = f"в «{self.chat_title}»" if self.chat_title else "в чужом чате"
        who = self.peer_name or "какой-то тип"
        limit = _MAX_LOOKUP_DIGEST if self.kind is SocialInteractionKind.WEB_LOOKUP else _MAX_TEXT_PREVIEW
        text = self.text.strip()
        if len(text) > limit:
            text = text[:limit].rstrip() + "…"
        return _DIARY_TEMPLATES[self.kind].format(where=where, who=who, text=text)

    def render_for_diary(self) -> str:
        """
        Текст записи для векторной памяти — живой фразой от первого лица,
        а не протоколом (см. докстринг модуля). Метатеги идут отдельной
        строкой в конце, чтобы не ломать читаемость самой фразы.
        """
        return f"{self.render_for_prompt()}\n{' '.join(self.build_tags())}"


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

    async def record_web_lookup(
        self,
        *,
        query: str,
        digest: str,
        chat_id: int | None = None,
        thread_id: int | None = None,
    ) -> int:
        """
        Фиксирует поход в интернет как отдельный вид внешнего опыта.

        Вызывается прямо из efi.tools.web_tools.web_search.WebSearchTool в
        момент поиска, а не отложенно: результаты поиска приходят модели
        TOOL-сообщением, которое в историю чата не сохраняется вообще (см.
        комментарий у SocialInteractionKind.WEB_LOOKUP) — если не записать
        здесь, этот опыт не восстановится ниоткуда.

        Запись синхронная (await, а не fire-and-forget) сознательно: вставка
        в SQLite плюс локальный эмбеддинг — десятки миллисекунд на фоне
        секунд самого сетевого поиска, а взамен нет ни гонок, ни потерянных
        задач при остановке приложения посреди хода.
        """
        return await self.record(
            SocialInteraction(
                kind=SocialInteractionKind.WEB_LOOKUP,
                text=f"{query.strip()} — {digest.strip()}",
                chat_id=chat_id,
                thread_id=thread_id,
            )
        )

    async def context_lines_for_chat(
        self, chat_id: int, *, since: datetime, limit: int = 30
    ) -> list[str]:
        """
        Что произошло во внешнем мире в рамках ЭТОГО чата начиная с `since`,
        живыми фразами от первого лица — вход для новеллизации эпизода
        (efi/memory/pulse.py). Именно это склеивает опыт в одну личность:
        разговор, гуглёж по ходу разговора и оставленный комментарий
        осмысляются одной записью, а не тремя независимыми логами.

        Хронологический порядок (от старых к новым) — в отличие от recent(),
        где интересны как раз самые свежие: для пересказа эпизода нужен
        естественный ход времени.
        """
        rows = await self._database.fetch_all(
            """
            SELECT kind, chat_id, thread_id, peer_user_id, peer_name, text, created_at
            FROM social_interactions
            WHERE chat_id = ? AND created_at >= ?
            ORDER BY created_at ASC LIMIT ?
            """,
            (chat_id, since.isoformat(), limit),
        )
        return [_row_to_interaction(row).render_for_prompt() for row in rows]

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
    "TAG_WEB_LOOKUP",
]
