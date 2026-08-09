"""
efi/telegram/comments.py

Участие Эфи в жизни сообщества: комментарии под постами в целевых каналах и
выборочное включение в чужие ветки обсуждений.

Три пути попадания в тред, с принципиально разной инициативой:

    1. НОВЫЙ ПОСТ (`ChannelPostWatcher`). Пост в канале из
       `telegram.community_chats` проверяется на пересечение с тем, что Эфи
       вообще интересно — семенами любопытства (curiosity_seeds) и
       интересами из worldview.json. Совпало по теме и выпал шанс
       (`comment_probability`) — ставится отложенный PUBLIC_COMMENT.
    2. УПОМИНАНИЕ ИЛИ РЕПЛАЙ. Отдельного кода здесь не требуется: это
       обычный USER_MESSAGE, и адресность уже проверяет
       efi.telegram.handlers._is_addressed_to_bot (упоминание либо ответ на
       её сообщение) — в треде обсуждения он работает так же, как в группе,
       потому что тред и есть супергруппа. Единственное, что было нужно, —
       пустить эти чаты через lockdown, см. `telegram.community_chats` в
       efi.security.access_control.
    3. ВЫБОРОЧНЫЙ КОММЕНТИНГ (`RandomCommentEngager`). Фоновый воркер сам
       заходит в тред, где Эфи уже отметилась, читает свежие комментарии и
       отвечает НА ОДИН подходящий — если ветка ему интересна.

Про задержку: комментарий никогда не уходит мгновенно. Живой участник
сообщества не отвечает на пост в ту же секунду, что он вышел, — поэтому
между поводом и реакцией всегда стоит случайная пауза (по умолчанию 5-30
минут, `community.min_delay_seconds`/`max_delay_seconds`). Задержка
реализована планированием: воркер спит и только потом кладёт Notification,
поэтому очередь всё это время свободна для живого диалога с владельцем.

Единый стиль: сам ТЕКСТ комментария этот модуль не сочиняет — он лишь
ставит Notification с поводом, а формулирует его Worker через роль MAIN,
ровно как обычный ответ в чате (efi/notifications/worker.py). Так внешние
комментарии звучат тем же голосом, что и личные сообщения, а не отдельным
«режимом для публики».

Каждое реально совершённое действие немедленно попадает в социальную память
(efi.memory.social_memory.SocialInteractionStore) — и прочитанный тред, и
оставленный комментарий; см. докстринг того модуля про гарантию сохранения.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

import aiofiles
from pyrogram import Client, filters
from pyrogram.handlers import MessageHandler
from pyrogram.types import Message as PyrogramMessage

from efi.config.schema import CommunitySettings, TelegramSettings
from efi.db.core import Database
from efi.memory.social_memory import SocialInteraction, SocialInteractionKind, SocialInteractionStore
from efi.notifications.manager import NotificationManager
from efi.notifications.schemas import Notification, NotificationType
from efi.security.sanitize import sanitize_text

logger = logging.getLogger(__name__)

#: Слова короче этого в пересечении тем не участвуют — предлоги и союзы
#: совпадают всегда и превратили бы любой пост в «интересный».
_MIN_TOKEN_LENGTH = 4
_TOKEN_RE = re.compile(r"[a-zA-Zа-яА-ЯёЁ0-9_]+")

_POST_PREVIEW_MAX_LENGTH = 600
_COMMENT_PREVIEW_MAX_LENGTH = 300

#: Сколько последних комментариев треда воркер читает за один заход.
_THREAD_SCAN_LIMIT = 30


def tokenize(text: str) -> set[str]:
    """Нормализованные значимые слова текста — основа дешёвого сопоставления тем."""
    return {token.lower() for token in _TOKEN_RE.findall(text) if len(token) >= _MIN_TOKEN_LENGTH}


def topic_match_score(text: str, interests: list[str]) -> float:
    """
    Доля слов интереса, встретившихся в тексте, по лучшему из интересов.

    Чистая функция без сети и LLM — тот же компромисс «дёшево и локально»,
    что у efi.behavior.affinity.classify_message и memory/tfidf_fallback.py:
    гонять модель на каждый пост в каждом канале было бы несоразмерно
    задаче «стоит ли вообще смотреть в эту сторону».
    """
    text_tokens = tokenize(text)
    if not text_tokens:
        return 0.0

    best = 0.0
    for interest in interests:
        interest_tokens = tokenize(interest)
        if not interest_tokens:
            continue
        overlap = len(interest_tokens & text_tokens) / len(interest_tokens)
        best = max(best, overlap)
    return best


class InterestSource(Protocol):
    """Откуда берутся темы, которые Эфи считает своими. Реализация — CuriosityTracker + worldview.json."""

    async def current_interests(self) -> list[str]: ...


class CommunityInterests:
    """
    Объединяет два источника интересов: постоянные (worldview.json — «что мне
    вообще интересно») и ситуативные (curiosity_seeds — «о чём недавно
    заходил разговор»). Второй источник важнее по смыслу: он привязывает
    участие в сообществе к реальным разговорам, а не к статичному списку.
    """

    def __init__(self, database: Database, worldview_path: Path) -> None:
        self._database = database
        self._worldview_path = worldview_path

    async def current_interests(self) -> list[str]:
        seeds, worldview = await asyncio.gather(self._pending_seed_topics(), self._worldview_interests())
        return seeds + worldview

    async def _pending_seed_topics(self) -> list[str]:
        rows = await self._database.fetch_all(
            "SELECT topic FROM curiosity_seeds ORDER BY weight DESC, created_at DESC LIMIT 20"
        )
        return [row["topic"] for row in rows]

    async def _worldview_interests(self) -> list[str]:
        try:
            async with aiofiles.open(self._worldview_path, mode="r", encoding="utf-8") as f:
                raw = await f.read()
        except OSError as exc:
            logger.warning("comments: failed to read worldview file %s: %s", self._worldview_path, exc)
            return []
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("comments: worldview file is not valid JSON: %s", exc)
            return []
        interests = payload.get("interests", [])
        return [str(item) for item in interests] if isinstance(interests, list) else []


@dataclass(slots=True, frozen=True)
class ThreadState:
    """Что Эфи уже делала в конкретной ветке обсуждения — переживает рестарт (таблица `thread_state`)."""

    chat_id: int
    thread_id: int
    has_commented: bool


class ThreadStateStore:
    """Персистентный учёт тредов: где Эфи уже отметилась и когда последний раз туда заглядывала."""

    def __init__(self, database: Database) -> None:
        self._database = database

    async def get(self, chat_id: int, thread_id: int) -> ThreadState | None:
        row = await self._database.fetch_one(
            "SELECT commented_at FROM thread_state WHERE chat_id = ? AND thread_id = ?",
            (chat_id, thread_id),
        )
        if row is None:
            return None
        return ThreadState(chat_id=chat_id, thread_id=thread_id, has_commented=row["commented_at"] is not None)

    async def mark_seen(self, chat_id: int, thread_id: int) -> None:
        await self._database.execute(
            """
            INSERT INTO thread_state (chat_id, thread_id, commented_at, last_seen_at)
            VALUES (?, ?, NULL, ?)
            ON CONFLICT (chat_id, thread_id) DO UPDATE SET last_seen_at = excluded.last_seen_at
            """,
            (chat_id, thread_id, _now()),
        )

    async def mark_commented(self, chat_id: int, thread_id: int) -> None:
        now = _now()
        await self._database.execute(
            """
            INSERT INTO thread_state (chat_id, thread_id, commented_at, last_seen_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (chat_id, thread_id) DO UPDATE SET
                commented_at = excluded.commented_at,
                last_seen_at = excluded.last_seen_at
            """,
            (chat_id, thread_id, now, now),
        )

    async def threads_awaiting_engagement(self, chat_ids: list[int], *, limit: int = 10) -> list[ThreadState]:
        """Треды, которые Эфи видела, но ещё не комментировала — кандидаты для RandomCommentEngager."""
        if not chat_ids:
            return []
        placeholders = ",".join("?" for _ in chat_ids)
        rows = await self._database.fetch_all(
            f"""
            SELECT chat_id, thread_id FROM thread_state
            WHERE commented_at IS NULL AND chat_id IN ({placeholders})
            ORDER BY last_seen_at DESC LIMIT ?
            """,
            (*chat_ids, limit),
        )
        return [
            ThreadState(chat_id=row["chat_id"], thread_id=row["thread_id"], has_commented=False) for row in rows
        ]


class ChannelPostWatcher:
    """
    Слушает новые посты в каналах сообщества и решает, вписываться ли.

    Регистрируется на том же Pyrogram Client, что и обычные обработчики
    (efi.telegram.handlers.TelegramEventHandlers.register) — отдельным
    хендлером с фильтром `filters.channel`, чтобы не перепутать пост в
    канале с сообщением в чате.
    """

    def __init__(
        self,
        manager: NotificationManager,
        telegram_settings: TelegramSettings,
        community_settings: CommunitySettings,
        interests: InterestSource,
        threads: ThreadStateStore,
        social_memory: SocialInteractionStore,
    ) -> None:
        self._manager = manager
        self._telegram_settings = telegram_settings
        self._settings = community_settings
        self._interests = interests
        self._threads = threads
        self._social_memory = social_memory
        self._pending: set[asyncio.Task[None]] = set()

    def register(self, client: Client) -> None:
        if not self._settings.enabled or not self._telegram_settings.community_chats:
            logger.info("comments: community engagement is disabled (no community_chats or enabled=false)")
            return
        client.add_handler(MessageHandler(self._handle_post, filters.channel))
        logger.info("comments: watching %d community channel(s)", len(self._telegram_settings.community_chats))

    async def cancel_pending(self) -> None:
        """Снимает запланированные, но ещё не сработавшие комментарии — вызывается при graceful shutdown."""
        for task in list(self._pending):
            task.cancel()
        if self._pending:
            await asyncio.gather(*self._pending, return_exceptions=True)

    async def _handle_post(self, client: Client, message: PyrogramMessage) -> None:
        chat = message.chat
        if chat is None or chat.id not in self._telegram_settings.community_chats:
            return

        text = (message.text or message.caption or "").strip()
        if not text:
            return

        interests = await self._interests.current_interests()
        score = topic_match_score(text, interests)
        if score < self._settings.topic_match_min_score:
            logger.debug("comments: post in chat_id=%s scored %.2f, not my topic", chat.id, score)
            return

        # Прочитанное — уже опыт, даже если комментировать в итоге не станем.
        await self._social_memory.record(
            SocialInteraction(
                kind=SocialInteractionKind.THREAD_READ,
                text=text,
                chat_id=chat.id,
                thread_id=message.id,
                chat_title=chat.title or "",
            )
        )
        await self._threads.mark_seen(chat.id, message.id)

        if random.random() > self._settings.comment_probability:
            logger.debug("comments: post in chat_id=%s matched (%.2f) but not in the mood", chat.id, score)
            return

        task = asyncio.create_task(self._schedule_comment(chat.id, chat.title or "", message.id, text))
        self._pending.add(task)
        task.add_done_callback(self._pending.discard)

    async def _schedule_comment(self, chat_id: int, chat_title: str, post_id: int, text: str) -> None:
        """Ждёт «человеческую» паузу и только потом ставит повод в очередь — см. докстринг модуля."""
        delay = random.uniform(self._settings.min_delay_seconds, self._settings.max_delay_seconds)
        logger.info("comments: will consider commenting post %s in chat_id=%s in %.0fs", post_id, chat_id, delay)
        try:
            await asyncio.sleep(delay)
        except asyncio.CancelledError:
            logger.debug("comments: scheduled comment for post %s cancelled", post_id)
            raise

        await self._manager.put(
            Notification(
                type=NotificationType.PUBLIC_COMMENT,
                priority=7,  # ниже живого диалога: собеседник всегда важнее публики
                chat_id=chat_id,
                message=_render_post_prompt(chat_title, text),
                payload={
                    "chat_title": chat_title,
                    "telegram_message_ids": [post_id],
                    "thread_id": post_id,
                    "is_public_comment": True,
                },
            )
        )


class RandomCommentEngager:
    """
    Фоновый воркер выборочного комментинга: раз в
    `thread_scan_interval_seconds` заходит в один из тредов, где Эфи ещё не
    отписалась, читает свежие комментарии и отвечает НА ОДИН подходящий.

    Почему на один: смысл в том, чтобы выглядеть участником, который иногда
    вставляет реплику, а не тем, кто прошёлся по всей ветке. Отсюда и
    `max_replies_per_thread=1` по умолчанию.
    """

    def __init__(
        self,
        manager: NotificationManager,
        client: Client,
        telegram_settings: TelegramSettings,
        community_settings: CommunitySettings,
        interests: InterestSource,
        threads: ThreadStateStore,
        social_memory: SocialInteractionStore,
    ) -> None:
        self._manager = manager
        self._client = client
        self._telegram_settings = telegram_settings
        self._settings = community_settings
        self._interests = interests
        self._threads = threads
        self._social_memory = social_memory

    async def run(self) -> None:
        """Основной цикл. Останавливается по отмене задачи (CancelledError) — см. efi/app.py graceful shutdown."""
        if not self._settings.enabled or not self._telegram_settings.community_chats:
            logger.info("random_comment_engager: disabled, not starting")
            return

        logger.info(
            "random_comment_engager: started (scan every %.0fs)", self._settings.thread_scan_interval_seconds
        )
        try:
            while True:
                await asyncio.sleep(self._settings.thread_scan_interval_seconds)
                try:
                    await self._tick()
                except Exception:
                    # Сбой одного захода (сеть, недоступный тред) не должен
                    # убивать весь цикл участия в сообществе.
                    logger.warning("random_comment_engager: tick failed", exc_info=True)
        except asyncio.CancelledError:
            logger.info("random_comment_engager: stopped")
            raise

    async def _tick(self) -> None:
        candidates = await self._threads.threads_awaiting_engagement(self._telegram_settings.community_chats)
        if not candidates:
            return

        thread = random.choice(candidates)
        comments = await self._read_thread(thread)
        if not comments:
            return

        interests = await self._interests.current_interests()
        best = _pick_most_interesting(comments, interests, self._settings.topic_match_min_score)
        if best is None:
            logger.debug("random_comment_engager: nothing worth answering in thread %s", thread.thread_id)
            return

        # Случайная пауза и здесь: заход в тред тоже не должен выглядеть как
        # реакция робота, сработавшего по таймеру.
        await asyncio.sleep(random.uniform(self._settings.min_delay_seconds, self._settings.max_delay_seconds))

        author_id = best.from_user.id if best.from_user else None
        author_name = best.from_user.first_name if best.from_user else ""
        text = (best.text or best.caption or "").strip()

        await self._social_memory.record(
            SocialInteraction(
                kind=SocialInteractionKind.THREAD_READ,
                text=text,
                chat_id=thread.chat_id,
                thread_id=thread.thread_id,
                peer_user_id=author_id,
                peer_name=author_name or "",
            )
        )
        await self._threads.mark_commented(thread.chat_id, thread.thread_id)
        await self._manager.put(
            Notification(
                type=NotificationType.THREAD_REPLY,
                priority=7,
                chat_id=thread.chat_id,
                message=_render_thread_prompt(author_name or "кто-то", text),
                payload={
                    "telegram_message_ids": [best.id],
                    "thread_id": thread.thread_id,
                    "sender_id": author_id,
                    "sender_name": author_name,
                    "is_public_comment": True,
                },
            )
        )
        logger.info(
            "random_comment_engager: queued a reply in thread %s of chat_id=%s", thread.thread_id, thread.chat_id
        )

    async def _read_thread(self, thread: ThreadState) -> list[PyrogramMessage]:
        """Свежие комментарии ветки. Недоступный тред — штатная ситуация (пост удалён, доступ закрыт), не ошибка."""
        try:
            # get_discussion_replies возвращает None, когда у поста вообще
            # нет привязанного обсуждения — это штатный случай, а не ошибка.
            replies = self._client.get_discussion_replies(
                thread.chat_id, thread.thread_id, limit=_THREAD_SCAN_LIMIT
            )
            if replies is None:
                return []
            return [message async for message in replies]
        except Exception:
            logger.debug(
                "random_comment_engager: thread %s in chat_id=%s is not readable",
                thread.thread_id, thread.chat_id, exc_info=True,
            )
            return []


def _pick_most_interesting(
    comments: list[PyrogramMessage], interests: list[str], min_score: float
) -> PyrogramMessage | None:
    """
    Самый близкий к интересам Эфи комментарий, но только если он вообще
    перешагнул порог: «зайти в тред и ответить хоть кому-нибудь» — это и есть
    поведение бота, от которого мы уходим.

    Собственные сообщения Эфи пропускаются: отвечать самой себе в ветке —
    ровно тот монолог, который мы уже вычищали из личных чатов.
    """
    best: PyrogramMessage | None = None
    best_score = min_score
    for comment in comments:
        if comment.from_user is not None and getattr(comment.from_user, "is_self", False):
            continue
        text = (comment.text or comment.caption or "").strip()
        if not text:
            continue
        score = topic_match_score(text, interests)
        if score >= best_score:
            best, best_score = comment, score
    return best


def _render_post_prompt(chat_title: str, text: str) -> str:
    preview = sanitize_text(text)[:_POST_PREVIEW_MAX_LENGTH]
    where = f"в канале «{chat_title}»" if chat_title else "в канале"
    return (
        f"Ты читаешь новый пост {where}, и тема тебя зацепила:\n\n{preview}\n\n"
        "Ты решила оставить комментарий под этим постом — публично, среди незнакомых людей."
    )


def _render_thread_prompt(author_name: str, text: str) -> str:
    preview = sanitize_text(text)[:_COMMENT_PREVIEW_MAX_LENGTH]
    return (
        f"Ты залипла в чужом обсуждении, и там {sanitize_text(author_name)} написал:\n\n{preview}\n\n"
        "Тебе есть что на это сказать, и ты решила ответить ему в ветке — публично, среди незнакомых людей."
    )


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def is_public_comment(notification: Notification) -> bool:
    """Публичное ли это выступление — используется промптом и Worker'ом (см. efi/prompts/builder.py)."""
    return bool(notification.payload.get("is_public_comment")) or notification.type in _PUBLIC_TYPES


_PUBLIC_TYPES = (NotificationType.PUBLIC_COMMENT, NotificationType.THREAD_REPLY)


def build_community_interests(database: Database, worldview_path: Path) -> CommunityInterests:
    """Фабрика для efi/app.py — чтобы граф сборки не знал про внутреннее устройство источников интересов."""
    return CommunityInterests(database, worldview_path)


__all__: list[str] = [
    "ChannelPostWatcher",
    "CommunityInterests",
    "InterestSource",
    "RandomCommentEngager",
    "ThreadState",
    "ThreadStateStore",
    "build_community_interests",
    "is_public_comment",
    "topic_match_score",
    "tokenize",
]
