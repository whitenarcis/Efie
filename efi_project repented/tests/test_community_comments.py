"""
Тесты для efi.telegram.comments (сопоставление тем, учёт тредов, выбор
комментария для ответа) и для доступа чатов сообщества через lockdown.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from pyrogram.types import Message

from efi.config.schema import CommunitySettings, LockdownMode, TelegramSettings
from efi.db.core import Database
from efi.db.models import MIGRATIONS
from efi.memory.social_memory import SocialInteractionStore
from efi.notifications.manager import NotificationManager
from efi.notifications.schemas import NotificationType
from efi.security.access_control import ChatAccessInfo, is_chat_accessible
from efi.telegram.comments import (
    ChannelPostWatcher,
    RandomCommentEngager,
    ThreadStateStore,
    _pick_most_interesting,
    tokenize,
    topic_match_score,
)

_INTERESTS = ["async-паттерны и подводные камни в Python", "локальные LLM на слабом железе"]


# -- сопоставление тем ----------------------------------------------------------


def test_tokenize_drops_short_words() -> None:
    tokens = tokenize("я и он про async в Python")
    assert "async" in tokens and "python" in tokens
    assert "я" not in tokens and "и" not in tokens


def test_matching_post_scores_high() -> None:
    score = topic_match_score("разбираем подводные камни async в Python на реальных паттернах", _INTERESTS)
    assert score > 0.5


def test_unrelated_post_scores_zero() -> None:
    assert topic_match_score("сегодня продаём картошку оптом недорого", _INTERESTS) == 0.0


def test_empty_text_scores_zero() -> None:
    assert topic_match_score("", _INTERESTS) == 0.0


def test_no_interests_scores_zero() -> None:
    assert topic_match_score("async python паттерны", []) == 0.0


# -- выбор комментария для ответа ------------------------------------------------


def _comment(text: str, *, is_self: bool = False) -> Message:
    user = SimpleNamespace(id=1, first_name="Кто-то", is_self=is_self)
    return cast(Message, SimpleNamespace(id=10, text=text, caption=None, from_user=user))


def test_picks_the_most_relevant_comment() -> None:
    comments = [
        _comment("да норм пост"),
        _comment("а вот подводные камни async в Python это отдельная боль, паттерны там странные"),
    ]
    best = _pick_most_interesting(comments, _INTERESTS, min_score=0.3)
    assert best is not None
    assert "async" in (best.text or "")


def test_returns_none_when_nothing_is_interesting() -> None:
    """«Зайти в тред и ответить хоть кому-нибудь» — ровно то поведение бота, от которого мы уходим."""
    comments = [_comment("ага"), _comment("+"), _comment("картошка оптом")]
    assert _pick_most_interesting(comments, _INTERESTS, min_score=0.3) is None


def test_skips_own_messages() -> None:
    """Отвечать самой себе в ветке — тот же монолог, что уже вычищали из личных чатов."""
    comments = [_comment("подводные камни async в Python и паттерны вокруг них", is_self=True)]
    assert _pick_most_interesting(comments, _INTERESTS, min_score=0.3) is None


# -- персистентный учёт тредов ----------------------------------------------------


def _threads(tmp_path: Path, *, db_name: str = "test.db") -> ThreadStateStore:
    return ThreadStateStore(Database(tmp_path / db_name, migrations=MIGRATIONS))


async def test_unknown_thread_is_none(tmp_path: Path) -> None:
    assert await _threads(tmp_path).get(-100, 5) is None


async def test_seen_thread_is_awaiting_engagement(tmp_path: Path) -> None:
    threads = _threads(tmp_path)
    await threads.mark_seen(-100, 5)

    state = await threads.get(-100, 5)
    assert state is not None and state.has_commented is False
    assert [t.thread_id for t in await threads.threads_awaiting_engagement([-100])] == [5]


async def test_commented_thread_drops_out_of_candidates(tmp_path: Path) -> None:
    threads = _threads(tmp_path)
    await threads.mark_seen(-100, 5)
    await threads.mark_commented(-100, 5)

    state = await threads.get(-100, 5)
    assert state is not None and state.has_commented is True
    assert await threads.threads_awaiting_engagement([-100]) == []


async def test_thread_state_survives_a_restart(tmp_path: Path) -> None:
    first = _threads(tmp_path, db_name="shared.db")
    await first.mark_seen(-100, 5)
    await first.mark_commented(-100, 5)

    second = _threads(tmp_path, db_name="shared.db")
    state = await second.get(-100, 5)
    assert state is not None and state.has_commented is True


async def test_candidates_are_scoped_to_the_given_chats(tmp_path: Path) -> None:
    threads = _threads(tmp_path)
    await threads.mark_seen(-100, 5)
    await threads.mark_seen(-200, 7)
    assert [t.thread_id for t in await threads.threads_awaiting_engagement([-100])] == [5]


async def test_no_chats_means_no_candidates(tmp_path: Path) -> None:
    threads = _threads(tmp_path)
    await threads.mark_seen(-100, 5)
    assert await threads.threads_awaiting_engagement([]) == []


# -- доступ чатов сообщества через lockdown -----------------------------------------


def _settings(**overrides: object) -> TelegramSettings:
    defaults: dict[str, object] = dict(
        api_id=1, api_hash="x", owner_id=111, lockdown_mode=LockdownMode.OWNER_ONLY
    )
    defaults.update(overrides)
    return TelegramSettings(**defaults)  # type: ignore[arg-type]


def _chat(chat_id: int) -> ChatAccessInfo:
    return ChatAccessInfo(chat_id=chat_id, is_owner=False, is_contact=False, is_private_chat=False)


def test_community_chat_passes_owner_only_lockdown() -> None:
    allowed, _ = is_chat_accessible(_chat(-1001), _settings(community_chats=[-1001]))
    assert allowed is True


def test_lockdown_still_closes_everything_else() -> None:
    """Список сообщества сужает исключение, а не отменяет режим."""
    allowed, reason = is_chat_accessible(_chat(-9999), _settings(community_chats=[-1001]))
    assert allowed is False
    assert reason is not None


def test_without_community_chats_efi_stays_a_personal_bot() -> None:
    allowed, _ = is_chat_accessible(_chat(-1001), _settings())
    assert allowed is False


# -- куда фактически уходит комментарий -------------------------------------------
#
# Регрессия: и пост в канале, и реплика в треде ставились в очередь с chat_id
# КАНАЛА, хотя комментарии живут в привязанной группе обсуждения. Обычный
# аккаунт в канал писать не может, а message_id в payload был при этом id
# сообщения из группы — пара (chat_id, message_id) была рассогласована, и
# комментирование сообщества не работало ни по одному из двух путей.

_CHANNEL_ID = -1001
_DISCUSSION_ID = -1002


def _community() -> CommunitySettings:
    # Пренебрежимые задержки: тест проверяет адрес назначения, а не «человеческую» паузу.
    return CommunitySettings(min_delay_seconds=0.0, max_delay_seconds=0.001, comment_probability=1.0)


async def test_channel_post_comment_goes_to_the_discussion_group(tmp_path: Path) -> None:
    manager = NotificationManager(worker_count=1)
    watcher = ChannelPostWatcher(
        manager,
        _settings(community_chats=[_CHANNEL_ID]),
        _community(),
        cast("object", SimpleNamespace(current_interests=_async_return([]))),  # type: ignore[arg-type]
        _threads(tmp_path),
        SocialInteractionStore(Database(tmp_path / "social.db", migrations=MIGRATIONS)),
    )
    discussion_message = SimpleNamespace(id=77, chat=SimpleNamespace(id=_DISCUSSION_ID))
    watcher._client = cast(  # type: ignore[assignment]
        "object", SimpleNamespace(get_discussion_message=_async_return(discussion_message))
    )

    await watcher._schedule_comment(_CHANNEL_ID, "Канал", post_id=5, text="пост про async")

    notification = await manager.get(0)
    assert notification.type is NotificationType.PUBLIC_COMMENT
    assert notification.chat_id == _DISCUSSION_ID, "комментарий обязан уйти в группу обсуждения, а не в канал"
    assert notification.payload["telegram_message_ids"] == [77], "reply-цель — экземпляр поста в группе"
    assert notification.payload["thread_id"] == 5, "идентичность треда — id поста в канале"
    assert notification.payload["force_reply"] is True


async def test_post_without_a_linked_discussion_is_skipped(tmp_path: Path) -> None:
    """Без привязанного обсуждения комментировать нечем — повод молча пропускается."""
    manager = NotificationManager(worker_count=1)
    watcher = ChannelPostWatcher(
        manager,
        _settings(community_chats=[_CHANNEL_ID]),
        _community(),
        cast("object", SimpleNamespace(current_interests=_async_return([]))),  # type: ignore[arg-type]
        _threads(tmp_path),
        SocialInteractionStore(Database(tmp_path / "social.db", migrations=MIGRATIONS)),
    )
    watcher._client = cast("object", SimpleNamespace(get_discussion_message=_async_return(None)))  # type: ignore[assignment]

    await watcher._schedule_comment(_CHANNEL_ID, "Канал", post_id=5, text="пост про async")

    with pytest.raises(TimeoutError):
        await asyncio.wait_for(manager.get(0), timeout=0.05)


async def test_thread_reply_goes_to_the_chat_the_comment_lives_in(tmp_path: Path) -> None:
    manager = NotificationManager(worker_count=1)
    threads = _threads(tmp_path)
    await threads.mark_seen(_CHANNEL_ID, 5)

    comment = SimpleNamespace(
        id=42,
        text="подводные камни async в Python и паттерны вокруг них",
        caption=None,
        chat=SimpleNamespace(id=_DISCUSSION_ID),
        from_user=SimpleNamespace(id=9, first_name="Гость", is_self=False),
    )
    engager = RandomCommentEngager(
        manager,
        cast("object", SimpleNamespace()),  # type: ignore[arg-type]
        _settings(community_chats=[_CHANNEL_ID]),
        _community(),
        cast("object", SimpleNamespace(current_interests=_async_return(_INTERESTS))),  # type: ignore[arg-type]
        threads,
        SocialInteractionStore(Database(tmp_path / "social.db", migrations=MIGRATIONS)),
    )
    engager._read_thread = _async_return([comment])  # type: ignore[assignment]

    await engager._tick()

    notification = await manager.get(0)
    assert notification.type is NotificationType.THREAD_REPLY
    assert notification.chat_id == _DISCUSSION_ID, "отвечаем туда, где лежит комментарий"
    assert notification.payload["telegram_message_ids"] == [42]
    assert notification.payload["community_chat_id"] == _CHANNEL_ID
    assert notification.payload["force_reply"] is True


def _async_return(value: object):  # noqa: ANN202 — тестовый хелпер, тип возврата не несёт смысла
    """Асинхронная заглушка, отдающая заранее заданное значение на любой вызов."""

    async def _call(*_args: object, **_kwargs: object) -> object:
        return value

    return _call
