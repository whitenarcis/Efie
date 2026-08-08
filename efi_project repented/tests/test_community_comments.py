"""
Тесты для efi.telegram.comments (сопоставление тем, учёт тредов, выбор
комментария для ответа) и для доступа чатов сообщества через lockdown.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import cast

from pyrogram.types import Message

from efi.config.schema import LockdownMode, TelegramSettings
from efi.db.core import Database
from efi.db.models import MIGRATIONS
from efi.security.access_control import ChatAccessInfo, is_chat_accessible
from efi.telegram.comments import (
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
