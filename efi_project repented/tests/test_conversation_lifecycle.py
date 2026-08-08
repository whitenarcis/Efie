"""
Тесты для efi.behavior.conversation_lifecycle: разграничение владельца и
посторонних, молчаливое завершение диалога и персистентность накопленной
навязчивости между перезапусками.
"""

from __future__ import annotations

from pathlib import Path

from efi.behavior.conversation_lifecycle import (
    ANNOYANCE_THRESHOLD,
    ConversationLifecycle,
    ConversationStatus,
    UserTier,
    is_farewell,
    is_terse,
    score_annoyance,
)
from efi.db.core import Database
from efi.db.models import MIGRATIONS

_OWNER_ID = 111
_STRANGER_ID = 222
_CHAT_ID = -100


def _lifecycle(tmp_path: Path, *, db_name: str = "test.db") -> ConversationLifecycle:
    database = Database(tmp_path / db_name, migrations=MIGRATIONS)
    return ConversationLifecycle(database, owner_id=_OWNER_ID)


# -- чистые функции ------------------------------------------------------------


def test_farewell_detection() -> None:
    assert is_farewell("ладно, пока") is True
    assert is_farewell("спокойной ночи") is True
    assert is_farewell("а что ты думаешь про это?") is False


def test_terse_detection() -> None:
    assert is_terse("ок") is True
    assert is_terse("ага") is True
    assert is_terse("ясно.") is True
    assert is_terse("ну расскажи подробнее что там было") is False


def test_annoyance_scoring() -> None:
    assert score_annoyance("привет, как дела") == 0.0
    assert score_annoyance("ОТВЕЧАЙ ЖЕ") > 0.0
    assert score_annoyance("ты тупая, заткнись") > score_annoyance("ответь")
    assert score_annoyance("ну что???") > 0.0


# -- разграничение владельца и посторонних ------------------------------------


def test_owner_is_primary_everyone_else_secondary(tmp_path: Path) -> None:
    lifecycle = _lifecycle(tmp_path)
    assert lifecycle.classify(_OWNER_ID) is UserTier.PRIMARY
    assert lifecycle.classify(_STRANGER_ID) is UserTier.SECONDARY
    assert lifecycle.classify(None) is UserTier.SECONDARY


def test_proactive_pings_are_owner_only(tmp_path: Path) -> None:
    lifecycle = _lifecycle(tmp_path)
    assert lifecycle.allows_proactive_ping(_OWNER_ID) is True
    assert lifecycle.allows_proactive_ping(_STRANGER_ID) is False
    assert lifecycle.allows_proactive_ping(None) is False


async def test_owner_conversation_never_ends(tmp_path: Path) -> None:
    """С владельцем диалог не завершается ни по прощанию, ни по навязчивости."""
    lifecycle = _lifecycle(tmp_path)
    for text in ("пока", "ок", "ты тупая, заткнись!!!"):
        decision = await lifecycle.evaluate(_OWNER_ID, _CHAT_ID, text)
        assert decision.should_disengage is False
        assert decision.tier is UserTier.PRIMARY


# -- завершение диалога с посторонним -----------------------------------------


async def test_farewell_ends_the_conversation(tmp_path: Path) -> None:
    lifecycle = _lifecycle(tmp_path)
    decision = await lifecycle.evaluate(_STRANGER_ID, _CHAT_ID, "ладно, пока")
    assert decision.should_disengage is True
    assert decision.reason == "farewell"


async def test_terse_streak_ends_the_conversation(tmp_path: Path) -> None:
    """Собеседник отписывается односложно — значит, ему не интересно."""
    lifecycle = _lifecycle(tmp_path)
    assert (await lifecycle.evaluate(_STRANGER_ID, _CHAT_ID, "ок")).should_disengage is False
    assert (await lifecycle.evaluate(_STRANGER_ID, _CHAT_ID, "ага")).should_disengage is False
    assert (await lifecycle.evaluate(_STRANGER_ID, _CHAT_ID, "ясно")).should_disengage is True


async def test_annoyance_accumulates_until_the_threshold(tmp_path: Path) -> None:
    lifecycle = _lifecycle(tmp_path)
    decision = None
    for _ in range(5):
        decision = await lifecycle.evaluate(_STRANGER_ID, _CHAT_ID, "ты тупая, отвечай!!!")
    assert decision is not None
    assert decision.annoyance_score >= ANNOYANCE_THRESHOLD
    assert decision.should_disengage is True


async def test_calm_messages_decay_annoyance(tmp_path: Path) -> None:
    """Один плохой день не должен закрывать человека навсегда."""
    lifecycle = _lifecycle(tmp_path)
    await lifecycle.evaluate(_STRANGER_ID, _CHAT_ID, "ты тупая")
    after_hostile = (await lifecycle.get_state(_STRANGER_ID, _CHAT_ID)).annoyance_score
    await lifecycle.evaluate(_STRANGER_ID, _CHAT_ID, "слушай, а как ты относишься к линуксу")
    after_calm = (await lifecycle.get_state(_STRANGER_ID, _CHAT_ID)).annoyance_score
    assert after_calm < after_hostile


async def test_closed_conversation_stays_closed_for_more_pressure(tmp_path: Path) -> None:
    lifecycle = _lifecycle(tmp_path)
    await lifecycle.evaluate(_STRANGER_ID, _CHAT_ID, "пока")
    decision = await lifecycle.evaluate(_STRANGER_ID, _CHAT_ID, "ЭЙ ОТВЕЧАЙ!!!")
    assert decision.should_disengage is True


async def test_substantive_message_reopens_a_closed_conversation(tmp_path: Path) -> None:
    lifecycle = _lifecycle(tmp_path)
    await lifecycle.evaluate(_STRANGER_ID, _CHAT_ID, "пока")
    decision = await lifecycle.evaluate(
        _STRANGER_ID, _CHAT_ID, "слушай, я вернулся, хотел спросить про твой проект"
    )
    assert decision.should_disengage is False


# -- персистентность -----------------------------------------------------------


async def test_annoyance_survives_a_restart(tmp_path: Path) -> None:
    """
    Регрессия по смыслу: без персистентности достаточно было бы
    перезапустить процесс, чтобы «закрытый» человек получил чистый лист.
    """
    first = _lifecycle(tmp_path, db_name="shared.db")
    for _ in range(5):
        await first.evaluate(_STRANGER_ID, _CHAT_ID, "ты тупая, отвечай!!!")

    # Новый экземпляр поверх той же БД = перезапуск приложения.
    second = _lifecycle(tmp_path, db_name="shared.db")
    state = await second.get_state(_STRANGER_ID, _CHAT_ID)
    assert state.annoyance_score >= ANNOYANCE_THRESHOLD
    assert state.status is ConversationStatus.CLOSED
