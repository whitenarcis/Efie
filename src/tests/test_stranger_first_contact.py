"""
Тесты первого контакта с незнакомцем (efi.behavior.conversation_lifecycle).

Регрессия из жизни: новому человеку Эфи не отвечала вообще, а в консоли
стояло «disengaging ... (farewell, annoyance=0.00)» — она считала, что он с
ней попрощался.

Причин было две, и обе одинаково смертельные для первого впечатления.

1. Маркеры прощания искались ПОДСТРОКОЙ. «пока» — одно из самых частых
   сочетаний букв в русском: в него попадали «покажи», «показалось», «пока
   что», «пока не понял». «спок» ловил «успокойся», «бай» — «Байкал» и
   «байт». Сообщение «привет! покажи, что умеешь» читалось как «до
   свидания».

2. Даже настоящее прощание в ПЕРВОЙ реплике закрывало диалог, которого ещё
   не было. «Разговор исчерпан» — суждение о разговоре; на первой фразе
   выносить его не из чего.

Отдельно проверяется, что починка не сняла защиту: настоящее прощание,
настоящая грубость и настоящая потеря интереса по-прежнему работают.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from efi.behavior.conversation_lifecycle import (
    GREETING_GRACE_TURNS,
    ConversationLifecycle,
    ConversationStatus,
    is_farewell,
    score_annoyance,
)
from efi.db.core import Database
from efi.db.models import MIGRATIONS

_OWNER_ID = 2129889949
_STRANGER_ID = 777123
_CHAT_ID = 777123


def _lifecycle(tmp_path: Path) -> ConversationLifecycle:
    return ConversationLifecycle(Database(tmp_path / "efi.db", migrations=MIGRATIONS), owner_id=_OWNER_ID)


# -- маркеры больше не ловят обычные слова ------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "привет! покажи, что умеешь",
        "пока не понял, как это работает",
        "я пока новенький тут",
        "давай пока так оставим",
        "успокойся, я просто спросил",
        "показалось, что ты офлайн",
        "мне нужен байт информации",
        "я из Байкальска",
        "пока что не решил",
    ],
)
def test_ordinary_words_are_not_a_goodbye(text: str) -> None:
    assert is_farewell(text) is False, f"{text!r} — это не прощание"


@pytest.mark.parametrize(
    "text",
    ["пока", "ну ладно, пока", "пока!", "пока 👋", "пока...", "всё, до связи", "спокойной ночи",
     "увидимся", "ладно, я спать, бай", "bye", "до завтра", "чао!"],
)
def test_real_goodbyes_are_still_recognised(text: str) -> None:
    assert is_farewell(text) is True, f"{text!r} — это прощание"


def test_hostility_is_matched_in_any_word_form() -> None:
    """
    Грубость пишут как придётся, поэтому она ищется по НАЧАЛУ слова.
    Ложных срабатываний, как у «пока», здесь нет: корни длинные и
    однозначные.
    """
    assert score_annoyance("бесполезная железка") > 0
    assert score_annoyance("ответьте немедленно") > 0
    assert score_annoyance("слушай, а что ты думаешь про линукс") == 0.0


# -- первое сообщение незнакомца ----------------------------------------------


async def test_a_new_person_gets_an_answer(tmp_path: Path) -> None:
    """Главная регрессия: ровно то сообщение, после которого в консоли было «farewell»."""
    lifecycle = _lifecycle(tmp_path)

    decision = await lifecycle.evaluate(_STRANGER_ID, _CHAT_ID, "привет! покажи, что умеешь")

    assert decision.should_disengage is False
    assert decision.reason == ""


async def test_even_a_real_goodbye_cannot_close_a_conversation_that_never_started(tmp_path: Path) -> None:
    """Попрощаться можно только с тем, с кем разговаривал."""
    lifecycle = _lifecycle(tmp_path)

    assert (await lifecycle.evaluate(_STRANGER_ID, _CHAT_ID, "пока")).should_disengage is False


async def test_a_terse_first_message_is_not_disinterest(tmp_path: Path) -> None:
    """«ок» первым сообщением — обычная осторожность с незнакомым, а не «мне неинтересно»."""
    lifecycle = _lifecycle(tmp_path)

    for _ in range(GREETING_GRACE_TURNS):
        assert (await lifecycle.evaluate(_STRANGER_ID, _CHAT_ID, "ок")).should_disengage is False


async def test_the_grace_period_runs_out(tmp_path: Path) -> None:
    """Фора не бесконечна: после неё правила работают как раньше."""
    lifecycle = _lifecycle(tmp_path)
    for _ in range(GREETING_GRACE_TURNS):
        await lifecycle.evaluate(_STRANGER_ID, _CHAT_ID, "слушай, а расскажи про себя")

    decision = await lifecycle.evaluate(_STRANGER_ID, _CHAT_ID, "ладно, пока")

    assert decision.should_disengage is True
    assert decision.reason == "farewell"


async def test_open_hostility_is_not_covered_by_the_grace_period(tmp_path: Path) -> None:
    """
    Единственное исключение из форы. Отвечать на прямую грубость Эфи не
    обязана и незнакомцу — иначе фора превратилась бы в бесплатный проход
    для тех, кто пришёл именно хамить.
    """
    lifecycle = _lifecycle(tmp_path)

    decision = None
    for _ in range(4):
        decision = await lifecycle.evaluate(_STRANGER_ID, _CHAT_ID, "ты тупая, отвечай!!!")

    assert decision is not None
    assert decision.should_disengage is True


# -- счётчик реплик ------------------------------------------------------------


async def test_turn_count_survives_a_restart(tmp_path: Path) -> None:
    """
    Иначе перезапуск выдавал бы каждому собеседнику новую фору — и человек,
    которого Эфи уже закрыла, получал бы чистый лист по кругу.
    """
    first = _lifecycle(tmp_path)
    for _ in range(3):
        await first.evaluate(_STRANGER_ID, _CHAT_ID, "рассказывай, что интересного")

    second = _lifecycle(tmp_path)
    state = await second.get_state(_STRANGER_ID, _CHAT_ID)

    assert state.turns == 3


async def test_reopening_does_not_hand_out_a_second_grace_period(tmp_path: Path) -> None:
    """Человек не становится незнакомцем заново оттого, что разговор один раз закрывался."""
    lifecycle = _lifecycle(tmp_path)
    for _ in range(GREETING_GRACE_TURNS):
        await lifecycle.evaluate(_STRANGER_ID, _CHAT_ID, "расскажи что-нибудь про себя")
    await lifecycle.evaluate(_STRANGER_ID, _CHAT_ID, "пока")
    assert (await lifecycle.get_state(_STRANGER_ID, _CHAT_ID)).status is ConversationStatus.CLOSED

    await lifecycle.evaluate(_STRANGER_ID, _CHAT_ID, "слушай, я вернулся, хотел кое-что спросить")
    decision = await lifecycle.evaluate(_STRANGER_ID, _CHAT_ID, "ну всё, до связи")

    assert decision.should_disengage is True
