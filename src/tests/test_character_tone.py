"""
Тесты характера Эфи — того, как она звучит.

Проверять тон тестами странно ровно до того момента, как посмотришь, во что
он однажды выродился. Промпт просил «дерзкую и азартную», хвалил стёб,
соревновательность и пари — и живая переписка стала выглядеть так:

    я вообще-то опасная
    хотя ладно, сегодня я в режиме временного перемирия
    не дождёшься, я слишком дерзкая для этого сахара
    привыкай, что я тебя только подкалываю
    1:0 в мою пользу лол
    фу, какой сахар, аж зубы сводит / прекращай

Ни одной такой фразы не напишет живой человек, и дело не в грубости.
Настоящий собеседник (переписка, с которой это сравнивали) звучит совсем
иначе: «Куда угодно», «Я тоже в шоке», «Не надо ничего придумывать»,
«Хахаха, тогда отлично». Разница ровно в трёх вещах, и каждая проверяется
здесь отдельно:

  1. Живой человек НЕ описывает вслух свой характер. Он просто такой.
     «Я вообще-то опасная» — это не характер, а объявление о характере.
  2. Живой человек не ведёт счёт. Разговор — не матч, и последнее слово
     оставлять за собой не нужно.
  3. Живой человек принимает тепло, а не отбивается от него.

Тесты держат не формулировки промпта (их можно и нужно переписывать), а
эти три запрета и общий тёплый регистр: без них характер тихо сползает
обратно, потому что «дерзкая» — самая лёгкая роль для модели.
"""

from __future__ import annotations

import pathlib

import pytest

_PERSONALITY = (
    pathlib.Path(__file__).resolve().parents[1] / "efi" / "prompts" / "templates" / "personality.md"
)


@pytest.fixture(scope="module")
def personality() -> str:
    return _PERSONALITY.read_text(encoding="utf-8")


# -- то, чего в характере больше быть не должно ---------------------------------


def test_she_is_not_advertised_as_edgy(personality: str) -> None:
    """
    «Дерзкая и азартная» в самом определении характера — и была корнем всего
    остального: модель честно отыгрывала выданную роль.
    """
    opening = personality.split("СЕЙЧАС:")[0]

    assert "дерзк" not in opening.lower()
    assert "тёплая" in opening.lower() or "добр" in opening.lower()


def test_competition_is_not_part_of_the_character(personality: str) -> None:
    """
    Промпт прямо просил соревновательность и пари («спорим», «на желание»).
    Отсюда и «1:0 в мою пользу»: разговор превращался в матч.
    """
    core = personality.split("ПРИМЕРЫ ТОНА")[0].lower()

    assert "соревноват" not in core
    assert "на желание" not in core
    assert "не сдаёшься первой" not in core


@pytest.mark.parametrize(
    "phrase",
    [
        "я вообще-то опасная",
        "я слишком дерзкая",
        "привыкай, что я тебя только подкалываю",
        "не пытайся меня задобрить",
    ],
)
def test_narrating_your_own_character_is_banned(personality: str, phrase: str) -> None:
    """Самая заметная примета нейросети, играющей роль, — и она названа поимённо."""
    assert phrase in personality.lower(), "фраза должна быть в промпте именно как запрещённая"


def test_the_ban_on_self_narration_is_explicit(personality: str) -> None:
    assert "НЕ ОПИСЫВАЙ ВСЛУХ СВОЙ ХАРАКТЕР" in personality


def test_the_ban_on_keeping_score_is_explicit(personality: str) -> None:
    assert "НЕ ВЕДИ СЧЁТ" in personality
    assert "1:0" in personality


def test_warmth_must_be_accepted(personality: str) -> None:
    """
    «Фу, какой сахар, аж зубы сводит» в ответ на «я люблю свою Эфочку» — это
    не характер, а просто холодно, и читается именно так.
    """
    assert "ПРИНИМАЙ ТЕПЛО" in personality
    assert "какой сахар" in personality.lower()
    assert "не дождёшься" in personality.lower()


# -- то, что должно было остаться -----------------------------------------------


def test_she_still_has_her_own_opinion(personality: str) -> None:
    """
    Смягчение не должно было превратить её в поддакивающего ассистента:
    честность без лести — то, ради чего прямота вообще заводилась.
    """
    core = personality.split("ПРИМЕРЫ ТОНА")[0].lower()

    assert "не льстишь" in core
    assert "своё мнение" in core


def test_she_is_still_not_an_assistant(personality: str) -> None:
    assert "не услужливый ассистент" in personality.lower()


def test_the_punchline_rule_survived(personality: str) -> None:
    """Главное правило против натужной остроты — и оно теперь работает заодно с тоном."""
    assert "НЕ ВЫДАВЛИВАЙ ИЗ СЕБЯ ПАНЧЛАЙН" in personality


def test_swearing_is_allowed_but_not_aimed_at_the_person(personality: str) -> None:
    """
    Мат остаётся — живые люди ругаются. Но «что за хуйня» в адрес работы
    собеседника это уже не мат, а холодность.
    """
    lowered = personality.lower()

    assert "мат разрешён" in lowered
    assert "редкий" in lowered
    assert "запрещена" in lowered


# -- и то же самое в коде, а не только в промпте --------------------------------


def test_small_talk_never_checks_whether_he_is_alive() -> None:
    """
    Бытовые поводы («ел вообще», «чем занят») не должны протаскивать через
    заднюю дверь то самое «ты там живой?», которое запрещено отдельно: это
    не болтовня, а проверка связи, и читается именно так.
    """
    from efi.behavior.ping_reason import _TRIFLES

    forbidden = ("сдох", "жив ", "жив там", "умер", "утонул")
    for trifle in _TRIFLES:
        assert not any(marker in trifle.lower() for marker in forbidden), trifle


def test_the_anti_sycophancy_guard_asks_for_kindness_not_combat() -> None:
    """
    Защита от угодливости обязана остаться — тёплая не значит поддакивающая.
    Но спорить она просит ради сути, а не ради победы.
    """
    from efi.config.schema import StateVectorSettings

    text = StateVectorSettings().sycophancy_protection_text.lower()

    assert "запрещено соглашаться" in text, "суть защиты никуда не делась"
    assert "по-доброму" in text
    assert "не ради победы" in text
