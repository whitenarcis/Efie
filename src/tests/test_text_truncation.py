"""
Тесты обрыва ответа по лимиту токенов (efi.utils.text + Response.was_truncated).

Регрессия: дневник заполнялся записями вида «...хотя ещё лет пять назад это»
— оборванными на полуслове. Причина оказалась не в дневнике: `finish_reason`
приходил от провайдера, был объявлен в схеме Response, и НИ ОДНА строка кода
его не читала. Ответ, упёршийся в `max_tokens`, ничем не отличался от
законченного и сохранялся как есть.

Кириллица делала это регулярным, а не редким: у токенизаторов бесплатных
моделей русский текст стоит в 2-3 раза дороже английского, поэтому бюджеты,
подобранные на глаз (256 и 128 токенов на «1-3 предложения»), выбирались
на середине второй фразы.
"""

from __future__ import annotations

import pytest

from efi.llm.schemas import Choice, Message, Response, Role
from efi.utils.text import looks_unfinished, salvage_truncated, trim_to_last_sentence

# -- Response.was_truncated ---------------------------------------------------


def _response(finish_reason: str | None) -> Response:
    return Response(
        choices=[Choice(index=0, message=Message(role=Role.ASSISTANT, content="текст"),
                        finish_reason=finish_reason)]
    )


@pytest.mark.parametrize("reason", ["length", "max_tokens", "max_output_tokens", "token_limit"])
def test_limit_finish_reasons_are_recognised_as_truncation(reason: str) -> None:
    """Разные провайдеры называют это по-разному — все варианты должны читаться одинаково."""
    assert _response(reason).was_truncated is True


@pytest.mark.parametrize("reason", ["stop", "end_turn", "tool_calls", None, ""])
def test_normal_finish_reasons_are_not_truncation(reason: str | None) -> None:
    assert _response(reason).was_truncated is False


def test_finish_reason_is_matched_case_and_space_insensitively() -> None:
    """Прокси иногда отдают " LENGTH " — это тот же самый обрыв."""
    assert _response(" LENGTH ").was_truncated is True


def test_truncation_in_any_choice_counts() -> None:
    response = Response(
        choices=[
            Choice(index=0, message=Message(role=Role.ASSISTANT, content="a"), finish_reason="stop"),
            Choice(index=1, message=Message(role=Role.ASSISTANT, content="b"), finish_reason="length"),
        ]
    )

    assert response.was_truncated is True


def test_response_without_choices_is_not_truncated() -> None:
    assert Response(choices=[]).was_truncated is False


# -- trim_to_last_sentence ----------------------------------------------------


def test_broken_tail_is_cut_at_the_last_complete_sentence() -> None:
    text = (
        "Меня давно занимает, почему старые дома строили из кирпича. "
        "Это дороже и медленнее, чем панель. Хотя ещё лет пять назад это"
    )

    kept = trim_to_last_sentence(text)

    assert kept.endswith("чем панель.")
    assert "лет пять назад" not in kept


def test_complete_text_survives_untouched() -> None:
    text = "Сегодня читала про акустику залов. Оказалось, что дерево там не для красоты."

    assert trim_to_last_sentence(text) == text


@pytest.mark.parametrize(
    "text",
    [
        "Мне кажется, что вся эта история с ретрофутуризмом на самом деле про",
        "хотя ещё лет пять назад это",
        "",
        "   ",
    ],
)
def test_nothing_salvageable_returns_empty(text: str) -> None:
    """
    Половина мысли хуже её отсутствия: отсутствие видно, а обрубок
    притворяется целым — и потом всплывает в промпте как «воспоминание».
    """
    assert trim_to_last_sentence(text) == ""


def test_a_scrap_left_over_from_a_long_text_is_discarded() -> None:
    """Уцелела одна короткая фраза из большого абзаца — это уже не запись."""
    text = "Ага. " + "Дальше шла длинная мысль, которую модель не успела закончить и оборвала " * 4

    assert trim_to_last_sentence(text) == ""


def test_closing_quote_stays_inside_the_sentence() -> None:
    """Резать перед кавычкой нельзя — фраза без неё выглядит так же оборванной."""
    text = 'Она сказала мне тогда: "это всё равно не работает." А потом добавила что-то про'

    assert trim_to_last_sentence(text) == 'Она сказала мне тогда: "это всё равно не работает."'


def test_ellipsis_counts_as_a_finished_sentence() -> None:
    text = "Я так и не поняла, зачем им вообще сдался этот шрифт… Но, пожалуй, дело в том, что"

    assert trim_to_last_sentence(text).endswith("сдался этот шрифт…")


def test_short_survivor_is_dropped_by_the_absolute_floor() -> None:
    """
    Даже законченная фраза короче 40 символов содержания не несёт — а в
    дневнике будет неотличима от полноценной записи.
    """
    text = "Ну да. " + "И дальше пошла мысль, которую модель не успела закончить"

    assert len(text.split(".")[0]) < 40
    assert trim_to_last_sentence(text) == ""


def test_thresholds_are_configurable() -> None:
    text = "Да. И потом ещё много-много слов, которые оборвались вот прямо тут"

    assert trim_to_last_sentence(text) == ""
    assert trim_to_last_sentence(text, min_chars=3, min_ratio=0.0) == "Да."


# -- looks_unfinished ---------------------------------------------------------


@pytest.mark.parametrize("text", ["оборвалось прямо тут", "и потом,", "", "хвост без знака"])
def test_unfinished_tails_are_detected(text: str) -> None:
    assert looks_unfinished(text) is True


@pytest.mark.parametrize("text", ["Это законченная мысль.", "Правда?!", "И всё…", 'Он сказал "нет."'])
def test_finished_tails_are_not_flagged(text: str) -> None:
    assert looks_unfinished(text) is False


# -- salvage_truncated --------------------------------------------------------


def test_provider_saying_stop_is_believed_even_without_a_full_stop() -> None:
    """
    Не дело эвристики решать, что автор обязан ставить точку. Если провайдер
    сказал «закончил сам» — текст берётся как есть.
    """
    text = "короткая мысль без точки"

    assert salvage_truncated(text, truncated=False) == text


def test_truncated_text_is_trimmed() -> None:
    text = "Первая мысль дописана до конца, вот она целиком. А вторая оборвалась на"

    assert salvage_truncated(text, truncated=True) == "Первая мысль дописана до конца, вот она целиком."


def test_truncation_landing_exactly_on_a_sentence_end_changes_nothing() -> None:
    """Лимит иногда совпадает с концом фразы — резать тогда нечего."""
    text = "Модель уложилась в бюджет ровно по границе предложения."

    assert salvage_truncated(text, truncated=True) == text


def test_truncated_with_nothing_salvageable_returns_empty() -> None:
    assert salvage_truncated("обрывок без единой законченной фразы", truncated=True) == ""


def test_empty_input_is_empty_regardless_of_the_flag() -> None:
    assert salvage_truncated("  ", truncated=False) == ""
    assert salvage_truncated("  ", truncated=True) == ""
