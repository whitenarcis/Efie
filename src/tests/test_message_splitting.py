"""Тесты для efi.humanizer.message_splitting: разбивка на бабблы + компенсация времени LLM для первого."""

from __future__ import annotations

from efi.config.schema import HumanizerSettings
from efi.humanizer.message_splitting import first_chunk_typing_delay, split_into_messages

_SETTINGS = HumanizerSettings(typing_wpm_min=120, typing_wpm_max=120)  # фиксированная скорость — детерминированный тест


def _settings(**overrides: object) -> HumanizerSettings:
    return HumanizerSettings(typing_wpm_min=120, typing_wpm_max=120, **overrides)  # type: ignore[arg-type]


def test_split_into_messages_respects_explicit_delimiter() -> None:
    assert split_into_messages("привет /// как дела", _SETTINGS) == ["привет", "как дела"]


def test_split_into_messages_single_message_without_delimiter() -> None:
    assert split_into_messages("просто одно сообщение", _SETTINGS) == ["просто одно сообщение"]


def test_split_into_messages_empty_text() -> None:
    assert split_into_messages("   ", _SETTINGS) == []


def test_first_chunk_typing_delay_is_zero_when_llm_was_slow_enough() -> None:
    chunk = "привет"  # 6 символов
    cps_min, cps_max = _SETTINGS.characters_per_second_range()
    target_delay = len(chunk) / cps_min  # верхняя граница идеального времени печати при фикс. скорости
    delay = first_chunk_typing_delay(chunk, _SETTINGS, llm_generation_time=target_delay + 10.0)
    assert delay == 0.0


def test_first_chunk_typing_delay_is_positive_when_llm_was_fast() -> None:
    chunk = "довольно длинный первый баббл ответа, чтобы точно занять заметное время печати"
    delay = first_chunk_typing_delay(chunk, _SETTINGS, llm_generation_time=0.0)
    assert delay > 0.0


def test_first_chunk_typing_delay_partial_credit() -> None:
    chunk = "довольно длинный первый баббл ответа, чтобы точно занять заметное время печати"
    cps_min, cps_max = _SETTINGS.characters_per_second_range()
    naive_delay = len(chunk) / cps_min
    partial_delay = first_chunk_typing_delay(chunk, _SETTINGS, llm_generation_time=1.0)
    assert 0.0 < partial_delay < naive_delay


def test_a_thought_per_line_becomes_a_message_per_line() -> None:
    """
    В мессенджере перевод строки — это место, где человек отпускает Enter.
    Пока разбивались только пустые строки, ответ, где модель писала мысль на
    строку, уходил одной простынёй на пятнадцать строк.
    """
    text = (
        "давай уточним: ищем на конкретных сайтах или просто гуглим?\n"
        "и какие форматы предпочитаешь — flac, wav, ape?\n"
        "либо сразу всё подряд, а потом фильтруем"
    )

    parts = split_into_messages(text, _settings(), long_message_threshold=40)

    assert len(parts) == 3
    assert parts[0].startswith("давай уточним")


def test_a_turn_has_a_ceiling_on_how_much_she_dumps_at_once() -> None:
    """
    Человек в переписке не выдаёт полторы тысячи символов подряд. Лишние
    реплики отбрасываются целиком: недосказанная мысль читается нормально, а
    оборванная на полуслове — как сбой.
    """
    settings = _settings(max_reply_chars_per_turn=200)
    text = " /// ".join(f"мысль номер {index} " + "с подробностями" * 5 for index in range(10))

    parts = split_into_messages(text, settings)

    assert parts, "ответ не должен пропадать целиком"
    assert sum(len(part) for part in parts) <= 200 or len(parts) == 1
    assert all(part.strip() for part in parts)


def test_a_single_giant_bubble_is_trimmed_at_a_sentence() -> None:
    settings = _settings(max_reply_chars_per_turn=120)
    text = (
        "Первая мысль тут закончена и она короткая. "
        "Вторая мысль тоже вполне закончена и осмысленна. "
        "Третья мысль уже точно не поместится в отведённый бюджет символов."
    )

    parts = split_into_messages(text, settings)

    assert len(parts) == 1
    assert len(parts[0]) <= 120
    assert parts[0].endswith("."), "обрыв на полуслове читается как сбой"
