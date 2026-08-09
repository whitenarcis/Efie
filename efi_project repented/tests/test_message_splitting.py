"""Тесты для efi.humanizer.message_splitting: разбивка на бабблы + компенсация времени LLM для первого."""

from __future__ import annotations

from efi.config.schema import HumanizerSettings
from efi.humanizer.message_splitting import first_chunk_typing_delay, split_into_messages

_SETTINGS = HumanizerSettings(typing_wpm_min=120, typing_wpm_max=120)  # фиксированная скорость — детерминированный тест


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
