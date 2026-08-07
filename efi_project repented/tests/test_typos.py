"""Тесты для efi.humanizer.typos: три вида алгоритмических опечаток."""

from __future__ import annotations

import random

from efi.config.schema import HumanizerSettings
from efi.humanizer.typos import (
    TypoKind,
    _replace_with_neighbor,
    _skip_character,
    _transpose_adjacent,
    inject_typo,
)

_NEIGHBORS = {"а": ["в"], "б": ["г"]}


def test_skip_character_removes_one_letter() -> None:
    text = "привет"
    random.seed(1)
    result = _skip_character(text)
    assert len(result) == len(text) - 1
    assert any(text[:i] + text[i + 1 :] == result for i in range(len(text)))


def test_skip_character_noop_without_letters() -> None:
    assert _skip_character("123 456") == "123 456"


def test_replace_with_neighbor_uses_configured_layout() -> None:
    result = _replace_with_neighbor("а", _NEIGHBORS)
    assert result == "в"


def test_replace_with_neighbor_preserves_case() -> None:
    result = _replace_with_neighbor("А", _NEIGHBORS)
    assert result == "В"


def test_replace_with_neighbor_noop_without_known_letters() -> None:
    assert _replace_with_neighbor("ъ", _NEIGHBORS) == "ъ"


def test_transpose_adjacent_swaps_two_letters() -> None:
    random.seed(1)
    result = _transpose_adjacent("ab")
    assert result == "ba"


def test_transpose_adjacent_noop_without_adjacent_letters() -> None:
    assert _transpose_adjacent("a1") == "a1"
    assert _transpose_adjacent("a") == "a"


def test_inject_typo_never_touches_short_text() -> None:
    settings = HumanizerSettings(typo_probability=1.0, typo_min_text_length=100)
    assert inject_typo("привет", settings) == "привет"


def test_inject_typo_never_fires_at_zero_probability() -> None:
    settings = HumanizerSettings(typo_probability=0.0, typo_min_text_length=1)
    text = "это достаточно длинный текст для опечатки"
    assert inject_typo(text, settings) == text


def test_inject_typo_always_fires_at_full_probability() -> None:
    settings = HumanizerSettings(typo_probability=1.0, typo_min_text_length=1)
    text = "это достаточно длинный текст для опечатки"
    mutated = {inject_typo(text, settings) for _ in range(50)}
    assert any(candidate != text for candidate in mutated)


def test_typo_kind_has_three_members() -> None:
    assert set(TypoKind) == {TypoKind.SKIP, TypoKind.NEIGHBOR, TypoKind.TRANSPOSE}
