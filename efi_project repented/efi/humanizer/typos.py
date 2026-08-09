"""
efi/humanizer/typos.py

Генерация редких "человеческих" опечаток — пост-обработка уже чистого текста
от LLM перед отправкой (модель сама опечатки не имитирует, это осознанно
отдельный, детерминированно тестируемый слой). Три равновероятных вида,
имитирующих реальные ошибки набора текста, а не случайный шум:
    - пропуск символа (не успела нажать клавишу);
    - замена буквы на соседнюю по раскладке клавиатуры (промахнулась мимо
      клавиши — использует HumanizerSettings.keyboard_neighbors, заполняется
      из behavior.toml, см. efi/config/schema.py; если не задан — встроенная
      раскладка ЙЦУКЕН+QWERTY по умолчанию);
    - перестановка двух соседних букв (напечатала не в том порядке).
"""

from __future__ import annotations

import random
from enum import Enum

from efi.config.schema import HumanizerSettings

# Встроенная раскладка на случай, если behavior.toml не переопределяет
# keyboard_neighbors — базовое покрытие ЙЦУКЕН (кириллица) и QWERTY (латиница),
# соседи считаются по физическому расположению клавиш, а не по алфавиту.
_DEFAULT_KEYBOARD_NEIGHBORS: dict[str, list[str]] = {
    # ЙЦУКЕН — верхний ряд
    "й": ["ц", "ф"], "ц": ["й", "у", "ф", "ы"], "у": ["ц", "к", "ы", "в"],
    "к": ["у", "е", "в", "а"], "е": ["к", "н", "а", "п"], "н": ["е", "г", "п", "р"],
    "г": ["н", "ш", "р", "о"], "ш": ["г", "щ", "о", "л"], "щ": ["ш", "з", "л", "д"],
    "з": ["щ", "х", "д", "ж"], "х": ["з", "ъ", "ж", "э"],
    # ЙЦУКЕН — средний ряд
    "ф": ["й", "ц", "ы", "я"], "ы": ["ц", "у", "ф", "в", "я", "ч"], "в": ["у", "к", "ы", "а", "ч", "с"],
    "а": ["к", "е", "в", "п", "с", "м"], "п": ["е", "н", "а", "р", "м", "и"], "р": ["н", "г", "п", "о", "и", "т"],
    "о": ["г", "ш", "р", "л", "т", "ь"], "л": ["ш", "щ", "о", "д", "ь", "б"], "д": ["щ", "з", "л", "ж", "б", "ю"],
    "ж": ["з", "х", "д", "э", "ю"], "э": ["х", "ъ", "ж"],
    # ЙЦУКЕН — нижний ряд
    "я": ["ф", "ы", "ч"], "ч": ["ы", "в", "я", "с"], "с": ["в", "а", "ч", "м"],
    "м": ["а", "п", "с", "и"], "и": ["п", "р", "м", "т"], "т": ["р", "о", "и", "ь"],
    "ь": ["о", "л", "т", "б"], "б": ["л", "д", "ь", "ю"], "ю": ["д", "ж", "б"],
    # QWERTY (латиница)
    "q": ["w", "a"], "w": ["q", "e", "a", "s"], "e": ["w", "r", "s", "d"],
    "r": ["e", "t", "d", "f"], "t": ["r", "y", "f", "g"], "y": ["t", "u", "g", "h"],
    "u": ["y", "i", "h", "j"], "i": ["u", "o", "j", "k"], "o": ["i", "p", "k", "l"], "p": ["o", "l"],
    "a": ["q", "w", "s", "z"], "s": ["w", "e", "a", "d", "z", "x"], "d": ["e", "r", "s", "f", "x", "c"],
    "f": ["r", "t", "d", "g", "c", "v"], "g": ["t", "y", "f", "h", "v", "b"], "h": ["y", "u", "g", "j", "b", "n"],
    "j": ["u", "i", "h", "k", "n", "m"], "k": ["i", "o", "j", "l", "m"], "l": ["o", "p", "k"],
    "z": ["a", "s", "x"], "x": ["s", "d", "z", "c"], "c": ["d", "f", "x", "v"],
    "v": ["f", "g", "c", "b"], "b": ["g", "h", "v", "n"], "n": ["h", "j", "b", "m"], "m": ["j", "k", "n"],
}


class TypoKind(str, Enum):
    """Вид алгоритмической опечатки — см. докстринг модуля."""

    SKIP = "skip"
    NEIGHBOR = "neighbor"
    TRANSPOSE = "transpose"


def inject_typo(text: str, settings: HumanizerSettings) -> str:
    """
    С вероятностью `settings.typo_probability` (рекомендованный диапазон —
    3-5%, см. HumanizerSettings.typo_probability) накладывает ОДНУ
    случайно выбранную опечатку одного из трёх видов (TypoKind). Короткие
    тексты (короче `typo_min_text_length`) не трогает — опечатка в короткой
    реплике выглядит неестественно чаще, чем естественно.

    Если для выбранного вида в тексте не нашлось подходящей позиции
    (например, текст без единой известной буквы раскладки) — возвращает
    текст как есть, не пытаясь силой применить другой вид: редкий частичный
    промах не стоит того, чтобы усложнять эту в остальном простую функцию.

    Чистая функция (кроме обращения к random) — без I/O, легко тестируется
    отдельно от Worker'а/отправки сообщений.
    """
    if len(text) < settings.typo_min_text_length:
        return text
    if random.random() > settings.typo_probability:
        return text

    keyboard_neighbors = settings.keyboard_neighbors or _DEFAULT_KEYBOARD_NEIGHBORS
    kind = random.choice(list(TypoKind))

    if kind is TypoKind.SKIP:
        return _skip_character(text)
    if kind is TypoKind.NEIGHBOR:
        return _replace_with_neighbor(text, keyboard_neighbors)
    return _transpose_adjacent(text)


def _skip_character(text: str) -> str:
    """Пропускает один случайный буквенный символ — как будто не успела нажать клавишу."""
    positions = [i for i, ch in enumerate(text) if ch.isalpha()]
    if not positions:
        return text
    position = random.choice(positions)
    return text[:position] + text[position + 1 :]


def _replace_with_neighbor(text: str, keyboard_neighbors: dict[str, list[str]]) -> str:
    """Заменяет один случайный буквенный символ на соседнюю по раскладке клавишу."""
    positions = [i for i, ch in enumerate(text) if ch.lower() in keyboard_neighbors]
    if not positions:
        return text
    position = random.choice(positions)
    original_char = text[position]
    replacement = random.choice(keyboard_neighbors[original_char.lower()])
    if original_char.isupper():
        replacement = replacement.upper()
    return text[:position] + replacement + text[position + 1 :]


def _transpose_adjacent(text: str) -> str:
    """Меняет местами два соседних буквенных символа — как будто напечатала не в том порядке."""
    positions = [i for i in range(len(text) - 1) if text[i].isalpha() and text[i + 1].isalpha()]
    if not positions:
        return text
    position = random.choice(positions)
    chars = list(text)
    chars[position], chars[position + 1] = chars[position + 1], chars[position]
    return "".join(chars)


__all__ = ["TypoKind", "inject_typo"]
