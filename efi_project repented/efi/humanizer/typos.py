"""
efi/humanizer/typos.py

Генерация редких "человеческих" опечаток — замена буквы на соседнюю по
раскладке клавиатуры, а не случайный символ (так на самом деле печатают
люди). Использует HumanizerSettings.keyboard_neighbors (заполняется из
behavior.toml, см. efi/config/schema.py); если он не задан — используется
встроенная раскладка ЙЦУКЕН+QWERTY по умолчанию.
"""

from __future__ import annotations

import random

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


def inject_typo(text: str, settings: HumanizerSettings) -> str:
    """
    С вероятностью `settings.typo_probability` заменяет один случайный
    буквенный символ текста на соседнюю по раскладке клавишу. Короткие тексты
    (короче `typo_min_text_length`) не трогает — опечатка в короткой реплике
    выглядит неестественно чаще, чем естественно.

    Чистая функция (кроме обращения к random) — без I/O, легко тестируется
    отдельно от Worker'а/отправки сообщений.
    """
    if len(text) < settings.typo_min_text_length:
        return text
    if random.random() > settings.typo_probability:
        return text

    keyboard_neighbors = settings.keyboard_neighbors or _DEFAULT_KEYBOARD_NEIGHBORS
    candidate_positions = [i for i, ch in enumerate(text) if ch.lower() in keyboard_neighbors]
    if not candidate_positions:
        return text

    position = random.choice(candidate_positions)
    original_char = text[position]
    replacement = random.choice(keyboard_neighbors[original_char.lower()])
    if original_char.isupper():
        replacement = replacement.upper()

    return text[:position] + replacement + text[position + 1 :]


__all__ = ["inject_typo"]
