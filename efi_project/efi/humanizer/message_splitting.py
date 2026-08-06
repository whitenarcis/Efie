"""
efi/humanizer/message_splitting.py

Разбивка итогового текста ответа на цепочку отдельных сообщений — перенос
"///"-разрывов из текущей реализации Эфи: модель сама размечает, где
заканчивается одна "мысль"/реплика и начинается следующая, вместо того чтобы
всегда отправлять один длинный монолитный текст. Живой человек в мессенджере
почти никогда не пишет один сплошной абзац на пять предложений — он рвёт
мысль на несколько сообщений подряд.
"""

from __future__ import annotations

import re

from efi.config.schema import HumanizerSettings

_EXPLICIT_DELIMITER_RE = re.compile(r"\s*///\s*")
_PARAGRAPH_DELIMITER_RE = re.compile(r"\n\s*\n")

#: Порог длины (символов), начиная с которого пробуем мягкую разбивку по
#: абзацам, если модель не расставила явные "///" — примерно 2-3 обычных
#: телеграм-реплики.
_DEFAULT_LONG_MESSAGE_THRESHOLD = 280


def split_into_messages(
    text: str,
    settings: HumanizerSettings,
    *,
    long_message_threshold: int = _DEFAULT_LONG_MESSAGE_THRESHOLD,
) -> list[str]:
    """
    Разбивает `text` на список сообщений для последовательной отправки.

    Порядок разбора:
        1. Явные разделители "///" — если модель их расставила, это самый
           надёжный сигнал ("вот здесь я нарочно хочу разбить на реплики").
        2. Если явных разделителей нет, а текст длиннее
           `long_message_threshold` — пробуем разбить по пустым строкам
           (абзацам); более мягкая эвристика, применяется только к заметно
           длинным ответам, чтобы короткие реплики не резались зря.
        3. Иначе — одно сообщение как есть.

    Результат всегда обрезан до `settings.max_messages_per_burst` элементов
    (защита от чрезмерного спама, даже если модель расставила разделители
    слишком щедро) — "лишние" куски склеиваются в последнее сообщение, текст
    не отбрасывается.
    """
    stripped = text.strip()
    if not stripped:
        return []

    if _EXPLICIT_DELIMITER_RE.search(stripped):
        parts = _EXPLICIT_DELIMITER_RE.split(stripped)
    elif len(stripped) > long_message_threshold and _PARAGRAPH_DELIMITER_RE.search(stripped):
        parts = _PARAGRAPH_DELIMITER_RE.split(stripped)
    else:
        parts = [stripped]

    cleaned_parts = [part.strip() for part in parts if part.strip()]
    if not cleaned_parts:
        return []

    return _cap_message_count(cleaned_parts, settings.max_messages_per_burst)


def _cap_message_count(parts: list[str], max_count: int) -> list[str]:
    if max_count < 1 or len(parts) <= max_count:
        return parts
    head = parts[: max_count - 1]
    tail = "\n\n".join(parts[max_count - 1 :])
    return [*head, tail]


__all__ = ["split_into_messages"]
