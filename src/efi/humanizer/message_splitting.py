"""
efi/humanizer/message_splitting.py

Разбивка итогового текста ответа на цепочку отдельных сообщений — перенос
"///"-разрывов из текущей реализации Эфи: модель сама размечает, где
заканчивается одна "мысль"/реплика и начинается следующая, вместо того чтобы
всегда отправлять один длинный монолитный текст. Живой человек в мессенджере
почти никогда не пишет один сплошной абзац на пять предложений — он рвёт
мысль на несколько сообщений подряд.

`first_chunk_typing_delay` — компенсация времени генерации LLM (обычно
5-10 секунд) для ПЕРВОГО баббла цепочки: пока модель думает, Worker уже
транслирует статус TYPING (см. efi/notifications/worker.py), так что реальное
время ожидания ответа собеседником УЖЕ засчитывается как "печатает". Если
генерация заняла дольше, чем заняла бы естественная печать первого куска —
дополнительная искусственная пауза не нужна, кусок уходит сразу. Для
последующих кусков цепочки (после "///") это не применяется — они всегда
идут через обычный calculate_typing_delay (efi/humanizer/typing_simulation.py).
"""

from __future__ import annotations

import random
import re

from efi.config.schema import HumanizerSettings
from efi.utils.text import trim_to_last_sentence

_EXPLICIT_DELIMITER_RE = re.compile(r"\s*///\s*")
_PARAGRAPH_DELIMITER_RE = re.compile(r"\n\s*\n")

#: Одиночный перевод строки. В мессенджере это ровно то место, где человек
#: отпускает Enter и отправляет реплику: строка = мысль. Пока разбивались
#: только пустые строки, ответ, где модель писала мысль на строку, уходил
#: одной простынёй на пятнадцать строк — то есть ровно тем, чего разбивка
#: должна не допускать.
_LINE_DELIMITER_RE = re.compile(r"\n+")

#: Порог длины (символов), начиная с которого пробуем мягкую разбивку по
#: абзацам, если модель не расставила явные "///" — примерно 2-3 обычных
#: телеграм-реплики.
_DEFAULT_LONG_MESSAGE_THRESHOLD = 280

#: Сколько слов считается «коротышом» — бабблом, который человек с телефона
#: не набирает, а выстреливает («прикинь», «я ток щас узнала», «а ты?»).
#: Такие идут почти встык, а не через полноценную паузу по WPM.
_SHORT_BUBBLE_MAX_WORDS = 3


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

    `settings.max_messages_per_burst` — не «нормальная длина ответа», а
    аварийный потолок. Раньше он стоял на 5 и работал как настоящий лимит:
    «поток мыслей» из 8 коротких реплик («прикинь /// фрустрация, это когда
    тип не может достичь цели /// я ток щас узнала /// а ты?») схлопывался в
    5 сообщений, где последнее было слипшимся комом из всего остатка. Живой
    человек, который делится находкой или эмоционирует, спокойно шлёт
    подряд 5-10 коротких реплик, поэтому потолок поднят и должен срабатывать
    только на явно неадекватной разметке. «Лишние» куски по-прежнему
    склеиваются в последнее сообщение, а не отбрасываются.

    Сколько бабблов уместно в конкретном ответе — решает модель (правила в
    системном промпте: бытовая переписка — 1-2 сообщения, рассказ или
    эмоция — свободная серия), а не эта функция: здесь нет контекста, чтобы
    отличить «ага» от рассказа с форума.
    """
    stripped = text.strip()
    if not stripped:
        return []

    if _EXPLICIT_DELIMITER_RE.search(stripped):
        parts = _EXPLICIT_DELIMITER_RE.split(stripped)
    elif len(stripped) > long_message_threshold and _PARAGRAPH_DELIMITER_RE.search(stripped):
        parts = _PARAGRAPH_DELIMITER_RE.split(stripped)
    elif len(stripped) > long_message_threshold and _LINE_DELIMITER_RE.search(stripped):
        parts = _LINE_DELIMITER_RE.split(stripped)
    else:
        parts = [stripped]

    cleaned_parts = [part.strip() for part in parts if part.strip()]
    if not cleaned_parts:
        return []

    capped = _cap_message_count(cleaned_parts, settings.max_messages_per_burst)
    return _fit_into_budget(capped, settings.max_reply_chars_per_turn)


def _cap_message_count(parts: list[str], max_count: int) -> list[str]:
    if max_count < 1 or len(parts) <= max_count:
        return parts
    head = parts[: max_count - 1]
    tail = "\n\n".join(parts[max_count - 1 :])
    return [*head, tail]


def _fit_into_budget(parts: list[str], budget: int) -> list[str]:
    """
    Оставляет столько реплик, сколько помещается в бюджет одного хода.

    Лишние отбрасываются ЦЕЛИКОМ, а не режутся: оборванная на полуслове
    мысль читается как сбой, а недосказанная — как нормальная человеческая
    реплика, к которой можно вернуться следующим сообщением. Первая реплика
    отбрасыванию не подлежит никогда (иначе ответа не будет вовсе) — если она
    одна длиннее бюджета, у неё отрезается хвост по последнему законченному
    предложению.

    Смысл всего этого в одной строчке: человек в переписке не выдаёт полторы
    тысячи символов подряд. Тот, кто выдаёт, — не собеседник, а лектор.
    """
    if budget < 1 or not parts:
        return parts

    kept: list[str] = []
    used = 0
    for part in parts:
        if kept and used + len(part) > budget:
            break
        kept.append(part)
        used += len(part)

    head = kept[0]
    if len(head) > budget:
        trimmed = trim_to_last_sentence(head[:budget])
        kept[0] = trimmed or head[:budget].rstrip()
    return kept


def is_short_bubble(text: str, *, max_words: int = _SHORT_BUBBLE_MAX_WORDS) -> bool:
    """Баббл в 1-3 слова — реплика, которую выстреливают, а не набирают."""
    return 0 < len(text.split()) <= max_words


def short_bubble_delay(settings: HumanizerSettings) -> float:
    """
    Задержка перед коротышом — доли секунды вместо полноценной паузы по WPM.

    Без этого серия из коротких реплик шла в том же темпе, что и абзац
    текста: `calculate_typing_delay` прибавляет паузу «на подумать» (1-2.2с)
    и зажимает результат снизу `typing_delay_min_seconds` (1.8с), так что
    «а ты?» уходило через две секунды после предыдущего баббла. Серия из
    шести таких растягивалась на четверть минуты и читалась как медленный
    бот, а не как быстрая печать с телефона.
    """
    return random.uniform(
        settings.short_bubble_delay_min_seconds,
        settings.short_bubble_delay_max_seconds,
    )


def first_chunk_typing_delay(chunk: str, settings: HumanizerSettings, *, llm_generation_time: float) -> float:
    """
    Идеальное время печати первого куска (`target_delay = len(chunk) / chars_per_sec`,
    БЕЗ паузы "на подумать" — этот момент уже покрыт временем самой генерации
    LLM, в отличие от calculate_typing_delay для остальных кусков), за
    вычетом того, что уже "напечатано" за время ожидания ответа модели.

    Возвращает 0.0, если `llm_generation_time` не меньше идеального времени
    печати — тогда первый баббл уходит сразу после получения ответа, без
    наложения ещё одной искусственной паузы поверх уже прошедшего ожидания.
    """
    cps_min, cps_max = settings.characters_per_second_range()
    chars_per_second = random.uniform(cps_min, cps_max)
    target_delay = len(chunk) / chars_per_second if chars_per_second > 0 else 0.0
    return max(target_delay - llm_generation_time, 0.0)


__all__ = ["first_chunk_typing_delay", "is_short_bubble", "short_bubble_delay", "split_into_messages"]
