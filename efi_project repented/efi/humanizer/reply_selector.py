"""
efi/humanizer/reply_selector.py

Разбор тега привязки `[reply:message_id]` в начале баббла и решение, стоит ли
эту привязку вообще применять.

Формат, который размечает модель:

    [reply:102] это про вторую строчку /// а это просто продолжение мысли

Первый баббл уйдёт явным Reply на сообщение 102, второй — обычным текстом.
Тег допустим только в начале баббла: середина текста — это уже содержание
реплики, и `[reply:...]` там почти наверняка не разметка, а случайное
совпадение.

УМЕСТНОСТЬ — половина смысла этого модуля. Reply в личке, где разговор идёт
одной нитью, не добавляет ничего: собеседник и так знает, на что ему
ответили, а каждая реплика со свайпом выглядит как переписка с саппортом.
Промпт учит модель ставить тег редко, но на промпт полагаться нельзя —
модель охотно размечает вообще всё. Поэтому здесь есть и код:

    - тег на сообщение, которого нет в текущей пачке, снимается (модель
      выдумала id — сослаться на произвольное старое сообщение она не может,
      их id ей нигде не показываются);
    - если во входящей пачке было ровно одно сообщение, привязка снимается
      вся: reply на единственную реплику, на которую ты и так отвечаешь, —
      ровно тот шум, от которого мы уходим;
    - если модель разметила ВСЕ бабблы на одно и то же сообщение, это не
      выборочный ответ, а привычка — привязка тоже снимается.

Снимается именно тег, не текст: содержание баббла в любом случае доходит.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Collection, Sequence

from efi.telegram.queue import OutboundBubble

logger = logging.getLogger(__name__)

#: Тег привязки в начале баббла: [reply:12345]. Пробелы вокруг и внутри
#: скобок модель ставит как попало, поэтому терпим их.
_REPLY_TAG_RE = re.compile(r"^\s*\[\s*reply\s*:\s*(-?\d+)\s*\]\s*", re.IGNORECASE)


def parse_reply_tags(chunks: Sequence[str], *, incoming_message_ids: Collection[int]) -> list[OutboundBubble]:
    """
    Превращает куски ответа в бабблы, вынимая из них теги привязки.

    `incoming_message_ids` — id сообщений ТЕКУЩЕЙ входящей пачки; всё
    остальное считается выдумкой модели. Пустая коллекция означает, что
    привязываться не к чему (проактивный пинг, ночная задача), и все теги
    снимаются.
    """
    parsed = [_split_tag(chunk) for chunk in chunks]
    bubbles = [
        OutboundBubble(text=text, reply_to_message_id=_validate_target(target, incoming_message_ids))
        for target, text in parsed
        if text
    ]
    return _drop_redundant_replies(bubbles, incoming_message_ids)


def _split_tag(chunk: str) -> tuple[int | None, str]:
    """Отделяет тег привязки от текста баббла. Тега нет — target=None, текст как есть."""
    match = _REPLY_TAG_RE.match(chunk)
    if match is None:
        return None, chunk.strip()
    return int(match.group(1)), chunk[match.end() :].strip()


def _validate_target(target: int | None, incoming_message_ids: Collection[int]) -> int | None:
    if target is None:
        return None
    if target not in incoming_message_ids:
        logger.info("reply_selector: dropping [reply:%s] — no such message in the current batch", target)
        return None
    return target


def _drop_redundant_replies(
    bubbles: list[OutboundBubble], incoming_message_ids: Collection[int]
) -> list[OutboundBubble]:
    """
    Снимает привязку там, где она ничего не сообщает: пачка из одного
    сообщения и «reply на всё подряд» — см. докстринг модуля.
    """
    targets = {bubble.reply_to_message_id for bubble in bubbles if bubble.reply_to_message_id is not None}
    if not targets:
        return bubbles

    single_incoming = len(set(incoming_message_ids)) <= 1
    every_bubble_tagged = len(targets) == 1 and all(bubble.reply_to_message_id is not None for bubble in bubbles)
    if not (single_incoming or every_bubble_tagged):
        return bubbles

    logger.debug(
        "reply_selector: stripping redundant reply tags (single_incoming=%s, every_bubble_tagged=%s)",
        single_incoming, every_bubble_tagged,
    )
    return [OutboundBubble(text=bubble.text) for bubble in bubbles]


__all__ = ["parse_reply_tags"]
