"""
efi/telegram/queue.py

Очередь исходящих бабблов одного хода: что Эфи собирается сказать, что из
этого уже доставлено и что было сброшено, когда ход прервали.

Зачем отдельная сущность, а не просто цикл по списку кусков. Серия бабблов
растянута во времени: между ними идут паузы по WPM, и на длинном «потоке
мыслей» из 8 сообщений ход живёт десятки секунд. Всё это время он может быть
снят как устаревший (efi/telegram/chat_orchestrator.py), и тогда важны две
вещи, которых у голого списка нет:

    1. ЧТО УЖЕ УШЛО. Доставленные бабблы отозвать нельзя — собеседник их
       прочитал. Значит, они обязаны попасть в историю как сказанное, иначе
       следующая генерация соберёт контекст без них и повторится. Раньше
       текст записывался в историю ОДНОЙ строкой после успешной отправки
       всей серии — при отмене на середине история не получала ничего, хотя
       половина ответа уже висела в чате.
    2. ЧТО НЕ УШЛО. Остаток сбрасывается молча, но он должен быть виден в
       логах: «прервали на 3-м баббле из 7» — это единственный способ понять
       постфактум, почему ответ выглядит оборванным.

Сама механика отправки (typing, опечатки, самокоррекция) живёт в
efi/telegram/client.py — здесь только состояние очереди и учёт.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class OutboundBubble:
    """
    Один баббл серии.

    `reply_to_message_id` проставляется на КОНКРЕТНЫЙ баббл, а не на всю
    серию: в пачке из нескольких входящих реплик Эфи может ответить общим
    текстом, а одну строчку прокомментировать явным reply (см.
    efi/humanizer/reply_selector.py).
    """

    text: str
    reply_to_message_id: int | None = None


class BubbleQueue:
    """
    Состояние отправки одной серии бабблов.

    Использование (см. efi.telegram.client.TelegramClientWrapper.send_message):

        queue = BubbleQueue(bubbles)
        while (bubble := queue.next_bubble()) is not None:
            ...  # typing, задержка, фактическая отправка
            queue.mark_delivered()

    При отмене посреди цикла очередь остаётся консистентной: `delivered`
    содержит ровно то, что реально ушло, `pending` — сброшенный остаток.
    """

    def __init__(self, bubbles: Sequence[OutboundBubble]) -> None:
        self._pending: deque[OutboundBubble] = deque(bubbles)
        self._delivered: list[OutboundBubble] = []
        self._total = len(bubbles)
        self._in_flight: OutboundBubble | None = None

    def __len__(self) -> int:
        return len(self._pending)

    @property
    def total(self) -> int:
        """Сколько бабблов было в серии изначально."""
        return self._total

    @property
    def delivered(self) -> list[OutboundBubble]:
        """Бабблы, реально дошедшие до собеседника, в порядке отправки."""
        return list(self._delivered)

    @property
    def delivered_texts(self) -> list[str]:
        """То же, но только текстами — в таком виде это уходит в историю диалога."""
        return [bubble.text for bubble in self._delivered]

    @property
    def pending(self) -> list[OutboundBubble]:
        """
        Ещё не отправленное. После отмены — ровно тот остаток, который был
        сброшен (баббл, отправка которого не успела подтвердиться, тоже
        считается неотправленным: подтверждает её только mark_delivered).
        """
        in_flight = [self._in_flight] if self._in_flight is not None else []
        return in_flight + list(self._pending)

    @property
    def is_first(self) -> bool:
        """
        Печатается ли сейчас первый баббл серии — только ему засчитывается
        время генерации LLM как уже прошедшая «печать»
        (efi/humanizer/message_splitting.py::first_chunk_typing_delay).
        """
        return not self._delivered

    def next_bubble(self) -> OutboundBubble | None:
        """Берёт следующий баббл в работу. None — серия закончилась."""
        if self._in_flight is not None:
            raise RuntimeError("предыдущий баббл ещё не подтверждён через mark_delivered()")
        if not self._pending:
            return None
        self._in_flight = self._pending.popleft()
        return self._in_flight

    def mark_delivered(self, *, text: str | None = None) -> None:
        """
        Подтверждает доставку взятого в работу баббла.

        `text` позволяет записать в доставленные то, что РЕАЛЬНО ушло, а не
        то, что собирались отправить: гуманизатор может подмешать в текст
        опечатку (efi/humanizer/typos.py), и в историю должна попасть
        отправленная версия.
        """
        if self._in_flight is None:
            raise RuntimeError("нет баббла в работе — mark_delivered() без next_bubble()")
        bubble = self._in_flight if text is None else OutboundBubble(text, self._in_flight.reply_to_message_id)
        self._delivered.append(bubble)
        self._in_flight = None

    def log_interruption(self, chat_id: int) -> None:
        """Единая формулировка для логов: серию сняли, часть уже у собеседника."""
        logger.info(
            "bubble_queue: interrupted in chat_id=%s after %d of %d bubbles, dropping %d",
            chat_id, len(self._delivered), self._total, len(self.pending),
        )


def bubbles_from_texts(texts: Iterable[str]) -> list[OutboundBubble]:
    """Простая серия без reply-привязок — путь для проактивных сообщений и служебных уведомлений."""
    return [OutboundBubble(text) for text in texts]


__all__ = ["BubbleQueue", "OutboundBubble", "bubbles_from_texts"]
