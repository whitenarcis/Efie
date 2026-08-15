"""
efi/notifications/schemas.py

Единая модель события для NotificationManager. Все проактивные и реактивные
триггеры Эфи — входящее сообщение, спонтанный пинг, ночная задача, follow-up,
пинг после тишины — становятся Notification и идут через одну и ту же
очередь и одних и тех же воркеров (efi/notifications/worker.py), которые
всегда собирают полный контекст личности перед обращением к LLM.

Это прямое воплощение принципа, уже проверенного на текущей Эфи и заложенного
в референс: все проактивные выводы должны проходить через полную модель
личности, а не через отдельный облегчённый путь — единая точка входа не
оставляет для этого лазейки.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class NotificationType(StrEnum):
    """Тип события. Определяет, какой промпт-шаблон/логику возьмёт Worker при обработке."""

    #: Входящее сообщение от пользователя в чате — основной реактивный путь.
    USER_MESSAGE = "user_message"
    #: Спонтанный пинг — Эфи сама решает написать первой (аналог try_spontaneous_ping).
    SPONTANEOUS_PING = "spontaneous_ping"
    #: Плановая ночная/фоновая задача (консолидация дневника, memoirs и т.п.).
    NIGHTLY_TASK = "nightly_task"
    #: Отложенное продолжение темы, на которую Эфи обещала вернуться (resume-callback).
    FOLLOW_UP = "follow_up"
    #: Реакция на длительное молчание в чате (аналог silence_monitor_lifecycle).
    SILENCE_PING = "silence_ping"
    #: Публичный комментарий под постом в канале сообщества (efi.telegram.comments).
    PUBLIC_COMMENT = "public_comment"
    #: Включение в чужую ветку обсуждения (RandomCommentEngager).
    THREAD_REPLY = "thread_reply"
    #: Ход работы над проектом: короткая реплика о процессе или готовый
    #: релиз со ссылкой (efi.dev.reporter.DevReporter). Не «инициатива из
    #: воздуха», как спонтанный пинг: повод конкретный — она правда только
    #: что это сделала.
    DEV_UPDATE = "dev_update"


class Notification(BaseModel):
    """
    Единица работы в очереди NotificationManager.

    `priority`: чем МЕНЬШЕ значение, тем раньше событие будет обработано —
    это соответствует нативному порядку asyncio.PriorityQueue (наименьший
    элемент забирается первым), поэтому значения не нужно инвертировать при
    постановке в очередь. Рекомендуемая шкала: 0 (real-time, например
    USER_MESSAGE) .. 9 (фоновое, например NIGHTLY_TASK); по умолчанию 5.

    `chat_id`: чат, к которому относится событие. None — для событий без
    привязки к конкретному чату (например, NIGHTLY_TASK). См. `routing_key`.

    `message`: короткое естественно-языковое описание события для LLM —
    например, "Тебе написал Рома: {текст}" или "Прошло 6 часов тишины в этом
    чате, возможно, стоит написать первой самой".

    `payload`: произвольные структурированные данные, нужные Worker'у и
    инструментам для обработки конкретно этого события (id входящего
    сообщения в Telegram, тема follow-up'а и т.п.) — намеренно свободная
    форма, типизируется предметно тем кодом, который создаёт Notification
    конкретного типа.
    """

    id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    type: NotificationType
    priority: int = Field(default=5, ge=0, le=9)
    chat_id: int | None = None
    message: str
    payload: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    #: Номер попытки доставки, считая с нуля. Проактивное событие — это
    #: намерение Эфи что-то сказать; если LLM не ответил (таймаут на
    #: бесплатном тире — обычное дело), намерение переставляется в очередь
    #: заново, а не теряется. См. NotificationManager.retry_later.
    #: `created_at` при этом НЕ обновляется: повод возник тогда, когда возник,
    #: и по нему решается, не протух ли он.
    attempt: int = Field(default=0, ge=0)

    @property
    def routing_key(self) -> str:
        """
        Ключ маршрутизации к воркеру (см. NotificationManager.worker_index_for).
        События одного чата всегда получают один и тот же ключ и,
        следовательно, один и тот же воркер — гарантия последовательной
        обработки без гонок за контекст. У событий без chat_id — синтетический
        ключ по типу события, чтобы разные фоновые задачи не толпились на
        одном воркере без необходимости.
        """
        if self.chat_id is not None:
            return f"chat:{self.chat_id}"
        return f"type:{self.type.value}"


__all__ = ["NotificationType", "Notification"]
