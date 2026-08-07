"""
efi/tools/telegram_actions/send_message.py

Инструмент отправки сообщения в Telegram-чат — то, чем модель фактически
"говорит" с собеседником.

Гуманизация разделена по двум слоям с чёткой границей ответственности:
    - ЗДЕСЬ (политика): проверка на самоповтор через AntiRepeatTracker. Если
      кандидат слишком похож на недавние ответы в этом же чате, отправка
      блокируется, а модели возвращается текст ошибки с просьбой
      перефразировать — мы всё ещё внутри tool-calling цикла Worker'а
      (efi/notifications/worker.py), и у модели есть возможность попробовать
      снова, а не молча получить забракованное сообщение.
    - efi.telegram.client.TelegramClientWrapper (механика): typing-индикатор,
      задержка по WPM, редкие опечатки — то, что относится к ФАКТИЧЕСКОЙ
      отправке, а не к решению "стоит ли отправлять именно этот текст".

Дополнительно уведомляет SilenceMonitor об исходящей активности — иначе
собственные сообщения Эфи не засчитывались бы как "активность" в чате, и
монитор тишины мог бы запинговать чат сразу после того, как она сама в нём написала.

`context.extra["llm_generation_time"]` (если есть — кладёт туда
efi.notifications.worker.Worker._run_with_tool_calls) прокидывается в
send_message как есть: сколько реально заняла генерация ответа LLM до этого
момента, чтобы TelegramClientWrapper мог зачесть это время как "печать"
первого баббла (efi/humanizer/message_splitting.py::first_chunk_typing_delay)
вместо того, чтобы наслаивать ещё одну искусственную паузу поверх уже
прошедшего ожидания.

Каждый успешно отправленный текст ЕЩЁ И накапливается в
`context.extra["sent_texts"]` — это единственное место, где реально видно,
что модель сказала собеседнику. Личность обязана вызывать этот инструмент
как ПОСЛЕДНЕЕ действие хода (см. personality.md), поэтому финальный ответ
LLM в цикле tool-calling (после TOOL-результата этого вызова) часто пустой
или служебный ("готово") — если сохранять в историю именно его (как было
раньше), персистентная память вообще не видела бы реального текста ответа
Эфи. efi.notifications.worker.Worker читает `sent_texts` после цикла
tool-calling и сохраняет ИХ, а не последнее сырое сообщение модели — см.
Worker._handle.
"""

from __future__ import annotations

import logging
from typing import Any, Protocol

from efi.humanizer.anti_repeat import AntiRepeatTracker
from efi.tools.base import Tool, ToolContext

logger = logging.getLogger(__name__)


class MessageSender(Protocol):
    """Абстракция отправки сообщения. Конкретная реализация — efi.telegram.client.TelegramClientWrapper."""

    async def send_message(
        self,
        chat_id: int,
        text: str,
        *,
        reply_to_message_id: int | None = None,
        llm_generation_time: float | None = None,
    ) -> None: ...


class ActivityRecorder(Protocol):
    """Абстракция учёта активности чата. Конкретная реализация — efi.behavior.silence_monitor.SilenceMonitor."""

    def record_activity(self, chat_id: int) -> None: ...


class SendMessageTool(Tool):
    """Отправляет текстовое сообщение в чат, из которого пришло текущее уведомление."""

    name = "send_telegram_message"
    description = (
        "Отправляет текстовое сообщение в текущий чат. Это единственный способ, "
        "которым твои слова доходят до собеседника — простой текстовый ответ "
        "без вызова этого инструмента никуда не отправляется."
    )
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Текст сообщения"},
            "reply_to_current": {
                "type": "boolean",
                "description": (
                    "Если true — отправит явным Reply на то сообщение (или последнее из пачки, "
                    "если собеседник прислал несколько подряд), которое сейчас вызвало твой ответ. "
                    "Используй, когда важно явно показать, на что именно ты отвечаешь — например, "
                    "в оживлённой группе, где без этого не всегда очевидно, кому адресована реплика. "
                    "Для обычного ответа в личке это не обязательно."
                ),
            },
        },
        "required": ["text"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        sender: MessageSender,
        *,
        anti_repeat: AntiRepeatTracker | None = None,
        activity_recorder: ActivityRecorder | None = None,
    ) -> None:
        self._sender = sender
        self._anti_repeat = anti_repeat
        self._activity_recorder = activity_recorder

    def is_available(self, context: ToolContext) -> bool:
        # Отправка сообщения осмысленна только тогда, когда уведомление вообще
        # привязано к чату — NIGHTLY_TASK, например, может быть глобальной задачей.
        return context.chat_id is not None

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        text = str(arguments.get("text", "")).strip()
        if not text:
            return "error: text must not be empty"
        if context.chat_id is None:
            return "error: no chat_id in the current context, nowhere to send the message"

        if self._anti_repeat is not None and await self._anti_repeat.is_repetitive(context.chat_id, text):
            logger.info("send_message: blocked repetitive candidate for chat_id=%s", context.chat_id)
            return (
                "error: this is too similar to what you already said recently in this chat — "
                "try rephrasing or saying something genuinely different"
            )

        reply_to_message_id = self._resolve_reply_target(arguments, context)
        llm_generation_time = context.extra.get("llm_generation_time")
        await self._sender.send_message(
            context.chat_id,
            text,
            reply_to_message_id=reply_to_message_id,
            llm_generation_time=llm_generation_time,
        )

        if self._anti_repeat is not None:
            self._anti_repeat.record(context.chat_id, text)
        if self._activity_recorder is not None:
            self._activity_recorder.record_activity(context.chat_id)
        context.extra.setdefault("sent_texts", []).append(text)

        logger.info("send_message: sent %d chars to chat_id=%s", len(text), context.chat_id)
        return "Message sent successfully. Warning: you have sent a message. Consider not spamming with repeated calls."

    def _resolve_reply_target(self, arguments: dict[str, Any], context: ToolContext) -> int | None:
        """
        Reply возможен только на сообщение(я), которые реально вызвали текущий
        Notification (их telegram_message_id кладёт efi.telegram.handlers в
        payload) — модель не может сослаться на произвольное сообщение из
        более старой истории, потому что history/Session не хранят исходные
        Telegram message_id (см. известное ограничение в резюме этого шага).
        Если пришла пачка (дебаунс сгруппировал несколько сообщений) — Reply
        ставится на самое последнее из них.
        """
        if not bool(arguments.get("reply_to_current", False)):
            return None
        message_ids = context.notification.payload.get("telegram_message_ids")
        if not message_ids:
            return None
        return int(message_ids[-1])


__all__ = ["MessageSender", "ActivityRecorder", "SendMessageTool"]
