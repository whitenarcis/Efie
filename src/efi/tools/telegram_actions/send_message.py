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
import re
from collections.abc import Callable, Collection
from typing import Any, Protocol

from efi.humanizer.anti_repeat import AntiRepeatTracker
from efi.notifications.schemas import NotificationType
from efi.telegram.client import UnknownChatError
from efi.tools.base import Tool, ToolContext

logger = logging.getLogger(__name__)

#: Сколько реплик подряд допустимо, когда пишешь первой. Две-три коротких —
#: это ровно то, как выглядит живое «привет / как ты». Больше — уже поток,
#: которым заваливают человека, ничего у тебя не спросившего.
_MAX_PROACTIVE_BUBBLES = 3

#: И потолок на весь инициативный ход. Человек, который пишет первым, не
#: присылает абзац: он бросает пару строчек и ждёт.
_MAX_PROACTIVE_CHARS = 220

#: Уведомления, где Эфи пишет первой. Здесь ответ обязан быть одной репликой:
#: серия бабблов от того, кому ещё не ответили, читается как нетерпение.
#: FOLLOW_UP входит: напоминание — тоже сообщение без реплики собеседника.
#: Ходы, где она пишет первой. Для них ход ограничивается по длине и числу
#: реплик (см. _collapse_bubbles_if_proactive) — имя оставлено историческим,
#: чтобы не расходиться с логами и конфигом.
_SINGLE_BUBBLE_TYPES = frozenset(
    {
        NotificationType.SPONTANEOUS_PING,
        NotificationType.SILENCE_PING,
        NotificationType.FOLLOW_UP,
    }
)

#: Тот же разделитель, что размечает модель (см. efi/humanizer/message_splitting.py).
_BUBBLE_DELIMITER_RE = re.compile(r"\s*///\s*")


class MessageSender(Protocol):
    """Абстракция отправки сообщения. Конкретная реализация — efi.telegram.client.TelegramClientWrapper."""

    async def send_message(
        self,
        chat_id: int,
        text: str,
        *,
        reply_to_message_id: int | None = None,
        llm_generation_time: float | None = None,
        incoming_message_ids: Collection[int] = (),
        on_bubble_sent: Callable[[str], None] | None = None,
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

        text = _collapse_bubbles_if_proactive(text, context)

        if self._anti_repeat is not None and await self._anti_repeat.is_repetitive(context.chat_id, text):
            logger.info("send_message: blocked repetitive candidate for chat_id=%s", context.chat_id)
            return (
                "error: this is too similar to what you already said recently in this chat — "
                "try rephrasing or saying something genuinely different"
            )

        reply_to_message_id = self._resolve_reply_target(arguments, context)
        llm_generation_time = context.extra.get("llm_generation_time")
        # sent_texts наполняется ПОБАББЛЬНО, а не одной строкой после успеха
        # всей серии. Ход может быть снят как устаревший посреди отправки
        # (efi/telegram/chat_orchestrator.py), и тогда история обязана знать
        # ровно то, что собеседник успел прочитать: раньше при отмене на
        # середине она не получала ничего, хотя половина ответа уже висела
        # в чате, и следующая генерация повторяла сказанное.
        delivered: list[str] = context.extra.setdefault("sent_texts", [])
        try:
            await self._sender.send_message(
                context.chat_id,
                text,
                reply_to_message_id=reply_to_message_id,
                llm_generation_time=llm_generation_time,
                incoming_message_ids=self._incoming_message_ids(context),
                on_bubble_sent=delivered.append,
            )
        except UnknownChatError:
            # Штатная ситуация, а не сбой: этот аккаунт не видит такой чат
            # (никогда в нём не был, его удалили, Эфи оттуда вышли). Раньше
            # сюда прилетал сырой KeyError из недр Pyrogram и ToolRegistry
            # печатал полный трейсбек как ERROR на каждой попытке пинга.
            logger.warning("send_message: chat_id=%s is unreachable, dropping the message", context.chat_id)
            return "error: this chat is not reachable — do not retry sending here"

        if self._anti_repeat is not None:
            self._anti_repeat.record(context.chat_id, text)
        if self._activity_recorder is not None:
            self._activity_recorder.record_activity(context.chat_id)

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

        `force_reply` в payload уведомления делает реплай обязательным,
        независимо от того, попросила ли о нём модель: под постом в канале
        комментарий — это ИМЕННО реплай на экземпляр поста в группе
        обсуждения, и без него сообщение уходит отдельной репликой в группу,
        никак не привязанной к посту (см. efi/telegram/comments.py).
        """
        forced = bool(context.notification.payload.get("force_reply"))
        if not forced and not bool(arguments.get("reply_to_current", False)):
            return None
        message_ids = self._incoming_message_ids(context)
        return message_ids[-1] if message_ids else None

    @staticmethod
    def _incoming_message_ids(context: ToolContext) -> list[int]:
        """
        id сообщений текущей входящей пачки — область допустимых целей для
        reply. Модель может привязать баббл к КОНКРЕТНОЙ реплике из пачки
        тегом `[reply:id]` (см. efi/humanizer/reply_selector.py); всё, чего
        в этом списке нет, снимается как выдумка — id старых сообщений ей
        нигде не показываются, сослаться на них она не может.
        """
        raw = context.notification.payload.get("telegram_message_ids") or []
        return [int(item) for item in raw]


def _collapse_bubbles_if_proactive(text: str, context: ToolContext) -> str:
    """
    Инициативное сообщение — коротко, но живой речью, а не одной фразой.

    История этого места стоит того, чтобы её знать. Сначала здесь не было
    ничего, и инициатива выглядела так:

        эй
        ты там ещё не утонул в своём коде?

    Тогда бабблы стали СКЛЕИВАТЬСЯ в одну реплику — и лечение оказалось хуже
    болезни. Проблема была не в том, что реплик две, а в том, что сказать
    было нечего: первая только объявляла, что сейчас будет сообщение. После
    склейки получилось другое, но такое же ненастоящее: одно длинное
    предложение, которым в мессенджере не пишет никто.

    Правильное правило — не «одна реплика», а «мало и коротко»: пара коротких
    строчек, как их и пишет человек с телефона. Лишние бабблы отбрасываются
    (а не склеиваются в ком), длина хода ограничена. Содержание при этом
    держит повод (efi/behavior/ping_reason.py) — без него никакие ограничения
    формы не спасали.

    На ОТВЕТЫ правило не распространяется — там серия коротких реплик как раз
    и есть живая речь (см. efi/humanizer/message_splitting.py).
    """
    if context.notification.type not in _SINGLE_BUBBLE_TYPES:
        return text

    parts = [part.strip() for part in _BUBBLE_DELIMITER_RE.split(text) if part.strip()]
    if not parts:
        return text.strip()

    kept: list[str] = []
    used = 0
    for part in parts[:_MAX_PROACTIVE_BUBBLES]:
        if kept and used + len(part) > _MAX_PROACTIVE_CHARS:
            break
        kept.append(part)
        used += len(part)

    if len(kept) < len(parts):
        logger.info(
            "send_message: инициативный ход укорочен до %d реплик(и) для chat_id=%s",
            len(kept), context.chat_id,
        )
    return " /// ".join(kept)


__all__ = ["MessageSender", "ActivityRecorder", "SendMessageTool"]
