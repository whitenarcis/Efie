"""
efi/notifications/worker.py

Worker — обрабатывает уведомления, закреплённые за ним NotificationManager'ом,
строго последовательно (одно за другим, в порядке приоритета внутри своей
подочереди). Это и есть гарантия отсутствия гонок за контекст одного чата:
пока Worker обрабатывает Notification для chat_id=X, следующее уведомление
для того же chat_id физически лежит в ЕГО ЖЕ подочереди и просто ждёт своей
очереди — никакой другой Worker его в это время не подхватит.

На каждое уведомление Worker (порядок ВАЖЕН — см. efi/behavior/busy_engine.py):
    1. Считает и выжидает `ignore_delay` (BusyEngine) — симуляция занятости.
       До этого момента Worker НЕ делает ни одного обращения к Telegram, ни
       видимого (typing/read), ни любого другого — значит, Telegram не
       покажет клиента "в сети", а сообщение остаётся "не прочитано".
    2. Для USER_MESSAGE — заходит в чат (mark_as_read): именно в ЭТОТ момент
       появляется статус "прочитано" и, как следствие, онлайн-присутствие —
       не раньше.
    3. Собирает Session (история + текст события, прогнанный через
       security.sanitize) и просит системный промпт у SystemPromptBuilder
       (личность, время, working memory, RAG, ограничения безопасности,
       мета-тема/эмпатия — Worker в это не вникает, см. efi/prompts/builder.py).
    4. Прогоняет цикл tool-calling через LLMRouter (см. _run_with_tool_calls),
       транслируя TYPING все время ожидания ответа модели (см. _typing_pulse) —
       живой статус "печатает" вместо тишины между "прочитано" и первым
       сообщением. Если запрос падает как "слишком большой" (413 — реальный
       случай на узких TPM-лимитах бесплатных тиров), один раз пробует заново
       с урезанной историей вместо того, чтобы сразу сдаваться (см.
       _chat_with_size_retry).
    5. Сохраняет финальное сообщение ассистента в историю диалога.

Если ВСЕ кандидаты LLMRouter (включая retry) всё равно отказали — раньше это
приводило к полной тишине: ошибка просто логировалась, а собеседник не
получал вообще ничего и не понимал, что случилось. Теперь, если передан
`telegram`, Worker отправляет короткое нейтральное сообщение о сбое
напрямую (в обход LLM, который как раз и недоступен).

ВАЖНО про сохранение шага 5: `response.message` — это последний ответ LLM в
цикле tool-calling, а не обязательно то, что реально увидел собеседник.
Личность обязана вызывать send_telegram_message как ПОСЛЕДНЕЕ действие хода
(см. personality.md) — значит, реальный текст ответа уходит в аргументе
ЭТОГО вызова, а раунд после него (после TOOL-результата "Message sent
successfully") часто возвращает пустой/служебный `content`, потому что
модель уже "сказала своё" через инструмент. Если бы Worker сохранял в
историю именно `response.message.content` не глядя, персистентная память
хранила бы не то, что Эфи реально сказала, а обрывок технического
финального хода — со временем разговор для самой Эфи выглядел бы так,
будто она в основном отвечала пустотой. Поэтому `_handle` собирает
`tool_context.extra["sent_texts"]` (см. efi.tools.telegram_actions.
send_message.SendMessageTool) — тексты всех реально отправленных сообщений
за этот ход — и сохраняет ИХ как содержимое ASSISTANT-сообщения истории,
откатываясь на `response.message` только если инструмент отправки вообще
не вызывался (например, будущий тип уведомления без обязательной отправки).

С Шага 6 Worker больше не знает про RAGMemory/WorkingMemory напрямую — эта
логика переехала в SystemPromptBuilder (efi/prompts/builder.py). Worker
остаётся ответственным только за оркестрацию: занятость, историю,
санитайзинг входа и цикл tool-calling — и не заботится о содержании промпта.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Protocol

from efi.behavior.busy_engine import BusyEngine
from efi.config.schema import TaskRole
from efi.llm.errors import LLMError
from efi.llm.router import LLMRouter
from efi.llm.schemas import LLMParams, Message, Response, Role, Session
from efi.notifications.manager import NotificationManager
from efi.notifications.schemas import Notification, NotificationType
from efi.security.sanitize import sanitize_text
from efi.tools.base import ToolContext
from efi.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_PAYLOAD_TOO_LARGE_STATUS = 413
_TRIMMED_HISTORY_KEEP_LAST = 6
_FAILURE_NOTICE_TEXT = "уф, что-то у меня заглючило только что — попробуй написать ещё раз через минуту"

#: Пауза между повторными TYPING-пингами во время ожидания LLM — короче TTL
#: статуса "печатает" в Telegram (~5-6с, см. efi/telegram/typing_tracker.py),
#: чтобы статус не успевал погаснуть между пингами.
_TYPING_PULSE_INTERVAL_SECONDS = 4.0


class HistoryRepository(Protocol):
    """Абстракция истории диалога. Конкретная реализация — efi.db.history_repository.SqliteHistoryRepository."""

    async def get_recent(self, chat_id: int, limit: int = 20) -> Session: ...

    async def append(self, chat_id: int, message: Message) -> None: ...


class SystemPromptBuilder(Protocol):
    """Абстракция сборки системного промпта. Конкретная реализация — efi.prompts.builder.EfiSystemPromptBuilder."""

    async def build(self, notification: Notification, history: Session) -> str: ...


class TelegramNotifier(Protocol):
    """
    Всё, что Worker'у нужно от телеграм-слоя. Конкретная реализация —
    efi.telegram.client.TelegramClientWrapper (уже реализует все три метода).
    Объединено в один Protocol, а не три отдельных: у Worker'а нет сценария,
    где были бы доступны одни методы этой связки без других — это одна и та
    же "видимая" поверхность одного и того же Telegram-клиента.
    """

    async def send_message(
        self,
        chat_id: int,
        text: str,
        *,
        reply_to_message_id: int | None = None,
        llm_generation_time: float | None = None,
    ) -> None: ...

    async def mark_as_read(self, chat_id: int) -> None: ...

    async def send_typing_action(self, chat_id: int) -> None: ...


class Worker:
    """Обрабатывает уведомления из своей подочереди NotificationManager'а."""

    def __init__(
        self,
        worker_index: int,
        manager: NotificationManager,
        *,
        llm_router: LLMRouter,
        tool_registry: ToolRegistry,
        history: HistoryRepository,
        system_prompt_builder: SystemPromptBuilder,
        busy_engine: BusyEngine,
        main_role: TaskRole = TaskRole.MAIN,
        max_tool_call_rounds: int = 8,
        history_limit: int = 20,
        telegram: TelegramNotifier | None = None,
    ) -> None:
        self._worker_index = worker_index
        self._manager = manager
        self._llm_router = llm_router
        self._tool_registry = tool_registry
        self._history = history
        self._system_prompt_builder = system_prompt_builder
        self._busy_engine = busy_engine
        self._main_role = main_role
        self._max_tool_call_rounds = max_tool_call_rounds
        self._history_limit = history_limit
        self._telegram = telegram

    async def run(self) -> None:
        """
        Основной цикл: забирает уведомления из своей подочереди и
        обрабатывает их одно за другим. Ошибка обработки ОДНОГО уведомления
        не должна останавливать Worker целиком — она логируется, и цикл идёт
        дальше. Останавливается только по внешней отмене задачи (CancelledError).
        """
        logger.info("worker[%d]: started", self._worker_index)
        try:
            while True:
                notification = await self._manager.get(self._worker_index)
                try:
                    await self._handle(notification)
                except Exception:
                    logger.exception(
                        "worker[%d]: unhandled error while processing notification %s (%s)",
                        self._worker_index, notification.id, notification.type.value,
                    )
                finally:
                    self._manager.task_done(self._worker_index)
        except asyncio.CancelledError:
            logger.info("worker[%d]: stopped", self._worker_index)
            raise

    async def _handle(self, notification: Notification) -> None:
        await self._apply_busy_delay(notification)

        history = (
            await self._history.get_recent(notification.chat_id, limit=self._history_limit)
            if notification.chat_id is not None
            else Session()
        )
        session = _finalize_session(history, notification)

        if notification.type is NotificationType.USER_MESSAGE and notification.chat_id is not None:
            # Реплика собеседника — единственное, что делает историю ИСТОРИЕЙ
            # ДИАЛОГА, а не монологом Эфи с самой собой: раньше сюда попадал
            # только response.message (см. ниже), а сообщение пользователя
            # так и оставалось только в одноразовой Session ЭТОГО вызова и
            # никогда не сохранялось в БД. get_recent() в следующий раз
            # отдавал бы историю из одних только прошлых ответов Эфи — модель
            # буквально продолжала бы саму себя, что выглядит как спор с
            # призраком. Пишем ДО обращения к LLM: реплика человека должна
            # остаться в истории, даже если сам запрос к LLM ниже провалится.
            await self._history.append(notification.chat_id, session.messages[-1])

        if (
            notification.type is NotificationType.USER_MESSAGE
            and notification.chat_id is not None
            and self._telegram is not None
        ):
            # "Заход в чат" — именно тут, а не раньше: до этого момента
            # Worker не совершил ни одного видимого телеграм-действия (см.
            # докстринг модуля и efi/behavior/busy_engine.py).
            await self._telegram.mark_as_read(notification.chat_id)

        system_prompt = await self._system_prompt_builder.build(notification, history)

        tool_context = ToolContext(notification=notification)
        params = LLMParams(
            model="",  # роутер сам подставит модель кандидата по main_role — см. efi.llm.router.LLMRouter
            system_prompt=system_prompt,
            tools=self._tool_registry.as_openai_tools(tool_context),
        )

        try:
            response = await self._run_with_typing_pulse(notification.chat_id, params, session, tool_context)
        except LLMError:
            await self._notify_failure(notification)
            raise  # даём run() залогировать полный трейсбек, как и раньше

        if notification.chat_id is not None:
            await self._history.append(notification.chat_id, _message_to_persist(response, tool_context))

    async def _apply_busy_delay(self, notification: Notification) -> None:
        delay = await self._busy_engine.compute_ignore_delay(notification.chat_id)
        if delay <= 0:
            return
        logger.debug(
            "worker[%d]: ignoring notification %s for %.1fs (busy simulation)",
            self._worker_index, notification.id, delay,
        )
        await asyncio.sleep(delay)

    async def _notify_failure(self, notification: Notification) -> None:
        """
        Короткое нейтральное уведомление о сбое напрямую — иначе собеседник
        ничего не получает и не понимает, в чём дело.

        Персистим его в историю точно так же, как обычный ответ (см.
        _message_to_persist): иначе следующий запрос в этом чате не будет
        знать, что Эфи вообще что-то говорила про сбой — с точки зрения
        истории собеседник получил бы сообщение "из ниоткуда", а сама Эфи
        в следующий раз не будет помнить, что уже извинялась.
        """
        if self._telegram is None or notification.chat_id is None:
            return
        try:
            await self._telegram.send_message(notification.chat_id, _FAILURE_NOTICE_TEXT)
        except Exception:
            logger.exception(
                "worker[%d]: failed to send fallback failure notice to chat_id=%s",
                self._worker_index, notification.chat_id,
            )
            return
        await self._history.append(
            notification.chat_id, Message(role=Role.ASSISTANT, content=_FAILURE_NOTICE_TEXT)
        )

    async def _run_with_typing_pulse(
        self, chat_id: int | None, params: LLMParams, session: Session, tool_context: ToolContext
    ) -> Response:
        """
        Оборачивает цикл tool-calling фоновым "пульсом" TYPING, пока Worker
        ждёт LLM (см. _typing_pulse) — без этого собеседник видел бы просто
        тишину между "прочитано" и первым сообщением, что менее естественно,
        чем живой статус "печатает".
        """
        if chat_id is None or self._telegram is None:
            return await self._run_with_tool_calls(params, session, tool_context)

        pulse_task = asyncio.create_task(self._typing_pulse(chat_id))
        try:
            return await self._run_with_tool_calls(params, session, tool_context)
        finally:
            pulse_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await pulse_task

    async def _typing_pulse(self, chat_id: int) -> None:
        assert self._telegram is not None  # проверено вызывающей стороной (_run_with_typing_pulse)
        try:
            while True:
                await self._telegram.send_typing_action(chat_id)
                await asyncio.sleep(_TYPING_PULSE_INTERVAL_SECONDS)
        except asyncio.CancelledError:
            raise

    async def _run_with_tool_calls(self, params: LLMParams, session: Session, tool_context: ToolContext) -> Response:
        """
        Цикл tool-calling: запросить ответ у LLM -> если модель вызвала
        инструменты, выполнить их и вернуть результаты в модель как
        TOOL-сообщения -> повторить, пока модель не даст финальный ответ без
        tool_calls. `max_tool_call_rounds` — защита от зацикливания (модель
        может застрять, бесконечно вызывая инструменты).

        Перед исполнением каждого раунда tool-вызовов кладёт накопленное
        время генерации в `tool_context.extra["llm_generation_time"]` — это
        читает efi.tools.telegram_actions.send_message.SendMessageTool, чтобы
        зачесть уже прошедшее ожидание LLM как время печати первого баббла
        (см. efi/humanizer/message_splitting.py::first_chunk_typing_delay).
        `tool_context.extra` — обычный dict, мутация на месте безопасна:
        ToolContext заморожен только на уровне переприсваивания полей, не
        содержимого изменяемых полей (см. efi/tools/base.py).
        """
        start_time = asyncio.get_running_loop().time()
        response: Response | None = None
        for _round_number in range(self._max_tool_call_rounds):
            response = await self._chat_with_size_retry(params, session)
            message = response.message
            session.append(message)

            if not message.has_tool_calls:
                return response

            tool_context.extra["llm_generation_time"] = asyncio.get_running_loop().time() - start_time
            for tool_call in message.tool_calls:
                result_text = await self._tool_registry.execute(tool_call, tool_context)
                session.append(Message(role=Role.TOOL, content=result_text, tool_call_id=tool_call.id))

        logger.warning(
            "worker[%d]: reached max_tool_call_rounds=%d while processing %s, returning last response as-is",
            self._worker_index, self._max_tool_call_rounds, tool_context.notification.id,
        )
        assert response is not None  # цикл выполняется минимум один раз, т.к. max_tool_call_rounds задаётся >= 1
        return response

    async def _chat_with_size_retry(self, params: LLMParams, session: Session) -> Response:
        """
        Один повторный запрос: если он падает как "слишком большой" (413 —
        реальный случай на узких TPM-лимитах бесплатных тиров, а не только
        теоретический), пробуем ещё раз с урезанной историей вместо того,
        чтобы сразу сдаваться. Мутирует `session.messages` НА МЕСТЕ при
        успешном урезании — дальнейшие раунды tool-calling в этом же ходу
        тоже должны видеть уже урезанную историю, а не полную при следующем вызове.
        """
        try:
            return await self._llm_router.chat(self._main_role, params, session)
        except LLMError as exc:
            if exc.status_code != _PAYLOAD_TOO_LARGE_STATUS:
                raise
            if len(session.messages) <= _TRIMMED_HISTORY_KEEP_LAST:
                raise  # урезать уже нечего — тот же результат, повторять бессмысленно

            logger.warning(
                "worker[%d]: request failed as too large (%s), retrying with trimmed session (%d -> %d messages)",
                self._worker_index, exc, len(session.messages), _TRIMMED_HISTORY_KEEP_LAST,
            )
            session.messages[:] = session.messages[-_TRIMMED_HISTORY_KEEP_LAST:]
            return await self._llm_router.chat(self._main_role, params, session)


def _message_to_persist(response: Response, tool_context: ToolContext) -> Message:
    """
    Что реально сохранить в персистентную историю как реплику Эфи — см.
    докстринг модуля про `response.message` vs `sent_texts`. Если за этот ход
    хоть раз успешно сработал send_telegram_message, история должна помнить
    ИМЕННО отправленный текст (в порядке отправки, если бабблов было
    несколько), а не последний служебный ход LLM после этого. Если инструмент
    ни разу не вызывался (например, модель ответила текстом без вызова —
    такой ответ до собеседника не долетел, но сохранить хоть что-то лучше,
    чем ничего) — откатываемся на response.message как раньше.
    """
    sent_texts = tool_context.extra.get("sent_texts")
    if not sent_texts:
        return response.message
    return Message(role=Role.ASSISTANT, content="\n".join(sent_texts))


def _finalize_session(history: Session, notification: Notification) -> Session:
    """
    История чата + текущее сообщение события. `notification.message` —
    внешний ввод (то, что написал собеседник, или сформулированный текст
    триггера) и прогоняется через sanitize_text ПЕРЕД тем, как попасть в
    Session — единственная точка, где сырой внешний текст превращается в
    USER-сообщение LLM.
    """
    session = history.model_copy(deep=True)
    session.append(Message(role=Role.USER, content=sanitize_text(notification.message)))
    return session


__all__ = ["Worker", "HistoryRepository", "SystemPromptBuilder", "TelegramNotifier"]
