"""
efi/notifications/worker.py

Worker — обрабатывает уведомления, закреплённые за ним NotificationManager'ом,
строго последовательно (одно за другим, в порядке приоритета внутри своей
подочереди). Это и есть гарантия отсутствия гонок за контекст одного чата:
пока Worker обрабатывает Notification для chat_id=X, следующее уведомление
для того же chat_id физически лежит в ЕГО ЖЕ подочереди и просто ждёт своей
очереди — никакой другой Worker его в это время не подхватит.

На каждое уведомление Worker (порядок ВАЖЕН — см. efi/behavior/busy_engine.py):
    1. Спрашивает у BusyEngine решение (`decide`): сколько ждать и идёт ли
       уже активный разговор в этом чате.
       - Разговор УЖЕ идёт: сообщение отмечается прочитанным СРАЗУ, до
         всякой задержки. Эфи физически смотрит в этот чат прямо сейчас, и
         держать реплику непрочитанной было бы искусственным удержанием в
         непрочитанных, а не симуляцией занятости.
       - Разговора нет (первое сообщение после паузы): Worker выжидает
         `ignore_delay`, НЕ делая ни одного обращения к Telegram — значит,
         Telegram не покажет клиента "в сети", а сообщение остаётся
         непрочитанным, — и только потом заходит в чат (mark_as_read).
         Именно в ЭТОТ момент появляется "прочитано" и онлайн-присутствие.
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
       _chat_with_size_retry). Бюджет раундов зависит от типа уведомления:
       USER_MESSAGE получает полный `max_tool_call_rounds` (реальный диалог,
       может понадобиться несколько шагов — поиск, потом ответ), а
       проактивные уведомления (спонтанный пинг, follow-up, тишина, ночная
       задача) — заметно урезанный `proactive_max_tool_call_rounds`: без
       настоящей реплики собеседника несколько полных раундов подряд иначе
       выливаются в монолог с самой собой ("эй, ты там?" / "алло" / ...).
       Кроме бюджета раундов, цикл ЖЁСТКО останавливается сразу, как только
       send_telegram_message реально сработал хотя бы раз за ход (см.
       _run_with_tool_calls) — раньше это было только пожеланием в
       personality.md, и модель, уже отправив ответ, иногда продолжала
       генерировать ещё реплики как ни в чём не бывало ("чё, молчишь?" —
       через несколько раундов ПОСЛЕ уже доставленного ответа).
    5. Для USER_MESSAGE — проверяет (_ensure_reply_was_sent), что модель
       реально вызвала send_telegram_message хотя бы раз за ход. Ничто в
       контракте LLM это не гарантирует — модель может формально завершить
       ход текстом без единого tool_call, и тогда собеседник получил бы
       "прочитано" (шаг 2) и полную тишину после, БЕЗ единой ошибки в логах.
       Один явный раунд-напоминание, затем — нейтральное сообщение напрямую,
       если и это не помогло.
    6. Сохраняет финальное сообщение ассистента в историю диалога.

Если ход всё равно провалился — раньше это приводило к полной тишине: ошибка
просто логировалась, а собеседник не получал вообще ничего и не понимал, что
случилось. Теперь, если передан `telegram`, Worker отправляет короткое
нейтральное сообщение о сбое напрямую (в обход LLM, который как раз и мог
быть недоступен).

Важно, что это касается ЛЮБОГО сбоя, а не только LLMError: смысл уведомления
не в том, что подвёл именно LLM, а в том, что собеседник уже получил
"прочитано" (шаг 2) и обязан получить после него хоть что-то. Пока здесь
ловилась одна LLMError, всё остальное — баг в инструменте, ответ провайдера
без choices, сбой БД — давало ровно ту "прочитано и тишину", ради
предотвращения которой существует и `_ensure_reply_was_sent`. Отмена хода
(CancelledError) исключение из правила: ход сняли как устаревший, это не сбой,
и сообщение о глюке легло бы поверх ответа на новую реплику.

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
from efi.behavior.conversation_lifecycle import ConversationLifecycle
from efi.config.schema import TaskRole
from efi.llm.errors import LLMError
from efi.llm.router import LLMRouter
from efi.llm.schemas import LLMParams, Message, Response, Role, Session
from efi.memory.social_memory import SocialInteraction, SocialInteractionKind, SocialInteractionStore
from efi.notifications.manager import NotificationManager
from efi.notifications.schemas import Notification, NotificationType
from efi.security.sanitize import sanitize_text
from efi.telegram.chat_orchestrator import ChatOrchestrator
from efi.tools.base import ToolContext
from efi.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)

_PAYLOAD_TOO_LARGE_STATUS = 413
_TRIMMED_HISTORY_KEEP_LAST = 6
_FAILURE_NOTICE_TEXT = "уф, что-то у меня заглючило только что — попробуй написать ещё раз через минуту"

#: Личность обязана вызывать send_telegram_message как последнее действие
#: хода (см. personality.md), но ничто на уровне LLM-контракта не гарантирует
#: этого — модель может формально завершить ход текстом без tool_calls вовсе.
#: На практике это выглядит как "прочитано, и тишина" — собеседник получает
#: read receipt (Worker уже сходил в mark_as_read) и не получает ничего
#: больше, а в логах при этом всё чисто: с точки зрения кода ошибки не было.
_NO_REPLY_REMINDER_TEXT = (
    "[системное напоминание] Ты не отправила ответ собеседнику в этом ходу — обязательно вызови "
    "send_telegram_message с текстом ответа прямо сейчас, иначе он ничего не получит."
)

#: Пауза между повторными TYPING-пингами во время ожидания LLM — короче TTL
#: статуса "печатает" в Telegram (~5-6с, см. efi/telegram/typing_tracker.py),
#: чтобы статус не успевал погаснуть между пингами.
_TYPING_PULSE_INTERVAL_SECONDS = 4.0

#: Уведомления, где Эфи пишет ПЕРВОЙ. Разрешены только владельцу —
#: см. ConversationLifecycle.allows_proactive_ping и Worker._should_disengage.
_PROACTIVE_NOTIFICATION_TYPES = frozenset(
    {NotificationType.SPONTANEOUS_PING, NotificationType.SILENCE_PING, NotificationType.FOLLOW_UP}
)

#: Публичные выступления — их результат идёт в социальную память как
#: #public_comment (см. Worker._record_social_interaction).
_PUBLIC_COMMENT_TYPES = frozenset({NotificationType.PUBLIC_COMMENT, NotificationType.THREAD_REPLY})


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
        proactive_max_tool_call_rounds: int = 2,
        history_limit: int = 20,
        telegram: TelegramNotifier | None = None,
        lifecycle: ConversationLifecycle | None = None,
        social_memory: SocialInteractionStore | None = None,
        orchestrator: ChatOrchestrator | None = None,
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
        self._proactive_max_tool_call_rounds = proactive_max_tool_call_rounds
        self._history_limit = history_limit
        self._telegram = telegram
        self._lifecycle = lifecycle
        self._social_memory = social_memory
        self._orchestrator = orchestrator

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
                    await self._run_cancellable(notification)
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

    async def _run_cancellable(self, notification: Notification) -> None:
        """
        Обработка одного уведомления как отменяемого таска.

        Без оркестратора — обычный await, как было раньше. С ним обработка
        живёт в отдельном asyncio.Task, который снимается, как только в этом
        чате появляется новая реплика: пока Эфи думала или печатала серию
        бабблов, разговор мог уйти вперёд, и договаривать ответ на устаревший
        вопрос — ровно тот эффект «запоздалого бота», от которого уходим
        (см. efi/telegram/chat_orchestrator.py).

        Отмена ЭТОЙ генерации не должна выглядеть как ошибка и не должна
        останавливать воркер: он просто берёт следующее уведомление, которое
        уже содержит всю актуальную пачку.
        """
        if self._orchestrator is None:
            await self._handle(notification)
            return
        await self._orchestrator.run(notification.chat_id, self._handle(notification))

    async def _handle(self, notification: Notification) -> None:
        tool_context = ToolContext(notification=notification)
        try:
            await self._handle_inner(notification, tool_context)
        except asyncio.CancelledError:
            # Ход сняли как устаревший. Бабблы, которые собеседник УЖЕ
            # прочитал, отозвать нельзя — значит, они обязаны попасть в
            # историю: следующая генерация иначе соберёт контекст без них и
            # повторит сказанное. sent_texts наполняется побаббльно именно
            # ради этого случая (см. SendMessageTool).
            await self._persist_partial_reply(notification, tool_context)
            raise

    async def _persist_partial_reply(self, notification: Notification, tool_context: ToolContext) -> None:
        """Сохраняет в историю то, что успело уйти до отмены. Ошибка здесь не должна подменять саму отмену."""
        sent_texts = tool_context.extra.get("sent_texts")
        if not sent_texts or notification.chat_id is None:
            return
        try:
            await self._history.append(
                notification.chat_id, Message(role=Role.ASSISTANT, content="\n".join(sent_texts))
            )
        except Exception:
            logger.warning(
                "worker[%d]: failed to persist %d already-delivered bubbles after interruption",
                self._worker_index, len(sent_texts), exc_info=True,
            )

    async def _handle_inner(self, notification: Notification, tool_context: ToolContext) -> None:
        if await self._should_disengage(notification):
            return

        decision = await self._busy_engine.decide(notification.chat_id)
        is_user_message = notification.type is NotificationType.USER_MESSAGE and notification.chat_id is not None

        # Если Эфи УЖЕ в контексте активного чата, сообщение отмечается
        # прочитанным сразу, ДО какой-либо задержки: она физически смотрит в
        # этот чат прямо сейчас, и держать реплику непрочитанной несколько
        # секунд — не "занятость", а искусственное удержание в непрочитанных.
        # Полноценная задержка "не сразу взяла телефон" осмысленна только
        # тогда, когда разговор ещё не идёт (см. efi/behavior/busy_engine.py).
        marked_as_read = False
        if is_user_message and decision.is_active_conversation and self._telegram is not None:
            assert notification.chat_id is not None  # гарантировано is_user_message
            await self._telegram.mark_as_read(notification.chat_id)
            marked_as_read = True

        await self._apply_busy_delay(notification, decision.delay_seconds)

        history = (
            await self._history.get_recent(notification.chat_id, limit=self._history_limit)
            if notification.chat_id is not None
            else Session()
        )
        session = _finalize_session(history, notification)

        if is_user_message and notification.chat_id is not None:
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

        if is_user_message and not marked_as_read and self._telegram is not None:
            # "Заход в чат" после паузы — именно тут, а не раньше: до этого
            # момента Worker не совершил ни одного видимого телеграм-действия
            # (см. докстринг модуля и efi/behavior/busy_engine.py). Внутри
            # активного разговора отметка уже проставлена выше, до задержки.
            assert notification.chat_id is not None  # гарантировано is_user_message
            await self._telegram.mark_as_read(notification.chat_id)

        system_prompt = await self._system_prompt_builder.build(notification, history)

        params = LLMParams(
            model="",  # роутер сам подставит модель кандидата по main_role — см. efi.llm.router.LLMRouter
            system_prompt=system_prompt,
            tools=self._tool_registry.as_openai_tools(tool_context),
        )

        # Проактивные уведомления (спонтанный пинг, пинг по затишью, follow-up,
        # ночная задача) не имеют настоящей "реплики собеседника" — это
        # синтетический повод, а не реальный туда-обратно диалог. С полным
        # бюджетом раундов (max_tool_call_rounds=8, как у обычного ответа)
        # модель может провести несколько раундов подряд БЕЗ единой реакции
        # от собеседника и уйти в монолог с самой собой ("эй, ты там?" / "алло"
        # / "ты в коме?" / ... одним потоком) — с точки зрения истории это
        # выглядит как обычный разговор, но по факту никто не отвечал.
        # Для таких уведомлений бюджет раундов заметно ниже: одна попытка
        # позвать/поделиться и один раунд на wrap-up, не восемь.
        max_rounds = (
            self._max_tool_call_rounds
            if notification.type is NotificationType.USER_MESSAGE
            else self._proactive_max_tool_call_rounds
        )

        try:
            response = await self._run_with_typing_pulse(
                notification.chat_id, params, session, tool_context, max_rounds
            )
            if is_user_message:
                response = await self._ensure_reply_was_sent(params, session, tool_context, response)
        except asyncio.CancelledError:
            # Ход сняли как устаревший — не сбой. Уведомлять "у меня заглючило"
            # здесь нельзя: сообщение ушло бы поверх ответа на новую реплику.
            raise
        except LLMError:
            await self._notify_failure(notification)
            raise  # даём run() залогировать полный трейсбек, как и раньше
        except Exception:
            # Ловим ЛЮБОЙ сбой, а не только LLMError. Смысл уведомления не в
            # том, что подвёл именно LLM, а в том, что собеседник уже получил
            # "прочитано" (шаг 2) и обязан получить хоть что-то после него.
            # Раньше здесь стоял только `except LLMError`, и любая другая
            # ошибка на пути ответа — баг в инструменте, некорректный ответ
            # провайдера, сбой БД при записи истории — давала ровно ту самую
            # "прочитано и тишина", ради предотвращения которой существуют и
            # _ensure_reply_was_sent, и _notify_failure.
            if is_user_message:
                await self._notify_failure(notification)
            raise  # даём run() залогировать полный трейсбек, как и раньше

        if notification.type in _PROACTIVE_NOTIFICATION_TYPES and not tool_context.extra.get("sent_texts"):
            # Инициативный пинг, на котором модель ничего не отправила.
            # Запасной текст здесь подставлять нельзя (в отличие от
            # USER_MESSAGE): никто ничего не спрашивал, и «уф, у меня
            # заглючило» из ниоткуда выглядит хуже молчания. Но и тихо
            # ронять нельзя — иначе это опять неотличимо от «не работает».
            # В историю тоже ничего не пишем: до собеседника не долетело
            # ничего, а сохранённая реплика заставила бы Эфи в следующий раз
            # считать, что она это сказала, и продолжать с несуществующего места.
            logger.warning(
                "worker[%d]: proactive %s for chat_id=%s finished without sending anything "
                "(модель не вызвала send_telegram_message)",
                self._worker_index, notification.type.value, notification.chat_id,
            )
        elif notification.chat_id is not None:
            await self._history.append(notification.chat_id, _message_to_persist(response, tool_context))

        await self._record_social_interaction(notification, tool_context)

    async def _record_social_interaction(self, notification: Notification, tool_context: ToolContext) -> None:
        """
        Немедленно фиксирует внешнее взаимодействие в социальной памяти —
        публичный комментарий и переписку с посторонним (см. докстринг
        efi/memory/social_memory.py про гарантию сохранения).

        Пишется именно то, что РЕАЛЬНО ушло собеседнику (`sent_texts`), а не
        то, что модель собиралась сказать: журнал внешнего опыта должен
        совпадать с тем, что видели другие люди. Сбой записи не срывает уже
        доставленный ответ — только логируется.
        """
        if self._social_memory is None or notification.chat_id is None:
            return

        sent_texts = tool_context.extra.get("sent_texts")
        if not sent_texts:
            return

        sender_id = notification.payload.get("sender_id")
        sender_id = sender_id if isinstance(sender_id, int) else None
        is_public = bool(notification.payload.get("is_public_comment")) or notification.type in _PUBLIC_COMMENT_TYPES
        is_stranger_dm = (
            not is_public
            and notification.payload.get("chat_type") == "PRIVATE"
            and self._lifecycle is not None
            and not self._lifecycle.allows_proactive_ping(sender_id)
        )
        if not is_public and not is_stranger_dm:
            return

        try:
            await self._social_memory.record(
                SocialInteraction(
                    kind=SocialInteractionKind.PUBLIC_COMMENT if is_public else SocialInteractionKind.STRANGER_DM,
                    text="\n".join(sent_texts),
                    chat_id=notification.chat_id,
                    thread_id=notification.payload.get("thread_id"),
                    peer_user_id=sender_id,
                    peer_name=str(notification.payload.get("sender_name") or ""),
                    chat_title=str(notification.payload.get("chat_title") or ""),
                )
            )
        except Exception:
            logger.warning(
                "worker[%d]: failed to record social interaction for notification %s",
                self._worker_index, notification.id, exc_info=True,
            )

    async def _should_disengage(self, notification: Notification) -> bool:
        """
        Молчаливый выход из разговора с посторонним — см.
        efi.behavior.conversation_lifecycle.ConversationLifecycle.

        Проверяется ПЕРВЫМ, до busy-задержки и до любого обращения к
        Telegram: если Эфи решила не отвечать, она и не должна засветиться
        ни статусом "прочитано", ни "печатает" — молчание должно выглядеть
        как молчание, а не как начатый и брошенный ответ.

        Инициативные пинги посторонним отсекаются здесь же: писать первой
        тому, кто об этом не просил, — навязчивость по определению.
        """
        if self._lifecycle is None:
            return False

        sender_id = notification.payload.get("sender_id")
        sender_id = sender_id if isinstance(sender_id, int) else None

        if notification.type in _PROACTIVE_NOTIFICATION_TYPES and not self._lifecycle.allows_proactive_ping_to_chat(
            notification.chat_id, sender_id
        ):
            # INFO, а не DEBUG: это единственный след того, что инициативный
            # пинг был отброшен. Пока он был отладочным, «Эфи не пишет первой»
            # выглядело как отсутствие функции, а не как решение кода.
            logger.info(
                "worker[%d]: skipping proactive %s for chat_id=%s — "
                "чат не в telegram.allowed_chats и это не личка владельца",
                self._worker_index, notification.type.value, notification.chat_id,
            )
            return True

        if notification.type is not NotificationType.USER_MESSAGE or notification.chat_id is None:
            return False

        decision = await self._lifecycle.evaluate(sender_id, notification.chat_id, notification.message)
        return decision.should_disengage

    async def _ensure_reply_was_sent(
        self, params: LLMParams, session: Session, tool_context: ToolContext, response: Response
    ) -> Response:
        """
        Гарантия "прочитано ≠ тишина" для входящих сообщений — см. докстринг
        `_NO_REPLY_REMINDER_TEXT`. Если за весь ход `send_telegram_message` ни
        разу не вызвался (проверяем по `tool_context.extra["sent_texts"]`,
        которое наполняет сам инструмент — см. efi.tools.telegram_actions.
        send_message.SendMessageTool), даём модели ОДИН явный шанс исправиться
        прямым напоминанием; если и это не помогло — отправляем нейтральное
        сообщение напрямую, чтобы молчание не выглядело как обрыв связи.
        LLMError на этой повторной попытке намеренно не ловится здесь —
        поднимается в `_handle`, где обрабатывается точно так же, как обычный
        сбой LLM (см. `except LLMError` вызывающей стороны).
        """
        if tool_context.extra.get("sent_texts"):
            return response

        logger.warning(
            "worker[%d]: notification %s finished without calling send_telegram_message, nudging once",
            self._worker_index, tool_context.notification.id,
        )
        session.append(Message(role=Role.SYSTEM, content=_NO_REPLY_REMINDER_TEXT))
        response = await self._run_with_tool_calls(params, session, tool_context, self._max_tool_call_rounds)

        if not tool_context.extra.get("sent_texts") and self._telegram is not None and tool_context.chat_id is not None:
            logger.warning(
                "worker[%d]: notification %s still produced no reply after nudge, sending fallback notice",
                self._worker_index, tool_context.notification.id,
            )
            await self._telegram.send_message(tool_context.chat_id, _FAILURE_NOTICE_TEXT)
            tool_context.extra["sent_texts"] = [_FAILURE_NOTICE_TEXT]

        return response

    async def _apply_busy_delay(self, notification: Notification, delay: float) -> None:
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
        self, chat_id: int | None, params: LLMParams, session: Session, tool_context: ToolContext, max_rounds: int
    ) -> Response:
        """
        Оборачивает цикл tool-calling фоновым "пульсом" TYPING, пока Worker
        ждёт LLM (см. _typing_pulse) — без этого собеседник видел бы просто
        тишину между "прочитано" и первым сообщением, что менее естественно,
        чем живой статус "печатает".
        """
        if chat_id is None or self._telegram is None:
            return await self._run_with_tool_calls(params, session, tool_context, max_rounds)

        pulse_task = asyncio.create_task(self._typing_pulse(chat_id))
        try:
            return await self._run_with_tool_calls(params, session, tool_context, max_rounds)
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

    async def _run_with_tool_calls(
        self, params: LLMParams, session: Session, tool_context: ToolContext, max_rounds: int
    ) -> Response:
        """
        Цикл tool-calling: запросить ответ у LLM -> если модель вызвала
        инструменты, выполнить их и вернуть результаты в модель как
        TOOL-сообщения -> повторить, пока модель не даст финальный ответ без
        tool_calls. `max_rounds` — защита от зацикливания (модель может
        застрять, бесконечно вызывая инструменты) и, для проактивных
        уведомлений без реальной реплики собеседника, защита от монолога с
        самой собой (см. вызывающую сторону — `_handle`).

        ВАЖНО: как только в каком-то раунде реально сработал
        send_telegram_message (`tool_context.extra["sent_texts"]` стало
        непустым), цикл останавливается СРАЗУ, не дожидаясь, пока модель
        сама решит закончить ход. personality.md и так требует вызывать
        send_telegram_message ПОСЛЕДНИМ действием хода, но это лишь
        инструкция в промпте — ничто не мешало модели, уже отправив
        сообщение и получив TOOL-результат "Message sent successfully",
        продолжить как ни в чём не бывало: сгенерировать ещё реплику, будто
        прошло время и собеседник промолчал ("чё, молчишь?"), отправить её,
        и повторить это до max_rounds раз — что на практике выглядело не
        как один ответ (пусть даже из нескольких bubble'ов через "///"), а
        как самостоятельный внутренний диалог поверх уже доставленного
        ответа. Одного успешного send_telegram_message достаточно для
        целого хода: multi-bubble ответы уже поддерживаются ВНУТРИ одного
        вызова (см. efi/humanizer/message_splitting.py), отдельный
        повторный вызов инструмента для этого не нужен.

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
        for _round_number in range(max_rounds):
            response = await self._chat_with_size_retry(params, session)
            message = response.message
            session.append(message)

            if not message.has_tool_calls:
                return response

            tool_context.extra["llm_generation_time"] = asyncio.get_running_loop().time() - start_time
            for tool_call in message.tool_calls:
                result_text = await self._tool_registry.execute(tool_call, tool_context)
                session.append(Message(role=Role.TOOL, content=result_text, tool_call_id=tool_call.id))

            if tool_context.extra.get("sent_texts"):
                return response

        logger.warning(
            "worker[%d]: reached max_rounds=%d while processing %s, returning last response as-is",
            self._worker_index, max_rounds, tool_context.notification.id,
        )
        assert response is not None  # цикл выполняется минимум один раз, т.к. max_rounds задаётся >= 1
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
