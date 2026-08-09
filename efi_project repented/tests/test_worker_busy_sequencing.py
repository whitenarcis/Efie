"""
Тесты для efi.notifications.worker.Worker: порядок операций для входящего
сообщения — ignore_delay -> mark_as_read -> системный промпт -> LLM (с
TYPING-пульсом) -> llm_generation_time в ToolContext.extra, и (регрессия на
критический баг) — что реплика собеседника действительно попадает в
персистентную историю, а не только ответ самой Эфи.
"""

from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable, Collection

import pytest

from efi.behavior.busy_engine import BusyDecision
from efi.behavior.conversation_lifecycle import LifecycleDecision
from efi.config.schema import TaskRole
from efi.llm.errors import LLMServerError
from efi.llm.schemas import Choice, LLMParams, Message, Response, Role, Session, ToolCall, ToolCallFunction
from efi.memory.social_memory import SocialInteraction, SocialInteractionKind
from efi.notifications.manager import NotificationManager
from efi.notifications.schemas import Notification, NotificationType
from efi.notifications.worker import _FAILURE_NOTICE_TEXT, Worker
from efi.telegram.chat_orchestrator import ChatOrchestrator
from efi.tools.base import Tool, ToolContext
from efi.tools.registry import ToolRegistry
from efi.tools.telegram_actions.send_message import SendMessageTool


class _FakeBusyEngine:
    def __init__(self, delay: float = 0.0, *, is_active_conversation: bool = False) -> None:
        self.delay = delay
        self.is_active_conversation = is_active_conversation
        self.calls: list[int | None] = []

    async def decide(self, chat_id: int | None) -> BusyDecision:
        self.calls.append(chat_id)
        return BusyDecision(delay_seconds=self.delay, is_active_conversation=self.is_active_conversation)


class _FakeHistoryRepository:
    def __init__(self) -> None:
        self.appended: list[tuple[int, Message]] = []

    async def get_recent(self, chat_id: int, limit: int = 20) -> Session:
        return Session()

    async def append(self, chat_id: int, message: Message) -> None:
        self.appended.append((chat_id, message))


class _FakeSystemPromptBuilder:
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def build(self, notification: Notification, history: Session) -> str:
        self._events.append("build_prompt")
        return "system prompt"


class _FakeTelegramNotifier:
    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.typing_calls = 0

    async def send_message(
        self,
        chat_id: int,
        text: str,
        *,
        reply_to_message_id: int | None = None,
        llm_generation_time: float | None = None,
        incoming_message_ids: Collection[int] = (),
        on_bubble_sent: Callable[[str], None] | None = None,
    ) -> None:
        self._events.append("send_message")
        # Реальный клиент подтверждает КАЖДЫЙ доставленный баббл — на этом
        # держится и история диалога, и учёт уже сказанного при отмене хода.
        if on_bubble_sent is not None:
            on_bubble_sent(text)

    async def mark_as_read(self, chat_id: int) -> None:
        self._events.append("mark_as_read")

    async def send_typing_action(self, chat_id: int) -> None:
        self.typing_calls += 1
        self._events.append("typing_action")


class _FakeLLMRouter:
    def __init__(self, events: list[str], *, error: Exception | None = None) -> None:
        self._events = events
        self._error = error

    async def chat(self, role: TaskRole, params: LLMParams, session: Session) -> Response:
        self._events.append("llm_chat")
        # Уступаем event loop хотя бы раз — иначе фоновая typing-pulse задача
        # (asyncio.create_task) не успеет выполнить свою первую итерацию до
        # того, как _run_with_typing_pulse её отменит.
        await asyncio.sleep(0.01)
        if self._error is not None:
            raise self._error
        return Response(choices=[Choice(message=Message(role=Role.ASSISTANT, content="привет"))])


def _make_worker(
    events: list[str], *, busy_delay: float = 0.0, llm_router: _FakeLLMRouter | None = None
) -> tuple[Worker, _FakeTelegramNotifier, _FakeBusyEngine, _FakeHistoryRepository]:
    manager = NotificationManager(worker_count=1)
    telegram = _FakeTelegramNotifier(events)
    busy_engine = _FakeBusyEngine(busy_delay)
    history = _FakeHistoryRepository()
    worker = Worker(
        0,
        manager,
        llm_router=llm_router or _FakeLLMRouter(events),  # type: ignore[arg-type]
        tool_registry=ToolRegistry(),
        history=history,
        system_prompt_builder=_FakeSystemPromptBuilder(events),
        busy_engine=busy_engine,  # type: ignore[arg-type]
        telegram=telegram,  # type: ignore[arg-type]
    )
    return worker, telegram, busy_engine, history


async def test_user_message_marks_as_read_after_busy_delay_and_before_llm() -> None:
    events: list[str] = []
    worker, telegram, busy_engine, _history = _make_worker(events)
    notification = Notification(type=NotificationType.USER_MESSAGE, chat_id=42, message="привет")

    await worker._handle(notification)

    assert busy_engine.calls == [42]
    assert events.index("mark_as_read") < events.index("build_prompt") < events.index("llm_chat")
    assert telegram.typing_calls >= 1


async def test_active_conversation_marks_as_read_before_any_delay() -> None:
    """
    Регрессия: внутри уже идущего разговора сообщение не должно "висеть"
    непрочитанным на время задержки — Эфи и так смотрит в этот чат. Отметка
    прочитанного обязана уйти ДО сна busy-задержки, а не после.
    """
    events: list[str] = []
    manager = NotificationManager(worker_count=1)
    telegram = _FakeTelegramNotifier(events)
    # Заметная задержка + активный разговор: если бы mark_as_read шёл после
    # сна, тест занял бы эти секунды и порядок событий был бы другим.
    busy_engine = _FakeBusyEngine(5.0, is_active_conversation=True)
    worker = Worker(
        0,
        manager,
        llm_router=_FakeLLMRouter(events),  # type: ignore[arg-type]
        tool_registry=ToolRegistry(),
        history=_FakeHistoryRepository(),
        system_prompt_builder=_FakeSystemPromptBuilder(events),
        busy_engine=busy_engine,  # type: ignore[arg-type]
        telegram=telegram,  # type: ignore[arg-type]
    )
    notification = Notification(type=NotificationType.USER_MESSAGE, chat_id=42, message="привет")

    read_at_start = asyncio.create_task(worker._handle(notification))
    await asyncio.sleep(0.05)  # заведомо меньше busy-задержки в 5с
    assert "mark_as_read" in events, "внутри активного разговора читать надо сразу, не дожидаясь задержки"

    read_at_start.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await read_at_start


async def test_inactive_chat_still_marks_as_read_only_after_the_delay() -> None:
    """Первое сообщение после паузы — прежнее поведение: сначала задержка, только потом 'прочитано'."""
    events: list[str] = []
    manager = NotificationManager(worker_count=1)
    telegram = _FakeTelegramNotifier(events)
    busy_engine = _FakeBusyEngine(5.0, is_active_conversation=False)
    worker = Worker(
        0,
        manager,
        llm_router=_FakeLLMRouter(events),  # type: ignore[arg-type]
        tool_registry=ToolRegistry(),
        history=_FakeHistoryRepository(),
        system_prompt_builder=_FakeSystemPromptBuilder(events),
        busy_engine=busy_engine,  # type: ignore[arg-type]
        telegram=telegram,  # type: ignore[arg-type]
    )
    notification = Notification(type=NotificationType.USER_MESSAGE, chat_id=42, message="привет")

    task = asyncio.create_task(worker._handle(notification))
    await asyncio.sleep(0.05)
    assert "mark_as_read" not in events, "без активного разговора отметка должна ждать окончания задержки"

    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


async def test_non_user_message_does_not_mark_as_read() -> None:
    events: list[str] = []
    worker, telegram, _busy_engine, _history = _make_worker(events)
    notification = Notification(type=NotificationType.SPONTANEOUS_PING, chat_id=42, message="напиши первой")

    await worker._handle(notification)

    assert "mark_as_read" not in events


async def test_handle_without_chat_id_skips_read_receipt_and_typing() -> None:
    events: list[str] = []
    worker, telegram, busy_engine, _history = _make_worker(events)
    notification = Notification(type=NotificationType.NIGHTLY_TASK, chat_id=None, message="ночная задача")

    await worker._handle(notification)

    assert busy_engine.calls == [None]
    assert "mark_as_read" not in events
    assert telegram.typing_calls == 0


async def test_user_message_persists_both_sides_of_the_conversation() -> None:
    """
    Регрессия на критический баг: раньше в историю попадал только ответ Эфи,
    реплика собеседника не сохранялась вовсе — следующий запрос видел бы
    историю из одних её собственных сообщений (монолог, не диалог). Роутер
    здесь реально вызывает send_telegram_message (см.
    _ScriptedToolCallingLLMRouter) — plain-текстовый ответ без tool_calls
    теперь сам по себе ловится _ensure_reply_was_sent (см. тесты ниже) и не
    подходит для проверки именно порядка/состава персистентной истории.
    """
    events: list[str] = []
    manager = NotificationManager(worker_count=1)
    telegram = _FakeTelegramNotifier(events)
    busy_engine = _FakeBusyEngine(0.0)
    history = _FakeHistoryRepository()
    tool_registry = ToolRegistry()
    tool_registry.register(SendMessageTool(telegram))
    worker = Worker(
        0,
        manager,
        llm_router=_ScriptedToolCallingLLMRouter(events, actual_reply_text="привет"),  # type: ignore[arg-type]
        tool_registry=tool_registry,
        history=history,
        system_prompt_builder=_FakeSystemPromptBuilder(events),
        busy_engine=busy_engine,  # type: ignore[arg-type]
        telegram=telegram,  # type: ignore[arg-type]
    )
    notification = Notification(type=NotificationType.USER_MESSAGE, chat_id=42, message="привет, как дела?")

    await worker._handle(notification)

    assert len(history.appended) == 2
    (chat_id_1, first), (chat_id_2, second) = history.appended
    assert chat_id_1 == chat_id_2 == 42
    assert first.role is Role.USER
    assert first.content == "привет, как дела?"
    assert second.role is Role.ASSISTANT
    assert second.content == "привет"


async def test_user_turn_is_persisted_even_when_llm_fails() -> None:
    """
    Реплика собеседника не должна теряться, даже если сам запрос к LLM в
    итоге упал — и точно так же не должен теряться fallback-текст о сбое,
    который Worker отправляет напрямую (в обход LLM): следующий запрос в
    этом чате должен видеть, что Эфи уже что-то сказала про сбой, а не
    находить в истории провал без объяснения.
    """
    events: list[str] = []
    failing_router = _FakeLLMRouter(events, error=LLMServerError("boom", provider="test"))
    worker, _telegram, _busy_engine, history = _make_worker(events, llm_router=failing_router)
    notification = Notification(type=NotificationType.USER_MESSAGE, chat_id=42, message="привет")

    with pytest.raises(LLMServerError):
        await worker._handle(notification)

    assert len(history.appended) == 2
    (chat_id_1, user_message), (chat_id_2, failure_message) = history.appended
    assert chat_id_1 == chat_id_2 == 42
    assert user_message.role is Role.USER
    assert user_message.content == "привет"
    assert failure_message.role is Role.ASSISTANT
    assert failure_message.content == _FAILURE_NOTICE_TEXT


async def test_non_user_message_does_not_fabricate_a_user_turn() -> None:
    """Триггер SPONTANEOUS_PING и т.п. — не то, что сказал собеседник, и не должен так выглядеть в истории."""
    events: list[str] = []
    worker, _telegram, _busy_engine, history = _make_worker(events)
    notification = Notification(type=NotificationType.SPONTANEOUS_PING, chat_id=42, message="напиши первой")

    await worker._handle(notification)

    assert len(history.appended) == 1
    _chat_id, message = history.appended[0]
    assert message.role is Role.ASSISTANT


class _ScriptedToolCallingLLMRouter:
    """
    Симулирует реальную двухраундовую механику tool-calling: раунд 1 несёт
    и черновик ответа в content, и вызов send_telegram_message; раунд 2 —
    пустой служебный ход БЕЗ tool_calls (типичный для многих моделей после
    TOOL-результата "Message sent successfully").
    """

    def __init__(self, events: list[str], *, actual_reply_text: str) -> None:
        self._events = events
        self._actual_reply_text = actual_reply_text
        self._round = 0

    async def chat(self, role: TaskRole, params: LLMParams, session: Session) -> Response:
        self._events.append("llm_chat")
        self._round += 1
        if self._round == 1:
            tool_call = ToolCall(
                id="call_1",
                type="function",
                function=ToolCallFunction(
                    name="send_telegram_message",
                    arguments=f'{{"text": "{self._actual_reply_text}"}}',
                ),
            )
            return Response(
                choices=[
                    Choice(
                        message=Message(
                            role=Role.ASSISTANT,
                            content="<response>черновик, не отправлено напрямую</response>",
                            tool_calls=[tool_call],
                        )
                    )
                ]
            )
        # Раунд 2: модель уже "сказала своё" через инструмент — многие модели
        # возвращают пустой/служебный content здесь, без новых tool_calls.
        return Response(choices=[Choice(message=Message(role=Role.ASSISTANT, content=""))])


async def test_history_persists_actually_sent_text_not_the_trailing_empty_round() -> None:
    """
    Регрессия: раньше Worker сохранял в историю response.message последнего
    раунда цикла tool-calling — а не текст, который реально ушёл собеседнику
    через send_telegram_message. Личность обязана вызывать этот инструмент
    как ПОСЛЕДНИМ действием хода (см. personality.md), поэтому раунд ПОСЛЕ
    вызова инструмента часто пустой — и раньше именно эта пустота попадала
    в персистентную историю вместо реального ответа.
    """
    events: list[str] = []
    manager = NotificationManager(worker_count=1)
    telegram = _FakeTelegramNotifier(events)
    busy_engine = _FakeBusyEngine(0.0)
    history = _FakeHistoryRepository()

    tool_registry = ToolRegistry()
    tool_registry.register(SendMessageTool(telegram))

    llm_router = _ScriptedToolCallingLLMRouter(events, actual_reply_text="привет, у меня всё хорошо!")

    worker = Worker(
        0,
        manager,
        llm_router=llm_router,  # type: ignore[arg-type]
        tool_registry=tool_registry,
        history=history,
        system_prompt_builder=_FakeSystemPromptBuilder(events),
        busy_engine=busy_engine,  # type: ignore[arg-type]
        telegram=telegram,  # type: ignore[arg-type]
    )
    notification = Notification(type=NotificationType.USER_MESSAGE, chat_id=42, message="как дела?")

    await worker._handle(notification)

    assert len(history.appended) == 2
    (_chat_id, user_message), (_chat_id2, assistant_message) = history.appended
    assert user_message.role is Role.USER
    assert assistant_message.role is Role.ASSISTANT
    assert assistant_message.content == "привет, у меня всё хорошо!"
    assert assistant_message.content != ""


class _NeverCallsSendMessageRouter:
    """Модель, которая раз за разом отвечает текстом, ни разу не вызывая send_telegram_message."""

    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def chat(self, role: TaskRole, params: LLMParams, session: Session) -> Response:
        self._events.append("llm_chat")
        return Response(choices=[Choice(message=Message(role=Role.ASSISTANT, content="думаю про себя"))])


async def test_ensure_reply_sends_fallback_when_model_never_calls_send_message() -> None:
    """
    Регрессия: "прочитано, и тишина" без единой ошибки в логах — модель
    формально завершает ход текстом без tool_calls, ничего не долетает до
    собеседника, а Worker раньше считал это нормальным завершением обработки.
    """
    events: list[str] = []
    manager = NotificationManager(worker_count=1)
    telegram = _FakeTelegramNotifier(events)
    busy_engine = _FakeBusyEngine(0.0)
    history = _FakeHistoryRepository()
    tool_registry = ToolRegistry()
    tool_registry.register(SendMessageTool(telegram))
    worker = Worker(
        0,
        manager,
        llm_router=_NeverCallsSendMessageRouter(events),  # type: ignore[arg-type]
        tool_registry=tool_registry,
        history=history,
        system_prompt_builder=_FakeSystemPromptBuilder(events),
        busy_engine=busy_engine,  # type: ignore[arg-type]
        telegram=telegram,  # type: ignore[arg-type]
    )
    notification = Notification(type=NotificationType.USER_MESSAGE, chat_id=42, message="привет")

    await worker._handle(notification)

    assert events.count("llm_chat") == 2  # исходный раунд + один явный "нудж"
    assert events.count("send_message") == 1  # fallback ушёл напрямую, в обход инструмента
    assert len(history.appended) == 2
    _chat_id, assistant_message = history.appended[1]
    assert assistant_message.role is Role.ASSISTANT
    assert assistant_message.content == _FAILURE_NOTICE_TEXT


class _RepliesOnlyAfterNudgeRouter:
    """Забывает отправить ответ в первом раунде, но реагирует на системное напоминание."""

    def __init__(self, events: list[str], *, actual_reply_text: str) -> None:
        self._events = events
        self._actual_reply_text = actual_reply_text
        self._call_count = 0

    async def chat(self, role: TaskRole, params: LLMParams, session: Session) -> Response:
        self._events.append("llm_chat")
        self._call_count += 1
        if self._call_count == 1:
            return Response(choices=[Choice(message=Message(role=Role.ASSISTANT, content="забыла ответить"))])
        if self._call_count == 2:
            tool_call = ToolCall(
                id="call_1",
                type="function",
                function=ToolCallFunction(
                    name="send_telegram_message",
                    arguments=f'{{"text": "{self._actual_reply_text}"}}',
                ),
            )
            return Response(
                choices=[Choice(message=Message(role=Role.ASSISTANT, content="", tool_calls=[tool_call]))]
            )
        return Response(choices=[Choice(message=Message(role=Role.ASSISTANT, content=""))])


async def test_ensure_reply_recovers_after_a_single_nudge() -> None:
    """Если модель одумывается после напоминания — до fallback-сообщения дело не доходит."""
    events: list[str] = []
    manager = NotificationManager(worker_count=1)
    telegram = _FakeTelegramNotifier(events)
    busy_engine = _FakeBusyEngine(0.0)
    history = _FakeHistoryRepository()
    tool_registry = ToolRegistry()
    tool_registry.register(SendMessageTool(telegram))
    worker = Worker(
        0,
        manager,
        llm_router=_RepliesOnlyAfterNudgeRouter(events, actual_reply_text="ой прости, вот мой ответ"),  # type: ignore[arg-type]
        tool_registry=tool_registry,
        history=history,
        system_prompt_builder=_FakeSystemPromptBuilder(events),
        busy_engine=busy_engine,  # type: ignore[arg-type]
        telegram=telegram,  # type: ignore[arg-type]
    )
    notification = Notification(type=NotificationType.USER_MESSAGE, chat_id=42, message="привет")

    await worker._handle(notification)

    assert events.count("send_message") == 1
    _chat_id, assistant_message = history.appended[1]
    assert assistant_message.content == "ой прости, вот мой ответ"
    assert assistant_message.content != _FAILURE_NOTICE_TEXT


async def test_llm_generation_time_is_recorded_only_with_tool_calls() -> None:
    """Без tool_calls в ответе (наш FakeLLMRouter их не шлёт) llm_generation_time в extra не появляется."""
    events: list[str] = []
    manager = NotificationManager(worker_count=1)
    telegram = _FakeTelegramNotifier(events)
    busy_engine = _FakeBusyEngine(0.0)
    worker = Worker(
        0,
        manager,
        llm_router=_FakeLLMRouter(events),  # type: ignore[arg-type]
        tool_registry=ToolRegistry(),
        history=_FakeHistoryRepository(),
        system_prompt_builder=_FakeSystemPromptBuilder(events),
        busy_engine=busy_engine,  # type: ignore[arg-type]
        telegram=telegram,  # type: ignore[arg-type]
    )
    notification = Notification(type=NotificationType.USER_MESSAGE, chat_id=1, message="привет")
    tool_context = ToolContext(notification=notification)
    params = LLMParams(model="", system_prompt="x")
    session = Session()
    await worker._run_with_tool_calls(params, session, tool_context, 8)
    assert "llm_generation_time" not in tool_context.extra


class _NoopTool(Tool):
    """Инструмент-пустышка для тестов — не отправляет ничего, просто занимает раунд tool-calling."""

    name = "noop_tool"
    description = "test-only tool that does nothing"
    parameters: dict[str, object] = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}

    async def execute(self, arguments: dict[str, object], context: ToolContext) -> str:
        return "ok"


class _AlwaysSendsRouter:
    """Модель, которая на КАЖДЫЙ раунд заново вызывает send_telegram_message — 'никогда не считает себя законченной'."""

    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.call_count = 0

    async def chat(self, role: TaskRole, params: LLMParams, session: Session) -> Response:
        self._events.append("llm_chat")
        self.call_count += 1
        tool_call = ToolCall(
            id=f"call_{self.call_count}",
            type="function",
            function=ToolCallFunction(
                name="send_telegram_message",
                arguments=f'{{"text": "сообщение номер {self.call_count}"}}',
            ),
        )
        return Response(choices=[Choice(message=Message(role=Role.ASSISTANT, content="", tool_calls=[tool_call]))])


class _NeverSendsRouter:
    """Модель, которая каждый раунд вызывает посторонний инструмент, но НИКОГДА не отправляет сообщение."""

    def __init__(self, events: list[str]) -> None:
        self._events = events
        self.call_count = 0

    async def chat(self, role: TaskRole, params: LLMParams, session: Session) -> Response:
        self._events.append("llm_chat")
        self.call_count += 1
        tool_call = ToolCall(
            id=f"call_{self.call_count}", type="function", function=ToolCallFunction(name="noop_tool", arguments="{}")
        )
        return Response(choices=[Choice(message=Message(role=Role.ASSISTANT, content="", tool_calls=[tool_call]))])


@pytest.mark.parametrize("notification_type", [NotificationType.USER_MESSAGE, NotificationType.SPONTANEOUS_PING])
async def test_loop_stops_immediately_after_first_successful_send(notification_type: NotificationType) -> None:
    """
    Регрессия на "монолог поверх уже доставленного ответа": модель, отправив
    сообщение, раньше могла продолжать генерировать ещё реплики как ни в чём
    не бывало ("чё, молчишь?" — уже ПОСЛЕ того, как ответ дошёл), пока не
    кончится бюджет раундов. Один успешный send_telegram_message теперь
    обрывает цикл сразу — вне зависимости от типа уведомления.
    """
    events: list[str] = []
    manager = NotificationManager(worker_count=1)
    telegram = _FakeTelegramNotifier(events)
    busy_engine = _FakeBusyEngine(0.0)
    history = _FakeHistoryRepository()
    tool_registry = ToolRegistry()
    tool_registry.register(SendMessageTool(telegram))
    router = _AlwaysSendsRouter(events)
    worker = Worker(
        0,
        manager,
        llm_router=router,  # type: ignore[arg-type]
        tool_registry=tool_registry,
        history=history,
        system_prompt_builder=_FakeSystemPromptBuilder(events),
        busy_engine=busy_engine,  # type: ignore[arg-type]
        telegram=telegram,  # type: ignore[arg-type]
    )
    notification = Notification(type=notification_type, chat_id=42, message="привет")

    await worker._handle(notification)

    assert router.call_count == 1
    assert events.count("send_message") == 1


async def test_proactive_notification_round_budget_caps_a_model_that_never_sends() -> None:
    """
    Бэкстоп на случай, если модель вообще не отправляет сообщение (а просто
    вызывает посторонние инструменты раунд за раундом) — для проактивных
    уведомлений бюджет раундов заметно меньше (по умолчанию 2), а не
    max_tool_call_rounds=8, чтобы не жечь раунды впустую на монолог с самой
    собой без единого реального сообщения собеседнику.
    """
    events: list[str] = []
    manager = NotificationManager(worker_count=1)
    telegram = _FakeTelegramNotifier(events)
    busy_engine = _FakeBusyEngine(0.0)
    history = _FakeHistoryRepository()
    tool_registry = ToolRegistry()
    tool_registry.register(SendMessageTool(telegram))
    tool_registry.register(_NoopTool())
    router = _NeverSendsRouter(events)
    worker = Worker(
        0,
        manager,
        llm_router=router,  # type: ignore[arg-type]
        tool_registry=tool_registry,
        history=history,
        system_prompt_builder=_FakeSystemPromptBuilder(events),
        busy_engine=busy_engine,  # type: ignore[arg-type]
        telegram=telegram,  # type: ignore[arg-type]
        proactive_max_tool_call_rounds=2,
    )
    notification = Notification(
        type=NotificationType.SPONTANEOUS_PING, chat_id=42, message="у тебя есть желание написать первой"
    )

    await worker._handle(notification)

    assert router.call_count == 2
    assert events.count("send_message") == 0


async def test_user_message_keeps_the_full_tool_call_round_budget() -> None:
    """Тот же 'никогда не отправляющий' раутер, но напрямую через _run_with_tool_calls — бюджет остаётся полным (8)."""
    events: list[str] = []
    manager = NotificationManager(worker_count=1)
    telegram = _FakeTelegramNotifier(events)
    busy_engine = _FakeBusyEngine(0.0)
    tool_registry = ToolRegistry()
    tool_registry.register(SendMessageTool(telegram))
    tool_registry.register(_NoopTool())
    router = _NeverSendsRouter(events)
    worker = Worker(
        0,
        manager,
        llm_router=router,  # type: ignore[arg-type]
        tool_registry=tool_registry,
        history=_FakeHistoryRepository(),
        system_prompt_builder=_FakeSystemPromptBuilder(events),
        busy_engine=busy_engine,  # type: ignore[arg-type]
        telegram=telegram,  # type: ignore[arg-type]
    )
    notification = Notification(type=NotificationType.USER_MESSAGE, chat_id=42, message="привет")
    tool_context = ToolContext(notification=notification)
    params = LLMParams(model="", system_prompt="x")
    session = Session()

    await worker._run_with_tool_calls(params, session, tool_context, 8)

    assert router.call_count == 8


class _FakeLifecycle:
    """Подменяет ConversationLifecycle: фиксированное решение + список запросов."""

    def __init__(self, *, disengage: bool = False, allow_proactive: bool = True) -> None:
        self._disengage = disengage
        self._allow_proactive = allow_proactive
        self.evaluated: list[tuple[int | None, int, str]] = []

    def allows_proactive_ping(self, user_id: int | None) -> bool:
        return self._allow_proactive

    async def evaluate(self, peer_user_id: int | None, chat_id: int, text: str) -> LifecycleDecision:
        self.evaluated.append((peer_user_id, chat_id, text))
        return LifecycleDecision(should_disengage=self._disengage, reason="test")


def _worker_with_lifecycle(
    events: list[str], lifecycle: _FakeLifecycle, *, social_memory: object | None = None
) -> Worker:
    telegram = _FakeTelegramNotifier(events)
    tool_registry = ToolRegistry()
    tool_registry.register(SendMessageTool(telegram))
    return Worker(
        0,
        NotificationManager(worker_count=1),
        llm_router=_ScriptedToolCallingLLMRouter(events, actual_reply_text="ответ"),  # type: ignore[arg-type]
        tool_registry=tool_registry,
        history=_FakeHistoryRepository(),
        system_prompt_builder=_FakeSystemPromptBuilder(events),
        busy_engine=_FakeBusyEngine(0.0),  # type: ignore[arg-type]
        telegram=telegram,  # type: ignore[arg-type]
        lifecycle=lifecycle,  # type: ignore[arg-type]
        social_memory=social_memory,  # type: ignore[arg-type]
    )


async def test_disengaged_conversation_produces_total_silence() -> None:
    """
    Молчание должно выглядеть как молчание: ни "прочитано", ни "печатает",
    ни обращения к LLM — иначе это читается как начатый и брошенный ответ.
    """
    events: list[str] = []
    worker = _worker_with_lifecycle(events, _FakeLifecycle(disengage=True))
    notification = Notification(
        type=NotificationType.USER_MESSAGE, chat_id=42, message="пока", payload={"sender_id": 999}
    )

    await worker._handle(notification)

    assert events == []


async def test_proactive_ping_to_a_stranger_is_dropped() -> None:
    """Писать первой тому, кто об этом не просил, — навязчивость по определению."""
    events: list[str] = []
    worker = _worker_with_lifecycle(events, _FakeLifecycle(allow_proactive=False))
    notification = Notification(type=NotificationType.SPONTANEOUS_PING, chat_id=42, message="напиши первой")

    await worker._handle(notification)

    assert events == []


async def test_proactive_ping_to_the_owner_goes_through() -> None:
    events: list[str] = []
    worker = _worker_with_lifecycle(events, _FakeLifecycle(allow_proactive=True))
    notification = Notification(type=NotificationType.SPONTANEOUS_PING, chat_id=42, message="напиши первой")

    await worker._handle(notification)

    assert events.count("send_message") == 1


class _RecordingSocialMemory:
    def __init__(self) -> None:
        self.recorded: list[SocialInteraction] = []

    async def record(self, interaction: SocialInteraction) -> int:
        self.recorded.append(interaction)
        return len(self.recorded)


async def test_public_comment_is_recorded_in_social_memory() -> None:
    """Внешний опыт фиксируется тем, что РЕАЛЬНО ушло людям, а не черновиком модели."""
    events: list[str] = []
    social_memory = _RecordingSocialMemory()
    worker = _worker_with_lifecycle(events, _FakeLifecycle(), social_memory=social_memory)
    notification = Notification(
        type=NotificationType.PUBLIC_COMMENT,
        chat_id=-1001,
        message="пост про async",
        payload={"is_public_comment": True, "thread_id": 55, "chat_title": "Канал"},
    )

    await worker._handle(notification)

    assert len(social_memory.recorded) == 1
    assert social_memory.recorded[0].kind is SocialInteractionKind.PUBLIC_COMMENT
    assert social_memory.recorded[0].text == "ответ"
    assert social_memory.recorded[0].thread_id == 55


async def test_owner_conversation_is_not_recorded_as_external_experience() -> None:
    """Личный разговор с владельцем — не «внешний опыт», ему в социальной памяти не место."""
    events: list[str] = []
    social_memory = _RecordingSocialMemory()
    worker = _worker_with_lifecycle(
        events, _FakeLifecycle(allow_proactive=True), social_memory=social_memory
    )
    notification = Notification(
        type=NotificationType.USER_MESSAGE,
        chat_id=42,
        message="привет",
        payload={"sender_id": 111, "chat_type": "PRIVATE"},
    )

    await worker._handle(notification)

    assert social_memory.recorded == []


# -- прерывание устаревшей генерации ------------------------------------------------
#
# Между приходом сообщения и последним бабблом проходят десятки секунд.
# Раньше Эфи договаривала ответ на устаревший вопрос, даже если разговор уже
# ушёл вперёд, — тот самый эффект «запоздалого бота» с отставанием на реплику.


class _SlowSendingTool(Tool):
    """Инструмент, который «печатает» серию бабблов и подтверждает каждый по отдельности."""

    name = "send_telegram_message"
    description = "test"
    parameters = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}

    def __init__(self, bubbles: list[str], *, per_bubble_delay: float = 0.05) -> None:
        self._bubbles = bubbles
        self._per_bubble_delay = per_bubble_delay

    async def execute(self, arguments: dict, context: ToolContext) -> str:
        delivered = context.extra.setdefault("sent_texts", [])
        for bubble in self._bubbles:
            await asyncio.sleep(self._per_bubble_delay)
            delivered.append(bubble)
        return "Message sent successfully."


def _worker_with_tool(tool: Tool, events: list[str]) -> tuple[Worker, _FakeHistoryRepository, ChatOrchestrator]:
    registry = ToolRegistry()
    registry.register(tool)
    history = _FakeHistoryRepository()
    orchestrator = ChatOrchestrator()

    class _ToolCallingRouter(_FakeLLMRouter):
        async def chat(self, role: TaskRole, params: LLMParams, session: Session) -> Response:
            await asyncio.sleep(0.01)
            call = ToolCall(id="1", function=ToolCallFunction(name=tool.name, arguments="{}"))
            return Response(choices=[Choice(message=Message(role=Role.ASSISTANT, content="", tool_calls=[call]))])

    worker = Worker(
        0,
        NotificationManager(worker_count=1),
        llm_router=_ToolCallingRouter(events),  # type: ignore[arg-type]
        tool_registry=registry,
        history=history,
        system_prompt_builder=_FakeSystemPromptBuilder(events),
        busy_engine=_FakeBusyEngine(0.0),  # type: ignore[arg-type]
        telegram=_FakeTelegramNotifier(events),  # type: ignore[arg-type]
        orchestrator=orchestrator,
    )
    return worker, history, orchestrator


async def test_interrupting_a_turn_persists_only_what_was_already_delivered() -> None:
    """
    Доставленные бабблы отозвать нельзя — собеседник их прочитал. Значит, они
    обязаны попасть в историю: иначе следующая генерация соберёт контекст без
    них и повторит сказанное. Раньше история при отмене не получала ничего.
    """
    events: list[str] = []
    worker, history, orchestrator = _worker_with_tool(_SlowSendingTool(["раз", "два", "три"]), events)
    notification = Notification(type=NotificationType.USER_MESSAGE, chat_id=42, message="привет")

    runner = asyncio.create_task(worker._run_cancellable(notification))
    await asyncio.sleep(0.09)  # успели уйти примерно первые бабблы, серия ещё идёт
    await orchestrator.interrupt(42)
    await runner

    persisted = [message.content for _chat_id, message in history.appended if message.role is Role.ASSISTANT]
    assert len(persisted) == 1
    delivered = persisted[0].split("\n")
    assert delivered == ["раз", "два", "три"][: len(delivered)], "в историю попало ровно доставленное, по порядку"
    assert len(delivered) < 3, "серию прервали — последний баббл уйти не успел"


async def test_interruption_does_not_stop_the_worker() -> None:
    """Отмена одной генерации — штатное событие, а не сбой: воркер обязан взять следующее уведомление."""
    events: list[str] = []
    worker, _history, orchestrator = _worker_with_tool(_SlowSendingTool(["раз", "два"], per_bubble_delay=0.1), events)
    notification = Notification(type=NotificationType.USER_MESSAGE, chat_id=42, message="привет")

    runner = asyncio.create_task(worker._run_cancellable(notification))
    await asyncio.sleep(0.05)
    await orchestrator.interrupt(42)

    await runner  # не должно бросить CancelledError наружу


async def test_uninterrupted_turn_persists_the_full_reply_once() -> None:
    events: list[str] = []
    worker, history, _orchestrator = _worker_with_tool(_SlowSendingTool(["раз", "два"], per_bubble_delay=0.0), events)
    notification = Notification(type=NotificationType.USER_MESSAGE, chat_id=42, message="привет")

    await worker._run_cancellable(notification)

    persisted = [message.content for _chat_id, message in history.appended if message.role is Role.ASSISTANT]
    assert persisted == ["раз\nдва"]
