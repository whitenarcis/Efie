"""
Тесты для efi.notifications.worker.Worker: порядок операций для входящего
сообщения — ignore_delay -> mark_as_read -> системный промпт -> LLM (с
TYPING-пульсом) -> llm_generation_time в ToolContext.extra, и (регрессия на
критический баг) — что реплика собеседника действительно попадает в
персистентную историю, а не только ответ самой Эфи.
"""

from __future__ import annotations

import asyncio

import pytest

from efi.config.schema import TaskRole
from efi.llm.errors import LLMServerError
from efi.llm.schemas import Choice, LLMParams, Message, Response, Role, Session, ToolCall, ToolCallFunction
from efi.notifications.manager import NotificationManager
from efi.notifications.schemas import Notification, NotificationType
from efi.notifications.worker import _FAILURE_NOTICE_TEXT, Worker
from efi.tools.base import ToolContext
from efi.tools.registry import ToolRegistry
from efi.tools.telegram_actions.send_message import SendMessageTool


class _FakeBusyEngine:
    def __init__(self, delay: float = 0.0) -> None:
        self.delay = delay
        self.calls: list[int | None] = []

    async def compute_ignore_delay(self, chat_id: int | None) -> float:
        self.calls.append(chat_id)
        return self.delay


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
    ) -> None:
        self._events.append("send_message")

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


class _AlwaysCallsToolRouter:
    """Модель, которая НИКОГДА не останавливается сама — на каждый раунд отвечает новым tool_call."""

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


async def test_proactive_notification_gets_a_much_smaller_tool_call_round_budget() -> None:
    """
    Регрессия на "монолог с самой собой": для SPONTANEOUS_PING (и прочих
    уведомлений без реальной реплики собеседника) раньше использовался тот
    же 8-раундовый бюджет, что и для обычного диалога — модель, которую
    ничего не останавливает, генерировала цепочку из нескольких сообщений
    подряд ("эй, ты там?" / "алло" / "ты в коме?" / ...). Бюджет для
    проактивных уведомлений должен быть заметно меньше (по умолчанию 2).
    """
    events: list[str] = []
    manager = NotificationManager(worker_count=1)
    telegram = _FakeTelegramNotifier(events)
    busy_engine = _FakeBusyEngine(0.0)
    history = _FakeHistoryRepository()
    tool_registry = ToolRegistry()
    tool_registry.register(SendMessageTool(telegram))
    router = _AlwaysCallsToolRouter(events)
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

    assert router.call_count == 2  # остановилась на proactive_max_tool_call_rounds, а не max_tool_call_rounds=8
    assert events.count("send_message") == 2


async def test_user_message_still_gets_the_full_tool_call_round_budget() -> None:
    """Тот же 'никогда не останавливающийся' раутер, но для USER_MESSAGE — бюджет должен остаться полным (8)."""
    events: list[str] = []
    manager = NotificationManager(worker_count=1)
    telegram = _FakeTelegramNotifier(events)
    busy_engine = _FakeBusyEngine(0.0)
    history = _FakeHistoryRepository()
    tool_registry = ToolRegistry()
    tool_registry.register(SendMessageTool(telegram))
    router = _AlwaysCallsToolRouter(events)
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
    notification = Notification(type=NotificationType.USER_MESSAGE, chat_id=42, message="привет")

    await worker._handle(notification)

    assert router.call_count == 8
