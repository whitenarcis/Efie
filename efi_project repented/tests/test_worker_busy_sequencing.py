"""
Тесты для efi.notifications.worker.Worker: порядок операций для входящего
сообщения — ignore_delay -> mark_as_read -> системный промпт -> LLM (с
TYPING-пульсом) -> llm_generation_time в ToolContext.extra.
"""

from __future__ import annotations

import asyncio

from efi.config.schema import TaskRole
from efi.llm.schemas import Choice, Message, Response, Role, Session
from efi.notifications.manager import NotificationManager
from efi.notifications.schemas import Notification, NotificationType
from efi.notifications.worker import Worker
from efi.tools.base import ToolContext
from efi.tools.registry import ToolRegistry


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
    def __init__(self, events: list[str]) -> None:
        self._events = events

    async def chat(self, role: TaskRole, params, session: Session) -> Response:
        self._events.append("llm_chat")
        # Уступаем event loop хотя бы раз — иначе фоновая typing-pulse задача
        # (asyncio.create_task) не успеет выполнить свою первую итерацию до
        # того, как _run_with_typing_pulse её отменит.
        await asyncio.sleep(0.01)
        return Response(choices=[Choice(message=Message(role=Role.ASSISTANT, content="привет"))])


def _make_worker(
    events: list[str], *, busy_delay: float = 0.0
) -> tuple[Worker, _FakeTelegramNotifier, _FakeBusyEngine]:
    manager = NotificationManager(worker_count=1)
    telegram = _FakeTelegramNotifier(events)
    busy_engine = _FakeBusyEngine(busy_delay)
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
    return worker, telegram, busy_engine


async def test_user_message_marks_as_read_after_busy_delay_and_before_llm() -> None:
    events: list[str] = []
    worker, telegram, busy_engine = _make_worker(events)
    notification = Notification(type=NotificationType.USER_MESSAGE, chat_id=42, message="привет")

    await worker._handle(notification)

    assert busy_engine.calls == [42]
    assert events.index("mark_as_read") < events.index("build_prompt") < events.index("llm_chat")
    assert telegram.typing_calls >= 1


async def test_non_user_message_does_not_mark_as_read() -> None:
    events: list[str] = []
    worker, telegram, _busy_engine = _make_worker(events)
    notification = Notification(type=NotificationType.SPONTANEOUS_PING, chat_id=42, message="напиши первой")

    await worker._handle(notification)

    assert "mark_as_read" not in events


async def test_handle_without_chat_id_skips_read_receipt_and_typing() -> None:
    events: list[str] = []
    worker, telegram, busy_engine = _make_worker(events)
    notification = Notification(type=NotificationType.NIGHTLY_TASK, chat_id=None, message="ночная задача")

    await worker._handle(notification)

    assert busy_engine.calls == [None]
    assert "mark_as_read" not in events
    assert telegram.typing_calls == 0


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
    from efi.llm.schemas import LLMParams

    params = LLMParams(model="", system_prompt="x")
    session = Session()
    await worker._run_with_tool_calls(params, session, tool_context)
    assert "llm_generation_time" not in tool_context.extra
