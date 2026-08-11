"""
Тесты повторной доставки проактивных уведомлений.

Регрессия из жизни: человек попросил «напиши мне через 10 минут», через
десять минут напоминание сработало, ушло в очередь — и на генерации ответа
все три кандидата роли main не уложились в общий бюджет
(`LLMTimeoutError: role main: exceeded overall budget of 50s across 3
candidate(s)`). Трейсбек попал в лог, воркер выжил и взял следующее
уведомление, а обещание исчезло: напоминание уже помечено сработавшим
(ReminderStore.mark_fired), второго таймера у него нет.

С точки зрения человека это ровно то же самое, что было до починки самих
обещаний: он попросил, ему ответили «хорошо», и не написали. Таймаут
провайдера на бесплатном тире — рядовое событие, а не исключительная
ситуация, поэтому намерение обязано пережить неудачу.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from efi.behavior.conversation_lifecycle import ConversationLifecycle
from efi.config.schema import TaskRole
from efi.db.core import Database
from efi.db.models import MIGRATIONS
from efi.llm.errors import LLMTimeoutError
from efi.llm.schemas import (
    Choice,
    LLMParams,
    Message,
    Response,
    Role,
    Session,
    ToolCall,
    ToolCallFunction,
)
from efi.notifications.manager import MAX_DELIVERY_ATTEMPTS, NotificationManager
from efi.notifications.schemas import Notification, NotificationType
from efi.notifications.worker import Worker
from efi.tools.base import Tool, ToolContext
from efi.tools.registry import ToolRegistry
from tests.test_proactive_delivery import (  # переиспользуем уже готовые дубли
    _OWNER_ID,
    _FakeBusyEngine,
    _FakeHistory,
    _FakePromptBuilder,
)


class _TimingOutRouter:
    """Провайдер, который не уложился в бюджет — тот самый случай со скриншота."""

    def __init__(self) -> None:
        self.requests = 0

    async def chat(self, role: TaskRole, params: LLMParams, session: Session) -> Response:
        self.requests += 1
        raise LLMTimeoutError("role main: exceeded overall budget of 50s across 3 candidate(s)", provider="main")


class _SilentRouter:
    """Модель ответила текстом, но не позвала send_telegram_message — до человека не дошло ничего."""

    async def chat(self, role: TaskRole, params: LLMParams, session: Session) -> Response:
        return Response(choices=[Choice(message=Message(role=Role.ASSISTANT, content="подумала и промолчала"))])


class _SendTool(Tool):
    name = "send_telegram_message"
    description = "отправляет сообщение"
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

    def __init__(self) -> None:
        self.sent: list[str] = []

    async def execute(self, arguments: dict[str, object], context: ToolContext) -> str:
        text = str(arguments.get("text", ""))
        self.sent.append(text)
        context.extra.setdefault("sent_texts", []).append(text)
        return "Message sent successfully"


class _SendThenFailRouter:
    """Отправила сообщение, и уже ПОСЛЕ этого сломалась. Повтор здесь был бы вторым сообщением."""

    def __init__(self) -> None:
        self.requests = 0

    async def chat(self, role: TaskRole, params: LLMParams, session: Session) -> Response:
        self.requests += 1
        if self.requests == 1:
            return Response(
                choices=[
                    Choice(
                        message=Message(
                            role=Role.ASSISTANT,
                            content="",
                            tool_calls=[
                                ToolCall(
                                    id="1",
                                    function=ToolCallFunction(
                                        name="send_telegram_message", arguments='{"text": "как и обещала"}'
                                    ),
                                )
                            ],
                        )
                    )
                ]
            )
        raise LLMTimeoutError("поздний сбой", provider="main")


def _make_worker(tmp_path: Path, router: object) -> tuple[Worker, NotificationManager, _SendTool]:
    database = Database(tmp_path / "efi.db", migrations=MIGRATIONS)
    lifecycle = ConversationLifecycle(database, owner_id=_OWNER_ID, proactive_chats=())
    manager = NotificationManager(worker_count=1)
    send_tool = _SendTool()
    registry = ToolRegistry()
    registry.register(send_tool)
    worker = Worker(
        0,
        manager,
        llm_router=router,  # type: ignore[arg-type]
        tool_registry=registry,
        history=_FakeHistory(),
        system_prompt_builder=_FakePromptBuilder(),  # type: ignore[arg-type]
        busy_engine=_FakeBusyEngine(),  # type: ignore[arg-type]
        lifecycle=lifecycle,
    )
    return worker, manager, send_tool


def _reminder(attempt: int = 0) -> Notification:
    return Notification(
        type=NotificationType.FOLLOW_UP,
        chat_id=_OWNER_ID,
        message="ты обещала написать через 10 минут",
        payload={"promise_text": "написать через 10 минут"},
        attempt=attempt,
    )


# -- сам баг ------------------------------------------------------------------


async def test_promise_survives_an_llm_timeout(tmp_path: Path) -> None:
    """Главная регрессия: обещание не должно исчезать из-за таймаута провайдера."""
    worker, manager, _send = _make_worker(tmp_path, _TimingOutRouter())

    with pytest.raises(LLMTimeoutError):
        await worker._handle(_reminder())

    assert manager.pending_retries == 1, "намерение написать обязано пережить неудачу"
    await manager.cancel_retries()


async def test_retry_actually_lands_back_in_the_queue(tmp_path: Path) -> None:
    """Не просто «таймер создан», а уведомление действительно возвращается воркеру."""
    manager = NotificationManager(worker_count=1)
    notification = _reminder()

    assert manager.retry_later(notification, delay=0.01) is True
    await asyncio.sleep(0.05)

    assert manager.qsize() == 1
    queued = await manager.get(0)
    assert queued.type is NotificationType.FOLLOW_UP
    assert queued.payload["promise_text"] == "написать через 10 минут"
    assert queued.attempt == 1, "счётчик попыток должен расти, иначе повторы бесконечны"


async def test_silent_model_also_gets_another_chance(tmp_path: Path) -> None:
    """
    «Модель не позвала send_telegram_message» с точки зрения человека
    неотличимо от сбоя: он попросил написать — ему не написали.
    """
    worker, manager, send_tool = _make_worker(tmp_path, _SilentRouter())

    await worker._handle(_reminder())

    assert send_tool.sent == []
    assert manager.pending_retries == 1
    await manager.cancel_retries()


# -- границы ------------------------------------------------------------------


async def test_attempts_are_not_endless() -> None:
    manager = NotificationManager(worker_count=1)

    assert manager.retry_later(_reminder(attempt=MAX_DELIVERY_ATTEMPTS - 1), delay=0.01) is False
    assert manager.pending_retries == 0


async def test_a_delivered_turn_is_never_repeated(tmp_path: Path) -> None:
    """
    Сообщение уже ушло, сбой случился после — повтор означал бы второе
    сообщение поверх доставленного.
    """
    worker, manager, send_tool = _make_worker(tmp_path, _SendThenFailRouter())

    await worker._handle(_reminder())

    assert send_tool.sent == ["как и обещала"]
    assert manager.pending_retries == 0


async def test_user_messages_are_never_retried(tmp_path: Path) -> None:
    """
    Собеседник уже получил «уф, у меня заглючило» и, скорее всего, написал
    снова. Ответ на его прошлую реплику через минуту пришёл бы поверх
    нового разговора.
    """
    worker, manager, _send = _make_worker(tmp_path, _TimingOutRouter())

    with pytest.raises(LLMTimeoutError):
        await worker._handle(
            Notification(
                type=NotificationType.USER_MESSAGE,
                chat_id=_OWNER_ID,
                message="привет",
                payload={"sender_id": _OWNER_ID},
            )
        )

    assert manager.pending_retries == 0


async def test_pending_retries_are_dropped_on_shutdown() -> None:
    """Таймеры спят минутами — держать на них выключение приложения незачем."""
    manager = NotificationManager(worker_count=1)
    manager.retry_later(_reminder(), delay=600.0)

    await manager.cancel_retries()

    assert manager.pending_retries == 0
    assert manager.qsize() == 0
