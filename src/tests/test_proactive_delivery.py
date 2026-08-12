"""
Тесты доставки инициативных сообщений — тех, где Эфи пишет ПЕРВОЙ
(SPONTANEOUS_PING, SILENCE_PING, FOLLOW_UP).

Регрессия на баг, из-за которого она не написала первой ни разу за всё
время работы. Инициативные уведомления ставились в очередь как положено
(«spontaneous_ping: queued for chat_id=...» в логах), но Worker отбрасывал
их все до единого: он спрашивал у ConversationLifecycle, разрешён ли пинг
ОТПРАВИТЕЛЮ, а у события, рождённого таймером, отправителя нет — payload
пустой. None трактовался как «посторонний», посторонним инициатива
запрещена, и пинг умирал на первой же строчке обработки, оставляя после
себя одну debug-запись.

Правило теперь формулируется через чат: личка владельца и то, что владелец
сам перечислил в telegram.allowed_chats.
"""

from __future__ import annotations

from pathlib import Path

from efi.behavior.busy_engine import BusyDecision
from efi.behavior.conversation_lifecycle import ConversationLifecycle
from efi.config.schema import TaskRole
from efi.db.core import Database
from efi.db.models import MIGRATIONS
from efi.llm.schemas import Choice, LLMParams, Message, Response, Role, Session, ToolCall, ToolCallFunction
from efi.notifications.manager import NotificationManager
from efi.notifications.schemas import Notification, NotificationType
from efi.notifications.worker import Worker
from efi.tools.base import Tool, ToolContext
from efi.tools.registry import ToolRegistry

_OWNER_ID = 2129889949
_ALLOWED_GROUP_ID = -4404120219
_STRANGER_CHAT_ID = 777000


class _FakeBusyEngine:
    async def decide(self, chat_id: int | None) -> BusyDecision:
        return BusyDecision(delay_seconds=0.0, is_active_conversation=False)


class _FakeHistory:
    def __init__(self) -> None:
        self.appended: list[tuple[int, Message]] = []

    async def get_recent(self, chat_id: int, limit: int = 20) -> Session:
        return Session()

    async def append(self, chat_id: int, message: Message) -> None:
        self.appended.append((chat_id, message))


class _FakePromptBuilder:
    async def build(self, notification: Notification, history: Session) -> str:
        return "system prompt"


class _RecordingSendTool(Tool):
    """Подставной send_telegram_message: фиксирует отправленное, ничего не зная про Telegram."""

    name = "send_telegram_message"
    description = "отправляет сообщение"
    parameters = {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}

    def __init__(self) -> None:
        self.sent: list[tuple[int | None, str]] = []

    async def execute(self, arguments: dict[str, object], context: ToolContext) -> str:
        text = str(arguments.get("text", ""))
        self.sent.append((context.chat_id, text))
        context.extra.setdefault("sent_texts", []).append(text)
        return "Message sent successfully"


class _SendingRouter:
    """LLM, которая на первый запрос вызывает инструмент отправки, а дальше молчит."""

    def __init__(self, *, calls_tool: bool = True) -> None:
        self.calls_tool = calls_tool
        self.requests = 0

    async def chat(self, role: TaskRole, params: LLMParams, session: Session) -> Response:
        self.requests += 1
        if not self.calls_tool:
            return Response(choices=[Choice(message=Message(role=Role.ASSISTANT, content="подумала и промолчала"))])
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
                                        name="send_telegram_message", arguments='{"text": "слушай, а я тут подумала"}'
                                    ),
                                )
                            ],
                        )
                    )
                ]
            )
        return Response(choices=[Choice(message=Message(role=Role.ASSISTANT, content=""))])


def _make_worker(
    tmp_path: Path,
    *,
    router: _SendingRouter | None = None,
    allowed_chats: tuple[int, ...] = (_ALLOWED_GROUP_ID,),
    promises: object | None = None,
) -> tuple[Worker, _RecordingSendTool, _FakeHistory]:
    database = Database(tmp_path / "efi.db", migrations=MIGRATIONS)
    lifecycle = ConversationLifecycle(database, owner_id=_OWNER_ID, proactive_chats=allowed_chats)
    send_tool = _RecordingSendTool()
    registry = ToolRegistry()
    registry.register(send_tool)
    history = _FakeHistory()
    worker = Worker(
        0,
        NotificationManager(worker_count=1),
        llm_router=router or _SendingRouter(),  # type: ignore[arg-type]
        tool_registry=registry,
        history=history,
        system_prompt_builder=_FakePromptBuilder(),  # type: ignore[arg-type]
        busy_engine=_FakeBusyEngine(),  # type: ignore[arg-type]
        lifecycle=lifecycle,
        working_memory=promises,  # type: ignore[arg-type]
    )
    return worker, send_tool, history


# -- сам баг -----------------------------------------------------------------


async def test_spontaneous_ping_to_the_owner_is_delivered(tmp_path: Path) -> None:
    """Главная регрессия: пинг в личку владельца обязан дойти, а не отсеяться из-за пустого payload."""
    worker, send_tool, _history = _make_worker(tmp_path)

    await worker._handle(
        Notification(type=NotificationType.SPONTANEOUS_PING, chat_id=_OWNER_ID, message="напиши первой", payload={})
    )

    assert send_tool.sent == [(_OWNER_ID, "слушай, а я тут подумала")]


async def test_ping_to_an_allowed_group_is_delivered(tmp_path: Path) -> None:
    """Чат, который владелец сам вписал в allowed_chats, — тоже «свой»: инициатива там разрешена."""
    worker, send_tool, _history = _make_worker(tmp_path)

    await worker._handle(
        Notification(
            type=NotificationType.SPONTANEOUS_PING, chat_id=_ALLOWED_GROUP_ID, message="напиши первой", payload={}
        )
    )

    assert [chat_id for chat_id, _ in send_tool.sent] == [_ALLOWED_GROUP_ID]


async def test_silence_ping_and_follow_up_are_delivered_too(tmp_path: Path) -> None:
    """У этих типов payload тоже пустой — они падали ровно по той же причине."""
    for notification_type in (NotificationType.SILENCE_PING, NotificationType.FOLLOW_UP):
        worker, send_tool, _history = _make_worker(tmp_path / notification_type.value)
        await worker._handle(
            Notification(type=notification_type, chat_id=_OWNER_ID, message="повод", payload={})
        )
        assert send_tool.sent, f"{notification_type.value} не дошёл"


# -- граница, которую баг «случайно» соблюдал --------------------------------


async def test_ping_to_a_stranger_chat_is_still_refused(tmp_path: Path) -> None:
    """Писать первой в чат, которого нет в allowed_chats, по-прежнему нельзя."""
    worker, send_tool, _history = _make_worker(tmp_path)

    await worker._handle(
        Notification(
            type=NotificationType.SPONTANEOUS_PING, chat_id=_STRANGER_CHAT_ID, message="напиши первой", payload={}
        )
    )

    assert send_tool.sent == []


async def test_explicit_stranger_sender_still_wins_over_the_chat(tmp_path: Path) -> None:
    """Если отправитель известен и это не владелец — решает он, а не то, что чат разрешён."""
    worker, send_tool, _history = _make_worker(tmp_path)

    await worker._handle(
        Notification(
            type=NotificationType.SPONTANEOUS_PING,
            chat_id=_ALLOWED_GROUP_ID,
            message="напиши первой",
            payload={"sender_id": 505},
        )
    )

    assert send_tool.sent == []


# -- ничего не отправлено: видно в логах, но не записывается в историю -------


async def test_silent_proactive_turn_is_not_persisted_as_a_reply(tmp_path: Path) -> None:
    """
    Модель может закончить ход без вызова инструмента. Тогда собеседник не
    получил ничего — и история не должна утверждать обратное, иначе в
    следующий раз Эфи продолжит с реплики, которой никто не видел.
    """
    worker, send_tool, history = _make_worker(tmp_path, router=_SendingRouter(calls_tool=False))

    await worker._handle(
        Notification(type=NotificationType.SPONTANEOUS_PING, chat_id=_OWNER_ID, message="повод", payload={})
    )

    assert send_tool.sent == []
    assert history.appended == []


async def test_delivered_proactive_turn_is_persisted(tmp_path: Path) -> None:
    worker, _send_tool, history = _make_worker(tmp_path)

    await worker._handle(
        Notification(type=NotificationType.SPONTANEOUS_PING, chat_id=_OWNER_ID, message="повод", payload={})
    )

    assert [message.content for _chat_id, message in history.appended] == ["слушай, а я тут подумала"]


# -- сам предикат ------------------------------------------------------------


def test_lifecycle_allows_only_owner_and_configured_chats(tmp_path: Path) -> None:
    lifecycle = ConversationLifecycle(
        Database(tmp_path / "efi.db", migrations=MIGRATIONS),
        owner_id=_OWNER_ID,
        proactive_chats=(_ALLOWED_GROUP_ID,),
    )

    assert lifecycle.allows_proactive_ping_to_chat(_OWNER_ID) is True
    assert lifecycle.allows_proactive_ping_to_chat(_ALLOWED_GROUP_ID) is True
    assert lifecycle.allows_proactive_ping_to_chat(_STRANGER_CHAT_ID) is False
    assert lifecycle.allows_proactive_ping_to_chat(None) is False
    # Явный отправитель важнее чата — в обе стороны.
    assert lifecycle.allows_proactive_ping_to_chat(_STRANGER_CHAT_ID, _OWNER_ID) is True
    assert lifecycle.allows_proactive_ping_to_chat(_ALLOWED_GROUP_ID, 505) is False


# -- закрытие обещания после доставки напоминания ----------------------------


async def test_delivered_follow_up_closes_the_promise(tmp_path: Path) -> None:
    """
    Замыкание цикла: сработало напоминание -> сообщение ушло -> обещание
    закрыто. Раньше обещание висело в состоянии навсегда, даже когда всё
    остальное отрабатывало.
    """
    from datetime import UTC, datetime, timedelta

    from efi.memory.working_memory import WorkingMemory

    working_memory = WorkingMemory(tmp_path / "wm.json")
    await working_memory.add_item(
        "написать про собеседование", due_at=datetime.now(UTC) + timedelta(minutes=10), chat_id=_OWNER_ID
    )

    worker, send_tool, _history = _make_worker(tmp_path, promises=working_memory)
    await worker._handle(
        Notification(
            type=NotificationType.FOLLOW_UP,
            chat_id=_OWNER_ID,
            message="пора написать",
            payload={"promise_text": "написать про собеседование", "reminder_id": 1},
        )
    )

    assert send_tool.sent, "напоминание обязано дойти до собеседника"
    assert (await working_memory.load()).items[0].done is True


async def test_undelivered_follow_up_keeps_the_promise_open(tmp_path: Path) -> None:
    """
    Модель промолчала — значит, обещание НЕ выполнено. Закрыть его здесь
    значило бы записать невыполненное как сделанное, и человек не получил бы
    ни сообщения, ни следа о том, что она задолжала.
    """
    from datetime import UTC, datetime, timedelta

    from efi.memory.working_memory import WorkingMemory

    working_memory = WorkingMemory(tmp_path / "wm.json")
    await working_memory.add_item(
        "написать про собеседование", due_at=datetime.now(UTC) + timedelta(minutes=10), chat_id=_OWNER_ID
    )

    worker, send_tool, _history = _make_worker(
        tmp_path, router=_SendingRouter(calls_tool=False), promises=working_memory
    )
    await worker._handle(
        Notification(
            type=NotificationType.FOLLOW_UP,
            chat_id=_OWNER_ID,
            message="пора написать",
            payload={"promise_text": "написать про собеседование"},
        )
    )

    assert send_tool.sent == []
    assert (await working_memory.load()).items[0].done is False
