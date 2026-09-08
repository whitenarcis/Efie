"""
Тесты про то, ЧТО за чат стоит за chat_id, и что из этого следует.

Разбор реального случая. В логах:

    spontaneous_ping: queued for chat_id=-2041871692
    worker[0]: proactive spontaneous_ping for chat_id=-2041871692 finished
               without sending anything

Это была не личка, а группа, где у Эфи админка. Сошлись три вещи:

  1. чат попал в allowed_chats (он «свой»), и этого оказалось достаточно,
     чтобы служба спонтанных пингов сочла его собеседником;
  2. у проактивного уведомления payload пустой, поэтому системный промпт
     собирался вообще без блока «[О чате]» — модель не могла узнать, что
     пишет в общий чат, и писала как человеку в личку;
  3. никакой другой источник знания о типе чата к этому моменту не
     существовал: тип приходит только с входящим сообщением Pyrogram, а
     здесь входящего сообщения нет по определению.

Соответственно и проверяется здесь всё три: классификация чата по id,
справочник чатов, восстановление контекста воркером и запрет непрошеной
инициативы в общем чате.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

from efi.behavior.busy_engine import BusyDecision
from efi.db.chat_directory import ChatDescriptor, ChatDirectory
from efi.db.core import Database
from efi.db.models import MIGRATIONS
from efi.llm.schemas import Choice, LLMParams, Message, Response, Role, Session
from efi.notifications.manager import NotificationManager
from efi.notifications.schemas import Notification, NotificationType
from efi.notifications.worker import Worker
from efi.prompts.builder import _build_chat_context_block
from efi.telegram.chat_scope import ChatKind, classify_chat_id, kind_from_chat_type, resolve_chat_kind
from efi.tools.registry import ToolRegistry

#: Тот самый чат из логов — «длинный айди, взявшийся непонятно откуда».
_REAL_GROUP_ID = -2041871692
_OWNER_ID = 2129889949
_SUPERGROUP_ID = -1001234567890


# -- классификация по id ------------------------------------------------------


def test_positive_id_is_a_private_chat() -> None:
    """id личного чата в Telegram совпадает с user_id и всегда положителен."""
    assert classify_chat_id(_OWNER_ID) is ChatKind.PRIVATE
    assert classify_chat_id(_OWNER_ID).is_one_on_one is True


def test_negative_id_is_never_a_private_chat() -> None:
    """Главное свойство: что бы ни было известно про чат, отрицательный id — уже не личка."""
    assert classify_chat_id(_REAL_GROUP_ID) is ChatKind.GROUP
    assert classify_chat_id(_REAL_GROUP_ID).is_one_on_one is False
    # «-100…» — общий формат супергруппы и канала, различить их по id нельзя.
    assert classify_chat_id(_SUPERGROUP_ID) is ChatKind.UNKNOWN
    assert classify_chat_id(_SUPERGROUP_ID).is_one_on_one is False


def test_unknown_chat_is_not_treated_as_private() -> None:
    assert classify_chat_id(None) is ChatKind.UNKNOWN
    assert ChatKind.UNKNOWN.is_one_on_one is False


def test_confirmed_type_beats_the_id() -> None:
    """Тип из Telegram точнее: он отличает супергруппу от канала, id — нет."""
    assert kind_from_chat_type("SUPERGROUP") is ChatKind.GROUP
    assert kind_from_chat_type("CHANNEL") is ChatKind.CHANNEL
    assert kind_from_chat_type("нечто") is None
    assert resolve_chat_kind("CHANNEL", _SUPERGROUP_ID) is ChatKind.CHANNEL
    assert resolve_chat_kind(None, _SUPERGROUP_ID) is ChatKind.UNKNOWN


# -- справочник чатов ---------------------------------------------------------


async def test_directory_remembers_the_chat_between_events(tmp_path: Path) -> None:
    """
    Тип чата виден только во входящем сообщении. Справочник существует ровно
    затем, чтобы проактивное событие — у которого входящего сообщения нет —
    всё равно могло его узнать.
    """
    database = Database(tmp_path / "efi.db", migrations=MIGRATIONS)
    directory = ChatDirectory(database)

    await directory.remember(_REAL_GROUP_ID, chat_type="SUPERGROUP", title="Эфи-лаборатория")

    # Новый экземпляр — то есть чтение из БД, а не из кэша: между пингом и
    # сообщением процесс мог быть перезапущен.
    reread = await ChatDirectory(database).describe(_REAL_GROUP_ID)
    assert reread == ChatDescriptor(chat_id=_REAL_GROUP_ID, chat_type="SUPERGROUP", title="Эфи-лаборатория")
    assert reread.kind is ChatKind.GROUP


async def test_directory_falls_back_to_the_id_for_unknown_chats(tmp_path: Path) -> None:
    """
    Накопленная ДО этой правки история — чаты, про которые в справочнике
    ничего нет. Род чата всё равно обязан определяться, иначе правка не
    чинит ровно тот случай, из-за которого написана.
    """
    directory = ChatDirectory(Database(tmp_path / "efi.db", migrations=MIGRATIONS))

    assert await directory.kind_of(_REAL_GROUP_ID) is ChatKind.GROUP
    assert await directory.kind_of(_OWNER_ID) is ChatKind.PRIVATE


async def test_partial_knowledge_does_not_erase_full(tmp_path: Path) -> None:
    """У лички нет названия, у канала нет отправителя — пустое значение не должно затирать известное."""
    directory = ChatDirectory(Database(tmp_path / "efi.db", migrations=MIGRATIONS))

    await directory.remember(_REAL_GROUP_ID, chat_type="GROUP", title="Эфи-лаборатория")
    await directory.remember(_REAL_GROUP_ID, chat_type=None, title=None)

    assert (await directory.describe(_REAL_GROUP_ID)).title == "Эфи-лаборатория"


# -- блок «[О чате]» ----------------------------------------------------------


def test_chat_context_block_is_built_from_the_id_alone() -> None:
    """
    Именно этого не было: у проактивного уведомления payload пуст, и блок про
    чат не собирался вовсе — модель считала общий чат личной перепиской.
    """
    block = _build_chat_context_block(
        Notification(type=NotificationType.SPONTANEOUS_PING, chat_id=_REAL_GROUP_ID, message="повод", payload={})
    )

    assert "групповой чат" in block
    assert "личная переписка" not in block


def test_chat_context_block_for_a_channel_mentions_the_audience() -> None:
    block = _build_chat_context_block(
        Notification(
            type=NotificationType.USER_MESSAGE,
            chat_id=_SUPERGROUP_ID,
            message="пост",
            payload={"chat_type": "CHANNEL", "chat_title": "Эфи пишет"},
        )
    )

    assert "канал" in block
    assert "Эфи пишет" in block


def test_chat_context_block_stays_empty_without_a_chat() -> None:
    """Ночная задача не привязана к чату — выдумывать ей род чата не из чего."""
    assert _build_chat_context_block(
        Notification(type=NotificationType.NIGHTLY_TASK, chat_id=None, message="консолидация", payload={})
    ) == ""


# -- воркер: контекст чата и запрет непрошеной инициативы ---------------------


class _FakeBusyEngine:
    async def decide(self, chat_id: int | None) -> BusyDecision:
        return BusyDecision(delay_seconds=0.0, is_active_conversation=False)


class _FakeHistory:
    async def get_recent(self, chat_id: int, limit: int = 20) -> Session:
        return Session()

    async def append(self, chat_id: int, message: Message) -> None:
        return None


class _CapturingPromptBuilder:
    """Запоминает уведомление, с которым его позвали: важен не промпт, а то, что воркер про чат знал."""

    def __init__(self) -> None:
        self.seen: Notification | None = None

    async def build(self, notification: Notification, history: Session) -> str:
        self.seen = notification
        return "system prompt"


class _SilentRouter:
    async def chat(self, role: Any, params: LLMParams, session: Session) -> Response:
        return Response(choices=[Choice(message=Message(role=Role.ASSISTANT, content=""))])


def _make_worker(directory: ChatDirectory | None) -> tuple[Worker, _CapturingPromptBuilder]:
    prompt_builder = _CapturingPromptBuilder()
    worker = Worker(
        0,
        NotificationManager(worker_count=1),
        llm_router=_SilentRouter(),  # type: ignore[arg-type]
        tool_registry=ToolRegistry(),
        history=_FakeHistory(),
        system_prompt_builder=prompt_builder,
        busy_engine=_FakeBusyEngine(),  # type: ignore[arg-type]
        chat_directory=directory,
    )
    return worker, prompt_builder


async def test_worker_restores_chat_context_for_a_proactive_event(tmp_path: Path) -> None:
    """
    Регрессия на «думает, что это просто чат с человеком»: к моменту сборки
    промпта в payload обязаны быть тип и название чата, даже если событие
    родилось из таймера с пустым payload.
    """
    directory = ChatDirectory(Database(tmp_path / "efi.db", migrations=MIGRATIONS))
    await directory.remember(_REAL_GROUP_ID, chat_type="SUPERGROUP", title="Эфи-лаборатория")
    worker, prompt_builder = _make_worker(directory)

    await worker._handle(
        Notification(type=NotificationType.FOLLOW_UP, chat_id=_REAL_GROUP_ID, message="повод", payload={})
    )

    assert prompt_builder.seen is not None
    assert prompt_builder.seen.payload["chat_type"] == "SUPERGROUP"
    assert prompt_builder.seen.payload["chat_title"] == "Эфи-лаборатория"


async def test_worker_refuses_an_unprompted_ping_into_a_group_without_any_directory() -> None:
    """
    Справочника может не быть (или чата в нём), но отказ обязан случиться и
    так: род чата виден по самому id. Без этого правка не чинила бы как раз
    те чаты, из-за которых написана — накопленные до неё.
    """
    worker, prompt_builder = _make_worker(None)

    await worker._handle(
        Notification(type=NotificationType.SPONTANEOUS_PING, chat_id=_REAL_GROUP_ID, message="повод", payload={})
    )

    assert prompt_builder.seen is None, "до сборки промпта дело дойти не должно"


async def test_worker_still_pings_a_private_chat() -> None:
    """Запрет ровно про общие чаты: в личке инициатива по-прежнему разрешена."""
    worker, prompt_builder = _make_worker(None)

    await worker._handle(
        Notification(type=NotificationType.SPONTANEOUS_PING, chat_id=_OWNER_ID, message="повод", payload={})
    )

    assert prompt_builder.seen is not None


# -- отбор кандидатов на спонтанный пинг --------------------------------------


class _FakeHistoryWithChats:
    def __init__(self, chat_ids: list[int]) -> None:
        self._chat_ids = chat_ids

    async def get_active_chat_ids(self, *, since: Any) -> list[int]:
        return self._chat_ids


async def test_candidate_chats_exclude_groups(tmp_path: Path) -> None:
    """
    Первый из двух заслонов (второй — гейт воркера выше): группа не должна
    вообще становиться кандидатом, даже будучи вписанной в allowed_chats.
    """
    import efi.app as app_module

    app = app_module.EfiApp.__new__(app_module.EfiApp)
    app._settings = SimpleNamespace(  # type: ignore[assignment]
        telegram=SimpleNamespace(owner_id=_OWNER_ID, allowed_chats=[_REAL_GROUP_ID, 505])
    )
    app._history = _FakeHistoryWithChats([_OWNER_ID, _REAL_GROUP_ID, 505, 909])  # type: ignore[assignment]
    app._chat_directory = ChatDirectory(Database(tmp_path / "efi.db", migrations=MIGRATIONS))

    # 909 отсеивается как чужой чат, _REAL_GROUP_ID — как группа;
    # личка владельца остаётся кандидатом, не будучи перечисленной явно.
    assert await app._active_chat_candidates() == [_OWNER_ID, 505]
