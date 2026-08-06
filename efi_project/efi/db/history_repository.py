"""
efi/db/history_repository.py

Конкретная реализация efi.notifications.worker.HistoryRepository поверх
efi.db.core.Database — история диалога по чатам, используемая Worker'ом при
сборке Session перед каждым обращением к LLM.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone

import aiosqlite

from efi.db.core import Database
from efi.llm.schemas import Message, Role, Session, ToolCall

logger = logging.getLogger(__name__)


class SqliteHistoryRepository:
    """
    Хранит историю сообщений в таблице `messages` (см. efi.db.models).

    `get_recent()` — критический путь, вызывается Worker'ом перед каждым
    ответом. `append()`/`clear()` — запись; безопасны для запуска через
    `asyncio.create_task()`, если вызывающей стороне не важно дожидаться
    подтверждения записи немедленно (Worker намеренно ждёт `append()` в конце
    обработки — так следующее событие того же чата гарантированно увидит
    актуальную историю, но в других сценариях fire-and-forget допустим).
    """

    def __init__(self, database: Database) -> None:
        self._database = database

    async def get_recent(self, chat_id: int, limit: int = 20) -> Session:
        """Возвращает последние `limit` сообщений чата в хронологическом порядке (от старых к новым)."""
        rows = await self._database.fetch_all(
            """
            SELECT role, content, tool_call_id, tool_calls FROM (
                SELECT role, content, tool_call_id, tool_calls, created_at, id
                FROM messages
                WHERE chat_id = ?
                ORDER BY created_at DESC, id DESC
                LIMIT ?
            )
            ORDER BY created_at ASC, id ASC
            """,
            (chat_id, limit),
        )
        return Session(messages=[_row_to_message(row) for row in rows])

    async def append(self, chat_id: int, message: Message) -> None:
        """Добавляет одно сообщение в историю чата."""
        tool_calls_json = (
            json.dumps([tool_call.model_dump(mode="json") for tool_call in message.tool_calls])
            if message.tool_calls
            else None
        )
        await self._database.execute(
            """
            INSERT INTO messages (chat_id, role, content, tool_call_id, tool_calls, created_at)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                chat_id,
                message.role.value,
                message.content,
                message.tool_call_id,
                tool_calls_json,
                datetime.now(timezone.utc).isoformat(),
            ),
        )

    async def clear(self, chat_id: int) -> None:
        """Полностью очищает историю чата (например, по команде вида "забудь наш разговор")."""
        await self._database.execute("DELETE FROM messages WHERE chat_id = ?", (chat_id,))


def _row_to_message(row: aiosqlite.Row) -> Message:
    tool_calls_raw = row["tool_calls"]
    tool_calls = [ToolCall.model_validate(item) for item in json.loads(tool_calls_raw)] if tool_calls_raw else []
    return Message(
        role=Role(row["role"]),
        content=row["content"] or "",
        tool_call_id=row["tool_call_id"],
        tool_calls=tool_calls,
    )


__all__ = ["SqliteHistoryRepository"]
