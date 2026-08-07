"""
efi/db/history_repository.py

Конкретная реализация efi.notifications.worker.HistoryRepository поверх
efi.db.core.Database — история диалога по чатам, используемая Worker'ом при
сборке Session перед каждым обращением к LLM.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone

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

    async def get_active_chat_ids(self, *, since: datetime) -> list[int]:
        """
        Список chat_id, где было хоть одно сообщение начиная с `since`.
        Используется ночной новеллизацией памяти (efi.memory.consolidation.
        DiaryConsolidator.novelize_recent_history) для обхода всех чатов, где
        вообще была активность — не критический путь ответа.
        """
        rows = await self._database.fetch_all(
            "SELECT DISTINCT chat_id FROM messages WHERE created_at >= ?",
            (since.isoformat(),),
        )
        return [row["chat_id"] for row in rows]

    async def get_since(self, chat_id: int, *, since: datetime) -> Session:
        """
        Все сообщения чата начиная с `since`, в хронологическом порядке — БЕЗ
        ограничения по количеству (в отличие от get_recent). Используется
        только ночной новеллизацией для конкретного чата за один проход, не
        критическим путём ответа, поэтому отсутствие лимита здесь не
        раздувает промпт обычных ответов.
        """
        rows = await self._database.fetch_all(
            """
            SELECT role, content, tool_call_id, tool_calls
            FROM messages
            WHERE chat_id = ? AND created_at >= ?
            ORDER BY created_at ASC, id ASC
            """,
            (chat_id, since.isoformat()),
        )
        return Session(messages=[_row_to_message(row) for row in rows])

    async def prune_old_messages(self, *, older_than_days: int = 90, keep_last_per_chat: int = 200) -> int:
        """
        Удаляет старые сообщения — без этого таблица `messages` растёт
        бесконечно на диске (не в контексте LLM: `get_recent()` всегда
        возвращает не больше `limit` последних сообщений на запрос, промпт
        не раздувается сам по себе — раздувается только файл БД со временем).

        Порог двойной, оба условия должны выполниться одновременно, чтобы не
        потерять недавнюю историю активного чата: сообщение удаляется, только
        если ОНО СТАРШЕ `older_than_days` И вне последних `keep_last_per_chat`
        сообщений СВОЕГО чата (не общего счётчика по всем чатам — иначе
        активный чат вытеснял бы историю тихого). Возвращает число удалённых строк.
        """
        cutoff = (datetime.now(timezone.utc) - timedelta(days=older_than_days)).isoformat()
        result = await self._database.execute_and_count_changes(
            """
            DELETE FROM messages
            WHERE created_at < ?
              AND id NOT IN (
                  SELECT id FROM messages AS recent
                  WHERE recent.chat_id = messages.chat_id
                  ORDER BY recent.created_at DESC, recent.id DESC
                  LIMIT ?
              )
            """,
            (cutoff, keep_last_per_chat),
        )
        if result:
            logger.info("history: pruned %d messages older than %d days", result, older_than_days)
        return result


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
