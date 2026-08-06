"""
efi/db/models.py

Схема БД: таблицы истории сообщений, фактов и проактивных задач, плюс список
миграций для efi.db.core.Database. Каждая миграция — идемпотентная
(CREATE TABLE/INDEX IF NOT EXISTS), поэтому безопасно применяется на каждом
старте приложения без отдельной таблицы версий схемы — для масштаба одного
локального SQLite-файла это оправданный компромисс простоты; при заметном
росте схемы стоит перейти на версионированные миграции с таблицей учёта.
"""

from __future__ import annotations

import aiosqlite

_MESSAGES_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id      INTEGER NOT NULL,
    role         TEXT NOT NULL,
    content      TEXT NOT NULL DEFAULT '',
    tool_call_id TEXT,
    tool_calls   TEXT,              -- JSON-массив ToolCall, если role='assistant' с вызовами инструментов
    created_at   TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_messages_chat_id_created_at ON messages (chat_id, created_at);
"""

_FACTS_SCHEMA = """
CREATE TABLE IF NOT EXISTS facts (
    entity_id   TEXT NOT NULL,
    fact_key    TEXT NOT NULL,
    fact_value  TEXT NOT NULL,
    confidence  REAL NOT NULL DEFAULT 1.0,
    updated_at  TEXT NOT NULL,
    PRIMARY KEY (entity_id, fact_key)
);
CREATE INDEX IF NOT EXISTS idx_facts_entity_id ON facts (entity_id);
"""

_PROACTIVE_TASKS_SCHEMA = """
CREATE TABLE IF NOT EXISTS proactive_tasks (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id       INTEGER,                       -- NULL для задач без привязки к чату (например, NIGHTLY_TASK)
    task_type     TEXT NOT NULL,                 -- значение efi.notifications.schemas.NotificationType
    scheduled_at  TEXT NOT NULL,                 -- когда задачу нужно превратить в Notification
    payload       TEXT NOT NULL DEFAULT '{}',    -- JSON, копируется в Notification.payload при срабатывании
    status        TEXT NOT NULL DEFAULT 'pending',  -- pending | done | cancelled
    created_at    TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_proactive_tasks_status_scheduled_at ON proactive_tasks (status, scheduled_at);
CREATE INDEX IF NOT EXISTS idx_proactive_tasks_chat_id ON proactive_tasks (chat_id);
"""


async def _migration_001_messages(conn: aiosqlite.Connection) -> None:
    await conn.executescript(_MESSAGES_SCHEMA)


async def _migration_002_facts(conn: aiosqlite.Connection) -> None:
    await conn.executescript(_FACTS_SCHEMA)


async def _migration_003_proactive_tasks(conn: aiosqlite.Connection) -> None:
    await conn.executescript(_PROACTIVE_TASKS_SCHEMA)


#: Применяются по порядку при первом получении соединения (см. efi.db.core.Database).
MIGRATIONS = [
    _migration_001_messages,
    _migration_002_facts,
    _migration_003_proactive_tasks,
]

__all__ = ["MIGRATIONS"]
