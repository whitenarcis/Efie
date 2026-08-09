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

_BELIEFS_SCHEMA = """
CREATE TABLE IF NOT EXISTS beliefs (
    topic             TEXT PRIMARY KEY,
    stance            TEXT NOT NULL,
    confidence_score  REAL NOT NULL DEFAULT 0.5,
    origin_date       TEXT NOT NULL
);
"""

_CHAT_AFFINITY_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_affinity (
    chat_id       INTEGER PRIMARY KEY,
    affinity      REAL NOT NULL DEFAULT 0.5,
    respect_level REAL NOT NULL DEFAULT 0.5,
    updated_at    TEXT NOT NULL
);
"""

_CURIOSITY_SEEDS_SCHEMA = """
CREATE TABLE IF NOT EXISTS curiosity_seeds (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    topic           TEXT NOT NULL,
    source_chat_id  INTEGER,                          -- NULL, если тема пришла не из конкретного чата
    weight          REAL NOT NULL DEFAULT 0.5,
    created_at      TEXT NOT NULL,
    status          TEXT NOT NULL DEFAULT 'pending'    -- pending | researched
);
CREATE INDEX IF NOT EXISTS idx_curiosity_seeds_status_weight ON curiosity_seeds (status, weight DESC);
"""


_PEOPLE_SCHEMA = """
CREATE TABLE IF NOT EXISTS people (
    user_id        INTEGER PRIMARY KEY,               -- Telegram user_id, а не chat_id
    display_name   TEXT NOT NULL DEFAULT '',
    affinity       REAL NOT NULL DEFAULT 0.5,
    respect_level  REAL NOT NULL DEFAULT 0.5,
    message_count  INTEGER NOT NULL DEFAULT 0,
    first_seen_at  TEXT NOT NULL,
    last_seen_at   TEXT NOT NULL,
    last_chat_id   INTEGER,                           -- где видела в последний раз
    last_chat_title TEXT,                             -- NULL для лички
    impression     TEXT NOT NULL DEFAULT ''           -- сформированное отношение, свободный текст
);
CREATE INDEX IF NOT EXISTS idx_people_last_seen ON people (last_seen_at DESC);
"""


_SOCIAL_INTERACTIONS_SCHEMA = """
CREATE TABLE IF NOT EXISTS social_interactions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    kind           TEXT NOT NULL,        -- значение efi.memory.social_memory.SocialInteractionKind
    chat_id        INTEGER,              -- канал/группа/ЛС, где это произошло
    thread_id      INTEGER,              -- id обсуждения под постом, если это тред
    peer_user_id   INTEGER,              -- с кем именно (NULL для чтения треда «вообще»)
    peer_name      TEXT NOT NULL DEFAULT '',
    text           TEXT NOT NULL,
    tags           TEXT NOT NULL DEFAULT '',  -- пробел-разделённые метатеги (#public_comment и т.п.)
    created_at     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_social_interactions_created ON social_interactions (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_social_interactions_peer ON social_interactions (peer_user_id);
CREATE INDEX IF NOT EXISTS idx_social_interactions_chat ON social_interactions (chat_id, thread_id);
"""

_CONVERSATION_STATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversation_state (
    peer_user_id    INTEGER NOT NULL,
    chat_id         INTEGER NOT NULL,
    annoyance_score REAL NOT NULL DEFAULT 0.0,
    status          TEXT NOT NULL DEFAULT 'active',   -- active | closed
    closed_reason   TEXT NOT NULL DEFAULT '',
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (peer_user_id, chat_id)
);
"""

_THREAD_STATE_SCHEMA = """
CREATE TABLE IF NOT EXISTS thread_state (
    chat_id        INTEGER NOT NULL,
    thread_id      INTEGER NOT NULL,
    commented_at   TEXT,                              -- когда Эфи там уже отписалась (NULL — ещё нет)
    last_seen_at   TEXT NOT NULL,
    PRIMARY KEY (chat_id, thread_id)
);
"""


async def _migration_001_messages(conn: aiosqlite.Connection) -> None:
    await conn.executescript(_MESSAGES_SCHEMA)


async def _migration_002_facts(conn: aiosqlite.Connection) -> None:
    await conn.executescript(_FACTS_SCHEMA)


async def _migration_003_proactive_tasks(conn: aiosqlite.Connection) -> None:
    await conn.executescript(_PROACTIVE_TASKS_SCHEMA)


async def _migration_004_beliefs(conn: aiosqlite.Connection) -> None:
    await conn.executescript(_BELIEFS_SCHEMA)


async def _migration_005_chat_affinity(conn: aiosqlite.Connection) -> None:
    await conn.executescript(_CHAT_AFFINITY_SCHEMA)


async def _migration_006_curiosity_seeds(conn: aiosqlite.Connection) -> None:
    await conn.executescript(_CURIOSITY_SEEDS_SCHEMA)


async def _migration_007_people(conn: aiosqlite.Connection) -> None:
    await conn.executescript(_PEOPLE_SCHEMA)


async def _migration_008_social_interactions(conn: aiosqlite.Connection) -> None:
    await conn.executescript(_SOCIAL_INTERACTIONS_SCHEMA)


async def _migration_009_conversation_state(conn: aiosqlite.Connection) -> None:
    await conn.executescript(_CONVERSATION_STATE_SCHEMA)


async def _migration_010_thread_state(conn: aiosqlite.Connection) -> None:
    await conn.executescript(_THREAD_STATE_SCHEMA)


#: Применяются по порядку при первом получении соединения (см. efi.db.core.Database).
MIGRATIONS = [
    _migration_001_messages,
    _migration_002_facts,
    _migration_003_proactive_tasks,
    _migration_004_beliefs,
    _migration_005_chat_affinity,
    _migration_006_curiosity_seeds,
    _migration_007_people,
    _migration_008_social_interactions,
    _migration_009_conversation_state,
    _migration_010_thread_state,
]

__all__ = ["MIGRATIONS"]
