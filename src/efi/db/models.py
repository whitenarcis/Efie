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

from pathlib import Path

import aiosqlite

#: DDL хранилища знаний живёт отдельным файлом — см. _migration_011_knowledge.
_SCHEMA_SQL_PATH = Path(__file__).resolve().parent / "schema.sql"

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
    turns           INTEGER NOT NULL DEFAULT 0,       -- сколько реплик написал собеседник
    updated_at      TEXT NOT NULL,
    PRIMARY KEY (peer_user_id, chat_id)
);
"""

_DEV_TASKS_SCHEMA = """
CREATE TABLE IF NOT EXISTS dev_tasks (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    chat_id     INTEGER,                          -- куда рассказывать о ходе работы; NULL — некому
    idea        TEXT NOT NULL,                    -- замысел словами, как он был сформулирован
    is_collab   INTEGER NOT NULL DEFAULT 0,       -- 1 = проект заказал человек, 0 = своя затея
    status      TEXT NOT NULL DEFAULT 'pending',  -- значение efi.dev.schemas.DevTaskStatus
    spec        TEXT NOT NULL DEFAULT '',         -- JSON ProjectSpec; пусто, пока не спроектировано
    repo_url    TEXT NOT NULL DEFAULT '',
    error       TEXT NOT NULL DEFAULT '',
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_dev_tasks_status ON dev_tasks (status, created_at);
"""

_CHAT_DIRECTORY_SCHEMA = """
CREATE TABLE IF NOT EXISTS chat_directory (
    chat_id     INTEGER PRIMARY KEY,
    chat_type   TEXT NOT NULL DEFAULT '',   -- имя pyrogram.enums.ChatType: PRIVATE/GROUP/SUPERGROUP/CHANNEL
    title       TEXT NOT NULL DEFAULT '',   -- пусто для лички: у неё нет названия
    updated_at  TEXT NOT NULL
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


async def _migration_011_knowledge(conn: aiosqlite.Connection) -> None:
    """
    Строгое хранилище знаний — DDL лежит в efi/db/schema.sql (см. его шапку
    про домены C/P/H и про то, почему оно вынесено в отдельный файл).
    """
    await conn.executescript(_SCHEMA_SQL_PATH.read_text(encoding="utf-8"))


#: Домен по умолчанию для каждой таблицы памяти, у которой его исторически не
#: было. Значения выбраны по природе таблицы, а не «чтобы не было NULL»:
#:   facts               — тройки о сущностях, то есть модель людей -> 'P';
#:   beliefs             — взгляды на мир; про людей их пишет PeopleStore и
#:                         сам проставляет 'P' -> по умолчанию 'C';
#:   social_interactions — журнал прожитого Эфи опыта -> всегда 'H';
#:   curiosity_seeds     — «хочу разобраться в теме», знание о мире -> 'C'.
#:
#: Таблиц `people`, `chat_affinity`, `conversation_state`, `thread_state`
#: здесь намеренно нет: это не память о содержании, а состояние отношений и
#: поведения. Их домен постоянен по построению ('P' у первых двух), и колонка
#: с одним и тем же значением в каждой строке ничего бы не давала ни поиску,
#: ни фильтрации — только создавала бы вид, что он там варьируется.
_DOMAIN_DEFAULTS = {
    "facts": "P",
    "beliefs": "C",
    "social_interactions": "H",
    "curiosity_seeds": "C",
}


async def _migration_012_memory_domains(conn: aiosqlite.Connection) -> None:
    """
    Добавляет обязательный `domain` в таблицы памяти, созданные до появления
    доменной маршрутизации.

    Через ALTER TABLE с ручной проверкой наличия колонки, а не через
    CREATE TABLE IF NOT EXISTS: у людей уже есть базы с накопленным дневником
    и историей, и пересоздавать таблицы ради новой колонки означало бы либо
    потерю данных, либо миграцию с копированием — несоразмерно тому, что
    ALTER TABLE ADD COLUMN в SQLite мгновенен и не переписывает файл.
    `ADD COLUMN` не идемпотентен (второй раз падает с "duplicate column
    name"), поэтому наличие колонки проверяется через PRAGMA.
    """
    for table, default_domain in _DOMAIN_DEFAULTS.items():
        if await _has_column(conn, table, "domain"):
            continue
        await conn.execute(
            f"ALTER TABLE {table} ADD COLUMN domain TEXT NOT NULL DEFAULT '{default_domain}'"  # noqa: S608
        )
        await conn.execute(f"CREATE INDEX IF NOT EXISTS idx_{table}_domain ON {table} (domain)")


async def _has_column(conn: aiosqlite.Connection, table: str, column: str) -> bool:
    # Имя таблицы подставляется в SQL текстом: PRAGMA не принимает параметры,
    # а сами имена — литералы из _DOMAIN_DEFAULTS, а не внешний ввод.
    async with conn.execute(f"PRAGMA table_info({table})") as cursor:  # noqa: S608
        rows = await cursor.fetchall()
    return any(row[1] == column for row in rows)


async def _migration_013_conversation_turns(conn: aiosqlite.Connection) -> None:
    """
    Счётчик реплик собеседника в `conversation_state`.

    Нужен, чтобы отличить «разговор исчерпан» от «разговора ещё не было»:
    без него любое совпадение с маркером прощания на ПЕРВОМ же сообщении
    незнакомца закрывало диалог навсегда — человек не получал ни одного
    ответа (см. GREETING_GRACE_TURNS в efi/behavior/conversation_lifecycle.py).

    Тем, кто уже переписывался, ставится 0, а не «сколько-то»: настоящего
    числа взять неоткуда, а 0 всего лишь даёт им те же две реплики форы,
    что и новичкам. Ошибка в безопасную сторону.
    """
    if not await _has_column(conn, "conversation_state", "turns"):
        await conn.execute("ALTER TABLE conversation_state ADD COLUMN turns INTEGER NOT NULL DEFAULT 0")


async def _migration_014_chat_directory(conn: aiosqlite.Connection) -> None:
    """
    Справочник чатов (см. efi/db/chat_directory.py): что за чат стоит за
    chat_id — личка, группа или канал.

    Заполняется по мере того, как в чатах приходят сообщения; для уже
    накопленной истории таблица останется пустой, и род чата будет
    выводиться из самого id (efi.telegram.chat_scope.classify_chat_id) —
    задним числом восстановить тип неоткуда, а на главный вопрос («личка или
    нет») id отвечает и без справочника.
    """
    await conn.executescript(_CHAT_DIRECTORY_SCHEMA)


async def _migration_015_dev_tasks(conn: aiosqlite.Connection) -> None:
    """
    Очередь задач разработки (см. efi/dev/store.py).

    Отдельная таблица, а не `proactive_tasks`: у той жизненный цикл «сработать
    в назначенный момент», а здесь — конвейер от замысла до запушенного
    репозитория, со своим статусом и спекой проекта.
    """
    await conn.executescript(_DEV_TASKS_SCHEMA)


async def _migration_016_dev_reviews(conn: aiosqlite.Connection) -> None:
    """
    Когда Эфи последний раз возвращалась к своему проекту и сколько правок
    внесла с тех пор (см. efi/dev/maintenance.py).

    ALTER TABLE, а не пересоздание: у того, кто уже включил разработку, в
    таблице лежат живые задачи со спеками и ссылками.
    """
    if not await _has_column(conn, "dev_tasks", "reviewed_at"):
        await conn.execute("ALTER TABLE dev_tasks ADD COLUMN reviewed_at TEXT NOT NULL DEFAULT ''")
    if not await _has_column(conn, "dev_tasks", "revisions"):
        await conn.execute("ALTER TABLE dev_tasks ADD COLUMN revisions INTEGER NOT NULL DEFAULT 0")


async def _migration_017_dev_attempts(conn: aiosqlite.Connection) -> None:
    """
    Сколько раз конвейер уже брался за эту задачу (см. efi/dev/worker.py).

    Без счётчика любой временный отказ — 429 на третьем файле из четырёх,
    оборванная сеть на пуше — хоронил проект навсегда: задача уходила в
    «не вышло» и больше не поднималась. На бесплатных тирах это самый частый
    конец работы, и он не имеет отношения ни к качеству замысла, ни к коду.
    """
    if not await _has_column(conn, "dev_tasks", "attempts"):
        await conn.execute("ALTER TABLE dev_tasks ADD COLUMN attempts INTEGER NOT NULL DEFAULT 0")


async def _migration_018_dev_artifacts(conn: aiosqlite.Connection) -> None:
    """
    Файлы, уже написанные по этой задаче (см. efi/dev/worker.py).

    Без них повторный заход переписывал проект с нуля: те же запросы к тому
    же исчерпанному лимиту и новый шанс разойтись с тем, что в прошлый раз
    уже сходилось. Работа, которая пережила заход, должна пережить и его
    провал.
    """
    if not await _has_column(conn, "dev_tasks", "artifacts"):
        await conn.execute("ALTER TABLE dev_tasks ADD COLUMN artifacts TEXT NOT NULL DEFAULT ''")


async def _migration_019_dev_revivals(conn: aiosqlite.Connection) -> None:
    """
    Сколько раз Эфи возвращалась к брошенному проекту (см. efi/dev/worker.py).

    Счётчик, а не флаг: возвращаться стоит, но не бесконечно. Замысел, который
    не собрался трижды подряд в двух заходах через день, — это уже не «не
    повезло с лимитами», и десятый круг по нему стоит квоты, за которую можно
    написать что-то новое.
    """
    if not await _has_column(conn, "dev_tasks", "revivals"):
        await conn.execute("ALTER TABLE dev_tasks ADD COLUMN revivals INTEGER NOT NULL DEFAULT 0")


async def _migration_020_dev_swe(conn: aiosqlite.Connection) -> None:
    """
    Работа с чужим кодом в той же очереди, что и свои проекты
    (см. efi/dev/swe_engine.py).

    Очередь одна намеренно: и то и другое — её работа, у неё общий счётчик
    заходов, общая занятость и общее место в дашборде. А вот конвейеры разные,
    и `kind` — то, что не даёт SWE-задаче случайно уехать в конвейер
    собственных проектов, где её попытались бы спроектировать с нуля.

    По умолчанию 'project': всё, что уже лежит в таблице, — это проекты.
    """
    if not await _has_column(conn, "dev_tasks", "kind"):
        await conn.execute("ALTER TABLE dev_tasks ADD COLUMN kind TEXT NOT NULL DEFAULT 'project'")
    if not await _has_column(conn, "dev_tasks", "source"):
        await conn.execute("ALTER TABLE dev_tasks ADD COLUMN source TEXT NOT NULL DEFAULT ''")
    if not await _has_column(conn, "dev_tasks", "branch"):
        await conn.execute("ALTER TABLE dev_tasks ADD COLUMN branch TEXT NOT NULL DEFAULT ''")


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
    _migration_011_knowledge,
    _migration_012_memory_domains,
    _migration_013_conversation_turns,
    _migration_014_chat_directory,
    _migration_015_dev_tasks,
    _migration_016_dev_reviews,
    _migration_017_dev_attempts,
    _migration_018_dev_artifacts,
    _migration_019_dev_revivals,
    _migration_020_dev_swe,
]

__all__ = ["MIGRATIONS"]
