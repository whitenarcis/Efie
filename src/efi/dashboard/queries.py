"""
efi/dashboard/queries.py

Read-only SQL для табличных разделов дашборда.

Почему не через доменные хранилища (`FactStore`, `CuriosityTracker` и
прочие): у них другой заказчик. `FactStore.all_facts(entity_id)` требует
знать сущность, `CuriosityTracker.pick_top_pending()` отдаёт ровно одно
семя — это ровно то, что нужно личности в разговоре, и совсем не то, что
нужно странице "покажи всё, что накопилось". Дописывать в каждое хранилище
метод "а теперь отдай мне вообще всё, постранично" значило бы тянуть
интерфейсы под нужды витрины; вместо этого дашборд читает те же таблицы
сам, ничего не изменяя.

Ни одна функция здесь не пишет в базу — дашборд наблюдает, а не управляет.
"""

from __future__ import annotations

from typing import Any

import aiosqlite

from efi.db.core import Database

#: Потолок на любую выборку: страница дашборда не должна уметь вытащить в
#: память всю историю переписки одним запросом.
MAX_ROWS = 500


def _rows_to_dicts(rows: list[aiosqlite.Row]) -> list[dict[str, Any]]:
    return [dict(row) for row in rows]


def _limit(value: int, default: int) -> int:
    if value <= 0:
        return default
    return min(value, MAX_ROWS)


async def table_counts(database: Database) -> dict[str, int]:
    """
    Число строк в каждой таблице — верхняя строка обзорной страницы.

    Один запрос вместо десяти отдельных COUNT(*): таблицы маленькие, но
    каждое обращение к `Database` открывает и закрывает своё соединение
    (см. докстринг efi.db.core.Database), и десять открытий ради десяти
    чисел — заметно дороже одного.
    """
    tables = (
        "messages",
        "facts",
        "proactive_tasks",
        "beliefs",
        "chat_affinity",
        "curiosity_seeds",
        "people",
        "social_interactions",
        "conversation_state",
        "thread_state",
    )
    selects = " UNION ALL ".join(f"SELECT '{table}' AS name, COUNT(*) AS total FROM {table}" for table in tables)
    rows = await database.fetch_all(selects)
    counts = {str(row["name"]): int(row["total"]) for row in rows}
    counts["chats"] = await _scalar(database, "SELECT COUNT(DISTINCT chat_id) FROM messages")
    return counts


async def _scalar(database: Database, sql: str, params: tuple[Any, ...] = ()) -> int:
    row = await database.fetch_one(sql, params)
    if row is None:
        return 0
    value = row[0]
    return int(value) if value is not None else 0


async def chats(database: Database, *, limit: int = 100) -> list[dict[str, Any]]:
    """
    Чаты, о которых Эфи вообще что-то знает: объём переписки, когда было
    последнее сообщение, накопленные близость и уважение.
    """
    rows = await database.fetch_all(
        """
        SELECT
            m.chat_id                                        AS chat_id,
            COUNT(*)                                         AS message_count,
            SUM(CASE WHEN m.role = 'assistant' THEN 1 ELSE 0 END) AS assistant_count,
            SUM(CASE WHEN m.role = 'user' THEN 1 ELSE 0 END)      AS user_count,
            MIN(m.created_at)                                AS first_at,
            MAX(m.created_at)                                AS last_at,
            a.affinity                                       AS affinity,
            a.respect_level                                  AS respect_level
        FROM messages AS m
        LEFT JOIN chat_affinity AS a ON a.chat_id = m.chat_id
        GROUP BY m.chat_id
        ORDER BY last_at DESC
        LIMIT ?
        """,
        (_limit(limit, 100),),
    )
    return _rows_to_dicts(rows)


async def messages(database: Database, chat_id: int, *, limit: int = 100, before_id: int | None = None) -> list[
    dict[str, Any]
]:
    """
    Последние сообщения чата в хронологическом порядке (самое старое —
    первым), с необязательной постраничной прокруткой «вглубь» через
    `before_id`.
    """
    params: tuple[Any, ...] = (chat_id,)
    condition = ""
    if before_id is not None:
        condition = "AND id < ?"
        params = (chat_id, before_id)
    rows = await database.fetch_all(
        f"""
        SELECT id, chat_id, role, content, tool_call_id, tool_calls, created_at
        FROM messages
        WHERE chat_id = ? {condition}
        ORDER BY id DESC
        LIMIT ?
        """,
        (*params, _limit(limit, 100)),
    )
    result = _rows_to_dicts(rows)
    result.reverse()
    return result


async def facts(database: Database, *, limit: int = 200, query: str = "") -> list[dict[str, Any]]:
    condition = ""
    params: tuple[Any, ...] = ()
    if query:
        condition = "WHERE entity_id LIKE ? OR fact_key LIKE ? OR fact_value LIKE ?"
        pattern = f"%{query}%"
        params = (pattern, pattern, pattern)
    rows = await database.fetch_all(
        f"""
        SELECT entity_id, fact_key, fact_value, confidence, updated_at
        FROM facts
        {condition}
        ORDER BY updated_at DESC
        LIMIT ?
        """,
        (*params, _limit(limit, 200)),
    )
    return _rows_to_dicts(rows)


async def curiosity_seeds(database: Database, *, limit: int = 100) -> list[dict[str, Any]]:
    rows = await database.fetch_all(
        """
        SELECT id, topic, source_chat_id, weight, created_at, status
        FROM curiosity_seeds
        ORDER BY (status = 'pending') DESC, weight DESC, created_at DESC
        LIMIT ?
        """,
        (_limit(limit, 100),),
    )
    return _rows_to_dicts(rows)


async def beliefs(database: Database, *, limit: int = 200) -> list[dict[str, Any]]:
    rows = await database.fetch_all(
        """
        SELECT topic, stance, confidence_score, origin_date
        FROM beliefs
        ORDER BY confidence_score DESC, topic ASC
        LIMIT ?
        """,
        (_limit(limit, 200),),
    )
    return _rows_to_dicts(rows)


async def people(database: Database, *, limit: int = 100) -> list[dict[str, Any]]:
    rows = await database.fetch_all(
        """
        SELECT user_id, display_name, affinity, respect_level, message_count,
               first_seen_at, last_seen_at, last_chat_id, last_chat_title, impression
        FROM people
        ORDER BY last_seen_at DESC
        LIMIT ?
        """,
        (_limit(limit, 100),),
    )
    return _rows_to_dicts(rows)


async def social_interactions(database: Database, *, limit: int = 100, kind: str = "") -> list[dict[str, Any]]:
    condition = "WHERE kind = ?" if kind else ""
    params: tuple[Any, ...] = (kind,) if kind else ()
    rows = await database.fetch_all(
        f"""
        SELECT id, kind, chat_id, thread_id, peer_user_id, peer_name, text, tags, created_at
        FROM social_interactions
        {condition}
        ORDER BY created_at DESC
        LIMIT ?
        """,
        (*params, _limit(limit, 100)),
    )
    return _rows_to_dicts(rows)


async def conversation_states(database: Database, *, limit: int = 100) -> list[dict[str, Any]]:
    rows = await database.fetch_all(
        """
        SELECT peer_user_id, chat_id, annoyance_score, status, closed_reason, updated_at
        FROM conversation_state
        ORDER BY updated_at DESC
        LIMIT ?
        """,
        (_limit(limit, 100),),
    )
    return _rows_to_dicts(rows)


async def thread_states(database: Database, *, limit: int = 100) -> list[dict[str, Any]]:
    rows = await database.fetch_all(
        """
        SELECT chat_id, thread_id, commented_at, last_seen_at
        FROM thread_state
        ORDER BY last_seen_at DESC
        LIMIT ?
        """,
        (_limit(limit, 100),),
    )
    return _rows_to_dicts(rows)


async def proactive_tasks(database: Database, *, limit: int = 100) -> list[dict[str, Any]]:
    rows = await database.fetch_all(
        """
        SELECT id, chat_id, task_type, scheduled_at, payload, status, created_at
        FROM proactive_tasks
        ORDER BY scheduled_at DESC
        LIMIT ?
        """,
        (_limit(limit, 100),),
    )
    return _rows_to_dicts(rows)


async def messages_per_day(database: Database, *, days: int = 14) -> list[dict[str, Any]]:
    """
    Сколько сообщений приходилось на каждый из последних дней — данные для
    спарклайна активности на обзорной странице.
    """
    rows = await database.fetch_all(
        """
        SELECT substr(created_at, 1, 10) AS day,
               COUNT(*)                  AS total,
               SUM(CASE WHEN role = 'assistant' THEN 1 ELSE 0 END) AS assistant_count
        FROM messages
        GROUP BY day
        ORDER BY day DESC
        LIMIT ?
        """,
        (max(1, min(days, 90)),),
    )
    result = _rows_to_dicts(rows)
    result.reverse()
    return result


__all__ = [
    "MAX_ROWS",
    "beliefs",
    "chats",
    "conversation_states",
    "curiosity_seeds",
    "facts",
    "messages",
    "messages_per_day",
    "people",
    "proactive_tasks",
    "social_interactions",
    "table_counts",
    "thread_states",
]
