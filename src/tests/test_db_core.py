"""
Тесты efi.db.core.Database.

Главная проверка — про утечку соединений на неудачных попытках. aiosqlite
держит под каждое соединение отдельный НЕ-daemon-поток, поэтому брошенное
соединение не просто занимает память: процесс с таким потоком не завершается
вообще, а выглядит это как «приложение отработало и зависло на выходе».
"""

from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import aiosqlite
import pytest

from efi.db.core import Database
from efi.db.models import MIGRATIONS

_WORKER_THREAD_MARKER = "_connection_worker_thread"


def _worker_threads() -> set[threading.Thread]:
    return {thread for thread in threading.enumerate() if _WORKER_THREAD_MARKER in thread.name}


async def _settle() -> None:
    """Поток aiosqlite замечает закрытие не мгновенно — он опрашивает очередь раз в 0.1 с."""
    for _ in range(30):
        if not _worker_threads():
            return
        await asyncio.sleep(0.05)


async def test_connection_is_closed_after_use(tmp_path: Path) -> None:
    before = _worker_threads()
    database = Database(tmp_path / "efi.db", migrations=MIGRATIONS)
    async with database.connection() as conn:
        await conn.execute("SELECT 1")
    await _settle()
    assert _worker_threads() <= before


async def test_failed_pragma_does_not_leak_a_connection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    Регрессия: раньше сбой на PRAGMA уносил с собой открытое соединение —
    `_acquire` уходил на следующую попытку, а поток от предыдущей оставался
    жить до конца процесса. На практике это ловилось при двух одновременных
    первых обращениях к свежей базе (PRAGMA journal_mode=WAL берёт
    исключительную блокировку, второй вызов получал "database is locked").
    """
    original_execute = aiosqlite.Connection.execute

    async def failing_execute(self: aiosqlite.Connection, sql: str, *args: object, **kwargs: object):  # type: ignore[no-untyped-def]
        if "journal_mode" in sql:
            raise aiosqlite.OperationalError("database is locked")
        return await original_execute(self, sql, *args, **kwargs)

    monkeypatch.setattr(aiosqlite.Connection, "execute", failing_execute)

    before = _worker_threads()
    database = Database(
        tmp_path / "efi.db", migrations=MIGRATIONS, acquire_retries=3, acquire_retry_delay_seconds=0.0
    )
    with pytest.raises(aiosqlite.OperationalError):
        async with database.connection():
            pass

    await _settle()
    assert _worker_threads() <= before


async def test_failed_migration_does_not_leak_a_connection(tmp_path: Path) -> None:
    async def broken_migration(conn: aiosqlite.Connection) -> None:
        raise RuntimeError("миграция не применилась")

    before = _worker_threads()
    database = Database(tmp_path / "efi.db", migrations=[broken_migration])
    with pytest.raises(RuntimeError):
        async with database.connection():
            pass

    await _settle()
    assert _worker_threads() <= before


async def test_concurrent_first_touch_of_a_fresh_database(tmp_path: Path) -> None:
    """
    Несколько запросов, одновременно пришедших к ещё не созданной базе
    (типичный случай для дашборда, который собирает раздел через gather),
    должны отработать и не оставить за собой ни одного потока.
    """
    before = _worker_threads()
    database = Database(tmp_path / "efi.db", migrations=MIGRATIONS)

    results = await asyncio.gather(*(database.fetch_all("SELECT COUNT(*) AS total FROM messages") for _ in range(6)))
    assert [dict(rows[0])["total"] for rows in results] == [0] * 6

    await _settle()
    assert _worker_threads() <= before
