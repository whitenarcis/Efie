"""
efi/db/core.py

Единый менеджер подключений aiosqlite для всего приложения: WAL-режим,
busy_timeout (защита от SQLITE_BUSY при параллельных Worker'ах, пишущих в
одну БД), ретраи на этапе получения соединения и однократный прогон миграций.

ОДНО долгоживущее соединение на весь процесс, а не по соединению на запрос.
Так было не всегда: раньше `connection()` открывал новое подключение под
каждый вызов и закрывал его следом. Формально корректно — в WAL-режиме
параллельные соединения к одному файлу допустимы, — но каждое подключение
aiosqlite поднимает СОБСТВЕННЫЙ поток и заново прогоняет PRAGMA
(journal_mode, busy_timeout, foreign_keys). На замере это давало 2.7 мс на
точечное чтение вместо десятых долей миллисекунды: полсотни лишних затрат на
каждый запрос, а запросов на один ответ собеседнику — под десяток, плюс
фоновые службы. На телефоне, где всё это и живёт, разница ощутима.

Доступ сериализован одним `asyncio.Lock`. Это не потеря: SQLite и так
допускает ровно одного писателя, aiosqlite исполняет всё на своём единственном
потоке соединения, а локальное чтение измеряется микросекундами — при таких
величинах очередь дешевле, чем открытие нового подключения. Зато блок
`async with db.connection()` из нескольких операторов теперь по-настоящему
атомарен: раньше два таких блока могли переплестись через разные соединения.

Известный баг (см. память проекта): retry-логика вокруг получения соединения
ломается, если `try/except` вокруг acquire охватывает и сам `yield` внутри
`@asynccontextmanager` — тогда исключение, брошенное КОДОМ ВНУТРИ `async with`
блока (то есть бизнес-логикой вызывающей стороны, а не проблемой соединения),
Python бросает обратно в генератор ИМЕННО в точке yield, и такой except
ошибочно трактует чужую ошибку как сбой соединения, требующий повторной
попытки. Здесь ретраи строго ограничены фазой ПОЛУЧЕНИЯ соединения
(`_acquire`, вызывается ДО yield); вокруг самого `yield` — только `finally`,
без единого `except`.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import aiofiles.os
import aiosqlite

logger = logging.getLogger(__name__)

_DEFAULT_BUSY_TIMEOUT_MS = 5_000
_DEFAULT_ACQUIRE_RETRIES = 3
_DEFAULT_ACQUIRE_RETRY_DELAY_SECONDS = 0.2

#: Экземпляры Database с ещё не закрытым соединением.
#:
#: Ссылки СИЛЬНЫЕ, и это не недосмотр. Слабые здесь не работают по существу:
#: объект, который вот-вот соберёт сборщик мусора, — это ровно тот объект,
#: чьё соединение надо закрыть, а из WeakSet он к этому моменту уже исчез.
#: Поток aiosqlite при этом остаётся жить и продолжает ломиться в закрытый
#: event loop — что и наблюдалось сотней предупреждений на прогоне тестов.
#:
#: Утечки из-за сильных ссылок нет: запись удаляется в `close()`, а сам
#: Database — небольшой объект (путь, список миграций, лок). В приложении он
#: вообще один на процесс.
_OPEN_DATABASES: set[Database] = set()


async def close_all_databases() -> None:
    """
    Закрывает соединения всех незакрытых Database.

    Для тестов и разовых скриптов, где владельца нет. У приложения владелец
    есть (`EfiApp.stop`), и полагаться на эту функцию ему незачем.
    """
    for database in list(_OPEN_DATABASES):
        await database.close()



#: Одна миграция — идемпотентная функция, применяющая свою часть схемы к соединению.
Migration = Callable[[aiosqlite.Connection], Awaitable[None]]


class Database:
    """
    Владеет путём к файлу БД, списком миграций и ЕДИНСТВЕННЫМ соединением.

    `connection()` отдаёт это соединение под локом: параллельные вызовы ждут
    своей очереди, а не открывают каждый своё. Обоснование — в докстринге
    модуля; коротко: открытие подключения стоит на два порядка дороже самого
    запроса, а сериализация локального SQLite не стоит почти ничего.

    `close()` обязателен при остановке приложения: соединение aiosqlite держит
    НЕ-daemon-поток, и незакрытое не даёт процессу завершиться.
    """

    def __init__(
        self,
        db_path: Path,
        *,
        # Sequence, а не list: list инвариантен, и список конкретных миграций
        # (list[Callable[..., Coroutine[...]]]) под list[Migration] не подходит.
        migrations: Sequence[Migration] | None = None,
        busy_timeout_ms: int = _DEFAULT_BUSY_TIMEOUT_MS,
        acquire_retries: int = _DEFAULT_ACQUIRE_RETRIES,
        acquire_retry_delay_seconds: float = _DEFAULT_ACQUIRE_RETRY_DELAY_SECONDS,
    ) -> None:
        self._db_path = db_path
        self._migrations: Sequence[Migration] = migrations or []
        self._busy_timeout_ms = busy_timeout_ms
        self._acquire_retries = acquire_retries
        self._acquire_retry_delay_seconds = acquire_retry_delay_seconds
        self._migrations_lock = asyncio.Lock()
        self._migrations_applied = False
        #: Единственное соединение и лок на него. Лок защищает не только сам
        #: доступ, но и создание: без него два одновременных первых запроса
        #: открыли бы по соединению, и одно осталось бы бесхозным вместе со
        #: своим потоком.
        self._conn: aiosqlite.Connection | None = None
        self._lock = asyncio.Lock()


    @asynccontextmanager
    async def connection(self) -> AsyncIterator[aiosqlite.Connection]:
        """
        Выдаёт общее соединение (WAL/busy_timeout выставлены, миграции
        применены) под локом. Ретраи acquire — целиком до `yield`; тело
        вызывающего `async with`-блока не может спровоцировать повторную
        попытку получения соединения, даже если само бросит исключение.

        Соединение НЕ закрывается по выходе из блока: оно общее и живёт до
        `close()`. А вот если работа с ним сорвалась по OperationalError
        (файл увели, диск отвалился), соединение считается непригодным и
        сбрасывается — следующий вызов откроет новое вместо того, чтобы
        вечно спотыкаться об одно и то же сломанное.
        """
        async with self._lock:
            conn = await self._acquire_shared()
            try:
                yield conn
            except aiosqlite.OperationalError:
                await self._discard(conn)
                raise

    async def _acquire_shared(self) -> aiosqlite.Connection:
        """Общее соединение, открывая его при первом обращении. Зовётся только под локом."""
        if self._conn is not None:
            return self._conn
        self._conn = await self._acquire()
        _OPEN_DATABASES.add(self)
        return self._conn

    async def _discard(self, conn: aiosqlite.Connection) -> None:
        """Закрывает сломанное соединение и забывает о нём. Зовётся только под локом."""
        if self._conn is conn:
            self._conn = None
            _OPEN_DATABASES.discard(self)
        try:
            await conn.close()
        except Exception:
            logger.warning("db: не удалось закрыть сломанное соединение", exc_info=True)

    async def close(self) -> None:
        """
        Закрывает общее соединение. Обязателен при остановке приложения:
        aiosqlite держит под соединение НЕ-daemon-поток, и незакрытое
        соединение не даёт процессу завершиться — `threading._shutdown` ждёт
        поток, а тот крутит свой цикл, потому что закрыть его больше некому.
        """
        async with self._lock:
            if self._conn is None:
                return
            conn, self._conn = self._conn, None
            _OPEN_DATABASES.discard(self)
            await conn.close()

    async def _acquire(self) -> aiosqlite.Connection:
        last_error: Exception | None = None
        for attempt in range(1, self._acquire_retries + 1):
            try:
                conn = await self._open_and_prepare()
            except aiosqlite.OperationalError as exc:
                last_error = exc
                logger.warning(
                    "db: acquire attempt %d/%d failed (%s), retrying in %.2fs",
                    attempt, self._acquire_retries, exc, self._acquire_retry_delay_seconds,
                )
                await asyncio.sleep(self._acquire_retry_delay_seconds)
                continue
            try:
                await self._ensure_migrations(conn)
            except BaseException:
                # Соединение уже открыто и, значит, уже держит свой поток
                # (см. комментарий в _open_and_prepare) — если миграции не
                # прошли, закрыть его должны мы: наружу уйдёт исключение, и
                # вызывающая сторона про это соединение уже не узнает.
                await conn.close()
                raise
            return conn
        assert last_error is not None  # цикл всегда либо возвращает, либо оставляет last_error перед выходом
        raise last_error

    async def _open_and_prepare(self) -> aiosqlite.Connection:
        """
        Открывает соединение и выставляет PRAGMA.

        Любой сбой ПОСЛЕ connect() обязан закрыть соединение. aiosqlite под
        каждое соединение поднимает отдельный НЕ-daemon-поток, и брошенное
        (не закрытое) соединение навсегда оставляет этот поток жить: процесс
        после такого не завершается вообще — `threading._shutdown` ждёт его,
        а тот крутит свой цикл, потому что закрыть его больше некому.

        Ровно это и происходило на ретраях `_acquire`: PRAGMA journal_mode=WAL
        берёт кратковременную исключительную блокировку, и при двух
        одновременных первых обращениях к свежей базе один из вызовов ловил
        "database is locked" — ретрай отрабатывал как задумано, но поток от
        неудачной попытки оставался.
        """
        await aiofiles.os.makedirs(self._db_path.parent, exist_ok=True)
        conn = await aiosqlite.connect(self._db_path)
        try:
            conn.row_factory = aiosqlite.Row
            await conn.execute("PRAGMA journal_mode=WAL")
            await conn.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
            await conn.execute("PRAGMA foreign_keys=ON")
        except BaseException:
            await conn.close()
            raise
        return conn

    async def _ensure_migrations(self, conn: aiosqlite.Connection) -> None:
        if self._migrations_applied:
            return
        async with self._migrations_lock:
            if self._migrations_applied:  # кто-то успел применить, пока мы ждали лок
                return
            for migration in self._migrations:
                await migration(conn)
            await conn.commit()
            self._migrations_applied = True
            logger.info("db: applied %d migration(s) to %s", len(self._migrations), self._db_path)

    # -- удобные шорткаты для одиночных запросов ---------------------------

    async def execute(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        """INSERT/UPDATE/DELETE в одну операцию, с commit, без явного `async with`."""
        async with self.connection() as conn:
            await conn.execute(sql, params)
            await conn.commit()

    async def execute_and_count_changes(self, sql: str, params: tuple[Any, ...] = ()) -> int:
        """
        Как execute(), но возвращает число затронутых строк (aiosqlite.Connection.total_changes) — для отчётности
        задач вроде очистки старых данных.
        """
        async with self.connection() as conn:
            cursor = await conn.execute(sql, params)
            await conn.commit()
            return cursor.rowcount if cursor.rowcount >= 0 else 0

    async def fetch_all(self, sql: str, params: tuple[Any, ...] = ()) -> list[aiosqlite.Row]:
        """SELECT, возвращающий все строки, в одну операцию."""
        async with self.connection() as conn:
            async with conn.execute(sql, params) as cursor:
                return list(await cursor.fetchall())

    async def fetch_one(self, sql: str, params: tuple[Any, ...] = ()) -> aiosqlite.Row | None:
        """SELECT, возвращающий одну (или ни одной) строку, в одну операцию."""
        async with self.connection() as conn:
            async with conn.execute(sql, params) as cursor:
                return await cursor.fetchone()


__all__ = ["Database", "Migration", "close_all_databases"]
