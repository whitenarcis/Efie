"""
efi/dev/store.py

Хранилище задач разработки поверх SQLite.

Почему персистентно. Проект пишется десятками минут: спека, пять-шесть
файлов по несколько запросов каждый, публикация. За это время процесс на
телефоне успевает и упасть, и быть перезапущенным. Задача, живущая в памяти
процесса, в этом случае исчезает молча — а человек, который полчаса назад
договорился с Эфи о совместном проекте, остаётся с обещанием, о котором она
больше не помнит. Ровно тот же разбор, что у efi/behavior/reminders.py: обещание
со сроком обязано пережить перезапуск.

Отдельная таблица, а не `proactive_tasks`: та описывает «когда сработать» и
несёт payload'ом текст напоминания, а здесь нужны статус конвейера, спека
проекта и адрес репозитория — другая форма и другой жизненный цикл (задача
живёт от идеи до пуша, а не до момента срабатывания).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

import aiosqlite
from pydantic import ValidationError

from efi.db.core import Database
from efi.dev.schemas import DevTask, DevTaskStatus, ProjectSpec
from efi.utils.json_utils import compact_json_dumps, safe_json_loads

logger = logging.getLogger(__name__)

#: Статусы, из которых задачу ещё можно двигать, — всё, кроме терминальных.
#: Считаются из перечисления, а не переписываются списком: новый этап
#: конвейера иначе молча выпал бы из «в работе».
_ACTIVE_STATUSES = tuple(status.value for status in DevTaskStatus if not status.is_terminal)

#: Этапы, на которых задачу уже кто-то взял. Именно они могут «зависнуть»,
#: если процесс умер посреди работы, — см. reclaim_stalled.
_IN_FLIGHT_STATUSES = tuple(
    status.value
    for status in DevTaskStatus
    if not status.is_terminal and status is not DevTaskStatus.PENDING
)


class DevTaskStore:
    """
    CRUD задач разработки. Без кэша: обращений мало (раз в цикл фонового
    воркера и раз на инструмент), а вот расхождение кэша со статусом,
    записанным другим экземпляром, стоило бы дорого — задачу могли бы взять
    в работу дважды.
    """

    def __init__(self, database: Database) -> None:
        self._database = database

    async def create(self, idea: str, *, chat_id: int | None = None, is_collab: bool = False) -> DevTask:
        """Заводит новую задачу в статусе PENDING — её подхватит ближайший тик DevWorker."""
        now = datetime.now(UTC)
        normalized = idea.strip()
        # Через connection(), а не execute(): нужен lastrowid, иначе задачу
        # пришлось бы искать обратным запросом по её же полям (тот же приём,
        # что в efi/behavior/reminders.py).
        async with self._database.connection() as conn:
            cursor = await conn.execute(
                """
                INSERT INTO dev_tasks
                    (chat_id, idea, is_collab, status, spec, repo_url, error, created_at, updated_at)
                VALUES (?, ?, ?, ?, '', '', '', ?, ?)
                """,
                (
                    chat_id,
                    normalized,
                    int(is_collab),
                    DevTaskStatus.PENDING.value,
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
            await conn.commit()
            task_id = cursor.lastrowid or 0

        logger.info(
            "dev_store: задача #%d — %s (%s)",
            task_id, normalized[:80], "совместная" if is_collab else "своя затея",
        )
        return DevTask(
            id=task_id,
            chat_id=chat_id,
            idea=normalized,
            is_collab=is_collab,
            status=DevTaskStatus.PENDING,
            created_at=now,
            updated_at=now,
        )

    async def next_pending(self) -> DevTask | None:
        """
        Самая старая задача, к которой конвейер ещё не подходил.

        Именно старая, а не новая: очередь идей — это очередь, и замысел,
        о котором договорились час назад, не должен вечно уступать место
        свежим.
        """
        row = await self._database.fetch_one(
            "SELECT * FROM dev_tasks WHERE status = ? ORDER BY created_at ASC, id ASC LIMIT 1",
            (DevTaskStatus.PENDING.value,),
        )
        return _row_to_task(row) if row is not None else None

    async def reclaim_stalled(self, *, older_than: timedelta) -> list[DevTask]:
        """
        Возвращает в очередь задачи, застрявшие в работе.

        Задача помечается SPECCING/CODING/PUBLISHING на время своего этапа, и
        если процесс в этот момент умер (упал, перезапустили, телефон убил
        фоновую задачу), статус так и останется — то есть задача навсегда
        выпадет из `next_pending`, оставшись при этом в «сейчас в работе».
        Снаружи это выглядит хуже, чем потеря: Эфи месяцами утверждает, что
        пишет проект, к которому никто не подходил.

        Порог по времени, а не безусловный сброс на старте: воркер один, но
        `updated_at` двигается на каждом переходе, и живая задача под порог
        не попадает.
        """
        cutoff = (datetime.now(UTC) - older_than).isoformat()
        placeholders = ", ".join("?" for _ in _IN_FLIGHT_STATUSES)
        rows = await self._database.fetch_all(
            f"SELECT * FROM dev_tasks WHERE status IN ({placeholders}) AND updated_at < ?",  # noqa: S608
            (*_IN_FLIGHT_STATUSES, cutoff),
        )
        reclaimed: list[DevTask] = []
        for row in rows:
            task = _row_to_task(row)
            logger.warning(
                "dev_store: задача #%s зависла в статусе %s с %s — возвращаю в очередь",
                task.id, task.status.value, task.updated_at.isoformat(timespec="minutes"),
            )
            reclaimed.append(await self.update(task, status=DevTaskStatus.PENDING))
        return reclaimed

    async def get(self, task_id: int) -> DevTask | None:
        row = await self._database.fetch_one("SELECT * FROM dev_tasks WHERE id = ?", (task_id,))
        return _row_to_task(row) if row is not None else None

    async def active(self) -> list[DevTask]:
        """Задачи в работе — то, чем Эфи «сейчас занята» (см. промпт и dev-инструменты)."""
        placeholders = ", ".join("?" for _ in _ACTIVE_STATUSES)
        rows = await self._database.fetch_all(
            f"SELECT * FROM dev_tasks WHERE status IN ({placeholders}) ORDER BY created_at ASC",  # noqa: S608
            _ACTIVE_STATUSES,
        )
        return [_row_to_task(row) for row in rows]

    async def recent_releases(self, *, limit: int = 5) -> list[DevTask]:
        """Последние доведённые до репозитория проекты — материал для показа и для «внешнего флекса»."""
        rows = await self._database.fetch_all(
            "SELECT * FROM dev_tasks WHERE status = ? AND repo_url != '' ORDER BY updated_at DESC LIMIT ?",
            (DevTaskStatus.DONE.value, limit),
        )
        return [_row_to_task(row) for row in rows]

    async def has_open_task_for(self, chat_id: int) -> bool:
        """Есть ли в этом чате уже начатая задача — чтобы одна и та же идея не превращалась в три проекта."""
        row = await self._database.fetch_one(
            f"SELECT 1 FROM dev_tasks WHERE chat_id = ? AND status IN "  # noqa: S608
            f"({', '.join('?' for _ in _ACTIVE_STATUSES)}) LIMIT 1",
            (chat_id, *_ACTIVE_STATUSES),
        )
        return row is not None

    async def update(
        self,
        task: DevTask,
        *,
        status: DevTaskStatus | None = None,
        spec: ProjectSpec | None = None,
        repo_url: str | None = None,
        error: str | None = None,
    ) -> DevTask:
        """Сохраняет продвижение задачи. Возвращает обновлённую копию — DevTask иммутабелен по смыслу."""
        updated = task.model_copy(
            update={
                "status": status if status is not None else task.status,
                "spec": spec if spec is not None else task.spec,
                "repo_url": repo_url if repo_url is not None else task.repo_url,
                "error": error if error is not None else task.error,
                "updated_at": datetime.now(UTC),
            }
        )
        await self._database.execute(
            """
            UPDATE dev_tasks
               SET status = ?, spec = ?, repo_url = ?, error = ?, updated_at = ?
             WHERE id = ?
            """,
            (
                updated.status.value,
                compact_json_dumps(updated.spec.model_dump(mode="json")) if updated.spec is not None else "",
                updated.repo_url,
                updated.error,
                updated.updated_at.isoformat(),
                updated.id,
            ),
        )
        return updated


def _row_to_task(row: aiosqlite.Row) -> DevTask:
    """
    Строка БД -> задача. Битая спека (руками поправили JSON, сменилась схема)
    не должна ронять чтение всей очереди: задача без спеки просто вернётся к
    этапу проектирования.
    """
    spec: ProjectSpec | None = None
    raw_spec = str(row["spec"] or "")
    if raw_spec:
        payload = safe_json_loads(raw_spec)
        if isinstance(payload, dict):
            try:
                spec = ProjectSpec.model_validate(payload)
            except ValidationError:
                logger.warning("dev_store: спека задачи #%s не читается, начну с проектирования", row["id"])

    return DevTask(
        id=int(row["id"]),
        chat_id=row["chat_id"],
        idea=str(row["idea"] or ""),
        is_collab=bool(row["is_collab"]),
        status=DevTaskStatus(str(row["status"])),
        spec=spec,
        repo_url=str(row["repo_url"] or ""),
        error=str(row["error"] or ""),
        created_at=datetime.fromisoformat(str(row["created_at"])),
        updated_at=datetime.fromisoformat(str(row["updated_at"])),
    )


__all__ = ["DevTaskStore"]
