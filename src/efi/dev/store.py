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
from efi.dev.schemas import DevTask, DevTaskKind, DevTaskStatus, ProjectSpec
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

    async def create(
        self,
        idea: str,
        *,
        chat_id: int | None = None,
        is_collab: bool = False,
        kind: DevTaskKind = DevTaskKind.PROJECT,
        source: str = "",
    ) -> DevTask:
        """
        Заводит новую задачу в статусе PENDING — её подхватит ближайший тик
        DevWorker.

        `kind` определяет, КАКОЙ конвейер её возьмёт: свой проект с нуля или
        работа с существующим кодом (см. DevTaskKind). `source` осмыслен
        только для второго — это ссылка на репозиторий или путь к нему.
        """
        now = datetime.now(UTC)
        normalized = idea.strip()
        # Через connection(), а не execute(): нужен lastrowid, иначе задачу
        # пришлось бы искать обратным запросом по её же полям (тот же приём,
        # что в efi/behavior/reminders.py).
        async with self._database.connection() as conn:
            cursor = await conn.execute(
                """
                INSERT INTO dev_tasks
                    (chat_id, idea, is_collab, status, spec, repo_url, error, kind, source,
                     created_at, updated_at)
                VALUES (?, ?, ?, ?, '', '', '', ?, ?, ?, ?)
                """,
                (
                    chat_id,
                    normalized,
                    int(is_collab),
                    DevTaskStatus.PENDING.value,
                    kind.value,
                    source,
                    now.isoformat(),
                    now.isoformat(),
                ),
            )
            await conn.commit()
            task_id = cursor.lastrowid or 0

        logger.info(
            "dev_store: задача #%d (%s) — %s (%s)",
            task_id, kind.value, normalized[:80], "совместная" if is_collab else "своя затея",
        )
        return DevTask(
            id=task_id,
            chat_id=chat_id,
            idea=normalized,
            is_collab=is_collab,
            kind=kind,
            source=source,
            status=DevTaskStatus.PENDING,
            created_at=now,
            updated_at=now,
        )

    async def next_pending(self, *, kind: DevTaskKind = DevTaskKind.PROJECT) -> DevTask | None:
        """
        Самая старая задача НУЖНОГО РОДА, к которой конвейер ещё не подходил.

        Именно старая, а не новая: очередь идей — это очередь, и замысел,
        о котором договорились час назад, не должен вечно уступать место
        свежим.

        Род обязателен и по умолчанию «свой проект»: SWE-задача, попавшая в
        конвейер собственных проектов, была бы спроектирована с нуля вместо
        того, чтобы починить чужой импорт.
        """
        row = await self._database.fetch_one(
            "SELECT * FROM dev_tasks WHERE status = ? AND kind = ? ORDER BY created_at ASC, id ASC LIMIT 1",
            (DevTaskStatus.PENDING.value, kind.value),
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

    async def due_for_review(self, *, not_reviewed_for: timedelta) -> list[DevTask]:
        """
        Выложенные проекты, к которым Эфи давно не возвращалась.

        «Давно» считается от последнего просмотра, а если его не было — от
        публикации: свежий проект незачем ревизовать на следующий день после
        релиза, он ровно такой, каким его дописали.

        Наличие ссылки НЕ требуется. Проект, написанный в локальном режиме
        (без токена GitHub), — такой же её проект: он лежит на диске, его
        можно перечитать и поправить, и коммит ляжет в локальную историю.
        Требовать `repo_url` значило бы, что у владельца без токена
        возвращения к своему коду не существует вовсе.
        """
        cutoff = (datetime.now(UTC) - not_reviewed_for).isoformat()
        rows = await self._database.fetch_all(
            """
            SELECT * FROM dev_tasks
             WHERE status = ? AND spec != ''
               AND (CASE WHEN reviewed_at = '' THEN updated_at ELSE reviewed_at END) < ?
             ORDER BY (CASE WHEN reviewed_at = '' THEN updated_at ELSE reviewed_at END) ASC
            """,
            (DevTaskStatus.DONE.value, cutoff),
        )
        return [_row_to_task(row) for row in rows]

    async def mark_reviewed(self, task: DevTask, *, revised: bool = False) -> DevTask:
        """
        Отмечает, что проект просмотрен. `revised` — была ли внесена правка;
        счётчик правок нужен и дашборду, и самой Эфи («я к этой штуке уже
        трижды возвращалась»).

        Отметка ставится ВСЕГДА, включая исход «всё нормально, трогать
        нечего»: без неё один и тот же проект пересматривался бы каждый тик,
        а остальные не дождались бы очереди никогда.
        """
        now = datetime.now(UTC)
        updated = task.model_copy(
            update={
                "reviewed_at": now,
                "revisions": task.revisions + (1 if revised else 0),
                "updated_at": now if revised else task.updated_at,
            }
        )
        await self._database.execute(
            "UPDATE dev_tasks SET reviewed_at = ?, revisions = ?, updated_at = ? WHERE id = ?",
            (now.isoformat(), updated.revisions, updated.updated_at.isoformat(), task.id),
        )
        return updated

    async def recent_releases(self, *, limit: int = 5) -> list[DevTask]:
        """Последние доведённые до репозитория проекты — материал для показа и для «внешнего флекса»."""
        rows = await self._database.fetch_all(
            "SELECT * FROM dev_tasks WHERE status = ? AND repo_url != '' ORDER BY updated_at DESC LIMIT ?",
            (DevTaskStatus.DONE.value, limit),
        )
        return [_row_to_task(row) for row in rows]

    async def finished_projects(self, *, limit: int = 20) -> list[DevTask]:
        """
        Всё, что уже написано и доведено до конца, со ссылкой или без.

        Нужно в двух местах, и оба про качество замысла: в промпт
        проектирования («вот это ты уже писала, придумай другое») и в проверку
        имени репозитория. Без первого она раз в неделю придумывает тот же
        разборщик логов, без второго — второй такой проект не публикуется
        вовсе: пуш в непустой репозиторий отклоняется.
        """
        rows = await self._database.fetch_all(
            "SELECT * FROM dev_tasks WHERE status = ? AND spec != '' ORDER BY updated_at DESC LIMIT ?",
            (DevTaskStatus.DONE.value, limit),
        )
        return [_row_to_task(row) for row in rows]

    async def taken_slugs(self) -> set[str]:
        """Имена репозиториев, которые уже заняты её же проектами, — в любом статусе, кроме провала."""
        rows = await self._database.fetch_all(
            "SELECT spec FROM dev_tasks WHERE spec != '' AND status != ?", (DevTaskStatus.FAILED.value,)
        )
        slugs: set[str] = set()
        for row in rows:
            payload = safe_json_loads(str(row["spec"] or ""))
            if isinstance(payload, dict):
                slug = str(payload.get("slug", "")).strip().lower()
                if slug:
                    slugs.add(slug)
        return slugs

    async def abandoned_worth_another_try(
        self, *, not_touched_for: timedelta, max_revivals: int
    ) -> DevTask | None:
        """
        Брошенный проект, к которому стоит вернуться на свежую голову.

        Условий три, и каждое отсекает бессмысленный повтор. Спека должна
        быть: без неё это не проект, а несложившийся замысел, и возвращаться
        не к чему. Времени должно пройти достаточно: провал чаще всего
        случается из-за упёршегося лимита, а лимит отпускает к следующему дню.
        И возвращений должно быть немного: замысел, который не собрался и на
        третий раз, стоит квоты, за которую пишется что-то новое.
        """
        cutoff = (datetime.now(UTC) - not_touched_for).isoformat()
        row = await self._database.fetch_one(
            """
            SELECT * FROM dev_tasks
             WHERE status = ? AND spec != '' AND revivals < ? AND updated_at < ?
             ORDER BY updated_at ASC LIMIT 1
            """,
            (DevTaskStatus.FAILED.value, max_revivals, cutoff),
        )
        return _row_to_task(row) if row is not None else None

    async def purge_empty_failures(self) -> int:
        """
        Убирает провалы, в которых не осталось ничего: ни замысла, ни спеки.

        Такие строчки появлялись, когда задача заводилась ПЕРЕД
        проектированием и падала на нём же (см. DevWorker._maybe_start_own_project):
        «замысел без названия — не вышло», и так по строчке каждые несколько
        часов. Ни идеи, ни кода, ни причины возвращаться в них нет — это шум,
        из-за которого на дашборде не видно настоящих проектов.

        Задачи с текстом замысла или со спекой не трогаются: там есть что
        показать и о чём вспомнить, даже если работа не удалась.
        """
        removed = await self._database.execute_and_count_changes(
            "DELETE FROM dev_tasks WHERE status = ? AND TRIM(idea) = '' AND spec = ''",
            (DevTaskStatus.FAILED.value,),
        )
        if removed:
            logger.info("dev_store: убрала %d пустых провал(ов) — в них не было ни замысла, ни кода", removed)
        return removed

    async def recent_code_work(self, *, limit: int = 5) -> list[DevTask]:
        """
        Законченная работа с чужим кодом: ветки, которые она сдала.

        Отдельно от `recent_releases`: там её собственные репозитории со
        ссылками, которыми можно хвастаться, а здесь — правки в чужих
        проектах. Смешивать их значило бы показывать чужую репу как свою.
        """
        rows = await self._database.fetch_all(
            "SELECT * FROM dev_tasks WHERE status = ? AND kind = ? ORDER BY updated_at DESC LIMIT ?",
            (DevTaskStatus.DONE.value, DevTaskKind.SWE.value, limit),
        )
        return [_row_to_task(row) for row in rows]

    async def recent_failures(self, *, limit: int = 5) -> list[DevTask]:
        """Недавно провалившиеся задачи — только для дашборда: в промпт неудачи не идут, ей о них напоминать незачем."""
        rows = await self._database.fetch_all(
            "SELECT * FROM dev_tasks WHERE status = ? ORDER BY updated_at DESC LIMIT ?",
            (DevTaskStatus.FAILED.value, limit),
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
        branch: str | None = None,
        attempts: int | None = None,
        artifacts: dict[str, str] | None = None,
        revivals: int | None = None,
    ) -> DevTask:
        """Сохраняет продвижение задачи. Возвращает обновлённую копию — DevTask иммутабелен по смыслу."""
        updated = task.model_copy(
            update={
                "status": status if status is not None else task.status,
                "spec": spec if spec is not None else task.spec,
                "repo_url": repo_url if repo_url is not None else task.repo_url,
                "error": error if error is not None else task.error,
                "branch": branch if branch is not None else task.branch,
                "attempts": attempts if attempts is not None else task.attempts,
                "artifacts": artifacts if artifacts is not None else task.artifacts,
                "revivals": revivals if revivals is not None else task.revivals,
                "updated_at": datetime.now(UTC),
            }
        )
        await self._database.execute(
            """
            UPDATE dev_tasks
               SET status = ?, spec = ?, repo_url = ?, error = ?, branch = ?, attempts = ?,
                   artifacts = ?, revivals = ?, updated_at = ?
             WHERE id = ?
            """,
            (
                updated.status.value,
                compact_json_dumps(updated.spec.model_dump(mode="json")) if updated.spec is not None else "",
                updated.repo_url,
                updated.error,
                updated.branch,
                updated.attempts,
                compact_json_dumps(updated.artifacts) if updated.artifacts else "",
                updated.revivals,
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
        kind=DevTaskKind(str(row["kind"] or DevTaskKind.PROJECT.value)),
        source=str(row["source"] or ""),
        branch=str(row["branch"] or ""),
        status=DevTaskStatus(str(row["status"])),
        spec=spec,
        repo_url=str(row["repo_url"] or ""),
        error=str(row["error"] or ""),
        created_at=datetime.fromisoformat(str(row["created_at"])),
        updated_at=datetime.fromisoformat(str(row["updated_at"])),
        reviewed_at=_parse_optional(row["reviewed_at"]),
        revisions=int(row["revisions"] or 0),
        attempts=int(row["attempts"] or 0),
        artifacts=_parse_artifacts(row["artifacts"]),
        revivals=int(row["revivals"] or 0),
    )


def _parse_artifacts(raw: object) -> dict[str, str]:
    """Файлы прошлого захода. Битый JSON — не повод не начать заход заново."""
    payload = safe_json_loads(str(raw or "")) if raw else None
    if not isinstance(payload, dict):
        return {}
    return {str(key): str(value) for key, value in payload.items()}


def _parse_optional(raw: object) -> datetime | None:
    """Пустая строка в колонке значит «не было ни разу» — это не дата и не ноль эпохи."""
    text = str(raw or "")
    return datetime.fromisoformat(text) if text else None


__all__ = ["DevTaskStore"]
