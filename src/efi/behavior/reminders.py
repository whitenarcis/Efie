"""
efi/behavior/reminders.py

Отложенные напоминания: «напиши мне через 10 минут» — и через десять минут
она действительно пишет.

ПОЧЕМУ ЭТОГО НЕ БЫЛО. Механизм выглядел собранным, но был разорван в трёх
местах сразу, и каждый разрыв по отдельности не бросался в глаза:

    1. `remember_promise` записывал только ТЕКСТ обещания в рабочую память.
       Ни времени, ни чата — то есть стикер на память, а не таймер. Ничто в
       системе не могло узнать, что «через 10 минут» вообще означает срок.
    2. `SilenceMonitor.schedule_follow_up()` умел принимать «вернись к теме в
       такое-то время» и порождать FOLLOW_UP — но его никто никогда не
       вызывал. Механизм был написан «на вырост» и так и остался мёртвым.
    3. Даже если бы вызывали: очередь follow-up'ов жила в памяти процесса и
       проверялась раз в 15 минут. Перезапуск стирал обещание молча, а
       «через 10 минут» в лучшем случае срабатывало через 15.

Итог с точки зрения человека: он попросил, Эфи согласилась, обещание повисло
в её состоянии навсегда, и ничего не произошло. Худший вид поломки — тот, где
всё выглядит работающим.

ЧТО ЗДЕСЬ. Напоминания хранятся в SQLite (таблица `proactive_tasks`, которая
до этого момента тоже не использовалась ни одной строчкой кода), поэтому
переживают перезапуск, и проверяются раз в полминуты, поэтому «через 10
минут» означает десять минут, а не «когда-нибудь в ближайшие четверть часа».

Тихие часы намеренно НЕ учитываются. Они существуют, чтобы Эфи не будила
человека по собственной инициативе; напоминание — не её инициатива, а прямая
просьба, и «ты просил напомнить в 23:40, но у меня тихий час» — это не
деликатность, а невыполненное обещание.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from efi.db.core import Database
from efi.notifications.manager import NotificationManager
from efi.notifications.schemas import Notification, NotificationType
from efi.utils.loops import run_periodically

logger = logging.getLogger(__name__)

#: Как часто проверять, не пора ли. Полминуты — компромисс между точностью
#: («через 10 минут» не должно превращаться в 25) и стоимостью: это один
#: индексированный запрос к локальному SQLite, а не поход в сеть.
DEFAULT_CHECK_INTERVAL_SECONDS = 30.0

#: Границы срока. Снизу — чтобы «напомни через секунду» не превращалось в
#: сообщение поверх ещё не отправленного ответа; сверху — здравый смысл:
#: обещание на полгода вперёд это не напоминание, а список дел.
MIN_DELAY = timedelta(minutes=1)
MAX_DELAY = timedelta(days=30)

#: Сколько ждать до того, как просроченное напоминание считать протухшим.
#: Приложение стояло неделю — человек уже не ждёт того сообщения, и получить
#: «ты просил напомнить» через семь дней тишины страннее, чем не получить.
STALE_AFTER = timedelta(hours=12)

_TASK_TYPE = NotificationType.FOLLOW_UP.value

_STATUS_PENDING = "pending"
_STATUS_DONE = "done"
_STATUS_CANCELLED = "cancelled"


@dataclass(slots=True, frozen=True)
class Reminder:
    """Одно запланированное напоминание."""

    id: int
    chat_id: int
    text: str
    scheduled_at: datetime
    created_at: datetime

    def is_stale(self, now: datetime) -> bool:
        return now - self.scheduled_at > STALE_AFTER

    def render_reason(self) -> str:
        """
        Повод для системного промпта — в тех же терминах, что и остальные
        проактивные поводы (см. efi/behavior/spontaneous_ping.py): не готовый
        текст сообщения, а объяснение, ПОЧЕМУ она сейчас пишет.
        """
        return (
            f"Ты обещала вернуться к этому именно сейчас: {self.text}. "
            "Время пришло — напиши собеседнику сама. Коротко, своими словами, без «как и обещала» "
            "и прочих канцеляризмов: просто сделай то, что обещала."
        )


class ReminderStore:
    """
    Хранилище напоминаний поверх таблицы `proactive_tasks`.

    Отдельная таблица не заводилась: `proactive_tasks` создана ровно под это
    (chat_id, task_type, scheduled_at, payload, status) и до сих пор стояла
    пустой — ей просто никто не пользовался.
    """

    def __init__(self, database: Database) -> None:
        self._database = database

    async def schedule(self, chat_id: int, text: str, *, due_at: datetime) -> Reminder:
        """
        Ставит напоминание, ЗАМЕНЯЯ уже существующее с тем же текстом в том
        же чате.

        Замена, а не второе напоминание: человек, повторивший «ну напиши
        через десять минут», уточняет срок, а не просит написать дважды. А
        модель, вызвавшая инструмент дважды за один ход (что она делает
        регулярно), не должна превращать одну просьбу в два сообщения.
        """
        now = datetime.now(UTC)
        normalized = text.strip()
        payload = json.dumps({"promise_text": normalized}, ensure_ascii=False)

        async with self._database.connection() as conn:
            await conn.execute(
                """
                UPDATE proactive_tasks SET status = ?
                WHERE chat_id = ? AND task_type = ? AND status = ? AND payload = ?
                """,
                (_STATUS_CANCELLED, chat_id, _TASK_TYPE, _STATUS_PENDING, payload),
            )
            cursor = await conn.execute(
                """
                INSERT INTO proactive_tasks (chat_id, task_type, scheduled_at, payload, status, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (chat_id, _TASK_TYPE, due_at.isoformat(), payload, _STATUS_PENDING, now.isoformat()),
            )
            await conn.commit()
            reminder_id = cursor.lastrowid or 0

        logger.info(
            "reminders: напоминание #%d для chat_id=%s на %s (%r)",
            reminder_id, chat_id, due_at.isoformat(timespec="seconds"), normalized,
        )
        return Reminder(
            id=reminder_id, chat_id=chat_id, text=normalized, scheduled_at=due_at, created_at=now
        )

    async def due(self, now: datetime | None = None) -> list[Reminder]:
        """Напоминания, чей срок наступил и которые ещё не сработали."""
        moment = now or datetime.now(UTC)
        rows = await self._database.fetch_all(
            """
            SELECT id, chat_id, payload, scheduled_at, created_at
            FROM proactive_tasks
            WHERE status = ? AND task_type = ? AND scheduled_at <= ?
            ORDER BY scheduled_at ASC
            """,
            (_STATUS_PENDING, _TASK_TYPE, moment.isoformat()),
        )
        return [reminder for reminder in map(_row_to_reminder, rows) if reminder is not None]

    async def pending(self, chat_id: int | None = None, *, limit: int = 50) -> list[Reminder]:
        """Ещё не сработавшие напоминания — для дашборда и для ответов инструмента."""
        condition = "AND chat_id = ?" if chat_id is not None else ""
        params: tuple[object, ...] = (
            (_STATUS_PENDING, _TASK_TYPE, chat_id, limit) if chat_id is not None
            else (_STATUS_PENDING, _TASK_TYPE, limit)
        )
        rows = await self._database.fetch_all(
            f"""
            SELECT id, chat_id, payload, scheduled_at, created_at
            FROM proactive_tasks
            WHERE status = ? AND task_type = ? {condition}
            ORDER BY scheduled_at ASC
            LIMIT ?
            """,  # noqa: S608 — условие собрано из литералов, значения идут параметрами
            params,
        )
        return [reminder for reminder in map(_row_to_reminder, rows) if reminder is not None]

    async def mark_fired(self, reminder_id: int) -> None:
        """
        Помечает напоминание сработавшим.

        Делается СРАЗУ при постановке в очередь, а не после того, как
        сообщение реально ушло. Иначе любая неудача обработки (модель
        промолчала, провайдер лёг) оставляла бы напоминание в очереди, и
        следующий тик поставил бы его снова — раз в полминуты, бесконечно.
        Само обещание в рабочей памяти при этом НЕ закрывается, пока
        сообщение не доставлено (см. efi/notifications/worker.py), поэтому
        невыполненное остаётся видимым и в промпте, и на дашборде.
        """
        await self._database.execute(
            "UPDATE proactive_tasks SET status = ? WHERE id = ?", (_STATUS_DONE, reminder_id)
        )

    async def cancel(self, reminder_id: int) -> None:
        await self._database.execute(
            "UPDATE proactive_tasks SET status = ? WHERE id = ?", (_STATUS_CANCELLED, reminder_id)
        )

    async def cancel_matching(self, chat_id: int, text_query: str) -> int:
        """
        Снимает напоминания чата, чей текст содержит подстроку. Нужно, когда
        обещание закрыли раньше срока: напоминать о сделанном — то же самое,
        что не напомнить о несделанном.
        """
        query = text_query.strip().lower()
        if not query:
            return 0
        cancelled = 0
        for reminder in await self.pending(chat_id):
            if query in reminder.text.lower():
                await self.cancel(reminder.id)
                cancelled += 1
        return cancelled


class ReminderScheduler:
    """
    Фоновый цикл: раз в `check_interval_seconds` забирает наступившие
    напоминания и кладёт их в общую очередь как FOLLOW_UP.

    Дальше событие обрабатывается ровно так же, как любое другое (см.
    efi/notifications/worker.py) — с полным контекстом личности, а не
    отдельным «облегчённым» путём отправки.
    """

    def __init__(
        self,
        manager: NotificationManager,
        store: ReminderStore,
        *,
        check_interval_seconds: float = DEFAULT_CHECK_INTERVAL_SECONDS,
    ) -> None:
        self._manager = manager
        self._store = store
        self._check_interval_seconds = check_interval_seconds

    async def run(self) -> None:
        await run_periodically(
            self.tick, interval_seconds=self._check_interval_seconds, name="reminders"
        )

    async def tick(self) -> int:
        """
        Один проход. Возвращает число поставленных в очередь напоминаний —
        так его удобно проверять тестом, не гоняя бесконечный цикл.

        Сбой одного напоминания не должен срывать остальные и уж тем более
        останавливать цикл: следующий тик придёт через полминуты, а вот
        упавший планировщик не придёт уже никогда.
        """
        now = datetime.now(UTC)
        try:
            due = await self._store.due(now)
        except Exception:
            logger.exception("reminders: не удалось прочитать очередь напоминаний")
            return 0

        queued = 0
        for reminder in due:
            try:
                if reminder.is_stale(now):
                    # Приложение стояло долго. Человек уже не ждёт того
                    # сообщения, и «ты просил напомнить» через сутки тишины
                    # страннее, чем молчание.
                    logger.info(
                        "reminders: напоминание #%d протухло (срок был %s), снимаю",
                        reminder.id, reminder.scheduled_at.isoformat(timespec="minutes"),
                    )
                    await self._store.mark_fired(reminder.id)
                    continue

                await self._manager.put(
                    Notification(
                        type=NotificationType.FOLLOW_UP,
                        # Приоритет выше спонтанного пинга (6) и пинга по
                        # тишине (7): у напоминания есть СРОК, который человек
                        # назвал сам, и опоздать с ним хуже, чем с «просто
                        # написать первой».
                        priority=3,
                        chat_id=reminder.chat_id,
                        message=reminder.render_reason(),
                        payload={"promise_text": reminder.text, "reminder_id": reminder.id},
                    )
                )
                await self._store.mark_fired(reminder.id)
                queued += 1
                logger.info("reminders: сработало напоминание #%d для chat_id=%s", reminder.id, reminder.chat_id)
            except Exception:
                logger.exception("reminders: не удалось поставить напоминание #%d", reminder.id)
        return queued


def resolve_due_at(minutes: float, *, now: datetime | None = None) -> datetime:
    """
    Срок из «через сколько минут», зажатый в разумные границы.

    Зажимается, а не отбраковывается: модель, написавшая `remind_in_minutes:
    0`, всё равно имела в виду «скоро», и превратить это в минуту полезнее,
    чем отказать и потерять обещание целиком.
    """
    moment = now or datetime.now(UTC)
    delay = timedelta(minutes=max(0.0, float(minutes)))
    delay = max(MIN_DELAY, min(delay, MAX_DELAY))
    return moment + delay


def _row_to_reminder(row: object) -> Reminder | None:
    mapping = dict(row)  # type: ignore[call-overload]  # aiosqlite.Row поддерживает dict()
    try:
        payload = json.loads(str(mapping["payload"]) or "{}")
    except json.JSONDecodeError:
        payload = {}
    text = str(payload.get("promise_text", "")).strip()
    if not text:
        # Строка без текста обещания — мусор из прошлых версий или чужая
        # запись в той же таблице. Пропускаем: напоминание без текста
        # превратилось бы в сообщение «я обещала », что хуже молчания.
        return None
    try:
        return Reminder(
            id=int(mapping["id"]),
            chat_id=int(mapping["chat_id"]),
            text=text,
            scheduled_at=datetime.fromisoformat(str(mapping["scheduled_at"])),
            created_at=datetime.fromisoformat(str(mapping["created_at"])),
        )
    except (TypeError, ValueError):
        logger.warning("reminders: строка #%s не разбирается как напоминание", mapping.get("id"))
        return None


__all__ = [
    "DEFAULT_CHECK_INTERVAL_SECONDS",
    "MAX_DELAY",
    "MIN_DELAY",
    "Reminder",
    "ReminderScheduler",
    "ReminderStore",
    "resolve_due_at",
]
