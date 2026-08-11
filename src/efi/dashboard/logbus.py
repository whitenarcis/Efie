"""
efi/dashboard/logbus.py

Кольцевой буфер записей логирования + живая рассылка новых записей
подписчикам (SSE-поток дашборда).

Почему обработчик логирования, а не чтение файла лога: файл лога
необязателен (`scripts/run.py --log-file` по умолчанию не задан), а даже
когда он есть — обратный разбор форматированных строк ради уровня и имени
логгера теряет то, что в `logging.LogRecord` уже лежит структурно: имя
задачи asyncio, модуль, трассировка исключения. Обработчик получает запись
до форматирования, поэтому дашборд показывает ровно то, что произошло, и
умеет фильтровать без regex по тексту.

Потокобезопасность. `emit()` может прийти НЕ из event loop: часть кода
проекта считает тяжёлые вещи через `asyncio.to_thread` и логирует оттуда
(например, `efi.memory.diary._score_entries` предупреждает о несовпадении
размерности эмбеддингов). Поэтому запись в сам буфер сделана на `deque`
(её `append` атомарен под GIL), а рассылка подписчикам уходит в петлю через
`call_soon_threadsafe` — трогать `asyncio.Queue` из чужого потока нельзя.
"""

from __future__ import annotations

import asyncio
import itertools
import logging
from collections import Counter, deque
from collections.abc import AsyncIterator, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

#: Сколько записей максимум ждёт в очереди одного SSE-подписчика. Переполнение
#: означает, что клиент читает медленнее, чем Эфи пишет логи — в этом случае
#: теряется самая старая запись подписчика, а не тормозится логирование.
_SUBSCRIBER_QUEUE_SIZE = 512

_DEFAULT_CAPACITY = 2000


@dataclass(slots=True, frozen=True)
class LogEntry:
    """Одна запись лога в том виде, в каком её отдаёт дашборд."""

    id: int
    timestamp: datetime
    level: str
    level_no: int
    logger_name: str
    message: str
    module: str = ""
    task: str | None = None
    exception: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "level": self.level,
            "level_no": self.level_no,
            "logger": self.logger_name,
            "message": self.message,
            "module": self.module,
            "task": self.task,
            "exception": self.exception,
        }

    def matches(self, query: str) -> bool:
        """Подстрочный регистронезависимый поиск по тексту, логгеру и трассировке."""
        if not query:
            return True
        needle = query.lower()
        return (
            needle in self.message.lower()
            or needle in self.logger_name.lower()
            or (self.exception is not None and needle in self.exception.lower())
        )


class LogBuffer(logging.Handler):
    """
    `logging.Handler`, который держит последние N записей в памяти и
    транслирует новые подписчикам.

    Ставится на корневой логгер через `install()` и снимается через
    `uninstall()` — владелец жизненного цикла (efi.app.EfiApp) обязан снять
    его при остановке, иначе обработчик переживёт приложение и продолжит
    копить записи от чужого кода в том же процессе (типичный случай — тесты).
    """

    def __init__(self, capacity: int = _DEFAULT_CAPACITY, *, level: int = logging.INFO) -> None:
        super().__init__(level=level)
        self._entries: deque[LogEntry] = deque(maxlen=capacity)
        self._ids = itertools.count(1)
        self._level_counts: Counter[str] = Counter()
        self._total = 0
        self._dropped = 0
        self._subscribers: set[asyncio.Queue[LogEntry]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._installed_on: logging.Logger | None = None

    # -- жизненный цикл ---------------------------------------------------

    def install(self, logger: logging.Logger | None = None) -> None:
        """
        Подключает буфер к логгеру (по умолчанию — корневому) и запоминает
        текущую петлю событий для рассылки подписчикам. Вызывать нужно уже
        внутри работающего loop.
        """
        target = logger if logger is not None else logging.getLogger()
        try:
            self._loop = asyncio.get_running_loop()
        except RuntimeError:
            self._loop = None
        if self._installed_on is None:
            target.addHandler(self)
            self._installed_on = target

    def uninstall(self) -> None:
        """Снимает буфер с логгера. Накопленные записи остаются доступными для чтения."""
        if self._installed_on is not None:
            self._installed_on.removeHandler(self)
            self._installed_on = None
        self._loop = None

    # -- приём записей -----------------------------------------------------

    def emit(self, record: logging.LogRecord) -> None:
        """
        Никогда не бросает наружу: сбой в дашборде не должен ронять вызов,
        который просто написал строку в лог (контракт logging.Handler).
        """
        try:
            entry = self._to_entry(record)
        except Exception:  # noqa: BLE001 — см. докстринг: обработчик обязан быть безопасным
            self.handleError(record)
            return

        self._entries.append(entry)
        self._level_counts[entry.level] += 1
        self._total += 1
        self._publish(entry)

    def _to_entry(self, record: logging.LogRecord) -> LogEntry:
        exception: str | None = None
        if record.exc_info is not None:
            exception = self.format_exception(record)
        elif record.exc_text:
            exception = record.exc_text

        return LogEntry(
            id=next(self._ids),
            timestamp=datetime.fromtimestamp(record.created, tz=UTC),
            level=record.levelname,
            level_no=record.levelno,
            logger_name=record.name,
            message=record.getMessage(),
            module=record.module,
            # taskName появился в logging только в Python 3.12, а проект
            # поддерживает 3.11 — отсюда getattr вместо прямого обращения.
            task=getattr(record, "taskName", None),
            exception=exception,
        )

    def format_exception(self, record: logging.LogRecord) -> str:
        formatter = self.formatter or logging.Formatter()
        return formatter.formatException(record.exc_info) if record.exc_info else ""

    def _publish(self, entry: LogEntry) -> None:
        if not self._subscribers or self._loop is None:
            return
        try:
            self._loop.call_soon_threadsafe(self._fanout, entry)
        except RuntimeError:
            # Петля уже закрыта (остановка приложения) — живая рассылка
            # больше никому не нужна, запись всё равно осталась в буфере.
            pass

    def _fanout(self, entry: LogEntry) -> None:
        for queue in tuple(self._subscribers):
            if queue.full():
                # Медленный клиент не имеет права затормозить логирование:
                # освобождаем место, жертвуя самой старой его записью.
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:  # pragma: no cover — очередь только что была полной
                    pass
                self._dropped += 1
            queue.put_nowait(entry)

    # -- чтение ------------------------------------------------------------

    def snapshot(
        self,
        *,
        min_level: int = 0,
        query: str = "",
        limit: int = 200,
        after_id: int | None = None,
    ) -> list[LogEntry]:
        """
        Последние записи, подходящие под фильтр, в хронологическом порядке.

        `after_id` отдаёт только то, что появилось после указанной записи —
        так страница долистывает "хвост" без повторной пересылки всего буфера,
        если SSE-соединение прерывалось.
        """
        selected: list[LogEntry] = []
        for entry in reversed(self._entries):
            if after_id is not None and entry.id <= after_id:
                break
            if entry.level_no < min_level or not entry.matches(query):
                continue
            selected.append(entry)
            if len(selected) >= max(1, limit):
                break
        selected.reverse()
        return selected

    def counts(self) -> dict[str, int]:
        """Сколько записей каждого уровня прошло через буфер за всё время работы."""
        return dict(self._level_counts)

    @property
    def total(self) -> int:
        """Всего записей с момента установки обработчика (включая вытесненные из буфера)."""
        return self._total

    @property
    def buffered(self) -> int:
        return len(self._entries)

    @property
    def capacity(self) -> int:
        return self._entries.maxlen or 0

    @property
    def last_id(self) -> int:
        return self._entries[-1].id if self._entries else 0

    # -- живая подписка ----------------------------------------------------

    @contextmanager
    def subscribe(self) -> Iterator[asyncio.Queue[LogEntry]]:
        """
        Очередь новых записей на время работы контекста.

        Синхронный контекстный менеджер вокруг асинхронного потребления —
        подписка/отписка не делают I/O, а `async with` тут только мешал бы
        использовать её внутри async-генератора SSE.
        """
        queue: asyncio.Queue[LogEntry] = asyncio.Queue(maxsize=_SUBSCRIBER_QUEUE_SIZE)
        self._subscribers.add(queue)
        try:
            yield queue
        finally:
            self._subscribers.discard(queue)

    async def stream(self) -> AsyncIterator[LogEntry]:
        """Бесконечный поток новых записей — тонкая обёртка над `subscribe()`."""
        with self.subscribe() as queue:
            while True:
                yield await queue.get()

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    @property
    def dropped(self) -> int:
        """Сколько записей не доехало до медленных подписчиков (сам буфер их сохранил)."""
        return self._dropped


__all__ = ["LogBuffer", "LogEntry"]
