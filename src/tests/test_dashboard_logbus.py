"""
Тесты кольцевого буфера логов дашборда (efi.dashboard.logbus).

Проверяется то, ради чего он и написан: запись не теряется, фильтры
работают, буфер не растёт бесконечно, живая подписка получает новое, а
медленный подписчик не может ни затормозить логирование, ни съесть память.
"""

from __future__ import annotations

import asyncio
import logging

from efi.dashboard.logbus import LogBuffer


def _record(message: str, *, level: int = logging.INFO, name: str = "efi.test") -> logging.LogRecord:
    return logging.LogRecord(name=name, level=level, pathname=__file__, lineno=1, msg=message, args=(), exc_info=None)


def test_entries_are_stored_with_structure() -> None:
    buffer = LogBuffer(capacity=10)
    buffer.emit(_record("привет", level=logging.WARNING, name="efi.telegram"))

    entries = buffer.snapshot()
    assert len(entries) == 1
    assert entries[0].message == "привет"
    assert entries[0].level == "WARNING"
    assert entries[0].logger_name == "efi.telegram"
    assert entries[0].as_dict()["level_no"] == logging.WARNING


def test_ring_buffer_drops_oldest_but_keeps_total() -> None:
    buffer = LogBuffer(capacity=3)
    for index in range(10):
        buffer.emit(_record(f"строка {index}"))

    entries = buffer.snapshot()
    assert [entry.message for entry in entries] == ["строка 7", "строка 8", "строка 9"]
    assert buffer.buffered == 3
    assert buffer.total == 10


def test_filters_by_level_and_substring() -> None:
    buffer = LogBuffer(capacity=50, level=logging.DEBUG)
    buffer.emit(_record("обычное сообщение", level=logging.INFO))
    buffer.emit(_record("подозрительная штука", level=logging.WARNING))
    buffer.emit(_record("всё сломалось", level=logging.ERROR))

    assert len(buffer.snapshot(min_level=logging.WARNING)) == 2
    assert [entry.message for entry in buffer.snapshot(query="сломалось")] == ["всё сломалось"]
    assert buffer.snapshot(query="такого нет") == []


def test_snapshot_limit_returns_the_newest() -> None:
    buffer = LogBuffer(capacity=50)
    for index in range(20):
        buffer.emit(_record(f"строка {index}"))

    entries = buffer.snapshot(limit=3)
    assert [entry.message for entry in entries] == ["строка 17", "строка 18", "строка 19"]


def test_after_id_returns_only_the_tail() -> None:
    buffer = LogBuffer(capacity=50)
    for index in range(5):
        buffer.emit(_record(f"строка {index}"))
    boundary = buffer.snapshot()[2].id

    entries = buffer.snapshot(after_id=boundary)
    assert [entry.message for entry in entries] == ["строка 3", "строка 4"]


def test_handler_level_filters_before_the_buffer() -> None:
    """DEBUG не должен попадать в ленту, пока dashboard.log_level = INFO."""
    buffer = LogBuffer(capacity=10, level=logging.INFO)
    logger = logging.getLogger("efi.test.level")
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    buffer.install(logger)
    try:
        logger.debug("отладочное")
        logger.info("информационное")
    finally:
        buffer.uninstall()

    assert [entry.message for entry in buffer.snapshot()] == ["информационное"]


async def test_subscriber_receives_new_entries() -> None:
    buffer = LogBuffer(capacity=10)
    buffer.install(logging.getLogger("efi.test.stream"))
    try:
        with buffer.subscribe() as queue:
            buffer.emit(_record("живая запись"))
            # Рассылка идёт через call_soon_threadsafe, поэтому она случается
            # на следующем обороте петли, а не синхронно внутри emit().
            entry = await asyncio.wait_for(queue.get(), timeout=2.0)
    finally:
        buffer.uninstall()

    assert entry.message == "живая запись"


async def test_slow_subscriber_does_not_block_logging() -> None:
    buffer = LogBuffer(capacity=5000)
    buffer.install(logging.getLogger("efi.test.slow"))
    try:
        with buffer.subscribe() as queue:
            for index in range(1500):  # больше, чем вмещает очередь подписчика
                buffer.emit(_record(f"строка {index}"))
            await asyncio.sleep(0)  # даём рассылке отработать
            assert queue.qsize() <= 512
            assert buffer.total == 1500
            assert buffer.dropped > 0
    finally:
        buffer.uninstall()


def test_exception_is_captured_as_text() -> None:
    buffer = LogBuffer(capacity=10)
    logger = logging.getLogger("efi.test.exc")
    logger.propagate = False
    buffer.install(logger)
    try:
        try:
            raise ValueError("подстава")
        except ValueError:
            logger.exception("упало")
    finally:
        buffer.uninstall()

    entry = buffer.snapshot()[0]
    assert entry.exception is not None
    assert "ValueError: подстава" in entry.exception


def test_uninstall_detaches_from_logger() -> None:
    buffer = LogBuffer(capacity=10)
    logger = logging.getLogger("efi.test.detach")
    logger.propagate = False
    buffer.install(logger)
    buffer.uninstall()

    logger.warning("после отключения")
    assert buffer.snapshot() == []
