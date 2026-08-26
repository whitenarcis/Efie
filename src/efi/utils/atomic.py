"""
efi/utils/atomic.py

Запись файла, которая переживает выключение телефона.

Обычная запись — это две операции: файл сначала обрезается в ноль, потом
наполняется заново. Между ними есть момент, когда на диске лежит пустой файл,
и если процесс в этот момент умер, содержимое потеряно навсегда. На сервере
такое случается раз в год; на телефоне это норма жизни — Android убивает
фоновые процессы по своему усмотрению, батарея садится посреди ночи, Termux
закрывается вместе с приложением.

Цена такого обрыва здесь не абстрактная. В `working_memory.json` лежат её
состояние, энергия и НЕЗАКРЫТЫЕ ОБЕЩАНИЯ — то, что человек ей сказал и чего
ждёт; в файле дневниковой записи — прожитый вечер. Потерять это из-за
неудачного момента выключения нельзя.

Приём стандартный: пишем во временный файл РЯДОМ (обязательно в тот же
каталог — иначе `replace` окажется межфайловым переносом и перестанет быть
атомарным) и подменяем им целевой одним системным вызовом. `os.replace`
атомарен на POSIX: в любой момент по целевому пути лежит либо старое
содержимое целиком, либо новое целиком, но никогда не половина.

Здесь же живёт `sweep_stale_files` — обратная сторона той же медали. Обрыв
посреди работы не только портит записи, но и оставляет мусор: файл,
скачанный за секунду до того, как Android убил Termux, не удалит уже никто.
"""

from __future__ import annotations

import contextlib
import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiofiles
import aiofiles.os

logger = logging.getLogger(__name__)


async def write_text_atomic(path: Path, content: str, *, encoding: str = "utf-8") -> None:
    """
    Атомарно записывает текст по пути `path`, создавая каталог при необходимости.

    Временный файл кладётся рядом и получает имя с pid: два процесса (а на
    практике — старый и только что перезапущенный) не должны драться за один
    и тот же временный путь.
    """
    await aiofiles.os.makedirs(path.parent, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        async with aiofiles.open(temporary, mode="w", encoding=encoding) as handle:
            await handle.write(content)
            # Данные должны дойти до диска ДО подмены: иначе после внезапного
            # выключения на месте старого файла окажется новый, но пустой.
            # fsync синхронный и быстрый (единицы миллисекунд на файл в
            # килобайты), поэтому уводить его в поток незачем.
            await handle.flush()
            os.fsync(handle.fileno())
        await aiofiles.os.replace(temporary, path)
    except OSError:
        # Прибираем за собой, но не проглатываем ошибку: не записалось —
        # значит, вызывающая сторона должна об этом знать.
        with contextlib.suppress(OSError):
            await aiofiles.os.remove(temporary)
        raise


def sweep_stale_files(directory: Path, *, older_than: timedelta) -> int:
    """
    Удаляет из каталога файлы старше `older_than` и возвращает их число.

    Для временных каталогов, которые прибираются по ходу дела и потому
    «не должны» накапливаться: кэш скачанных фото и голосовых. Обычный путь
    там честно удаляет файл сразу после распознавания, но обычный путь — не
    единственный: Android убивает Termux в произвольный момент, и файл,
    скачанный за секунду до этого, не удалит уже никто. На телефоне такие
    остатки копятся месяцами и незаметны, пока не кончится место.

    Синхронная по умыслу: зовётся на старте и изредка, а обходить каталог
    через поток исполнителя ради десятка файлов незачем. Ошибки на отдельных
    файлах не прерывают обход — смысл уборки в том, чтобы убрать что
    получится.
    """
    if not directory.is_dir():
        return 0
    cutoff = (datetime.now(UTC) - older_than).timestamp()
    removed = 0
    for path in directory.iterdir():
        try:
            if not path.is_file() or path.stat().st_mtime >= cutoff:
                continue
            path.unlink()
            removed += 1
        except OSError:
            logger.debug("atomic: не удалось убрать %s", path, exc_info=True)
    if removed:
        logger.info("atomic: убрала %d старых файлов из %s", removed, directory)
    return removed


__all__ = ["sweep_stale_files", "write_text_atomic"]
