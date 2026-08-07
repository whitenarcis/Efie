"""
efi/prompts/loader.py

Асинхронная загрузка .md-шаблонов промптов с перечитыванием при изменении.

Два режима работы, оба доступны одновременно:
    - Ленивая инвалидация по mtime (всегда работает, без доп. зависимостей):
      каждый `get()` сверяет mtime файла с последним прочитанным значением
      (дешёвый `stat()`) и перечитывает файл, только если он изменился.
      Критический путь (сборка системного промпта перед каждым ответом) почти
      всегда получает попадание в кэш без единого файлового I/O.
    - Push-инвалидация через watchfiles (`watch()`, опционально): фоновая
      задача, которая сбрасывает кэш немедленно при изменении файла на диске,
      не дожидаясь следующего `get()`. Полезно для долгоживущего процесса,
      когда промпты редактируются "на лету".
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import aiofiles
import aiofiles.os

logger = logging.getLogger(__name__)


class PromptLoader:
    """Кэширующий асинхронный загрузчик .md-шаблонов из каталога `templates_dir`."""

    def __init__(self, templates_dir: Path) -> None:
        self._templates_dir = templates_dir
        self._cache: dict[str, tuple[float, str]] = {}  # name -> (mtime, content)
        self._locks: dict[str, asyncio.Lock] = {}

    async def get(self, name: str) -> str:
        """
        Возвращает содержимое шаблона `<templates_dir>/<name>.md`.

        Бросает FileNotFoundError, если файла нет — отсутствие шаблона не
        должно молча приводить к пустому/неполному промпту где-то выше по
        стеку; вызывающая сторона (например, prompts/builder.py) сама решает,
        считать это фатальной ошибкой или откатиться на что-то другое.
        """
        path = self._path_for(name)
        lock = self._locks.setdefault(name, asyncio.Lock())
        async with lock:
            try:
                mtime = (await aiofiles.os.stat(path)).st_mtime
            except FileNotFoundError:
                self._cache.pop(name, None)
                raise

            cached = self._cache.get(name)
            if cached is not None and cached[0] == mtime:
                return cached[1]

            async with aiofiles.open(path, mode="r", encoding="utf-8") as f:
                content = await f.read()
            self._cache[name] = (mtime, content)
            return content

    def invalidate(self, name: str | None = None) -> None:
        """Сбрасывает кэш одного шаблона (или всех, если имя не указано)."""
        if name is None:
            self._cache.clear()
        else:
            self._cache.pop(name, None)

    async def watch(self) -> None:
        """
        Фоновая задача: следит за `templates_dir` через watchfiles и сбрасывает
        кэш затронутых шаблонов сразу при изменении файлов на диске.
        Предназначена для запуска через `asyncio.create_task(loader.watch())`
        при старте приложения; завершается по отмене задачи (CancelledError).
        Если watchfiles не установлен — просто логирует предупреждение и
        завершается: ленивая mtime-инвалидация в get() продолжает работать
        в любом случае, watch() — только оптимизация задержки обновления.
        """
        try:
            import watchfiles
        except ImportError:
            logger.warning(
                "prompts: watchfiles is not installed, falling back to lazy mtime-based invalidation only"
            )
            return

        try:
            async for changes in watchfiles.awatch(self._templates_dir):
                for _change_type, changed_path in changes:
                    name = Path(changed_path).stem
                    self.invalidate(name)
                    logger.debug("prompts: invalidated cached template %r after on-disk change", name)
        except asyncio.CancelledError:
            logger.info("prompts: watch() stopped")
            raise

    def _path_for(self, name: str) -> Path:
        return self._templates_dir / f"{name}.md"


__all__ = ["PromptLoader"]
