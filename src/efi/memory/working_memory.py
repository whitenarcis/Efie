"""
efi/memory/working_memory.py

Рабочая память Эфи — аналог `important_things_to_remember` из C++-референса:
короткий горизонт (по умолчанию 3 дня), эмоциональное и физическое состояние
персонажа, список открытых задач/обещаний/напоминаний.

В отличие от Diary (десятки-сотни независимых записей с семантическим
поиском), рабочая память — один эволюционирующий снимок, поэтому хранится как
единый JSON-файл, а не набор markdown-файлов.

"Verbatim" в требовании означает: текст открытого пункта не переписывается
при плановой ротации/консолидации — обновляется только `last_updated` (через
`touch_item`) или `done` (через `mark_done`). Изменить сам текст пункта можно
только явно удалив старый и добавив новый через `add_item`.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiofiles
import aiofiles.os
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)

_DEFAULT_HORIZON = timedelta(days=3)


class WorkingMemoryItem(BaseModel):
    """Один открытый пункт: обещание, напоминание или незавершённая задача."""

    text: str
    created_at: datetime
    last_updated: datetime
    done: bool = False


class WorkingMemorySnapshot(BaseModel):
    """
    Снимок рабочей памяти целиком — то, что фактически подставляется в
    промпт как блок `<things_to_remember>` (формирование самого промпта —
    забота efi/prompts/, здесь только структурированные данные).
    """

    emotional_state: str = ""
    physical_state: str = ""
    energy: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="Текущий уровень бодрости (0..1) — вход для efi.behavior.busy_engine.BusyEngine "
        "(низкая энергия удлиняет ignore_delay перед ответом).",
    )
    items: list[WorkingMemoryItem] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class WorkingMemory:
    """
    Асинхронный доступ к рабочей памяти.

    `load()` — критический путь: читает JSON-файл один раз и дальше отдаёт
    закэшированный снимок, пока не будет вызвана мутирующая операция (`save`,
    `add_item`, `touch_item`, `mark_done`, `prune`, `update_state`) — каждая
    из них сама персистит изменения на диск. Если вызывающей стороне не важен
    момент, когда запись реально долетит до диска, она вольна обернуть вызов в
    `asyncio.create_task(...)`.
    """

    def __init__(self, path: Path, *, horizon: timedelta = _DEFAULT_HORIZON) -> None:
        self._path = path
        self._horizon = horizon
        self._lock = asyncio.Lock()
        self._cache: WorkingMemorySnapshot | None = None

    async def load(self) -> WorkingMemorySnapshot:
        """Критический путь: возвращает текущий снимок, читая с диска только при первом обращении."""
        if self._cache is not None:
            return self._cache
        async with self._lock:
            if self._cache is not None:
                return self._cache
            self._cache = await self._read_from_disk()
            return self._cache

    async def _read_from_disk(self) -> WorkingMemorySnapshot:
        try:
            async with aiofiles.open(self._path, encoding="utf-8") as f:
                raw = await f.read()
        except FileNotFoundError:
            return WorkingMemorySnapshot()
        try:
            return WorkingMemorySnapshot.model_validate_json(raw)
        except ValidationError as exc:
            logger.warning("working_memory: corrupted snapshot at %s (%s), starting fresh", self._path, exc)
            return WorkingMemorySnapshot()

    async def save(self, snapshot: WorkingMemorySnapshot | None = None) -> WorkingMemorySnapshot:
        """
        Персистит снимок (текущий закэшированный, если явно не передан другой) на диск.

        Блокировка (`self._lock`) сериализует сами записи на диск — важно
        теперь, когда один экземпляр WorkingMemory может быть общим для
        нескольких параллельных Worker'ов (efi/notifications/worker.py):
        физическое/эмоциональное состояние у Эфи одно на всех чатов, а не
        по одному на чат. Если snapshot не передан явно, берём уже
        закэшированный объект НЕ через load() — load() сама может захватывать
        этот же лок при первом обращении, и вызов её изнутри уже занятого
        лока привёл бы к дедлоку (asyncio.Lock не реентерабельна).
        """
        if snapshot is None:
            snapshot = self._cache if self._cache is not None else await self.load()
        async with self._lock:
            snapshot.updated_at = datetime.now(UTC)
            await aiofiles.os.makedirs(self._path.parent, exist_ok=True)
            async with aiofiles.open(self._path, mode="w", encoding="utf-8") as f:
                await f.write(snapshot.model_dump_json(indent=2))
            self._cache = snapshot
        return snapshot

    async def update_state(
        self,
        *,
        emotional_state: str | None = None,
        physical_state: str | None = None,
        energy: float | None = None,
    ) -> WorkingMemorySnapshot:
        """Обновляет эмоциональное/физическое состояние и/или уровень энергии персонажа."""
        snapshot = await self.load()
        if emotional_state is not None:
            snapshot.emotional_state = emotional_state
        if physical_state is not None:
            snapshot.physical_state = physical_state
        if energy is not None:
            snapshot.energy = max(0.0, min(energy, 1.0))
        return await self.save(snapshot)

    async def add_item(self, text: str) -> WorkingMemoryItem:
        """Добавляет новый открытый пункт (обещание/напоминание/задачу)."""
        snapshot = await self.load()
        now = datetime.now(UTC)
        item = WorkingMemoryItem(text=text, created_at=now, last_updated=now)
        snapshot.items.append(item)
        await self.save(snapshot)
        return item

    async def touch_item(self, index: int) -> None:
        """
        Подтверждает актуальность пункта БЕЗ изменения текста (verbatim) —
        обновляет только `last_updated`, чтобы пункт не был вычищен `prune()`.
        """
        snapshot = await self.load()
        if 0 <= index < len(snapshot.items):
            snapshot.items[index].last_updated = datetime.now(UTC)
            await self.save(snapshot)

    async def mark_done(self, index: int) -> None:
        """Помечает пункт выполненным — он будет убран ближайшим `prune()`."""
        snapshot = await self.load()
        if 0 <= index < len(snapshot.items):
            snapshot.items[index].done = True
            await self.save(snapshot)

    async def find_and_mark_done(self, text_query: str) -> WorkingMemoryItem | None:
        """
        Находит первый ОТКРЫТЫЙ пункт, чей текст содержит `text_query`
        (регистронезависимая подстрока), и помечает его выполненным.
        Возвращает найденный пункт, либо None, если подходящего не нашлось.

        Текстовый поиск, а не индекс — предназначен для вызова инструментом
        модели (efi.tools.memory_tools.manage_promises.CompletePromiseTool),
        которой удобнее сослаться на обещание по смыслу, чем помнить его
        порядковый номер в списке; для короткого списка из нескольких
        открытых пунктов точного/подстрочного совпадения достаточно — тот же
        компромисс "дёшево и без ML", что и у memory/tfidf_fallback.py.
        """
        query = text_query.strip().lower()
        if not query:
            return None
        snapshot = await self.load()
        for item in snapshot.items:
            if not item.done and query in item.text.lower():
                item.done = True
                await self.save(snapshot)
                return item
        return None

    async def prune(self) -> int:
        """
        Убирает завершённые пункты и те, что не обновлялись дольше `horizon`
        (по умолчанию 3 дня — как в референсе). Возвращает число удалённых пунктов.
        """
        snapshot = await self.load()
        cutoff = datetime.now(UTC) - self._horizon
        kept = [item for item in snapshot.items if not item.done and item.last_updated >= cutoff]
        removed = len(snapshot.items) - len(kept)
        if removed:
            snapshot.items = kept
            await self.save(snapshot)
        return removed


__all__ = ["WorkingMemoryItem", "WorkingMemorySnapshot", "WorkingMemory"]
