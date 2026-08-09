"""
efi/memory/facts.py

Структурированное хранилище фактов о сущностях (людях, чатах, самой Эфи) —
тройки entity_id/key/value с confidence, поверх efi.db.core.Database.

С появлением efi/db/ (Шаг 6) FactStore больше не открывает собственные
соединения и не хранит собственную схему: таблица `facts` определена в
efi.db.models вместе с остальными таблицами приложения (история сообщений,
проактивные задачи) и создаётся через единый Database.connection() —
WAL/busy_timeout/ретраи/миграции настроены там же, в одном месте. Публичный
API (upsert/get/get_all/delete/all_facts) не изменился.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from efi.db.core import Database

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class Fact:
    """Один факт о сущности — вся хранимая информация, включая служебные поля."""

    entity_id: str
    key: str
    value: str
    confidence: float
    updated_at: datetime


class FactStore:
    """
    Асинхронное хранилище фактов поверх общего Database.

    `get`/`get_all` — критический путь (сборка контекста перед ответом).
    `upsert`/`delete` — запись, безопасны для запуска через
    `asyncio.create_task()`, если вызывающей стороне не нужно дожидаться
    подтверждения записи немедленно.
    """

    def __init__(self, database: Database) -> None:
        self._database = database

    async def upsert(self, entity_id: str, key: str, value: str, *, confidence: float = 1.0) -> None:
        """Записывает или обновляет факт (SQLite UPSERT по составному первичному ключу)."""
        now = datetime.now(UTC).isoformat()
        await self._database.execute(
            """
            INSERT INTO facts (entity_id, fact_key, fact_value, confidence, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT (entity_id, fact_key) DO UPDATE SET
                fact_value = excluded.fact_value,
                confidence = excluded.confidence,
                updated_at = excluded.updated_at
            """,
            (entity_id, key, value, confidence, now),
        )

    async def get(self, entity_id: str, key: str, default: str | None = None) -> str | None:
        """Критический путь: точечное чтение одного факта."""
        row = await self._database.fetch_one(
            "SELECT fact_value FROM facts WHERE entity_id = ? AND fact_key = ?", (entity_id, key)
        )
        return row["fact_value"] if row is not None else default

    async def get_all(self, entity_id: str) -> dict[str, str]:
        """Критический путь: все факты о сущности одним запросом (для сборки контекста)."""
        rows = await self._database.fetch_all(
            "SELECT fact_key, fact_value FROM facts WHERE entity_id = ?", (entity_id,)
        )
        return {row["fact_key"]: row["fact_value"] for row in rows}

    async def all_facts(self, entity_id: str) -> list[Fact]:
        """Полные записи (с confidence/updated_at) — для дашборда/консолидации, не для критического пути."""
        rows = await self._database.fetch_all(
            "SELECT entity_id, fact_key, fact_value, confidence, updated_at FROM facts WHERE entity_id = ?",
            (entity_id,),
        )
        return [
            Fact(
                entity_id=row["entity_id"],
                key=row["fact_key"],
                value=row["fact_value"],
                confidence=row["confidence"],
                updated_at=datetime.fromisoformat(row["updated_at"]),
            )
            for row in rows
        ]

    async def delete(self, entity_id: str, key: str) -> None:
        """Удаляет факт, если он существует; иначе — no-op."""
        await self._database.execute("DELETE FROM facts WHERE entity_id = ? AND fact_key = ?", (entity_id, key))


__all__ = ["Fact", "FactStore"]
