"""
efi/behavior/curiosity.py

Семена любопытства — темы, мимоходом всплывшие в разговоре, которые Эфи
захотелось бы поизучать в фоне. CuriosityTracker извлекает такие темы из
входящих сообщений ДЕШЁВОЙ regex-эвристикой — без LLM/сети на критическом
пути, тот же принцип, что и у efi.behavior.affinity.classify_message и
efi.memory.beliefs.BeliefStore.find_relevant. Найденные темы копятся в
таблице `curiosity_seeds` (efi.db.models) со статусом `pending`; дальше ими
распоряжается efi.behavior.life_engine.BackgroundLifeWorker — раз в N минут
забирает семя с наибольшим весом, исследует его и переводит в `researched`.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from efi.db.core import Database

logger = logging.getLogger(__name__)


class SeedStatus(str, Enum):
    """Статус семени любопытства — жизненный цикл дальше ведёт BackgroundLifeWorker."""

    PENDING = "pending"
    RESEARCHED = "researched"


@dataclass(slots=True, frozen=True)
class CuriositySeed:
    """Одно семя любопытства: тема, откуда она пришла и насколько Эфи ей заинтересовалась."""

    id: int
    topic: str
    source_chat_id: int | None
    weight: float
    created_at: datetime
    status: SeedStatus


#: Триггерные обороты, за которыми обычно следует тема, достойная любопытства.
#: Намеренно узкий список: лучше пропустить часть интересных тем, чем засорить
#: curiosity_seeds случайным мусором из каждой второй реплики собеседника.
_TRIGGER_RE = re.compile(
    r"(?:что такое|что за|слышал[а]? про|слышал[а]? о|расскажи про|расскажи о|"
    r"интересно,? (?:как|почему|что)|кто такой|кто такая|как работает|почему)\s+([^?!.,;\n]{3,60})",
    re.IGNORECASE,
)

_BASE_WEIGHT = 0.5
_QUESTION_MARK_BONUS = 0.2


def extract_topic(text: str) -> str | None:
    """
    Чистая функция без побочных эффектов — вынесена отдельно от
    CuriosityTracker ради тестируемости без Database/event loop.
    Возвращает нормализованную тему-кандидат либо None, если в тексте не
    нашлось ни одного триггерного оборота.
    """
    match = _TRIGGER_RE.search(text)
    if match is None:
        return None
    topic = match.group(1).strip()
    return topic or None


class CuriosityTracker:
    """
    Дешёвый детектор тем-кандидатов на фоновое исследование поверх общего
    Database. Подключается в efi/telegram/handlers.py тем же дак-тайпингом,
    что и AffinityTracker (см. `affinity_recorder`/`curiosity_recorder` в
    TelegramEventHandlers) — срабатывает только на сообщения, которые прошли
    авторизацию (в группах — только адресованные Эфи), это сознательное
    решение: не имеет смысла копить любопытство по репликам, которые Эфи
    даже не собиралась читать внимательно.
    """

    def __init__(self, database: Database) -> None:
        self._database = database

    async def consider_message(self, chat_id: int | None, text: str) -> CuriositySeed | None:
        """
        Ищет в тексте оборот-триггер любопытства; если нашёлся — сохраняет
        новое семя и возвращает его. None — обычный, самый частый исход
        (ничего любопытного не нашлось), а не ошибка.
        """
        topic = extract_topic(text)
        if topic is None:
            return None

        weight = min(_BASE_WEIGHT + (_QUESTION_MARK_BONUS if "?" in text else 0.0), 1.0)
        created_at = datetime.now(timezone.utc)

        async with self._database.connection() as conn:
            cursor = await conn.execute(
                """
                INSERT INTO curiosity_seeds (topic, source_chat_id, weight, created_at, status)
                VALUES (?, ?, ?, ?, ?)
                """,
                (topic, chat_id, weight, created_at.isoformat(), SeedStatus.PENDING.value),
            )
            await conn.commit()
            seed_id = cursor.lastrowid
            assert seed_id is not None  # AUTOINCREMENT PRIMARY KEY — lastrowid всегда есть после успешного INSERT

        logger.debug("curiosity: new seed #%s topic=%r weight=%.2f chat_id=%s", seed_id, topic, weight, chat_id)
        return CuriositySeed(
            id=seed_id, topic=topic, source_chat_id=chat_id, weight=weight, created_at=created_at,
            status=SeedStatus.PENDING,
        )

    async def pick_top_pending(self) -> CuriositySeed | None:
        """Критический путь efi.behavior.life_engine.BackgroundLifeWorker: pending-семя с наибольшим весом."""
        row = await self._database.fetch_one(
            """
            SELECT id, topic, source_chat_id, weight, created_at, status
            FROM curiosity_seeds
            WHERE status = ?
            ORDER BY weight DESC, created_at ASC
            LIMIT 1
            """,
            (SeedStatus.PENDING.value,),
        )
        if row is None:
            return None
        return CuriositySeed(
            id=row["id"],
            topic=row["topic"],
            source_chat_id=row["source_chat_id"],
            weight=row["weight"],
            created_at=datetime.fromisoformat(row["created_at"]),
            status=SeedStatus(row["status"]),
        )

    async def mark_researched(self, seed_id: int) -> None:
        """Переводит семя в статус `researched` — идемпотентно, повторный вызов на то же id безопасен."""
        await self._database.execute(
            "UPDATE curiosity_seeds SET status = ? WHERE id = ?", (SeedStatus.RESEARCHED.value, seed_id)
        )


__all__ = ["CuriositySeed", "CuriosityTracker", "SeedStatus", "extract_topic"]
