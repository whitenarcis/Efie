"""
efi/memory/beliefs.py

Граф убеждений — структурированная память мнений Эфи по темам: topic/stance/
confidence_score/origin_date, поверх общего efi.db.core.Database (та же схема
владения соединениями, что и у efi.memory.facts.FactStore).

Ключевая идея — эпистемическая инерция: чем выше confidence_score, тем менее
охотно Эфи должна соглашаться с переубеждением с ходу. Само подавление
угодливости реализовано не здесь (это дело системного промпта, см.
efi/prompts/builder.py), а этот модуль отвечает только за хранение и за
дешёвый, БЕЗ обращения к LLM/сети, поиск убеждений, относящихся к текущему
сообщению (`find_relevant`) — критический путь сборки промпта не должен
платить дополнительным сетевым/LLM round-trip'ом за то, что можно решить
пересечением слов по уже небольшой (десятки-сотни записей) таблице убеждений.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime

import aiosqlite

from efi.db.core import Database

logger = logging.getLogger(__name__)

#: Порог confidence_score, начиная с которого убеждение считается "укоренившимся"
#: — при попытке переубедить Эфи по такой теме модель должна проявлять скепсис,
#: а не соглашаться мгновенно (см. efi.prompts.builder._build_state_vector_block).
STRONG_BELIEF_THRESHOLD = 0.7

_WORD_RE = re.compile(r"[\w]+", re.UNICODE)
_MIN_WORD_LENGTH = 3  # короткие слова (предлоги, союзы) не считаются значимым совпадением темы

#: Частотные короткие предлоги/местоимения/союзы, которые проходят по длине
#: (>=3 символов), но сами по себе НЕ являются значимым совпадением темы —
#: без этого списка любое сообщение с "про"/"что"/"как" ложно триггерило бы
#: find_relevant на первое попавшееся убеждение с тем же служебным словом.
_STOPWORDS = frozenset(
    {
        "про", "что", "как", "это", "для", "вот", "или", "его", "она", "они", "мне", "мой",
        "моя", "все", "всё", "уже", "тут", "там", "так", "нет", "тебе", "меня", "себя", "если",
        "тоже", "того", "этот", "эта", "эти", "куда", "чем", "кто", "где", "when", "what", "that",
        "this", "with", "from", "your", "you", "are", "the", "and", "for",
    }
)


@dataclass(slots=True, frozen=True)
class Belief:
    """Одно убеждение Эфи по теме — вся хранимая информация, включая служебные поля."""

    topic: str
    stance: str
    confidence_score: float
    origin_date: datetime


class BeliefStore:
    """
    Асинхронное хранилище убеждений поверх общего Database.

    `get`/`all_beliefs`/`find_relevant` — критический путь (сборка промпта
    перед ответом), `upsert`/`delete` — запись, обычно вызывается инструментом
    модели (efi.tools.memory_tools.manage_belief.UpdateBeliefTool) в середине
    разговора, а не в фоне.
    """

    def __init__(self, database: Database) -> None:
        self._database = database

    async def upsert(self, topic: str, stance: str, *, confidence_score: float = 0.5) -> None:
        """
        Записывает или обновляет убеждение (SQLite UPSERT по topic).

        Намеренно НЕ реализует здесь "инерцию" в смысле отказа перезаписать
        значение — это был бы неверный уровень: инерция должна проявляться в
        том, ЧТО модель решает сказать/сохранить (после скепсиса из промпта),
        а не в том, что хранилище тихо игнорирует явную запись модели.
        `origin_date` обновляется на текущий момент при каждом upsert — новое
        закрепление позиции по теме сбрасывает отсчёт "с каких пор так думаю".
        """
        confidence_score = max(0.0, min(confidence_score, 1.0))
        now = datetime.now(UTC).isoformat()
        await self._database.execute(
            """
            INSERT INTO beliefs (topic, stance, confidence_score, origin_date)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (topic) DO UPDATE SET
                stance = excluded.stance,
                confidence_score = excluded.confidence_score,
                origin_date = excluded.origin_date
            """,
            (topic, stance, confidence_score, now),
        )

    async def get(self, topic: str) -> Belief | None:
        """Критический путь: точечное чтение одного убеждения по точному совпадению темы."""
        row = await self._database.fetch_one(
            "SELECT topic, stance, confidence_score, origin_date FROM beliefs WHERE topic = ?", (topic,)
        )
        return _row_to_belief(row) if row is not None else None

    async def all_beliefs(self) -> list[Belief]:
        """Все убеждения — используется find_relevant и фоновой консолидацией/отладкой."""
        rows = await self._database.fetch_all("SELECT topic, stance, confidence_score, origin_date FROM beliefs")
        return [_row_to_belief(row) for row in rows]

    async def find_relevant(self, text: str, *, limit: int = 3) -> list[Belief]:
        """
        Ищет убеждения, тема которых пересекается словами с `text` — дешёвая
        эвристика без эмбеддингов и без похода в LLM (сама таблица убеждений
        по масштабу — десятки-сотни записей, не корпус для полноценного RAG).

        Ранжирует по (числу совпавших слов, confidence_score) по убыванию —
        сильные, явно релевантные убеждения идут первыми, чтобы именно они
        попадали в промпт при `limit`, отсекающем длинный хвост.
        """
        message_words = _extract_words(text)
        if not message_words:
            return []

        scored: list[tuple[int, float, Belief]] = []
        for belief in await self.all_beliefs():
            topic_words = _extract_words(belief.topic)
            overlap = len(topic_words & message_words)
            if overlap == 0:
                continue
            scored.append((overlap, belief.confidence_score, belief))

        scored.sort(key=lambda item: (item[0], item[1]), reverse=True)
        return [belief for _overlap, _confidence, belief in scored[:limit]]

    async def delete(self, topic: str) -> None:
        """Удаляет убеждение, если оно существует; иначе — no-op."""
        await self._database.execute("DELETE FROM beliefs WHERE topic = ?", (topic,))


def _extract_words(text: str) -> set[str]:
    return {
        word
        for raw_word in _WORD_RE.findall(text)
        if len(raw_word) >= _MIN_WORD_LENGTH and (word := raw_word.lower()) not in _STOPWORDS
    }


def _row_to_belief(row: aiosqlite.Row) -> Belief:
    return Belief(
        topic=row["topic"],
        stance=row["stance"],
        confidence_score=row["confidence_score"],
        origin_date=datetime.fromisoformat(row["origin_date"]),
    )


__all__ = ["Belief", "BeliefStore", "STRONG_BELIEF_THRESHOLD"]
