"""
efi/behavior/affinity.py

Трекер близости и уважения к собеседнику по каждому активному чату —
`affinity` (насколько тепло/по-свойски Эфи относится к чату) и
`respect_level` (насколько всерьёз воспринимает конкретного собеседника),
обе на шкале [0.0, 1.0].

Классификация реплики (`classify_message`) — чистая, зависящая только от
текста функция без сети/LLM: намеренно дешёвая эвристика по ключевым словам и
форме сообщения (тролинг / глубокий тех-дискусс / пустая фраза / нейтральное),
а не полноценный классификатор — тот же компромисс "дёшево и без сети", что
у memory/tfidf_fallback.py и memory/beliefs.py.find_relevant. Даёт вход в
блок текущего состояния системного промпта (efi/prompts/builder.py):
низкое уважение -> сухость и ирония, высокое -> готовность делиться глубокими
гипотезами (см. STRONG_RESPECT_THRESHOLD/social_distance_label).

Персистентность (`chat_affinity`, efi.db.models) нужна, чтобы отношение к
собеседнику не обнулялось при каждом рестарте процесса, но она вторична —
in-memory кэш обслуживает горячий путь (сборка промпта перед каждым ответом),
а запись в БД идёт после обновления кэша, не блокируя его.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from efi.db.core import Database

logger = logging.getLogger(__name__)

#: Порог (среднее affinity и respect_level), начиная с которого собеседник
#: считается "близким" (close_peer) в социальной дистанции промпта.
CLOSE_PEER_THRESHOLD = 0.6

#: Порог respect_level, начиная с которого Эфи готова делиться глубокими
#: гипотезами, а не просто отвечать по существу.
HIGH_RESPECT_THRESHOLD = 0.7

#: Порог respect_level, ниже которого тон становится сухим и ироничным.
LOW_RESPECT_THRESHOLD = 0.3

_DEFAULT_AFFINITY = 0.5
_DEFAULT_RESPECT = 0.5

_PHATIC_MAX_LENGTH = 14
_DEEP_TECH_MIN_LENGTH = 80

_TROLLING_MARKERS = (
    "тупая", "тупой бот", "дура", "дурочка", "идиотка", "нейронка тупая", "бесполезн",
    "заткнись", "иди нахуй", "иди нах", "отстой", "хуйню несешь", "хуйню несёшь",
    "ты бот и всё", "ты просто бот", "туп ты",
)
_DEEP_TECH_MARKERS = (
    "```", "def ", "class ", "async ", "await ", "traceback", "стектрейс", "трейс",
    "алгоритм", "архитектур", "рефактор", "паттерн проектирования", "race condition",
    "гонка потоков", "бэкенд", "фронтенд", "runtime", "компилятор", "exception",
    "sql", "api", "деплой", "оптимизаци",
)
_PHATIC_PHRASES = (
    "привет", "хай", "хей", "ку", "йо", "ок", "окей", "норм", "лол", "кек", "ору",
    "хех", "ага", "угу", "неа", "да", "нет", "хм", "пон", "понял", "поняла",
    "как дела", "что делаешь", "что нового", "как ты", "спс", "спасибо", "плюс", "+",
)

_SHOUTING_RE = re.compile(r"[A-ZА-ЯЁ]{4,}")


class MessageKind(StrEnum):
    """Грубая классификация реплики собеседника для эвристики близости/уважения."""

    TROLLING = "trolling"
    DEEP_TECH = "deep_tech"
    PHATIC = "phatic"
    NEUTRAL = "neutral"


#: (delta affinity, delta respect_level) на одно сообщение данного вида.
_DELTAS: dict[MessageKind, tuple[float, float]] = {
    MessageKind.TROLLING: (-0.05, -0.08),
    MessageKind.DEEP_TECH: (0.02, 0.06),
    MessageKind.PHATIC: (0.01, -0.01),
    MessageKind.NEUTRAL: (0.015, 0.005),
}


@dataclass(slots=True, frozen=True)
class AffinitySnapshot:
    """Текущее состояние близости/уважения к чату — то, что видит промпт."""

    affinity: float = _DEFAULT_AFFINITY
    respect_level: float = _DEFAULT_RESPECT

    @property
    def social_distance_label(self) -> str:
        """`close_peer`, если средняя близость выше порога, иначе `acquaintance` — см. builder.py."""
        combined = (self.affinity + self.respect_level) / 2.0
        return "close_peer" if combined >= CLOSE_PEER_THRESHOLD else "acquaintance"


def classify_message(text: str) -> MessageKind:
    """
    Чистая функция без побочных эффектов — вынесена отдельно от
    AffinityTracker ради тестируемости без Database/event loop.
    """
    stripped = text.strip()
    lowered = stripped.lower()

    if any(marker in lowered for marker in _TROLLING_MARKERS):
        return MessageKind.TROLLING
    if _is_shouting(stripped):
        return MessageKind.TROLLING
    if any(marker in lowered for marker in _DEEP_TECH_MARKERS) and len(stripped) >= _DEEP_TECH_MIN_LENGTH:
        return MessageKind.DEEP_TECH
    if len(stripped) <= _PHATIC_MAX_LENGTH and lowered in _PHATIC_PHRASES:
        return MessageKind.PHATIC
    return MessageKind.NEUTRAL


def _is_shouting(text: str) -> bool:
    """Капслок-outburst: длинная реплика, где заметная доля букв — заглавные подряд (не аббревиатуры)."""
    if len(text) < 8:
        return False
    shouted_chars = sum(len(match) for match in _SHOUTING_RE.findall(text))
    return shouted_chars / len(text) > 0.4


class AffinityTracker:
    """
    Держит in-memory кэш AffinitySnapshot на чат и синхронизирует его с
    таблицей `chat_affinity`. Кэш — источник истины для горячего пути
    (get_snapshot читает из него, если запись уже была загружена в этом
    процессе); БД — только для восстановления состояния после рестарта и для
    сохранения обновлений.
    """

    def __init__(self, database: Database) -> None:
        self._database = database
        self._cache: dict[int, AffinitySnapshot] = {}

    async def get_snapshot(self, chat_id: int) -> AffinitySnapshot:
        """Критический путь (сборка системного промпта): кэш -> БД -> дефолт, в таком порядке."""
        cached = self._cache.get(chat_id)
        if cached is not None:
            return cached

        row = await self._database.fetch_one(
            "SELECT affinity, respect_level FROM chat_affinity WHERE chat_id = ?", (chat_id,)
        )
        snapshot = (
            AffinitySnapshot(affinity=row["affinity"], respect_level=row["respect_level"])
            if row is not None
            else AffinitySnapshot()
        )
        self._cache[chat_id] = snapshot
        return snapshot

    async def record_message(self, chat_id: int, text: str) -> AffinitySnapshot:
        """
        Классифицирует реплику собеседника и сдвигает affinity/respect_level
        соответствующим шагом. Обновляет кэш немедленно (следующий build()
        системного промпта в этом же ходу уже увидит новое значение) и
        персистит изменение в БД.
        """
        current = await self.get_snapshot(chat_id)
        kind = classify_message(text)
        affinity_delta, respect_delta = _DELTAS[kind]
        updated = await self._shift(chat_id, current, affinity_delta, respect_delta)
        logger.debug(
            "affinity: chat_id=%s kind=%s affinity=%.3f respect_level=%.3f",
            chat_id, kind.value, updated.affinity, updated.respect_level,
        )
        return updated

    async def apply_boost(
        self, chat_id: int, *, affinity_delta: float = 0.0, respect_delta: float = 0.0
    ) -> AffinitySnapshot:
        """
        Сдвигает affinity/respect_level на явно заданную величину, В ОБХОД
        classify_message — для сигналов, которые несут больше информации, чем
        обычная реплика (например efi.behavior.organic_ping.OrganicPingGenerator:
        собеседник ОТВЕТИЛ на находку, которую Эфи сама принесла по своей
        инициативе — это более сильный сигнал вовлечённости, чем рядовое
        сообщение, и его не стоит сводить к той же грубой эвристике).
        """
        current = await self.get_snapshot(chat_id)
        updated = await self._shift(chat_id, current, affinity_delta, respect_delta)
        logger.debug(
            "affinity: chat_id=%s boost affinity=%.3f respect_level=%.3f",
            chat_id, updated.affinity, updated.respect_level,
        )
        return updated

    async def _shift(
        self, chat_id: int, current: AffinitySnapshot, affinity_delta: float, respect_delta: float
    ) -> AffinitySnapshot:
        updated = AffinitySnapshot(
            affinity=_clamp(current.affinity + affinity_delta),
            respect_level=_clamp(current.respect_level + respect_delta),
        )
        self._cache[chat_id] = updated

        now = datetime.now(UTC).isoformat()
        await self._database.execute(
            """
            INSERT INTO chat_affinity (chat_id, affinity, respect_level, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT (chat_id) DO UPDATE SET
                affinity = excluded.affinity,
                respect_level = excluded.respect_level,
                updated_at = excluded.updated_at
            """,
            (chat_id, updated.affinity, updated.respect_level, now),
        )
        return updated


def _clamp(value: float) -> float:
    return max(0.0, min(value, 1.0))


__all__ = [
    "AffinitySnapshot",
    "AffinityTracker",
    "MessageKind",
    "classify_message",
    "CLOSE_PEER_THRESHOLD",
    "HIGH_RESPECT_THRESHOLD",
    "LOW_RESPECT_THRESHOLD",
]
