"""
efi/behavior/ambiguity.py

Разрешение сущностей и режим уточнения.

Проблема. «Скинь это в Феникс» — а «Феникс» это рабочий проект, кот и чат
одновременно. Пока код молча брал лучшего по счёту кандидата, ошибка не
проявлялась сразу: она оседала в памяти как факт («у Феникса дедлайн в
пятницу» записано про кота), а всплывала через неделю, когда Эфи начинала
уверенно говорить чушь. Испорченную запись потом не отличить от настоящей —
у неё та же структура и та же уверенность.

Решение — не угадывать. Если два и более кандидата практически равны,
разрешение прекращается и вместо записи порождается короткий уточняющий
вопрос, который Эфи задаёт обычным бабблом. Один вопрос сейчас дешевле, чем
искажённая память навсегда.

Почему уточнение собирается шаблоном, а не моделью: это одна фраза из двух
известных слов, и тратить на неё сетевой вызов внутри уже идущего хода
значит добавить секунды ожидания ради того, что прекрасно строится
детерминированно. Вариативность формулировок обеспечена набором заготовок,
выбор внутри которого привязан к самому упоминанию — один и тот же «Феникс»
всегда спрашивается одинаково, и это не читается как заедающая пластинка.
"""

from __future__ import annotations

import logging
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

logger = logging.getLogger(__name__)

#: Насколько близкими должны быть счета кандидатов, чтобы считать их
#: неразличимыми. 0.08 при счетах в диапазоне 0..1 — это «разница в пределах
#: погрешности сопоставления»; выше порог начинал бы спрашивать там, где
#: лидер очевиден, ниже — молча угадывать в спорных случаях.
DEFAULT_MARGIN = 0.08

#: Ниже этого счёта кандидат не рассматривается вовсе: совпадение по одной
#: букве — не кандидат. Если лучший кандидат ниже порога, сущность считается
#: неизвестной, и уточнять нечего: спрашивать «ты про X или про Y?», когда ни
#: X, ни Y на самом деле не подходят, — хуже молчания.
DEFAULT_MIN_SCORE = 0.35

#: Сколько живёт незакрытое уточнение. Человек либо отвечает почти сразу,
#: либо уже забыл вопрос — и подставлять его ответ в вопрос вчерашней
#: давности значит записать ещё более искажённые данные, чем при угадывании.
DEFAULT_TTL = timedelta(minutes=30)

_MAX_LISTED_CANDIDATES = 3

#: Заготовки уточнения. Все — короткие, разговорные и без служебного тона:
#: это реплика в чате, а не диалоговое окно.
_CLARIFICATION_TEMPLATES: tuple[str, ...] = (
    "погоди, {mention} — это {options}?",
    "а {mention} у тебя сейчас {options}?",
    "уточни, {mention} — {options}?",
    "это ты про {options}? а то {mention} у нас не один",
)


@dataclass(slots=True, frozen=True)
class EntityCandidate:
    """
    Один вариант того, о ком/чём речь.

    `hint` — короткое человеческое пояснение («кот», «рабочий проект»),
    которое попадает в уточняющий вопрос. Без него вопрос выродится в
    «ты про Феникс или про Феникс?».
    """

    entity_id: str
    label: str
    score: float
    hint: str = ""

    def describe(self) -> str:
        return f"{self.label} ({self.hint})" if self.hint else self.label


@dataclass(slots=True, frozen=True)
class Resolution:
    """
    Итог разрешения упоминания.

    Ровно три взаимоисключающих исхода, и вызывающая сторона обязана
    различать их: разрешено (пиши), неоднозначно (спроси, НЕ пиши),
    неизвестно (не пиши и не спрашивай).
    """

    entity_id: str | None = None
    clarification: str | None = None
    candidates: tuple[EntityCandidate, ...] = ()
    mention: str = ""

    @property
    def is_resolved(self) -> bool:
        return self.entity_id is not None

    @property
    def needs_clarification(self) -> bool:
        return self.clarification is not None

    @property
    def is_unknown(self) -> bool:
        return not self.is_resolved and not self.needs_clarification


class AmbiguityDetector:
    """
    Чистая логика разрешения: список кандидатов на входе, решение на выходе.
    Ни сети, ни БД, ни состояния — поэтому её поведение полностью
    воспроизводимо в тестах.
    """

    def __init__(self, *, margin: float = DEFAULT_MARGIN, min_score: float = DEFAULT_MIN_SCORE) -> None:
        self._margin = margin
        self._min_score = min_score

    def resolve(self, mention: str, candidates: list[EntityCandidate]) -> Resolution:
        viable = sorted(
            (candidate for candidate in candidates if candidate.score >= self._min_score),
            key=lambda candidate: candidate.score,
            reverse=True,
        )
        if not viable:
            return Resolution(mention=mention, candidates=tuple(candidates))

        best = viable[0]
        if len(viable) == 1:
            return Resolution(entity_id=best.entity_id, candidates=(best,), mention=mention)

        contenders = tuple(
            candidate for candidate in viable if best.score - candidate.score <= self._margin
        )
        if len(contenders) == 1:
            return Resolution(entity_id=best.entity_id, candidates=tuple(viable), mention=mention)

        logger.info(
            "ambiguity: %r неоднозначно — %d равновероятных кандидата(ов), перехожу в режим уточнения",
            mention, len(contenders),
        )
        return Resolution(
            clarification=render_clarification(mention, contenders),
            candidates=contenders,
            mention=mention,
        )


def render_clarification(mention: str, candidates: tuple[EntityCandidate, ...]) -> str:
    """
    Короткий естественный вопрос-баббл.

    Больше трёх вариантов не перечисляется: список из пяти пунктов в чате
    читается как анкета, а не как вопрос живого человека — а если их
    действительно пять, то и трёх хватит, чтобы человек понял, о чём его
    спрашивают, и ответил своими словами.
    """
    listed = candidates[:_MAX_LISTED_CANDIDATES]
    descriptions = [candidate.describe() for candidate in listed]
    if len(descriptions) == 1:  # pragma: no cover — уточнение с одним кандидатом не порождается
        options = descriptions[0]
    else:
        options = f"{', '.join(descriptions[:-1])} или {descriptions[-1]}"

    template = _CLARIFICATION_TEMPLATES[_stable_index(mention, len(_CLARIFICATION_TEMPLATES))]
    return template.format(mention=mention, options=options)


@dataclass(slots=True)
class PendingClarification:
    """Заданный и ещё не закрытый вопрос по одному упоминанию."""

    mention: str
    candidates: tuple[EntityCandidate, ...]
    asked_at: datetime
    question: str

    def is_expired(self, now: datetime, ttl: timedelta) -> bool:
        return now - self.asked_at > ttl


class PendingClarifications:
    """
    Реестр незакрытых уточнений по чатам.

    In-memory и без персистентности намеренно: незакрытый вопрос живёт
    минуты и осмыслен только внутри текущего разговора. Пережившее рестарт
    уточнение — это вопрос, который человек уже не помнит, и ответ на него
    сопоставится не с тем, что он имел в виду; ровно та ошибка, ради
    предотвращения которой модуль и написан.
    """

    def __init__(self, *, ttl: timedelta = DEFAULT_TTL) -> None:
        self._ttl = ttl
        self._pending: dict[int, PendingClarification] = {}

    def remember(self, chat_id: int, resolution: Resolution) -> None:
        """Фиксирует, что по этому чату задан уточняющий вопрос."""
        if not resolution.needs_clarification or resolution.clarification is None:
            return
        self._pending[chat_id] = PendingClarification(
            mention=resolution.mention,
            candidates=resolution.candidates,
            asked_at=datetime.now(UTC),
            question=resolution.clarification,
        )

    def peek(self, chat_id: int) -> PendingClarification | None:
        """Текущее незакрытое уточнение чата, если оно ещё не протухло."""
        pending = self._pending.get(chat_id)
        if pending is None:
            return None
        if pending.is_expired(datetime.now(UTC), self._ttl):
            del self._pending[chat_id]
            return None
        return pending

    def resolve_with_answer(self, chat_id: int, answer: str) -> EntityCandidate | None:
        """
        Пытается сопоставить ответ человека с одним из кандидатов.

        Сопоставление намеренно грубое (подстрока по метке и по подсказке):
        человек отвечает «кот» или «проект», а не полным идентификатором.
        Если ответ не опознан однозначно — уточнение НЕ закрывается и ничего
        не записывается: неопознанный ответ ничем не лучше первоначальной
        неоднозначности.
        """
        pending = self.peek(chat_id)
        if pending is None:
            return None

        normalized = _normalize(answer)
        if not normalized:
            return None

        matches = [
            candidate
            for candidate in pending.candidates
            if _normalize(candidate.label) in normalized
            or (candidate.hint and _normalize(candidate.hint) in normalized)
        ]
        if len(matches) != 1:
            return None

        del self._pending[chat_id]
        logger.info("ambiguity: уточнение по %r закрыто ответом -> %s", pending.mention, matches[0].entity_id)
        return matches[0]

    def discard(self, chat_id: int) -> None:
        """Снимает уточнение без ответа — например, когда разговор ушёл на другую тему."""
        self._pending.pop(chat_id, None)

    def __len__(self) -> int:
        return len(self._pending)


@dataclass(slots=True)
class ResolutionOutcome:
    """
    Что делать вызывающей стороне после разрешения пачки упоминаний.

    Отдельный тип, а не кортеж: у конвейера памяти два независимых исхода
    (что записывать и о чём спросить), и путать их местами не должно быть
    возможно.
    """

    resolved: dict[str, str] = field(default_factory=dict)
    clarifications: list[str] = field(default_factory=list)

    @property
    def has_questions(self) -> bool:
        return bool(self.clarifications)


def _stable_index(text: str, modulo: int) -> int:
    """Устойчивый выбор заготовки: одно и то же упоминание всегда спрашивается одинаково."""
    if modulo <= 0:  # pragma: no cover — набор заготовок непуст
        return 0
    return sum(ord(char) for char in text) % modulo


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", str(text or "")).strip().casefold()


__all__ = [
    "DEFAULT_MARGIN",
    "DEFAULT_MIN_SCORE",
    "AmbiguityDetector",
    "EntityCandidate",
    "PendingClarification",
    "PendingClarifications",
    "Resolution",
    "ResolutionOutcome",
    "render_clarification",
]
