"""
efi/memory/validator.py

Детерминированная валидация — ВТОРАЯ половина границы доверия и единственное
место, после которого кандидат считается фактом.

Правило: между «модель предложила» и «лежит в базе» стоит обычный Python без
единого обращения к LLM. Он нормализует (сущности к каноническому виду, даты
к UTC, значения к очищенному тексту), проверяет типы и границы и отбраковывает
всё, что не проходит. Никакой «умной» интерпретации: если решение о том,
записывать ли, принимает модель, границы доверия нет — есть её иллюзия.

Что именно ловится здесь и не может быть поймано выше:
    - служебные ключи (`last_novelized_at`, `incubated_thought`): модель,
      попросив «запомнить» такой ключ, переписала бы бухгалтерию приложения —
      отметку «докуда уже новеллизировано» или инкубированную мысль;
    - инъекции в значении: value проходит тот же sanitize_text, что и любой
      внешний текст, потому что значение факта потом уезжает В ПРОМПТ;
    - выдуманные домены и сущности не из того домена (человек в C, тема в P);
    - пустые, гигантские и мусорные значения;
    - даты из будущего и «вчера» словами.

Отбраковка не молчит: каждый отказ уходит и в лог, и в `knowledge_rejections`
(см. efi/db/schema.sql) — без журнала «Эфи не запомнила» и «Эфи запомнила
чушь» выглядят одинаково.
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

from efi.memory.parser import FactCandidate, PerceptionBatch
from efi.memory.router import MemoryDomain
from efi.security.sanitize import sanitize_text

logger = logging.getLogger(__name__)

#: Ключи, которыми приложение ведёт свою бухгалтерию в FactStore. Модель не
#: должна получить к ним доступ ни при каких формулировках: перезаписав
#: `last_novelized_at`, она сдвинула бы окно новеллизации и стёрла бы себе
#: память о целом дне.
RESERVED_ATTRIBUTES = frozenset(
    {
        "last_novelized_at",
        "incubated_thought",
        "last_pulse_at",
        "novelized_until",
    }
)

#: Сущности, которые приложение адресует само (см. efi/behavior/researcher.py).
RESERVED_ENTITIES = frozenset({"researcher", "system", "efi_internal"})

MIN_VALUE_CHARS = 2
MAX_VALUE_CHARS = 300
MAX_ATTRIBUTE_CHARS = 48
MAX_ENTITY_CHARS = 96

#: Насколько далеко в будущее допускается observed_at. Небольшой запас — на
#: расхождение часов между устройством и собеседником; всё, что дальше, —
#: выдумка модели, а не наблюдение.
_FUTURE_TOLERANCE = timedelta(hours=12)

_ATTRIBUTE_SANITIZE_RE = re.compile(r"[^\w\-]+", re.UNICODE)
_MULTISPACE_RE = re.compile(r"\s+")

#: Как собеседник называет владельца и себя — всё это один и тот же субъект.
_OWNER_ALIASES = frozenset(
    {"я", "мне", "меня", "пользователь", "владелец", "хозяин", "папик", "owner", "user", "me"}
)
#: Как модель называет саму Эфи.
_SELF_ALIASES = frozenset({"я_эфи", "эфи", "efi", "self", "себя", "assistant"})

#: Относительные даты, которые модель пишет словами вместо ISO.
_RELATIVE_DATES = {
    "сегодня": 0,
    "today": 0,
    "вчера": -1,
    "yesterday": -1,
    "позавчера": -2,
    "завтра": 1,
    "tomorrow": 1,
}

_ISO_DATE_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})")


@dataclass(slots=True, frozen=True)
class ValidatedFact:
    """
    Кандидат, прошедший проверку. Всё здесь уже нормализовано и безопасно для
    записи: строки очищены, домен разобран, дата — aware UTC.
    """

    domain: MemoryDomain
    entity_id: str
    attribute: str
    value: str
    confidence: float
    observed_at: datetime
    source: str
    normalized_hash: str

    @property
    def canonical_text(self) -> str:
        """
        Текст, по которому считается эмбеддинг при дедупликации. Один и тот
        же вид для одного и того же смысла, иначе «работа: монтажёр» и
        «монтажёр» разошлись бы по вектору только из-за формы записи.
        """
        return f"{self.entity_id} {self.attribute.replace('_', ' ')}: {self.value}"


@dataclass(slots=True, frozen=True)
class Rejection:
    """Отклонённый кандидат вместе с причиной — попадает в лог и в knowledge_rejections."""

    candidate: FactCandidate
    reason: str


@dataclass(slots=True)
class ValidationReport:
    """Итог проверки одной пачки кандидатов."""

    accepted: list[ValidatedFact] = field(default_factory=list)
    rejected: list[Rejection] = field(default_factory=list)

    @property
    def accepted_count(self) -> int:
        return len(self.accepted)

    @property
    def rejected_count(self) -> int:
        return len(self.rejected)


class FactValidator:
    """
    Проверяет и нормализует кандидатов. Ни одного обращения к сети и ни
    одного к БД: чистая, воспроизводимая функция от входа — поэтому её
    поведение целиком покрывается тестами, а не «проверяется на практике».
    """

    def __init__(self, *, owner_id: int | None = None, now: datetime | None = None) -> None:
        self._owner_id = owner_id
        #: Фиксированное «сейчас» для тестов; в бою — None, берётся текущее время.
        self._fixed_now = now

    # -- публичный интерфейс ------------------------------------------------

    def validate_batch(self, batch: PerceptionBatch) -> ValidationReport:
        report = ValidationReport()
        seen_hashes: set[str] = set()
        for candidate in batch.candidates:
            outcome = self.validate(candidate, source=batch.source)
            if isinstance(outcome, Rejection):
                report.rejected.append(outcome)
                continue
            if outcome.normalized_hash in seen_hashes:
                # Один и тот же факт, повторённый моделью дважды внутри
                # ОДНОГО ответа, — это не подтверждение, а её многословность.
                # Считать его двумя наблюдениями значило бы позволить модели
                # накручивать occurrence_count простым повтором.
                report.rejected.append(Rejection(candidate, "дубль внутри одной пачки"))
                continue
            seen_hashes.add(outcome.normalized_hash)
            report.accepted.append(outcome)
        return report

    def validate(self, candidate: FactCandidate, *, source: str = "") -> ValidatedFact | Rejection:
        """Проверка одного кандидата. Возвращает либо готовый к записи факт, либо причину отказа."""
        try:
            domain = MemoryDomain.parse(candidate.domain)
        except ValueError:
            return Rejection(candidate, f"неизвестный домен {candidate.domain!r}")

        entity_id, entity_error = self._normalize_entity(candidate.entity, domain)
        if entity_error:
            return Rejection(candidate, entity_error)

        attribute, attribute_error = _normalize_attribute(candidate.attribute)
        if attribute_error:
            return Rejection(candidate, attribute_error)

        value, value_error = _normalize_value(candidate.value)
        if value_error:
            return Rejection(candidate, value_error)

        domain_error = _check_domain_consistency(domain, entity_id)
        if domain_error:
            return Rejection(candidate, domain_error)

        observed_at = self._normalize_observed_at(candidate.observed_at)
        confidence = _clamp(candidate.confidence)

        return ValidatedFact(
            domain=domain,
            entity_id=entity_id,
            attribute=attribute,
            value=value,
            confidence=confidence,
            observed_at=observed_at,
            source=source,
            normalized_hash=compute_hash(domain, entity_id, attribute, value),
        )

    def normalize_entity_for_lookup(self, raw: str) -> str:
        """
        Та же нормализация сущности, что и при записи, но для ЧТЕНИЯ.

        Нужна инструменту recall_fact: модель спрашивает то «Рома», то
        «user:625207005», то «я» — и без приведения к одному виду чтение
        промахивалось бы мимо собственной же записи. Домен при чтении
        неизвестен, поэтому используется P (по нему адресуются люди —
        подавляющее большинство запросов «а что я знаю про X»).
        """
        entity_id, error = self._normalize_entity(raw, MemoryDomain.PERSONAL)
        return entity_id if not error else raw.strip()

    def normalize_attribute_for_lookup(self, raw: str) -> str:
        """Та же нормализация ключа, что и при записи: 'Любимый цвет' и 'любимый_цвет' — один атрибут."""
        attribute, error = _normalize_attribute(raw)
        return attribute if not error else raw.strip().lower()

    # -- нормализация ------------------------------------------------------

    def _normalize_entity(self, raw: str, domain: MemoryDomain) -> tuple[str, str]:
        """
        Приводит субъект к каноническому `<префикс>:<имя>`.

        Префикс не украшение: без него «Костя» из личного домена и «костя»
        как тема в общем домене — одна и та же строка, и дедупликация
        схлопнула бы человека с темой.
        """
        text = _collapse(raw).strip(" .,:;!?").lower()
        if not text:
            return "", "пустая сущность"
        if len(text) > MAX_ENTITY_CHARS:
            return "", f"слишком длинная сущность ({len(text)} символов)"

        normalized = text.replace(" ", "_")
        if normalized in RESERVED_ENTITIES:
            return "", f"служебная сущность {normalized!r} недоступна для записи"

        if text in _OWNER_ALIASES:
            if self._owner_id is None:
                return "", "владелец не сконфигурирован, ссылка на него неразрешима"
            return f"user:{self._owner_id}", ""
        if normalized in _SELF_ALIASES:
            return "self", ""

        # Уже адресованная сущность («user:123», «topic:sqlite») — оставляем.
        if ":" in normalized:
            prefix, _, rest = normalized.partition(":")
            if not rest:
                return "", "сущность с пустым идентификатором после префикса"
            return f"{prefix}:{rest}", ""

        prefix = "topic" if domain is MemoryDomain.COMMON else "person"
        return f"{prefix}:{normalized}", ""

    def _normalize_observed_at(self, raw: str) -> datetime:
        """
        Дата наблюдения к aware UTC.

        Понимает ISO-дату, относительные слова («вчера») и пустую строку.
        Всё, что не разобралось, — это «сейчас»: у наблюдения всегда есть
        время, и терять факт из-за того, что модель написала дату прозой,
        неразумно. Будущее подрезается: наблюдение не может произойти позже,
        чем его записали.
        """
        now = self._now()
        text = _collapse(raw).lower()
        if not text:
            return now

        relative = _RELATIVE_DATES.get(text)
        if relative is not None:
            return now + timedelta(days=relative) if relative <= 0 else now

        iso_match = _ISO_DATE_RE.search(text)
        if iso_match is not None:
            try:
                parsed = datetime(
                    int(iso_match.group(1)), int(iso_match.group(2)), int(iso_match.group(3)), tzinfo=UTC
                )
            except ValueError:
                return now
            return parsed if parsed <= now + _FUTURE_TOLERANCE else now

        try:
            parsed = datetime.fromisoformat(text)
        except ValueError:
            return now
        aware = parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)
        return aware if aware <= now + _FUTURE_TOLERANCE else now

    def _now(self) -> datetime:
        return self._fixed_now if self._fixed_now is not None else datetime.now(UTC)


# ---------------------------------------------------------------------------
# Чистые помощники
# ---------------------------------------------------------------------------


def compute_hash(domain: MemoryDomain, entity_id: str, attribute: str, value: str) -> str:
    """
    Хэш нормализованного факта — первая ступень дедупликации.

    Значение перед хэшированием сводится к «смысловому» виду (регистр,
    пробелы, знаки препинания по краям), чтобы «Монтажёр.» и «монтажёр» дали
    один хэш. Unicode нормализуется в NFKC: «ё» из разных источников бывает
    склеенной и разложенной, и без этого один и тот же факт двоился бы по
    невидимой глазом причине.
    """
    canonical_value = unicodedata.normalize("NFKC", value).strip(" .,;:!?").casefold()
    payload = f"{domain.value}|{entity_id}|{attribute}|{canonical_value}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _normalize_attribute(raw: str) -> tuple[str, str]:
    text = _collapse(raw).lower().strip(" .,:;!?")
    if not text:
        return "", "пустой атрибут"
    slug = _ATTRIBUTE_SANITIZE_RE.sub("_", text).strip("_")
    if not slug:
        return "", f"атрибут {raw!r} не содержит ни одного пригодного символа"
    if len(slug) > MAX_ATTRIBUTE_CHARS:
        return "", f"слишком длинный атрибут ({len(slug)} символов)"
    if slug in RESERVED_ATTRIBUTES:
        return "", f"служебный ключ {slug!r} недоступен для записи"
    return slug, ""


def _normalize_value(raw: str) -> tuple[str, str]:
    """
    Значение факта к безопасному виду.

    Две обработки, и обе обязательны. `sanitize_text` — общий санитайзер
    внешнего текста (невидимые символы, гомоглифы, маркеры чужих
    промпт-форматов). Замена квадратных скобок — уже НАША специфика:
    системный промпт Эфи размечен блоками вида `[Ограничения]`, а сам факт
    уезжает туда строкой `[ФАКТ: ...]`. Значение, содержащее квадратные
    скобки, способно подделать эту разметку и выдать себя за инструкцию;
    легитимный короткий факт в скобках не нуждается, поэтому дешевле
    заменить их на круглые, чем разбираться, подделка это или нет.
    """
    text = _collapse(sanitize_text(raw)).replace("[", "(").replace("]", ")")
    if len(text) < MIN_VALUE_CHARS:
        return "", "пустое или слишком короткое значение"
    if len(text) > MAX_VALUE_CHARS:
        return "", f"слишком длинное значение ({len(text)} символов)"
    return text, ""


def _check_domain_consistency(domain: MemoryDomain, entity_id: str) -> str:
    """
    Домен и субъект должны сходиться. Модель регулярно кладёт человека в C
    («domain: C, entity: Костя»), и без этой проверки личные данные оседали
    бы в общем знании — то есть подмешивались бы к техническим вопросам,
    ради предотвращения чего домены и вводились.
    """
    prefix = entity_id.split(":", 1)[0]
    if domain is MemoryDomain.COMMON and prefix in {"user", "person"}:
        return "личная сущность в домене C"
    if domain is MemoryDomain.PERSONAL and prefix == "topic":
        return "тема в домене P"
    return ""


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.5
    if numeric != numeric:  # NaN не сравнивается сам с собой
        return 0.5
    return max(low, min(numeric, high))


def _collapse(text: str) -> str:
    return _MULTISPACE_RE.sub(" ", unicodedata.normalize("NFKC", str(text or ""))).strip()


__all__ = [
    "MAX_ATTRIBUTE_CHARS",
    "MAX_VALUE_CHARS",
    "RESERVED_ATTRIBUTES",
    "RESERVED_ENTITIES",
    "FactValidator",
    "Rejection",
    "ValidatedFact",
    "ValidationReport",
    "compute_hash",
]
