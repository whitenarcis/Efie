"""
efi/memory/parser.py

Модель восприятия (Perception) — ПЕРВАЯ половина границы доверия.

Правило, ради которого модуль существует: LLM не пишет в память. Совсем.
Она не выполняет SQL, не трогает векторный индекс и не решает, что попадёт
в базу; её работа заканчивается на том, что она предлагает структурированных
кандидатов (DTO) в виде JSON. Дальше кандидатов принимает детерминированный
код (efi/memory/validator.py), и только он производит запись.

Почему так, а не инструментом «запомни факт». Инструмент отдаёт модели перо:
она сама выбирает entity_id, сама формулирует ключ, сама решает, что это
важно, — и любая её галлюцинация становится строкой в базе, которую потом
никто не отличит от настоящего наблюдения. Разделение на «предложить» и
«записать» стоит одного лишнего слоя, зато делает состав памяти следствием
правил, а не следствием того, в каком настроении сегодня модель.

Второй практический смысл — устойчивость к формату. Бесплатные модели
отвечают JSON'ом как получится: в ```-заборе, с преамбулой «Вот факты:», с
одиночным объектом вместо массива, с обёрткой {"facts": [...]}. Разбор
(`parse_payload`) написан под эту реальность и является чистой функцией —
его можно и нужно проверять тестами без сети.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from efi.config.schema import TaskRole
from efi.llm.errors import LLMError
from efi.llm.router import LLMRouter
from efi.llm.schemas import LLMParams, Message, Role, Session

logger = logging.getLogger(__name__)

#: Потолок на число кандидатов из одного эпизода. Модель, которой дали волю,
#: охотно «извлекает» два десятка фактов из трёх реплик — и половина из них
#: пересказ самой переписки. Ограничение сверху дешевле, чем чистка потом.
MAX_CANDIDATES = 12

#: Ищем самый внешний JSON-массив или объект в свободном тексте ответа.
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)


class CandidateKind(StrEnum):
    """Что именно предлагает запомнить модель восприятия."""

    ATTRIBUTE = "attribute"   # устойчивая характеристика: «работает монтажёром»
    PREFERENCE = "preference"  # предпочтение: «не любит ранние подъёмы»
    EVENT = "event"            # произошедшее: «вчера сдал проект»


class FactCandidate(BaseModel):
    """
    Кандидат на запись — DTO, а не факт.

    Ничего из этих полей не считается достоверным: имена приходят как есть,
    даты в свободной форме, домен моделью угадывается. Нормализацией и
    отбраковкой занимается валидатор; здесь только структура.
    """

    model_config = ConfigDict(
        extra="ignore",              # лишние поля от модели игнорируем, а не падаем
        str_strip_whitespace=True,
        frozen=True,
    )

    domain: str = Field(default="", description="C | P | H, как их понимает модель; проверяется валидатором")
    kind: str = Field(default=CandidateKind.ATTRIBUTE.value)
    entity: str = Field(default="")
    attribute: str = Field(default="")
    value: str = Field(default="")
    confidence: float = Field(default=0.5)
    observed_at: str = Field(default="", description="Свободная форма: '2026-08-09', 'вчера', 'на прошлой неделе'")
    source_quote: str = Field(default="")

    @field_validator("confidence", mode="before")
    @classmethod
    def _tolerate_non_numeric_confidence(cls, raw: object) -> float:
        """
        Модель регулярно пишет уверенность словами («high») или строкой
        («0.9»). Ронять из-за этого ВЕСЬ кандидат неразумно: уверенность —
        наименее ценное его поле, а факт при этом теряется целиком. Не
        разобранное значение становится нейтральным 0.5, а границы диапазона
        доводит валидатор.
        """
        if isinstance(raw, int | float) and not isinstance(raw, bool):
            return float(raw)
        try:
            return float(str(raw).strip().replace(",", "."))
        except (TypeError, ValueError):
            return 0.5

    @property
    def is_empty(self) -> bool:
        return not (self.entity and self.attribute and self.value)


class PerceptionBatch(BaseModel):
    """Результат одного прохода восприятия по эпизоду."""

    model_config = ConfigDict(frozen=True)

    candidates: list[FactCandidate] = Field(default_factory=list)
    raw_response: str = Field(default="", description="Сырой ответ модели — для журнала отбраковки и отладки")
    parse_error: str = Field(default="")
    source: str = Field(default="")
    observed_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    @property
    def failed(self) -> bool:
        return bool(self.parse_error)


_PERCEPTION_SYSTEM_PROMPT = (
    "Ты — модуль восприятия памяти. Твоя единственная работа: вычленить из переданного фрагмента "
    "разговора устойчивые факты и вернуть их СТРОГО как JSON-массив. Ничего не сохраняй сама, "
    "ничего не комментируй, не пиши текст до или после JSON.\n\n"
    "Формат каждого элемента:\n"
    '{"domain": "C|P|H", "kind": "attribute|preference|event", "entity": "...", '
    '"attribute": "...", "value": "...", "confidence": 0.0-1.0, "observed_at": "...", "source_quote": "..."}\n\n'
    "Домены:\n"
    "  C — знание о мире: технологии, концепции, общие факты. entity — тема ('sqlite', 'плёночный звук').\n"
    "  P — про конкретного человека: характеристики, предпочтения, обстоятельства. entity — кто это.\n"
    "  H — твой собственный эпизодический опыт: с кем общалась, как узнала, что почувствовала.\n\n"
    "Правила:\n"
    "  - только то, что ПРЯМО следует из текста; ничего не додумывай и не обобщай;\n"
    "  - attribute — короткий ключ из 1-3 слов ('работа', 'режим_сна', 'любимый_режиссёр');\n"
    "  - value — короткое значение, без пересказа диалога;\n"
    "  - мимолётное («сегодня устал») не факт; факт — устойчивое («часто не высыпается»);\n"
    "  - если фактов нет — верни пустой массив [];\n"
    f"  - максимум {MAX_CANDIDATES} элементов.\n"
    "  - source_quote — короткая цитата из текста, на которой основан факт."
)


class PerceptionParser:
    """
    Просит модель извлечь кандидатов и разбирает её ответ. Ничего не пишет —
    ни в SQLite, ни в векторную память: это физическая граница доверия, а не
    договорённость.
    """

    def __init__(
        self, router: LLMRouter, *, role: TaskRole = TaskRole.BACKGROUND, max_output_tokens: int = 900
    ) -> None:
        self._router = router
        self._role = role
        self._max_output_tokens = max_output_tokens

    async def extract(self, conversation_text: str, *, source: str = "") -> PerceptionBatch:
        """
        Один проход восприятия. Роль BACKGROUND: никто не ждёт этого ответа
        в чате, поэтому здесь допустимы медленная модель и большой таймаут
        (см. регламент ролей в efi/config/schema.py::TaskRole).
        """
        text = conversation_text.strip()
        if not text:
            return PerceptionBatch(source=source)

        params = LLMParams(
            model="",
            system_prompt=_PERCEPTION_SYSTEM_PROMPT,
            max_output_tokens=self._max_output_tokens,
            # Восприятие — не творчество: одинаковый вход должен давать
            # одинаковый разбор, иначе одна и та же переписка при повторном
            # прогоне даст другой состав памяти.
            temperature=0.0,
        )
        session = Session(messages=[Message(role=Role.USER, content=text)])
        try:
            response = await self._router.chat(self._role, params, session)
        except LLMError as exc:
            logger.warning("perception: extraction request failed: %s", exc)
            return PerceptionBatch(source=source, parse_error=f"llm: {exc}")

        raw = response.text
        candidates, error = parse_payload(raw)
        if error:
            logger.warning("perception: could not parse model output (%s)", error)
        return PerceptionBatch(candidates=candidates, raw_response=raw, parse_error=error, source=source)


def parse_payload(raw: str) -> tuple[list[FactCandidate], str]:
    """
    Разбирает ответ модели в список кандидатов. Чистая функция: ни сети, ни
    состояния — поэтому все форматные причуды провайдеров проверяются тестами.

    Возвращает `(кандидаты, ошибка)`. Ошибка непустая только когда JSON не
    удалось найти или разобрать вовсе; отдельные негодные элементы внутри
    валидного массива не считаются ошибкой разбора — они пропускаются, а
    остальные кандидаты доезжают до валидатора. Терять весь эпизод из-за
    одного кривого элемента — худший из возможных обменов.
    """
    text = (raw or "").strip()
    if not text:
        return [], "пустой ответ модели"

    fenced = _JSON_FENCE_RE.search(text)
    if fenced is not None:
        text = fenced.group(1).strip()

    snippet = _extract_json_snippet(text)
    if snippet is None:
        return [], "в ответе не найдено ни массива, ни объекта JSON"

    try:
        payload = json.loads(snippet)
    except json.JSONDecodeError as exc:
        return [], f"невалидный JSON: {exc}"

    items = _as_item_list(payload)
    if items is None:
        return [], f"неожиданная структура ответа: {type(payload).__name__}"

    candidates: list[FactCandidate] = []
    for index, item in enumerate(items[:MAX_CANDIDATES]):
        if not isinstance(item, dict):
            logger.debug("perception: skipping non-object item #%d (%r)", index, item)
            continue
        try:
            candidate = FactCandidate.model_validate(item)
        except ValidationError as exc:
            logger.debug("perception: skipping malformed candidate #%d: %s", index, exc)
            continue
        if candidate.is_empty:
            continue
        candidates.append(candidate)
    return candidates, ""


def _as_item_list(payload: Any) -> list[Any] | None:
    """
    Приводит разные формы ответа к списку элементов.

    Поддерживаются: голый массив; объект-обёртка ({"facts": [...]},
    {"items": [...]}, {"candidates": [...]}); одиночный объект-факт (модель
    нашла ровно один и не стала заворачивать его в массив).
    """
    if isinstance(payload, list):
        return payload
    if not isinstance(payload, dict):
        return None
    for key in ("facts", "items", "candidates", "results", "data"):
        nested = payload.get(key)
        if isinstance(nested, list):
            return nested
    if any(key in payload for key in ("entity", "attribute", "value")):
        return [payload]
    return None


def _extract_json_snippet(text: str) -> str | None:
    """
    Вырезает первый сбалансированный JSON-массив или объект.

    Балансировкой скобок, а не регуляркой: значения фактов — свободный текст
    и вполне могут содержать `{`, `]` и кавычки, а регулярка на «от первой
    скобки до последней» ломается о преамбулу вида «Вот что я нашла: [...]»
    ровно тогда, когда после JSON модель добавила ещё и пояснение.
    """
    start_positions = [position for position in (text.find("["), text.find("{")) if position != -1]
    if not start_positions:
        return None
    start = min(start_positions)
    opening = text[start]
    closing = "]" if opening == "[" else "}"

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opening:
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


__all__ = [
    "MAX_CANDIDATES",
    "CandidateKind",
    "FactCandidate",
    "PerceptionBatch",
    "PerceptionParser",
    "parse_payload",
]
