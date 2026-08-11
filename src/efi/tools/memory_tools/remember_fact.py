"""
efi/tools/memory_tools/remember_fact.py

Инструмент сохранения структурированного факта — теперь через границу
доверия, а не прямой записью.

Что изменилось и почему. Раньше инструмент брал аргументы модели и клал их в
`facts` как есть: модель сама выбирала entity_id, сама придумывала ключ, сама
решала, что это важно. Любая её галлюцинация становилась строкой в базе,
неотличимой от настоящего наблюдения, а служебные ключи приложения
(`last_novelized_at`) она могла перезаписать одним удачно сформулированным
вызовом.

Теперь аргументы модели — это КАНДИДАТ (efi/memory/parser.py::FactCandidate),
который проходит ту же детерминированную проверку, что и всё остальное
(efi/memory/validator.py), и попадает в память только через
efi/memory/dedup.py::KnowledgeStore. Инструмент остался у модели, но перо у
неё забрали: она предлагает, записывает код.

Побочный выигрыш — дедупликация: повторный вызов с тем же фактом больше не
перезаписывает запись, а увеличивает счётчик подтверждений, и модель об этом
узнаёт из ответа инструмента («уже знала, теперь подтверждено N раз»).
"""

from __future__ import annotations

from typing import Any

from efi.memory.dedup import KnowledgeStore, StoreAction
from efi.memory.parser import FactCandidate
from efi.memory.validator import FactValidator, Rejection
from efi.tools.base import Tool, ToolContext


class RememberFactTool(Tool):
    """Позволяет модели ПРЕДЛОЖИТЬ факт к запоминанию; решение о записи принимает валидатор."""

    name = "remember_fact"
    description = (
        "Сохраняет короткий структурированный факт о ком-то или о чём-то — например, "
        "любимый цвет, день рождения, кличку питомца, устойчивую привычку. Используй для точных, "
        "легко формулируемых фактов; для свободных воспоминаний и историй используй дневник (ask_diary). "
        "Факт проходит проверку: если он повторяет уже известное, вместо новой записи растёт счётчик "
        "подтверждений."
    )
    parameters = {
        "type": "object",
        "properties": {
            "domain": {
                "type": "string",
                "enum": ["C", "P", "H"],
                "description": (
                    "Домен памяти: C — знание о мире (технологии, концепции, общие факты), "
                    "P — про конкретного человека, H — твой собственный опыт и переживания"
                ),
            },
            "entity_id": {
                "type": "string",
                "description": "Кого/чего касается факт: имя человека, 'я' про собеседника, тема для домена C",
            },
            "key": {"type": "string", "description": "Название факта, например 'любимый_цвет'"},
            "value": {"type": "string", "description": "Значение факта"},
            "confidence": {
                "type": "number",
                "description": "Насколько ты уверена в этом факте, от 0 до 1 (по умолчанию 1.0 — точно знаешь)",
            },
        },
        "required": ["entity_id", "key", "value"],
        "additionalProperties": False,
    }

    def __init__(self, store: KnowledgeStore, validator: FactValidator) -> None:
        self._store = store
        self._validator = validator

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        entity = str(arguments.get("entity_id", "")).strip()
        key = str(arguments.get("key", "")).strip()
        value = str(arguments.get("value", "")).strip()
        if not entity or not key or not value:
            return "error: entity_id, key and value must all be non-empty"

        candidate = FactCandidate(
            # Домен по умолчанию — P: инструмент почти всегда зовут посреди
            # разговора про человека, и это самая безопасная догадка. Ошибку
            # догадки поймает проверка согласованности домена и сущности
            # (validator._check_domain_consistency), а не тишина.
            domain=str(arguments.get("domain", "") or "P"),
            entity=entity,
            attribute=key,
            value=value,
            confidence=_coerce_confidence(arguments.get("confidence", 1.0)),
        )
        source = f"chat:{context.chat_id}" if context.chat_id is not None else "tool"
        outcome = self._validator.validate(candidate, source=source)

        if isinstance(outcome, Rejection):
            await self._store.record_rejections([outcome], source=source)
            # Причина возвращается модели специально: она может исправиться в
            # том же ходу (например, переформулировать служебный ключ), а
            # молчаливый отказ ничему её не учит.
            return f"не записала: {outcome.reason}"

        stored = await self._store.remember(outcome)
        if stored.action is StoreAction.REINFORCED:
            return (
                f"это я уже знала — подтвердила ещё раз "
                f"(всего упоминаний: {stored.fact.occurrence_count})"
            )
        return f"запомнила: {stored.fact.entity_id}.{stored.fact.attribute} = {stored.fact.value!r}"


def _coerce_confidence(raw: Any) -> float:
    try:
        value = float(raw)
    except (TypeError, ValueError):
        value = 1.0
    return max(0.0, min(value, 1.0))


__all__ = ["RememberFactTool"]
