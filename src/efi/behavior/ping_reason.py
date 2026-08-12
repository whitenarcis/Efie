"""
efi/behavior/ping_reason.py

С чем именно Эфи приходит, когда пишет первой.

Раньше повода не было вовсе. Спонтанный пинг подставлял в уведомление строку
«У тебя есть желание написать первой, без особого повода — просто чтобы
напомнить о себе», и модель, получив ровно ноль содержания, честно
отрабатывала единственным доступным способом:

    эй
    ты там ещё не утонул в своём коде?

Претензий к модели тут нет: из «напомни о себе» ничего другого не следует.
«Как дела» и «ты там живой?» — это то, что говорят, когда сказать нечего, и
никакой инструкцией «пиши интереснее» это не лечится. Лечится поводом.

Поводы берутся из того, что Эфи реально помнит, в порядке убывания
ценности:

  1. Инкубированная мысль — она сама что-то нагуглила и обдумала
     (efi/behavior/researcher.py). Лучший повод из возможных: это её
     собственная мысль, а не вежливый интерес к чужой жизни.
  2. Факт о собеседнике из строгой памяти (efi/memory/dedup.py) — «он
     работает монтажёром», «у него кот Феникс». Из такого получается
     нормальный человеческий вопрос про конкретную вещь.

Если ни одного повода нет — сообщения тоже нет. Это главное правило модуля:
молчание лучше, чем «эй». Молчание читается как «занята своими делами»,
а пустой пинг — как навязчивость, и второй такой пинг уже раздражает.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Awaitable, Callable

from efi.memory.dedup import KnowledgeStore, StoredFact
from efi.memory.people import PeopleStore
from efi.memory.router import MemoryDomain

logger = logging.getLogger(__name__)

#: Сколько фактов о человеке рассматривается как возможный повод. Берём
#: несколько и выбираем случайный, а не самый частый: самый частый повод
#: заводил бы разговор об одном и том же каждый раз.
_FACT_POOL = 6

#: Атрибуты, по которым заговаривать не стоит. Это либо служебное, либо то,
#: что в лоб звучит как допрос («а ты всё ещё живёшь в Минске?»).
_BORING_ATTRIBUTES = frozenset({"имя", "name", "возраст", "город", "адрес", "телефон"})

#: Функция, возвращающая текст инкубированной мысли и СБРАСЫВАЮЩАЯ её (см.
#: BackgroundResearcher.consume_incubated_thought), либо None.
IncubatedThoughtProvider = Callable[[], Awaitable[str | None]]


class PingReasonBuilder:
    """
    Ищет повод написать первой. Не находит — значит, писать не о чем.

    Все источники необязательны: без них builder честно отвечает «повода
    нет», и инициативные службы молчат. Это правильный дефолт — молчащая Эфи
    выглядит занятой, а Эфи с «эй» выглядит навязчивой.
    """

    def __init__(
        self,
        *,
        knowledge: KnowledgeStore | None = None,
        people: PeopleStore | None = None,
        incubated_thought_provider: IncubatedThoughtProvider | None = None,
    ) -> None:
        self._knowledge = knowledge
        self._people = people
        self._incubated_thought_provider = incubated_thought_provider

    async def consume_incubated_thought(self) -> str | None:
        """
        Мысль, которую Эфи выносила сама. Одноразовая по своей природе:
        отданная одному чату, она не должна повториться в другом.
        """
        if self._incubated_thought_provider is None:
            return None
        try:
            return await self._incubated_thought_provider()
        except Exception:
            logger.warning("ping_reason: не удалось получить инкубированную мысль", exc_info=True)
            return None

    async def reason_for(self, chat_id: int) -> str | None:
        """
        Повод написать в этот конкретный чат, либо None.

        Сбой любого источника — это «повода нет», а не исключение наружу:
        инициатива не настолько важна, чтобы ронять из-за неё фоновый цикл.
        """
        fact = await self._recall_fact(chat_id)
        if fact is not None:
            return _render_fact_reason(fact)
        return None

    async def _recall_fact(self, chat_id: int) -> StoredFact | None:
        if self._knowledge is None or self._people is None:
            return None
        try:
            people = await self._people.recent(limit=20)
            entity_ids = [f"user:{profile.user_id}" for profile in people if profile.last_chat_id == chat_id]
            if not entity_ids:
                return None
            facts = await self._knowledge.recall(
                entity_ids=entity_ids, domains=[MemoryDomain.PERSONAL], limit=_FACT_POOL
            )
        except Exception:
            logger.warning("ping_reason: не удалось прочитать факты для chat_id=%s", chat_id, exc_info=True)
            return None

        usable = [fact for fact in facts if fact.attribute.lower() not in _BORING_ATTRIBUTES]
        return random.choice(usable) if usable else None


def render_incubated_reason(thought: str) -> str:
    """Повод-мысль: она сама до этого додумалась, и это лучшее, с чем можно прийти."""
    return (
        f"Пока было тихо, тебе самой пришла в голову мысль (сама погуглила и подумала): {thought} "
        "Поделись этим с собеседником как своей спонтанной идеей — а не дежурным 'как дела'."
    )


def _render_fact_reason(fact: StoredFact) -> str:
    """
    Повод-факт: спросить про конкретную вещь, которую Эфи о человеке знает.

    Именно «спроси про это», а не «упомяни это»: разница между «как там твой
    монтаж?» и «я помню, что ты монтажёр» — это разница между разговором и
    зачитыванием досье.
    """
    attribute = fact.attribute.replace("_", " ")
    return (
        f"Ты вспомнила про собеседника вот что — {attribute}: {fact.value}. "
        "Тебе стало интересно, как там с этим дела сейчас. Спроси про это конкретно, "
        "коротко и по-человечески, как будто правда вспомнила и решила узнать. "
        "Не пересказывай, что ты помнишь, — просто спроси."
    )


__all__ = ["IncubatedThoughtProvider", "PingReasonBuilder", "render_incubated_reason"]
