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
Проблема не в самих словах — «как дела» живые люди пишут постоянно, и ничего
плохого в этом нет. Проблема в том, что сказать было НЕЧЕГО, и «ты там ещё не
утонул?» оставалось единственным содержанием. Инструкцией «пиши интереснее»
это не лечится. Лечится поводом.

Поводы берутся из того, что Эфи реально помнит, в порядке убывания
ценности:

  1. Инкубированная мысль — она сама что-то нагуглила и обдумала
     (efi/behavior/researcher.py). Лучший повод из возможных: это её
     собственная мысль, а не вежливый интерес к чужой жизни.
  2. Незакрытое обещание (efi/memory/working_memory.py) — «обещала скинуть
     ссылку». Самый обязывающий повод: человек его помнит и ждёт.
  3. Факт о собеседнике из строгой памяти (efi/memory/dedup.py) — «он
     работает монтажёром», «у него кот Феникс». Из такого получается
     нормальный человеческий вопрос про конкретную вещь.
  4. Собственная запись дневника за последние сутки (efi/memory/diary.py) —
     то, что она сама прожила и о чём может рассказать. Самый слабый из
     поводов, но именно он не даёт ей замолчать, пока строгая память ещё
     пустая: дневник наполняется с первого же разговора.

Если ни одного повода нет — сообщения тоже нет. Это главное правило модуля:
молчание лучше, чем «эй». Молчание читается как «занята своими делами»,
а пустой пинг — как навязчивость, и второй такой пинг уже раздражает.

Порядок именно такой, потому что он от обязательства к болтовне: обещание
надо выполнить, факт — это интерес к человеку, а своя запись — это уже «мне
есть чем поделиться». Первый найденный и выигрывает; перебирать все и
выбирать «лучший» незачем, они не сравнимы между собой.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Awaitable, Callable
from datetime import UTC, datetime, timedelta

from efi.memory.dedup import KnowledgeStore, StoredFact
from efi.memory.diary import Diary
from efi.memory.people import PeopleStore
from efi.memory.router import MemoryDomain
from efi.memory.working_memory import WorkingMemory, WorkingMemoryItem

logger = logging.getLogger(__name__)

#: Сколько фактов о человеке рассматривается как возможный повод. Берём
#: несколько и выбираем случайный, а не самый частый: самый частый повод
#: заводил бы разговор об одном и том же каждый раз.
_FACT_POOL = 6

#: Атрибуты, по которым заговаривать не стоит. Это либо служебное, либо то,
#: что в лоб звучит как допрос («а ты всё ещё живёшь в Минске?»).
_BORING_ATTRIBUTES = frozenset({"имя", "name", "возраст", "город", "адрес", "телефон"})

#: Насколько свежей должна быть запись дневника, чтобы годиться в повод.
#: Сутки: «я вчера читала про X» — это ещё разговор, «я на прошлой неделе» —
#: уже натянуто.
_DIARY_FRESHNESS = timedelta(days=1)

#: Сколько свежих записей рассматривается. Больше незачем: выбирается одна
#: случайная, а длинный список только тянет память и время.
_DIARY_POOL = 8

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
        diary: Diary | None = None,
        working_memory: WorkingMemory | None = None,
        incubated_thought_provider: IncubatedThoughtProvider | None = None,
    ) -> None:
        self._knowledge = knowledge
        self._people = people
        self._diary = diary
        self._working_memory = working_memory
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
        promise = await self._pending_promise(chat_id)
        if promise is not None:
            return _render_promise_reason(promise)

        fact = await self._recall_fact(chat_id)
        if fact is not None:
            return _render_fact_reason(fact)

        entry = await self._recent_diary_entry()
        if entry is not None:
            return _render_diary_reason(entry)
        return None

    async def _pending_promise(self, chat_id: int) -> WorkingMemoryItem | None:
        """Незакрытое обещание, данное именно в этом чате."""
        if self._working_memory is None:
            return None
        try:
            snapshot = await self._working_memory.load()
        except Exception:
            logger.warning("ping_reason: не удалось прочитать рабочую память", exc_info=True)
            return None
        open_items = [
            item for item in snapshot.items if not item.done and item.chat_id in (chat_id, None)
        ]
        return open_items[0] if open_items else None

    async def _recent_diary_entry(self) -> str | None:
        """
        Свежая запись из её собственного дневника.

        Самый слабый повод из всех — но именно он не даёт Эфи замолчать,
        пока строгая память ещё пустая. Дневник наполняется с первого же
        разговора, а knowledge_facts — только когда в разговоре прозвучал
        устойчивый факт, что бывает далеко не каждый день.
        """
        if self._diary is None:
            return None
        try:
            entries = await self._diary.all_entries()
        except Exception:
            logger.warning("ping_reason: не удалось прочитать дневник", exc_info=True)
            return None

        cutoff = datetime.now(UTC) - _DIARY_FRESHNESS
        fresh = [
            entry.body.strip()
            for entry in entries
            if entry.metadata.created_at >= cutoff and entry.body.strip()
        ]
        return random.choice(fresh[-_DIARY_POOL:]) if fresh else None

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


def _render_promise_reason(item: WorkingMemoryItem) -> str:
    """
    Повод-обещание — самый обязывающий: человек его помнит и ждёт.

    Формулировка не «напомни, что обещала», а «сделай»: обещание закрывается
    выполнением, а не сообщением о том, что оно ещё висит.
    """
    return (
        f"Ты обещала вот что и до сих пор не сделала: {item.text}. Напиши по этому поводу — "
        "не отчётом «я помню, что обещала», а собственно тем, что обещала, или честным «не успела, "
        "но помню»."
    )


def _render_diary_reason(entry: str) -> str:
    """
    Повод-запись: у неё был свой день, и об этом можно рассказать.

    Именно «расскажи», а не «процитируй»: запись дневника написана для себя
    и звучит как дневник, а в чат нужна живая фраза.
    """
    return (
        f"Из того, что у тебя было за последнее время: {entry} "
        "Тебе захотелось этим поделиться. Расскажи коротко и своими словами, как рассказывают "
        "приятелю, — не зачитывай запись и не начинай с «а я тут»."
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
