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

Обещание — единственный повод с приоритетом: его человек ждёт, и оно
обязывает. Всё остальное выбирается СЛУЧАЙНО из того, что нашлось, с весами.
Строгая лестница «факт → дневник → ничего» давала предсказуемое однообразие:
пока в дневнике был код и музыка, каждое её сообщение было про код и музыку.
Живой человек пишет первым по десятку разных причин, и большинство из них
вообще ни о чём:

    «как ты?»                     — просто так, без содержания
    «доброе утро»                 — потому что утро
    «я сегодня никакая»           — потому что своё состояние тоже повод
    «мы сто лет не разговаривали» — потому что заметил паузу
    «ты ел вообще?»               — бытовая мелочь

Ни один из этих поводов не «интересный», и в этом всё дело: разговор между
людьми состоит из них процентов на восемьдесят. Поэтому они здесь наравне с
содержательными — и берутся не по остаточному принципу, а с нормальными
весами.

Молчание по-прежнему возможно и по-прежнему лучше навязчивости: если поводов
нет вовсе (не с кем и нечего), служба молчит.
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
from efi.utils.clock import local_now

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

#: Через сколько молчания уместно сказать «сто лет не общались». Двое суток:
#: раньше это звучит как претензия, позже — как констатация.
_LONG_SILENCE = timedelta(days=2)

#: Веса поводов. Не «важность», а частота: так и распределены поводы у живого
#: человека — содержательное случается реже, чем «как ты?».
#:
#: Обещания в таблице нет намеренно: оно не участвует в жеребьёвке, а
#: выигрывает сразу (см. reason_for).
_WEIGHTS: dict[str, int] = {
    "fact": 3,      # спросить про конкретную вещь из жизни человека
    "diary": 3,     # рассказать своё
    "check_in": 4,  # просто «как ты» — самое частое, что пишут люди
    "mood": 2,      # своё состояние прямо сейчас
    "time": 2,      # доброе утро / ты чего не спишь
    "silence": 2,   # давно не разговаривали
    "trifle": 3,    # бытовая мелочь ни о чём
}

#: Бытовые мелочи — то, с чем пишут, когда писать не о чем, и это нормально.
#: Хранится списком, а не генерируется моделью: список из живой переписки
#: короче и честнее любого «придумай бытовой вопрос».
#:
#: Две первые редакции списка («не сдох там от работы», «жив там») пришлось
#: убрать: это ровно тот самый вопрос «ты ещё здесь», который запрещён
#: отдельным правилом промпта, — только завёрнутый в бытовую обёртку.
_TRIFLES = (
    "ел сегодня вообще или опять на одном кофе",
    "чем занят",
    "как оно там",
    "что слушаешь",
    "выспался хоть",
    "как настроение",
    "погода у вас такая же мерзкая",
    "делаешь что-нибудь интересное или так, отдыхаешь",
    "как день прошёл",
    "не устал совсем",
)


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
        timezone: str = "",
    ) -> None:
        self._knowledge = knowledge
        self._people = people
        self._diary = diary
        self._working_memory = working_memory
        self._incubated_thought_provider = incubated_thought_provider
        # Пояс нужен ровно одному поводу — «доброе утро»/«ты чего не спишь», —
        # но без него этот повод опаснее, чем полезен: под proot и в cron
        # переменная TZ пуста, процесс живёт по UTC, и Эфи желает доброго утра
        # в час ночи. См. efi/utils/clock.py.
        self._timezone = timezone

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

    async def reason_for(self, chat_id: int, *, now: datetime | None = None) -> str | None:
        """
        Повод написать в этот конкретный чат, либо None.

        Обещание выигрывает сразу — его ждут. Всё остальное разыгрывается по
        весам среди того, что нашлось: без жеребьёвки её инициатива была
        предсказуемой до буквы («в дневнике код — значит, опять про код»), а
        у живого человека поводы разные и в основном пустяковые.

        Сбой любого источника — это «этого повода нет», а не исключение
        наружу: инициатива не настолько важна, чтобы ронять из-за неё фоновый
        цикл.
        """
        moment = now or datetime.now(UTC)

        promise = await self._pending_promise(chat_id)
        if promise is not None:
            return _render_promise_reason(promise)

        candidates: dict[str, str] = {}

        fact = await self._recall_fact(chat_id)
        if fact is not None:
            candidates["fact"] = _render_fact_reason(fact)

        entry = await self._recent_diary_entry()
        if entry is not None:
            candidates["diary"] = _render_diary_reason(entry)

        state = await self._self_state()
        if state:
            candidates["mood"] = _render_mood_reason(state)

        silence = await self._silence_since(chat_id, now=moment)
        if silence is not None:
            candidates["silence"] = _render_silence_reason(silence)

        # Бытовые поводы («как ты», «доброе утро») доступны только там, где
        # разговор УЖЕ был. В чате, где никто ни разу не писал, «как дела» —
        # это не болтовня, а сообщение от незнакомого номера; там по-прежнему
        # действует старое правило: нет содержательного повода — нет
        # сообщения.
        if await self._knows_this_chat(chat_id):
            candidates["check_in"] = _render_check_in_reason()
            candidates["trifle"] = _render_trifle_reason()
            time_reason = _render_time_reason(local_now(self._timezone, now=moment))
            if time_reason:
                candidates["time"] = time_reason

        return _pick_weighted(candidates)

    async def _knows_this_chat(self, chat_id: int) -> bool:
        """Разговаривали ли в этом чате вообще — см. reason_for."""
        return await self._last_seen_in(chat_id) is not None

    async def _self_state(self) -> str:
        """Как она себя сейчас чувствует — её собственное состояние тоже повод написать."""
        if self._working_memory is None:
            return ""
        try:
            snapshot = await self._working_memory.load()
        except Exception:
            logger.warning("ping_reason: не удалось прочитать состояние", exc_info=True)
            return ""
        parts = [snapshot.emotional_state.strip(), snapshot.physical_state.strip()]
        return "; ".join(part for part in parts if part)

    async def _silence_since(self, chat_id: int, *, now: datetime) -> timedelta | None:
        """Сколько молчит этот чат — если молчит достаточно долго, чтобы это заметить."""
        last_seen = await self._last_seen_in(chat_id)
        if last_seen is None:
            return None
        elapsed = now - last_seen
        return elapsed if elapsed >= _LONG_SILENCE else None

    async def _last_seen_in(self, chat_id: int) -> datetime | None:
        """Когда в этом чате последний раз кто-то писал. None — не писал никто и никогда."""
        if self._people is None:
            return None
        try:
            people = await self._people.recent(limit=20)
        except Exception:
            logger.warning("ping_reason: не удалось прочитать людей", exc_info=True)
            return None
        seen = [
            profile.last_seen_at
            for profile in people
            if profile.last_chat_id == chat_id and profile.last_seen_at is not None
        ]
        return max(seen) if seen else None

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


def _pick_weighted(candidates: dict[str, str]) -> str | None:
    """Случайный повод из найденных, с весами. Пусто — значит, писать не о чем."""
    if not candidates:
        return None
    kinds = list(candidates)
    weights = [_WEIGHTS.get(kind, 1) for kind in kinds]
    chosen = random.choices(kinds, weights=weights, k=1)[0]
    logger.debug("ping_reason: повод «%s»", chosen)
    return candidates[chosen]


def _render_check_in_reason() -> str:
    """
    Самый частый повод у живых людей и до сих пор отсутствовавший здесь:
    никакого содержания, просто спросить, как человек.
    """
    return (
        "Тебе просто захотелось узнать, как он там. Никакого повода, никакой темы — так и пиши: "
        "коротко спроси, как дела или чем занят, одной-двумя короткими репликами. "
        "Не придумывай себе содержательный предлог и не рассказывай ничего в довесок."
    )


def _render_trifle_reason() -> str:
    """Бытовая мелочь — то, с чем пишут, когда писать не о чем."""
    return (
        f"Тебе захотелось спросить какую-нибудь бытовую мелочь, например: «{random.choice(_TRIFLES)}». "
        "Спроси примерно об этом своими словами — коротко, без вступления и без пояснений, зачем "
        "спрашиваешь."
    )


def _render_mood_reason(state: str) -> str:
    """Своё состояние: люди пишут «я сегодня варёная» безо всякого повода."""
    return (
        f"Ты сейчас вот в каком состоянии: {state}. Тебе захотелось про это сказать — просто "
        "поделиться, а не пожаловаться и не объяснять причины. Одна короткая реплика, можно с "
        "вопросом в конце, если он сам напрашивается."
    )


def _render_silence_reason(elapsed: timedelta) -> str:
    """«Сто лет не общались» — повод, который замечает человек, а не календарь."""
    days = max(int(elapsed.total_seconds() // 86400), 2)
    return (
        f"Вы не разговаривали уже {days} дня(ей), и ты это заметила. Напиши об этом коротко и без "
        "упрёка — не «ты пропал», а по-человечески: соскучилась, интересно, как он. "
        "Одна-две короткие реплики."
    )


def _render_time_reason(now: datetime) -> str:
    """
    Время суток как повод. Работает только в свои часы: «доброе утро» в час
    дня — это не повод, а неловкость.

    Момент приходит уже приведённым к поясу Эфи (см. reason_for): считать час
    здесь самостоятельно нельзя — под proot процесс живёт по UTC.
    """
    hour = now.hour
    if 6 <= hour <= 10:
        return (
            "Раннее утро, и ты уже на ногах. Напиши что-нибудь утреннее и очень короткое — "
            "поздороваться, спросить, проснулся ли он. Без пожеланий хорошего дня открыткой."
        )
    if hour >= 23 or hour <= 3:
        return (
            "Глубокая ночь, и ты ещё не спишь. Напиши что-то ночное и короткое — что не спится, "
            "или спроси, чего он не спит. Без нравоучений про режим."
        )
    return ""


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
