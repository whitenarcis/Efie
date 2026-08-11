"""
efi/behavior/life_engine.py

Движок фоновой автономии — раз в N минут забирает семя любопытства
(efi.behavior.curiosity.CuriosityTracker) с наибольшим весом, исследует его
через веб-поиск и формулирует по результату СОБСТВЕННЫЙ вывод (InformedThought),
который сохраняется в дневник с тегом #autonomous_thought и передаётся
efi.behavior.organic_ping.OrganicPingGenerator — тот решает, достаточно ли
находка важна, чтобы Эфи сама написала о ней собеседнику.

Тот же принцип разделения ответственности, что и у efi.behavior.researcher.
BackgroundResearcher, но другой источник тем: BackgroundResearcher берёт темы
из worldview.json (личные интересы Эфи вообще), этот модуль — из
curiosity_seeds (темы, реально всплывшие в разговорах с собеседниками).
Формулировка вывода — единственное место, где идёт обращение к LLM (роль
FAST, как и у BackgroundResearcher/DiaryConsolidator); сам цикл живёт по
собственному нечастому расписанию, вне критического пути ответа собеседнику.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from typing import Protocol

from efi.behavior.curiosity import CuriositySeed, CuriosityTracker
from efi.config.schema import TaskRole
from efi.llm.errors import LLMError
from efi.llm.router import LLMRouter
from efi.llm.schemas import LLMParams, Message, Role, Session
from efi.memory.rag import RAGMemory
from efi.notifications.schemas import Notification, NotificationType
from efi.tools.base import ToolContext
from efi.tools.web_tools.web_search import WebSearchTool
from efi.utils.text import salvage_truncated

logger = logging.getLogger(__name__)

#: Дневниковая запись фонового исследования размечается этим тегом прямо в
#: тексте — front-matter записи (efi.llm.schemas.DiaryEntryMetadata) не хранит
#: произвольные теги, но текстовой метки достаточно, чтобы такие записи
#: находились простым поиском/грепом по дневнику.
AUTONOMOUS_THOUGHT_TAG = "#autonomous_thought"

_FINDING_CONFIDENCE = 0.6

#: Бюджеты вывода. Промпты просят 1-3 предложения и одну строчку — по
#: английским меркам 256/128 токенов хватало, но кириллица у токенизаторов
#: бесплатных моделей стоит в 2-3 раза дороже, и находка регулярно
#: обрывалась на полуслове. Взято по худшему курсу.
_FINDING_MAX_OUTPUT_TOKENS = 512
_REACTION_MAX_OUTPUT_TOKENS = 256

_FINDING_SYSTEM_PROMPT = (
    "Собеседник недавно упоминал или спрашивал про тему, которая тебе стала любопытна. Тебе показаны "
    "результаты веб-поиска по ней. Сформулируй короткий (1-3 предложения) личный вывод или находку от "
    "первого лица — то, что ты реально узнала и хочешь рассказать собеседнику, а не сухую справку. Пиши "
    "так, будто поделилась бы этим в переписке. Не пиши 'по данным поиска' и не перечисляй источники."
)

#: Второй, отдельный запрос по тем же результатам поиска — ЛИЧНОЕ отношение к
#: прочитанному. Это то, что превращает дневник из склада выжимок в дневник
#: прожитого опыта: не только "что узнала", но и "что я об этом думаю и
#: почувствовала". Без него запись фонового исследования читалась как
#: справочная карточка, а сама Эфи не могла сослаться на своё впечатление в
#: разговоре ("я тут вычитала, и меня это, честно, взбесило").
_REACTION_SYSTEM_PROMPT = (
    "Ты только что читала материалы в интернете по теме и сделала для себя вывод. Напиши ОДНО короткое "
    "предложение от первого лица — своё ЛИЧНОЕ отношение к прочитанному: что зацепило, удивило, "
    "разозлило, показалось бредом или, наоборот, восхитило. Это строчка в твой личный дневник, а не "
    "оценка для кого-то. Никаких 'важно отметить' и прочей аналитики — только живая реакция."
)


@dataclass(slots=True, frozen=True)
class InformedThought:
    """Результат одного цикла фонового исследования — то, что Эфи "выяснила" по семени любопытства."""

    seed_id: int
    topic: str
    source_chat_id: int | None
    finding: str
    weight: float
    #: Личное отношение к прочитанному (см. _REACTION_SYSTEM_PROMPT). Пустая
    #: строка — штатный случай: реакцию не удалось получить, а находка сама
    #: по себе всё равно ценна и должна дойти до дневника и до пинга.
    reaction: str = ""


class OrganicPingSink(Protocol):
    """Абстракция получателя готовых находок. Конкретная реализация — efi.behavior.organic_ping.OrganicPingGenerator."""

    async def notify(self, thought: InformedThought) -> None: ...


class BackgroundLifeWorker:
    """
    Основной цикл движка фоновой автономии. Предназначен для запуска через
    `asyncio.create_task(worker.run())` при старте приложения, как и
    остальные фоновые сервисы efi/behavior/. Отсутствие pending-семян, сбой
    поиска или LLM не должны ронять цикл — семя просто останется/станет
    researched без находки, следующий тик возьмёт следующее по весу.
    """

    def __init__(
        self,
        curiosity: CuriosityTracker,
        web_search: WebSearchTool,
        rag: RAGMemory,
        router: LLMRouter,
        organic_ping: OrganicPingSink,
        *,
        check_interval_seconds: float = 1800.0,
        finding_role: TaskRole = TaskRole.BACKGROUND,
    ) -> None:
        self._curiosity = curiosity
        self._web_search = web_search
        self._rag = rag
        self._router = router
        self._organic_ping = organic_ping
        self._check_interval_seconds = check_interval_seconds
        self._finding_role = finding_role
        self._is_researching = False

    @property
    def is_researching(self) -> bool:
        """
        True на всё время `_research()` (веб-поиск + формулировка находки) —
        вход для efi.behavior.busy_engine.BusyEngine: пока Эфи занята фоновым
        исследованием, ignore_delay перед реакцией на входящее сообщение
        увеличивается, она "не сразу отвлекается на телефон".
        """
        return self._is_researching

    async def run(self) -> None:
        """Основной цикл. Останавливается по отмене задачи (CancelledError) — см. efi/app.py graceful shutdown."""
        logger.info("life_engine: started (interval=%.0fs)", self._check_interval_seconds)
        try:
            while True:
                await asyncio.sleep(self._check_interval_seconds)
                await self._tick()
        except asyncio.CancelledError:
            logger.info("life_engine: stopped")
            raise

    async def _tick(self) -> None:
        seed = await self._curiosity.pick_top_pending()
        if seed is None:
            return

        self._is_researching = True
        try:
            thought = await self._research(seed)
        finally:
            self._is_researching = False

        # Помечаем researched СРАЗУ после попытки, независимо от её исхода —
        # неудачный поиск/LLM-сбой не должен держать семя pending вечно и
        # блокировать собой всё более весомые семена, появившиеся позже.
        await self._curiosity.mark_researched(seed.id)
        if thought is None:
            return

        entry = await self._rag.remember(_render_diary_entry(thought), confidence=_FINDING_CONFIDENCE)
        if entry is not None:
            logger.info("life_engine: researched seed #%s (%r) -> diary entry %s", seed.id, seed.topic, entry.id)

        await self._organic_ping.notify(thought)

    async def _research(self, seed: CuriositySeed) -> InformedThought | None:
        # search(), а не execute(): нужна причина неудачи, а не текст для
        # модели — см. тот же разбор в efi/behavior/researcher.py.
        outcome = await self._web_search.search(
            seed.topic, journal_context=_make_research_context(seed.topic)
        )
        if outcome.failed:
            logger.warning(
                "life_engine: search for seed #%s (%r) failed: %s", seed.id, seed.topic, outcome.error
            )
            return None
        if not outcome.results:
            logger.info("life_engine: search for seed #%s (%r) returned no results", seed.id, seed.topic)
            return None

        search_text = outcome.render()
        finding = await self._formulate_finding(seed.topic, search_text)
        if finding is None:
            return None

        # Личное отношение — вторым запросом, УЖЕ после того, как находка
        # получена: если этот запрос не удастся, находка всё равно уцелеет
        # (reaction останется пустой), а не потеряется вместе с ним.
        reaction = await self._formulate_reaction(seed.topic, search_text, finding)

        return InformedThought(
            seed_id=seed.id,
            topic=seed.topic,
            source_chat_id=seed.source_chat_id,
            finding=finding,
            weight=seed.weight,
            reaction=reaction or "",
        )

    async def _formulate_finding(self, topic: str, search_text: str) -> str | None:
        return await self._ask(
            _FINDING_SYSTEM_PROMPT, f"Тема: {topic}\n\n{search_text}", what="finding", topic=topic
        )

    async def _formulate_reaction(self, topic: str, search_text: str, finding: str) -> str | None:
        return await self._ask(
            _REACTION_SYSTEM_PROMPT,
            f"Тема: {topic}\n\nЧто ты вычитала:\n{search_text}\n\nТвой вывод: {finding}",
            what="reaction",
            topic=topic,
            max_output_tokens=_REACTION_MAX_OUTPUT_TOKENS,
        )

    async def _ask(
        self,
        system_prompt: str,
        user_content: str,
        *,
        what: str,
        topic: str,
        max_output_tokens: int = _FINDING_MAX_OUTPUT_TOKENS,
    ) -> str | None:
        """Один короткий запрос к фоновой роли; сбой — не исключение наружу, а None (цикл жизни не должен падать)."""
        params = LLMParams(model="", system_prompt=system_prompt, max_output_tokens=max_output_tokens)
        session = Session(messages=[Message(role=Role.USER, content=user_content)])
        try:
            response = await self._router.chat(self._finding_role, params, session)
        except LLMError as exc:
            logger.warning("life_engine: %s formulation for %r failed: %s", what, topic, exc)
            return None

        # Обрыв по лимиту — не ошибка запроса: ответ пришёл успешно, просто
        # он неполный. Без этой проверки обрубок уходил прямо в дневник.
        text = salvage_truncated(response.text, truncated=response.was_truncated)
        if response.was_truncated:
            logger.warning(
                "life_engine: %s for %r hit the output limit (%s tokens); %s",
                what,
                topic,
                max_output_tokens,
                "trimmed to the last complete sentence" if text else "nothing salvageable, dropping",
            )
        return text or None


def _render_diary_entry(thought: InformedThought) -> str:
    """
    Запись в дневник о прожитом фоновом опыте, а не выжимка из статьи.

    Формат намеренно повествовательный ("читала про X ... поняла ... и меня
    это ..."), потому что эта же запись потом находится RAG-поиском и
    подмешивается в системный промпт: из неё Эфи должна суметь сослаться на
    свой опыт живой фразой ("я тут вычитала про X"), а не зачитать карточку.
    """
    parts = [f"{AUTONOMOUS_THOUGHT_TAG} Читала сегодня про {thought.topic}. {thought.finding}"]
    if thought.reaction:
        parts.append(thought.reaction)
    return " ".join(parts)


def _make_research_context(topic: str) -> ToolContext:
    """Синтетический ToolContext для WebSearchTool вне tool-calling цикла Worker'а (см. behavior/researcher.py)."""
    return ToolContext(notification=Notification(type=NotificationType.NIGHTLY_TASK, message=topic))


__all__ = ["InformedThought", "OrganicPingSink", "BackgroundLifeWorker", "AUTONOMOUS_THOUGHT_TAG"]
