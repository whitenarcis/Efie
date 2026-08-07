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

logger = logging.getLogger(__name__)

#: Дневниковая запись фонового исследования размечается этим тегом прямо в
#: тексте — front-matter записи (efi.llm.schemas.DiaryEntryMetadata) не хранит
#: произвольные теги, но текстовой метки достаточно, чтобы такие записи
#: находились простым поиском/грепом по дневнику.
AUTONOMOUS_THOUGHT_TAG = "#autonomous_thought"

_FINDING_CONFIDENCE = 0.6

_FINDING_SYSTEM_PROMPT = (
    "Собеседник недавно упоминал или спрашивал про тему, которая тебе стала любопытна. Тебе показаны "
    "результаты веб-поиска по ней. Сформулируй короткий (1-3 предложения) личный вывод или находку от "
    "первого лица — то, что ты реально узнала и хочешь рассказать собеседнику, а не сухую справку. Пиши "
    "так, будто поделилась бы этим в переписке. Не пиши 'по данным поиска' и не перечисляй источники."
)


@dataclass(slots=True, frozen=True)
class InformedThought:
    """Результат одного цикла фонового исследования — то, что Эфи "выяснила" по семени любопытства."""

    seed_id: int
    topic: str
    source_chat_id: int | None
    finding: str
    weight: float


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
        finding_role: TaskRole = TaskRole.FAST,
    ) -> None:
        self._curiosity = curiosity
        self._web_search = web_search
        self._rag = rag
        self._router = router
        self._organic_ping = organic_ping
        self._check_interval_seconds = check_interval_seconds
        self._finding_role = finding_role

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

        thought = await self._research(seed)
        # Помечаем researched СРАЗУ после попытки, независимо от её исхода —
        # неудачный поиск/LLM-сбой не должен держать семя pending вечно и
        # блокировать собой всё более весомые семена, появившиеся позже.
        await self._curiosity.mark_researched(seed.id)
        if thought is None:
            return

        entry = await self._rag.remember(
            f"{AUTONOMOUS_THOUGHT_TAG} {thought.topic}: {thought.finding}", confidence=_FINDING_CONFIDENCE
        )
        if entry is not None:
            logger.info("life_engine: researched seed #%s (%r) -> diary entry %s", seed.id, seed.topic, entry.id)

        await self._organic_ping.notify(thought)

    async def _research(self, seed: CuriositySeed) -> InformedThought | None:
        search_text = await self._web_search.execute({"query": seed.topic}, _make_research_context(seed.topic))
        if search_text.startswith("error:") or "ничего не нашлось" in search_text:
            logger.info("life_engine: search for seed #%s (%r) yielded nothing useful", seed.id, seed.topic)
            return None

        finding = await self._formulate_finding(seed.topic, search_text)
        if finding is None:
            return None

        return InformedThought(
            seed_id=seed.id,
            topic=seed.topic,
            source_chat_id=seed.source_chat_id,
            finding=finding,
            weight=seed.weight,
        )

    async def _formulate_finding(self, topic: str, search_text: str) -> str | None:
        params = LLMParams(model="", system_prompt=_FINDING_SYSTEM_PROMPT, max_output_tokens=256)
        session = Session(messages=[Message(role=Role.USER, content=f"Тема: {topic}\n\n{search_text}")])
        try:
            response = await self._router.chat(self._finding_role, params, session)
        except LLMError as exc:
            logger.warning("life_engine: finding formulation for %r failed: %s", topic, exc)
            return None
        return response.text.strip() or None


def _make_research_context(topic: str) -> ToolContext:
    """Синтетический ToolContext для WebSearchTool вне tool-calling цикла Worker'а (см. behavior/researcher.py)."""
    return ToolContext(notification=Notification(type=NotificationType.NIGHTLY_TASK, message=topic))


__all__ = ["InformedThought", "OrganicPingSink", "BackgroundLifeWorker", "AUTONOMOUS_THOUGHT_TAG"]
