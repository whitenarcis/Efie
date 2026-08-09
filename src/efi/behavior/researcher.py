"""
efi/behavior/researcher.py

Автономные фоновые исследования — на моменты затишья Эфи сама выбирает тему
из worldview.json ("что мне вообще интересно"), гуглит её (WebSearchTool),
формулирует по результату СОБСТВЕННУЮ гипотезу (не сухой пересказ фактов) и
сохраняет её в дневник с повышенной уверенностью как "инкубированную мысль".
Следующий спонтанный пинг (efi.behavior.spontaneous_ping) забирает эту мысль
вместо дежурного "как дела" — см. consume_incubated_thought().

Формулировка гипотезы — единственное место в этом модуле, где идёт обращение
к LLM (роль FAST, как и остальная фоновая работа с памятью — см.
efi.memory.consolidation). Это не нарушает требование "zero-latency на
критическом пути": критический путь — это сборка системного промпта перед
ОТВЕТОМ собеседнику (efi/prompts/builder.py), а не независимый от него
фоновый цикл, который сам решает, когда и вызывать ли LLM вообще.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from pathlib import Path

import aiofiles

from efi.config.schema import TaskRole
from efi.llm.errors import LLMError
from efi.llm.router import LLMRouter
from efi.llm.schemas import LLMParams, Message, Role, Session
from efi.memory.facts import FactStore
from efi.memory.rag import RAGMemory
from efi.notifications.schemas import Notification, NotificationType
from efi.tools.base import ToolContext
from efi.tools.web_tools.web_search import WebSearchTool

logger = logging.getLogger(__name__)

#: FactStore-ключ инкубированной мысли — намеренно НЕ привязан к chat_id: мысль
#: рождается вне контекста конкретного разговора, её "усыновляет" тот чат,
#: которому в итоге достанется следующий спонтанный пинг.
_INCUBATED_THOUGHT_ENTITY = "researcher"
_INCUBATED_THOUGHT_KEY = "incubated_thought"

#: Гипотеза — не подтверждённый факт (иначе она бы не менялась ночной
#: консолидацией), но и не мимолётная теория с нуля: это то, что Эфи
#: специально нагуглила и обдумала, поэтому чуть выше дефолтной уверенности
#: обычной дневниковой записи (см. RememberDiaryEntryTool, где дефолт 0.7,
#: но там — прямое осознанное решение модели посреди разговора).
_HYPOTHESIS_CONFIDENCE = 0.6

_HYPOTHESIS_SYSTEM_PROMPT = (
    "Тебе показаны результаты веб-поиска по теме, которая тебе реально интересна. Сформулируй из них "
    "СОБСТВЕННУЮ гипотезу или личное мнение — коротко (1-3 предложения), от первого лица, с оценкой, а не "
    "сухим пересказом фактов. Не перечисляй источники и не пиши 'по данным поиска' — пиши так, будто это "
    "мысль, которая у тебя реально возникла, пока ты гуглила от скуки."
)


class BackgroundResearcher:
    """
    Раз в `check_interval_seconds`, с вероятностью `research_probability` —
    та же схема, что у SpontaneousPingScheduler, ровно затем, чтобы моменты
    исследования сами по себе выглядели как случайные, а не по расписанию —
    выбирает случайную тему из worldview.json, ищет её в вебе и формулирует
    гипотезу.

    Предназначена для запуска через `asyncio.create_task(researcher.run())`
    при старте приложения, как и остальные фоновые сервисы efi/behavior/.
    Отсутствие worldview.json, сбой поиска или LLM не должны ронять цикл —
    это необязательная "для настроения" функциональность, а не критический путь.
    """

    def __init__(
        self,
        worldview_path: Path,
        web_search: WebSearchTool,
        rag: RAGMemory,
        router: LLMRouter,
        facts: FactStore,
        *,
        check_interval_seconds: float = 2700.0,  # 45 минут
        research_probability: float = 0.25,
        hypothesis_role: TaskRole = TaskRole.BACKGROUND,
    ) -> None:
        self._worldview_path = worldview_path
        self._web_search = web_search
        self._rag = rag
        self._router = router
        self._facts = facts
        self._check_interval_seconds = check_interval_seconds
        self._research_probability = research_probability
        self._hypothesis_role = hypothesis_role

    async def run(self) -> None:
        """Основной цикл. Останавливается по отмене задачи (CancelledError) — см. efi/app.py graceful shutdown."""
        logger.info(
            "researcher: started (interval=%.0fs, p=%.2f)",
            self._check_interval_seconds, self._research_probability,
        )
        try:
            while True:
                await asyncio.sleep(self._check_interval_seconds)
                if random.random() <= self._research_probability:
                    await self._research_once()
        except asyncio.CancelledError:
            logger.info("researcher: stopped")
            raise

    async def consume_incubated_thought(self) -> str | None:
        """
        Забирает и СБРАСЫВАЕТ последнюю инкубированную мысль (одноразово —
        следующий вызов вернёт None, пока не появится новая). Вызывается
        efi.behavior.spontaneous_ping.SpontaneousPingScheduler перед тем, как
        сформулировать текст очередного спонтанного пинга.
        """
        thought = await self._facts.get(_INCUBATED_THOUGHT_ENTITY, _INCUBATED_THOUGHT_KEY)
        if thought is None:
            return None
        await self._facts.delete(_INCUBATED_THOUGHT_ENTITY, _INCUBATED_THOUGHT_KEY)
        return thought

    async def _research_once(self) -> None:
        topic = await self._pick_topic()
        if topic is None:
            return

        search_text = await self._web_search.execute({"query": topic}, _make_research_context(topic))
        if search_text.startswith("error:") or "ничего не нашлось" in search_text:
            logger.info("researcher: search for %r yielded nothing useful, skipping", topic)
            return

        hypothesis = await self._formulate_hypothesis(topic, search_text)
        if hypothesis is None:
            return

        entry = await self._rag.remember(hypothesis, confidence=_HYPOTHESIS_CONFIDENCE)
        if entry is not None:
            logger.info("researcher: incubated new hypothesis on %r -> diary entry %s", topic, entry.id)
        await self._facts.upsert(_INCUBATED_THOUGHT_ENTITY, _INCUBATED_THOUGHT_KEY, hypothesis)

    async def _pick_topic(self) -> str | None:
        try:
            async with aiofiles.open(self._worldview_path, encoding="utf-8") as f:
                raw = await f.read()
        except OSError as exc:
            logger.warning("researcher: failed to read worldview file %s: %s", self._worldview_path, exc)
            return None

        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning("researcher: worldview file is not valid JSON: %s", exc)
            return None

        raw_interests = data.get("interests", []) if isinstance(data, dict) else []
        interests = [str(item).strip() for item in raw_interests if str(item).strip()]
        if not interests:
            return None
        return random.choice(interests)

    async def _formulate_hypothesis(self, topic: str, search_text: str) -> str | None:
        params = LLMParams(model="", system_prompt=_HYPOTHESIS_SYSTEM_PROMPT, max_output_tokens=256)
        session = Session(messages=[Message(role=Role.USER, content=f"Тема: {topic}\n\n{search_text}")])
        try:
            response = await self._router.chat(self._hypothesis_role, params, session)
        except LLMError as exc:
            logger.warning("researcher: hypothesis formulation for %r failed: %s", topic, exc)
            return None
        return response.text.strip() or None


def _make_research_context(topic: str) -> ToolContext:
    """
    Синтетический ToolContext для вызова WebSearchTool вне обычного
    tool-calling цикла Worker'а. WebSearchTool.execute не читает контекст
    вообще (см. efi/tools/web_tools/web_search.py) — это просто соблюдение
    контракта Tool.execute(arguments, context), а не реальная зависимость.
    """
    return ToolContext(notification=Notification(type=NotificationType.NIGHTLY_TASK, message=topic))


__all__ = ["BackgroundResearcher"]
