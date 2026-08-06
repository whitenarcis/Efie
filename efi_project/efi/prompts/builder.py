"""
efi/prompts/builder.py

Реализация efi.notifications.worker.SystemPromptBuilder: собирает полный
системный промпт для одного обращения к LLM — личность, временной контекст,
рабочую память, RAG-факты и ограничения безопасности — в один связный текст.

С появлением этого модуля Worker больше не строит собственный "preface"
system-блок с working memory/RAG (как было до Шага 6) — эта логика переехала
сюда, потому что именно PromptBuilder знает, КАК личность должна
воспринимать эти данные (формулировки, порядок, акценты), а Worker остаётся
безразличным к содержанию промпта и занимается только оркестрацией истории и
tool-calling циклом (см. efi/notifications/worker.py).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from efi.config.schema import LockdownMode, Settings
from efi.llm.schemas import DiaryQueryOptions, DiaryQueryResult
from efi.memory.rag import RAGMemory
from efi.memory.working_memory import WorkingMemory, WorkingMemorySnapshot
from efi.notifications.schemas import Notification
from efi.prompts.loader import PromptLoader
from efi.security.sanitize import sanitize_text

logger = logging.getLogger(__name__)

_PERSONALITY_TEMPLATE_NAME = "personality"

_LOCKDOWN_DESCRIPTIONS: dict[LockdownMode, str] = {
    LockdownMode.NONE: "Ты можешь свободно общаться в любом чате.",
    LockdownMode.CONTACTS_ONLY: "Ты сейчас отвечаешь только людям из своих контактов — с незнакомцами держись настороже.",
    LockdownMode.OWNER_ONLY: "Ты в закрытом режиме: разговариваешь только с владельцем, во всех остальных чатах молчишь.",
}


class EfiSystemPromptBuilder:
    """
    Собирает системный промпт из пяти блоков, в порядке от самого
    стабильного (личность) к самому переменчивому (что нашлось в памяти
    именно сейчас): личность -> время -> рабочая память -> RAG -> безопасность.
    """

    def __init__(
        self,
        loader: PromptLoader,
        settings: Settings,
        rag: RAGMemory,
        working_memory: WorkingMemory,
    ) -> None:
        self._loader = loader
        self._settings = settings
        self._rag = rag
        self._working_memory = working_memory

    async def build(self, notification: Notification) -> str:
        """Критический путь: все источники читаются конкурентно (asyncio.gather), не последовательно."""
        personality_task = self._get_personality_text()
        rag_task = self._rag.search(
            notification.message,
            DiaryQueryOptions(
                max_entry_count=self._settings.memory.max_rag_results,
                min_relatedness=self._settings.memory.min_relatedness,
            ),
        )
        working_memory_task = self._working_memory.load()

        personality, rag_results, memory_snapshot = await asyncio.gather(
            personality_task, rag_task, working_memory_task
        )

        blocks = [
            personality.strip(),
            _build_time_block(),
            _build_working_memory_block(memory_snapshot),
            _build_rag_block(rag_results),
            _build_safety_block(self._settings.telegram.lockdown_mode),
        ]
        return "\n\n".join(block for block in blocks if block)

    async def _get_personality_text(self) -> str:
        """
        Личность по умолчанию берётся из behavior.toml (`settings.personality_prompt`)
        — так она и хранится в текущей реализации Эфи. Но если в каталоге
        шаблонов лежит `personality.md`, он имеет приоритет: это позволяет
        редактировать личность "на лету" через PromptLoader/watchfiles, не
        трогая остальной behavior.toml и не перезапуская процесс.
        """
        try:
            return await self._loader.get(_PERSONALITY_TEMPLATE_NAME)
        except FileNotFoundError:
            return self._settings.personality_prompt


def _build_time_block() -> str:
    now = datetime.now(timezone.utc).astimezone()
    return f"[Время] Сейчас {now.strftime('%A, %d %B %Y, %H:%M')} ({now.tzname() or 'UTC'})."


def _build_working_memory_block(snapshot: WorkingMemorySnapshot) -> str:
    parts: list[str] = []
    if snapshot.emotional_state or snapshot.physical_state:
        parts.append(
            f"эмоциональное состояние: {snapshot.emotional_state or 'не определено'}; "
            f"физическое состояние: {snapshot.physical_state or 'не определено'}"
        )
    open_items = [item for item in snapshot.items if not item.done]
    if open_items:
        parts.append("открытые задачи/обещания:\n" + "\n".join(f"  - {item.text}" for item in open_items))
    if not parts:
        return ""
    return "[Текущее состояние]\n" + "\n".join(parts)


def _build_rag_block(rag_results: list[DiaryQueryResult]) -> str:
    if not rag_results:
        return ""
    # sanitize_text — на случай, если в дневник когда-то попал текст,
    # содержащий фрагменты, похожие на служебную разметку (defense in depth:
    # даже "свой" контент проходит ту же обработку, что и внешний).
    lines = "\n".join(f"- {sanitize_text(result.entry.body.strip())}" for result in rag_results)
    return f"[Из долгосрочной памяти]\n{lines}"


def _build_safety_block(lockdown_mode: LockdownMode) -> str:
    return f"[Ограничения]\n{_LOCKDOWN_DESCRIPTIONS[lockdown_mode]}"


__all__ = ["EfiSystemPromptBuilder"]
