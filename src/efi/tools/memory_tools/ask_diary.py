"""
efi/tools/memory_tools/ask_diary.py

Инструмент "сверься с дневником" — явный способ для модели попросить более
точный/широкий поиск по долгосрочной памяти, чем то, что уже автоматически
подмешано в контекст (Worker._build_session кладёт top-N результатов RAG по
тексту самого уведомления в системный блок ДО обращения к модели). Здесь
модель сама формулирует поисковый запрос — полезно, когда ей нужно уточнить
детали или найти что-то по формулировке, которая отличается от исходного
сообщения пользователя.

Аналог tools/ask.h у референса.
"""

from __future__ import annotations

from typing import Any

from efi.llm.schemas import DiaryQueryOptions
from efi.memory.rag import RAGMemory
from efi.tools.base import Tool, ToolContext


class AskDiaryTool(Tool):
    """Позволяет модели самостоятельно сформулировать поисковый запрос к дневнику Эфи."""

    name = "ask_diary"
    description = (
        "Ищет в твоём долгосрочном дневнике записи, связанные с заданным запросом. "
        "Используй, когда контекста, уже подмешанного автоматически, недостаточно — "
        "например, чтобы уточнить детали или найти что-то конкретное из прошлого "
        "по формулировке, отличной от исходного сообщения собеседника."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Что именно нужно вспомнить или уточнить"},
            "max_results": {
                "type": "integer",
                "description": "Максимум записей в ответе (по умолчанию 5, не больше 20)",
            },
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(self, rag: RAGMemory, *, min_relatedness: float = 0.0) -> None:
        self._rag = rag
        self._min_relatedness = min_relatedness

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        query = str(arguments.get("query", "")).strip()
        if not query:
            return "error: query must not be empty"

        max_results = _coerce_max_results(arguments.get("max_results", 5))
        options = DiaryQueryOptions(max_entry_count=max_results, min_relatedness=self._min_relatedness)

        results = await self._rag.search(query, options)
        if not results:
            return "В дневнике не нашлось ничего связанного с этим запросом."

        lines = [f"- (relatedness={result.relatedness:.2f}) {result.entry.body.strip()}" for result in results]
        return "Найдено в дневнике:\n" + "\n".join(lines)


def _coerce_max_results(raw: Any) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = 5
    return max(1, min(value, 20))


__all__ = ["AskDiaryTool"]
