"""
efi/tools/registry.py

ToolRegistry — реестр всех инструментов Эфи: регистрация, фильтрация по
контексту уведомления и конвертация в формат OpenAI function calling для
LLMParams.tools. Аналог сборки OpenAITools у референса (AppBase::updateTools),
но вместо ручной пересборки набора инструментов под каждый вызов — единый
реестр, который сам решает, что показать модели, опираясь на Tool.is_available().
"""

from __future__ import annotations

import logging
from typing import Any

from efi.llm.schemas import ToolCall
from efi.tools.base import Tool, ToolContext

logger = logging.getLogger(__name__)


class ToolRegistry:
    """
    Реестр инструментов с фильтрацией по контексту и безопасным исполнением
    tool-вызовов модели: `execute()` никогда не пробрасывает исключение
    наружу — любая ошибка (неизвестный инструмент, недоступен в контексте,
    некорректные аргументы, исключение внутри execute()) превращается в
    текстовый результат для модели, чтобы не срывать весь tool-calling цикл
    в Worker'е из-за одного неудачного вызова.
    """

    def __init__(self) -> None:
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        """
        Регистрирует инструмент. Повторная регистрация того же имени — ошибка конфигурации, не должна проходить
        незаметно.
        """
        if tool.name in self._tools:
            raise ValueError(f"tool {tool.name!r} is already registered")
        self._tools[tool.name] = tool

    def register_all(self, tools: list[Tool]) -> None:
        """Регистрирует несколько инструментов разом — удобно при сборке реестра в app.py."""
        for tool in tools:
            self.register(tool)

    def available_tools(self, context: ToolContext) -> list[Tool]:
        """Инструменты, доступные в данном контексте (см. Tool.is_available)."""
        return [tool for tool in self._tools.values() if tool.is_available(context)]

    def as_openai_tools(self, context: ToolContext) -> list[dict[str, Any]]:
        """Список инструментов, доступных в контексте, в формате LLMParams.tools."""
        return [tool.as_openai_schema() for tool in self.available_tools(context)]

    async def execute(self, tool_call: ToolCall, context: ToolContext) -> str:
        """
        Выполняет один tool call модели по имени функции из `tool_call.function.name`.

        Если инструмент не зарегистрирован или недоступен в данном контексте
        (модель могла его "придумать" или вызвать то, что ей формально не
        предлагали), не бросает исключение — возвращает текст ошибки как
        результат TOOL-сообщения, чтобы модель могла среагировать сама и
        цикл tool-calling в Worker'е продолжился.
        """
        tool = self._tools.get(tool_call.function.name)
        if tool is None:
            logger.warning("tool_registry: unknown tool requested: %s", tool_call.function.name)
            return f"error: unknown tool {tool_call.function.name!r}"

        if not tool.is_available(context):
            logger.warning("tool_registry: tool %s requested outside of its availability context", tool.name)
            return f"error: tool {tool.name!r} is not available in this context"

        try:
            arguments = tool_call.function.parsed_arguments()
        except ValueError as exc:
            logger.warning("tool_registry: %s sent malformed arguments: %s", tool.name, exc)
            return f"error: malformed arguments: {exc}"

        try:
            return await tool.execute(arguments, context)
        except Exception as exc:  # noqa: BLE001 — намеренно широкий catch: баг в одном инструменте не должен ронять весь цикл
            logger.exception("tool_registry: tool %s raised during execute()", tool.name)
            return f"error: tool {tool.name!r} failed: {exc}"


__all__ = ["ToolRegistry"]
