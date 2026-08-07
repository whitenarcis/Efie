"""
efi/tools/base.py

Базовый класс инструмента LLM — перенос OpenAITools::Tool из Kuni на Python:
один инструмент = один класс (обычно — один файл) с именем, описанием,
JSON-схемой параметров и асинхронным execute(). ToolRegistry
(efi/tools/registry.py) собирает экземпляры Tool в реестр, фильтрует их по
контексту конкретного Notification и конвертирует в формат OpenAI function
calling для LLMParams.tools.

Изоляция инструментов (принцип из Kuni, который мы переносим как есть):
каждый инструмент ничего не знает о других инструментах, о NotificationManager
или о Worker — только о своих зависимостях, переданных через конструктор
(RAGMemory, MessageSender и т.п.), и о ToolContext, который ему дают на
исполнение. Это то же самое разделение, что в tools/*.h у референса, где
каждый tools::xxx() — независимая фабричная функция.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from efi.notifications.schemas import Notification


@dataclass(slots=True, frozen=True)
class ToolContext:
    """
    Всё, что может понадобиться инструменту при исполнении одного tool call.

    Помимо самого Notification (chat_id, тип события, payload), несёт
    `extra` — свободный словарь произвольных зависимостей на конкретный
    вызов (например, отправитель Telegram-сообщений), чтобы не раздувать
    сигнатуру ToolContext под каждую новую интеграцию и не тянуть сюда
    конкретные типы из telegram/ и подобных ещё не существующих модулей.
    """

    notification: Notification
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def chat_id(self) -> int | None:
        return self.notification.chat_id


class Tool(ABC):
    """
    Контракт одного инструмента LLM.

    `name`/`description`/`parameters` — статические атрибуты класса (аналог
    полей OpenAITools::Tool у референса); `execute()` — асинхронная
    реализация, обязательная к переопределению.

    `is_available()` — фильтр контекста. Вызывается ToolRegistry дважды: при
    построении списка инструментов для конкретного запроса к LLM (чтобы не
    предлагать модели то, чем сейчас нельзя воспользоваться) и повторно при
    исполнении tool call (чтобы модель не могла обойти фильтр, вызвав
    инструмент, который ей не предлагали, но имя которого могла угадать/
    запомнить из истории).
    """

    #: Имя инструмента, как оно видно модели. Должно быть уникальным в ToolRegistry.
    name: str
    #: Описание для модели — что делает инструмент и когда его использовать.
    description: str
    #: JSON-схема параметров в формате OpenAI function calling.
    parameters: dict[str, Any] = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}

    def is_available(self, context: ToolContext) -> bool:
        """По умолчанию инструмент доступен всегда; переопределяется там, где есть ограничения контекста."""
        return True

    @abstractmethod
    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        """
        Выполняет инструмент и возвращает текстовый результат — он пойдёт
        обратно в модель как содержимое TOOL-сообщения. Инструмент не должен
        бросать исключения наружу ради самого факта ошибки использования
        (неверные аргументы, недоступность внешнего сервиса и т.п.) — такие
        случаи стоит превращать в текст вида "error: ..." и возвращать его,
        чтобы модель могла среагировать сама. Настоящие программные ошибки
        (баги) допустимо не ловить — ToolRegistry.execute перехватит их на
        своём уровне и не даст уронить весь tool-calling цикл.
        """
        raise NotImplementedError

    def as_openai_schema(self) -> dict[str, Any]:
        """Формат, ожидаемый LLMParams.tools (OpenAI function calling)."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


__all__ = ["ToolContext", "Tool"]
