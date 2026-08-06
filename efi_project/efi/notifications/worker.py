"""
efi/notifications/worker.py

Worker — обрабатывает уведомления, закреплённые за ним NotificationManager'ом,
строго последовательно (одно за другим, в порядке приоритета внутри своей
подочереди). Это и есть гарантия отсутствия гонок за контекст одного чата:
пока Worker обрабатывает Notification для chat_id=X, следующее уведомление
для того же chat_id физически лежит в ЕГО ЖЕ подочереди и просто ждёт своей
очереди — никакой другой Worker его в это время не подхватит.

На каждое уведомление Worker:
    1. Собирает Session: история диалога (HistoryRepository) + текст события
       (сперва прогнанный через security.sanitize — это внешний ввод).
    2. Просит полный системный промпт у SystemPromptBuilder (личность, время,
       working memory, RAG, ограничения безопасности — Worker в это не
       вникает, см. efi/prompts/builder.py).
    3. Прогоняет цикл tool-calling через LLMRouter, пока модель не даст
       финальный ответ без tool_calls (см. _run_with_tool_calls).
    4. Сохраняет финальное сообщение ассистента в историю диалога.

С Шага 6 Worker больше не знает про RAGMemory/WorkingMemory напрямую — эта
логика переехала в SystemPromptBuilder (efi/prompts/builder.py), потому что
именно он решает, как результаты памяти должны формулироваться в промпте.
Worker остаётся ответственным только за оркестрацию: историю, санитайзинг
входа и цикл tool-calling — и не заботится о содержании промпта.

История диалога (HistoryRepository) и сборка системного промпта
(SystemPromptBuilder) заданы здесь как typing.Protocol; конкретные реализации
— efi.db.history_repository.SqliteHistoryRepository и
efi.prompts.builder.EfiSystemPromptBuilder соответственно.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Protocol

from efi.config.schema import TaskRole
from efi.llm.router import LLMRouter
from efi.llm.schemas import LLMParams, Message, Response, Role, Session
from efi.notifications.manager import NotificationManager
from efi.notifications.schemas import Notification
from efi.security.sanitize import sanitize_text
from efi.tools.base import ToolContext
from efi.tools.registry import ToolRegistry

logger = logging.getLogger(__name__)


class HistoryRepository(Protocol):
    """Абстракция истории диалога. Конкретная реализация — efi.db.history_repository.SqliteHistoryRepository."""

    async def get_recent(self, chat_id: int, limit: int = 20) -> Session: ...

    async def append(self, chat_id: int, message: Message) -> None: ...


class SystemPromptBuilder(Protocol):
    """Абстракция сборки системного промпта. Конкретная реализация — efi.prompts.builder.EfiSystemPromptBuilder."""

    async def build(self, notification: Notification) -> str: ...


class Worker:
    """Обрабатывает уведомления из своей подочереди NotificationManager'а."""

    def __init__(
        self,
        worker_index: int,
        manager: NotificationManager,
        *,
        llm_router: LLMRouter,
        tool_registry: ToolRegistry,
        history: HistoryRepository,
        system_prompt_builder: SystemPromptBuilder,
        main_role: TaskRole = TaskRole.MAIN,
        max_tool_call_rounds: int = 8,
        history_limit: int = 20,
    ) -> None:
        self._worker_index = worker_index
        self._manager = manager
        self._llm_router = llm_router
        self._tool_registry = tool_registry
        self._history = history
        self._system_prompt_builder = system_prompt_builder
        self._main_role = main_role
        self._max_tool_call_rounds = max_tool_call_rounds
        self._history_limit = history_limit

    async def run(self) -> None:
        """
        Основной цикл: забирает уведомления из своей подочереди и
        обрабатывает их одно за другим. Ошибка обработки ОДНОГО уведомления
        не должна останавливать Worker целиком — она логируется, и цикл идёт
        дальше. Останавливается только по внешней отмене задачи (CancelledError).
        """
        logger.info("worker[%d]: started", self._worker_index)
        try:
            while True:
                notification = await self._manager.get(self._worker_index)
                try:
                    await self._handle(notification)
                except Exception:
                    logger.exception(
                        "worker[%d]: unhandled error while processing notification %s (%s)",
                        self._worker_index, notification.id, notification.type.value,
                    )
                finally:
                    self._manager.task_done(self._worker_index)
        except asyncio.CancelledError:
            logger.info("worker[%d]: stopped", self._worker_index)
            raise

    async def _handle(self, notification: Notification) -> None:
        # Системный промпт (личность + время + working memory + RAG +
        # ограничения) и история диалога зависят от разных источников и не
        # блокируют друг друга — читаем конкурентно.
        session_task = self._build_session(notification)
        system_prompt_task = self._system_prompt_builder.build(notification)
        session, system_prompt = await asyncio.gather(session_task, system_prompt_task)

        tool_context = ToolContext(notification=notification)
        params = LLMParams(
            model="",  # роутер сам подставит модель кандидата по main_role — см. efi.llm.router.LLMRouter
            system_prompt=system_prompt,
            tools=self._tool_registry.as_openai_tools(tool_context),
        )

        response = await self._run_with_tool_calls(params, session, tool_context)

        if notification.chat_id is not None:
            await self._history.append(notification.chat_id, response.message)

    async def _build_session(self, notification: Notification) -> Session:
        """
        История чата + текущее сообщение события. `notification.message` —
        внешний ввод (то, что написал собеседник, или сформулированный текст
        триггера) и прогоняется через sanitize_text ПЕРЕД тем, как попасть в
        Session — единственная точка, где сырой внешний текст превращается в
        USER-сообщение LLM.
        """
        history = (
            await self._history.get_recent(notification.chat_id, limit=self._history_limit)
            if notification.chat_id is not None
            else Session()
        )
        session = history.model_copy(deep=True)
        session.append(Message(role=Role.USER, content=sanitize_text(notification.message)))
        return session

    async def _run_with_tool_calls(self, params: LLMParams, session: Session, tool_context: ToolContext) -> Response:
        """
        Цикл tool-calling: запросить ответ у LLM -> если модель вызвала
        инструменты, выполнить их и вернуть результаты в модель как
        TOOL-сообщения -> повторить, пока модель не даст финальный ответ без
        tool_calls. `max_tool_call_rounds` — защита от зацикливания (модель
        может застрять, бесконечно вызывая инструменты).
        """
        response: Response | None = None
        for _round_number in range(self._max_tool_call_rounds):
            response = await self._llm_router.chat(self._main_role, params, session)
            message = response.message
            session.append(message)

            if not message.has_tool_calls:
                return response

            for tool_call in message.tool_calls:
                result_text = await self._tool_registry.execute(tool_call, tool_context)
                session.append(Message(role=Role.TOOL, content=result_text, tool_call_id=tool_call.id))

        logger.warning(
            "worker[%d]: reached max_tool_call_rounds=%d while processing %s, returning last response as-is",
            self._worker_index, self._max_tool_call_rounds, tool_context.notification.id,
        )
        assert response is not None  # цикл выполняется минимум один раз, т.к. max_tool_call_rounds задаётся >= 1
        return response


__all__ = ["Worker", "HistoryRepository", "SystemPromptBuilder"]
