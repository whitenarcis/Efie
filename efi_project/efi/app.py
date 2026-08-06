"""
efi/app.py

EfiApp — точка сборки всего приложения: связывает Settings, Database,
LLMRouter, память (Diary/RAGMemory/WorkingMemory/FactStore), очередь событий
(NotificationManager + N Worker'ов), Telegram-слой и проактивные сервисы
(Scheduler/SpontaneousPingScheduler/SilenceMonitor) в один управляемый объект
с корректным graceful shutdown.

EfiApp сам не работает с сигналами ОС (SIGINT/SIGTERM) — это дело точки
входа (scripts/run.py), которая вызывает `request_stop()` из обработчика
сигнала. Такое разделение позволяет использовать EfiApp и в контексте, где
сигналы не нужны (тесты, встраивание в другой процесс).
"""

from __future__ import annotations

import asyncio
import logging
from datetime import time as dt_time

from pyrogram import Client

from efi.behavior.scheduler import ScheduledJob, Scheduler, seconds_until_next
from efi.behavior.silence_monitor import SilenceMonitor
from efi.behavior.spontaneous_ping import SpontaneousPingScheduler
from efi.config.schema import Settings, TaskRole
from efi.db.core import Database
from efi.db.history_repository import SqliteHistoryRepository
from efi.db.models import MIGRATIONS
from efi.humanizer.anti_repeat import AntiRepeatTracker
from efi.memory.consolidation import DiaryConsolidator
from efi.memory.diary import Diary
from efi.memory.facts import FactStore
from efi.memory.rag import RAGMemory
from efi.memory.tfidf_fallback import TfidfFallbackIndex
from efi.memory.working_memory import WorkingMemory
from efi.notifications.manager import NotificationManager
from efi.notifications.worker import Worker
from efi.prompts.builder import EfiSystemPromptBuilder
from efi.prompts.loader import PromptLoader
from efi.telegram.client import TelegramClientWrapper
from efi.telegram.handlers import TelegramEventHandlers
from efi.tools.base import Tool
from efi.tools.chat_management.join_chat import JoinChatTool
from efi.tools.chat_management.leave_chat import LeaveChatTool
from efi.tools.chat_management.search_chats import SearchChatsTool
from efi.tools.memory_tools.ask_diary import AskDiaryTool
from efi.tools.memory_tools.recall_fact import RecallFactTool
from efi.tools.memory_tools.remember_fact import RememberFactTool
from efi.tools.registry import ToolRegistry
from efi.tools.system_tools.device import GetBatteryStatusTool, TriggerVibrationTool
from efi.tools.telegram_actions.edit_message import EditMessageTool
from efi.tools.telegram_actions.forward_message import ForwardMessageTool
from efi.tools.telegram_actions.react_with_emoji import ReactWithEmojiTool
from efi.tools.telegram_actions.send_message import SendMessageTool
from efi.tools.telegram_actions.stickers import SendStickerTool
from efi.tools.web_tools.web_search import WebSearchTool

logger = logging.getLogger(__name__)

_DEFAULT_WORKER_COUNT = 3
_QUEUE_DRAIN_TIMEOUT_SECONDS = 30.0
_CONSOLIDATION_TRIGGER_AT = dt_time(hour=3, minute=30)


class EfiApp:
    """
    Владеет жизненным циклом всех подсистем приложения.

    Использование:
        app = EfiApp(settings)
        await app.start()
        await app.wait_until_stopped()  # блокируется до request_stop()
        await app.stop()
    """

    def __init__(self, settings: Settings, *, worker_count: int = _DEFAULT_WORKER_COUNT) -> None:
        self._settings = settings
        self._stop_event = asyncio.Event()
        self._background_tasks: list[asyncio.Task] = []
        self._worker_tasks: list[asyncio.Task] = []

        # -- инфраструктура ------------------------------------------------
        self._database = Database(settings.paths.db_path, migrations=MIGRATIONS)
        self._llm_router = settings.build_router()

        # -- память ----------------------------------------------------------
        diary_dir = settings.memory.resolve_diary_dir(settings.paths)
        self._diary = Diary(diary_dir, plagiarism_threshold=settings.memory.plagiarism_threshold)
        self._tfidf = TfidfFallbackIndex()
        self._rag = RAGMemory(self._diary, self._llm_router, self._tfidf)
        self._working_memory = WorkingMemory(settings.paths.data_dir / "working_memory.json")
        self._facts = FactStore(self._database)
        self._history = SqliteHistoryRepository(self._database)
        self._consolidator = DiaryConsolidator(self._diary, self._llm_router)

        # -- промпты -----------------------------------------------------
        templates_dir = settings.paths.base_dir / "efi" / "prompts" / "templates"
        self._prompt_loader = PromptLoader(templates_dir)
        self._prompt_builder = EfiSystemPromptBuilder(self._prompt_loader, settings, self._rag, self._working_memory)

        # -- humanizer / проактивность --------------------------------------
        self._anti_repeat = AntiRepeatTracker(settings.humanizer)
        self._notification_manager = NotificationManager(worker_count=worker_count)
        self._silence_monitor = SilenceMonitor(self._notification_manager)
        self._scheduler = Scheduler(self._notification_manager, _build_scheduled_jobs())
        self._spontaneous_ping = SpontaneousPingScheduler(self._notification_manager, self._active_chat_candidates)

        # -- telegram --------------------------------------------------------
        self._pyrogram_client = Client(
            settings.paths.session_name,
            api_id=settings.telegram.api_id,
            api_hash=settings.telegram.api_hash.get_secret_value(),
            phone_number=(
                settings.telegram.phone_number.get_secret_value() if settings.telegram.phone_number else None
            ),
            workdir=str(settings.paths.session_path.parent),
        )
        self._telegram_client = TelegramClientWrapper(self._pyrogram_client, settings.humanizer)
        self._telegram_handlers = TelegramEventHandlers(
            self._notification_manager,
            settings.telegram,
            self._llm_router,
            settings.paths.cache_dir,
            activity_recorder=self._silence_monitor,
        )

        # -- инструменты -------------------------------------------------
        self._web_search_tool = WebSearchTool()
        self._tool_registry = ToolRegistry()
        self._tool_registry.register_all(self._build_tools())

    def _build_tools(self) -> list[Tool]:
        return [
            AskDiaryTool(self._rag, min_relatedness=self._settings.memory.min_relatedness),
            RememberFactTool(self._facts),
            RecallFactTool(self._facts),
            SendMessageTool(
                self._telegram_client,
                anti_repeat=self._anti_repeat,
                activity_recorder=self._silence_monitor,
            ),
            EditMessageTool(self._telegram_client),
            ReactWithEmojiTool(self._telegram_client),
            ForwardMessageTool(self._telegram_client),
            SendStickerTool(self._telegram_client),
            JoinChatTool(self._telegram_client, enabled=self._settings.telegram.can_join_chats),
            LeaveChatTool(self._telegram_client, enabled=self._settings.telegram.can_leave_chats),
            SearchChatsTool(self._telegram_client),
            self._web_search_tool,
            GetBatteryStatusTool(),
            TriggerVibrationTool(),
            # GenerateImageTool/GenerateVoiceTool сознательно не подключены здесь:
            # им нужны appearance_prompt (Pollinations) и api_key/voice_id
            # (ElevenLabs), а этих полей пока нет в Settings — добавлять их
            # без явного запроса не стал (см. итоговое резюме шага). Сами
            # инструменты полностью реализованы и готовы к подключению, как
            # только появится конфигурация.
        ]

    async def _active_chat_candidates(self) -> list[int]:
        """
        Список чатов-кандидатов для спонтанного пинга.

        TODO: как только появится полноценный реестр известных диалогов
        (например, через Pyrogram get_dialogs, кэшируемый в efi/telegram/),
        заменить на реальный источник. Сейчас — allowed_chats из конфига;
        этого достаточно, чтобы функциональность была рабочей с первого дня.
        """
        return list(self._settings.telegram.allowed_chats)

    async def start(self) -> None:
        """Поднимает все подсистемы: Telegram-клиент, обработчики, воркеры, проактивные сервисы."""
        logger.info("app: starting")

        self._telegram_handlers.register(self._pyrogram_client)
        await self._telegram_client.start()

        for worker_index in range(self._notification_manager.worker_count):
            worker = Worker(
                worker_index,
                self._notification_manager,
                llm_router=self._llm_router,
                tool_registry=self._tool_registry,
                history=self._history,
                system_prompt_builder=self._prompt_builder,
                main_role=TaskRole.MAIN,
            )
            self._worker_tasks.append(asyncio.create_task(worker.run(), name=f"worker-{worker_index}"))

        self._background_tasks.extend(
            [
                asyncio.create_task(self._scheduler.run(), name="scheduler"),
                asyncio.create_task(self._silence_monitor.run(), name="silence_monitor"),
                asyncio.create_task(self._spontaneous_ping.run(), name="spontaneous_ping"),
                asyncio.create_task(self._prompt_loader.watch(), name="prompt_loader_watch"),
                asyncio.create_task(self._run_consolidation_loop(), name="diary_consolidation"),
            ]
        )

        logger.info("app: started (%d workers, %d background services)", len(self._worker_tasks), len(self._background_tasks))

    async def _run_consolidation_loop(self) -> None:
        """
        Программная (не диалоговая) консолидация дневника: dedup существующих
        записей + сжатие старых записей в мемуары через LLM (см.
        efi/memory/consolidation.py). Идёт своим отдельным ежедневным
        расписанием, НЕ через NotificationManager — это обслуживание данных,
        а не разговорный ответ модели. Дополняет, а не заменяет
        "nightly_consolidation" ScheduledJob (та даёт личности повод
        отрефлексировать день в разговорном формате).
        """
        try:
            while True:
                await asyncio.sleep(seconds_until_next(_CONSOLIDATION_TRIGGER_AT))

                removed = await self._consolidator.deduplicate(
                    plagiarism_threshold=self._settings.memory.plagiarism_threshold
                )
                logger.info("app: nightly dedup removed %d duplicate diary entries", removed)

                merged = await self._consolidator.summarize_stale_entries()
                if merged is not None:
                    logger.info("app: nightly consolidation created memoir entry %s", merged.id)
        except asyncio.CancelledError:
            logger.info("app: diary consolidation loop stopped")
            raise

    async def wait_until_stopped(self) -> None:
        """Блокируется, пока не будет вызван request_stop() (обычно — из обработчика сигнала ОС в scripts/run.py)."""
        await self._stop_event.wait()

    def request_stop(self) -> None:
        """Неблокирующий сигнал остановки — безопасно вызывать из обработчика сигнала ОС."""
        logger.info("app: stop requested")
        self._stop_event.set()

    async def stop(self) -> None:
        """
        Graceful shutdown, в таком порядке:
            1. Останавливаем всё, что ПОРОЖДАЕТ новую работу (проактивные
               сервисы, watcher шаблонов) — чтобы очередь перестала расти.
            2. Ждём, пока Worker'ы разберут то, что уже лежит в очереди
               (не обрубаем обработку уведомления на середине), с таймаутом.
            3. Останавливаем сами Worker'ы.
            4. Закрываем внешние соединения: Telegram-клиент, httpx-клиенты
               LLMRouter (WAL-файлы SQLite закрываются автоматически — каждое
               соединение efi.db.core.Database открывается и закрывается
               на одну операцию, долгоживущего соединения, которое нужно
               было бы закрывать явно, здесь нет).
        """
        logger.info("app: stopping")

        for task in self._background_tasks:
            task.cancel()
        await asyncio.gather(*self._background_tasks, return_exceptions=True)

        try:
            await asyncio.wait_for(self._notification_manager.join(), timeout=_QUEUE_DRAIN_TIMEOUT_SECONDS)
        except TimeoutError:
            logger.warning(
                "app: notification queues did not drain within %.0fs, stopping workers anyway",
                _QUEUE_DRAIN_TIMEOUT_SECONDS,
            )

        for task in self._worker_tasks:
            task.cancel()
        await asyncio.gather(*self._worker_tasks, return_exceptions=True)

        await self._telegram_client.stop()
        await self._llm_router.aclose()
        await self._web_search_tool.aclose()

        logger.info("app: stopped")


def _build_scheduled_jobs() -> list[ScheduledJob]:
    return [
        ScheduledJob(
            name="nightly_consolidation",
            trigger_at=dt_time(hour=3, minute=0),
            notification_message="Пора провести ночную консолидацию дневника и рабочей памяти за прошедший день.",
        ),
        ScheduledJob(
            name="morning_wakeup",
            trigger_at=dt_time(hour=8, minute=0),
            notification_message="Начинается новый день — просыпайся и загляни в чаты, где давно не отвечала.",
        ),
    ]


__all__ = ["EfiApp"]
