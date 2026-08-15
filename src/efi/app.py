"""
efi/app.py

EfiApp — точка сборки всего приложения: связывает Settings, Database,
LLMRouter, память (Diary/RAGMemory/WorkingMemory/FactStore), очередь событий
(NotificationManager + N Worker'ов), Telegram-слой, проактивные сервисы
(Scheduler/SpontaneousPingScheduler/SilenceMonitor) и веб-дашборд
(efi/dashboard/) в один управляемый объект с корректным graceful shutdown.

EfiApp сам не работает с сигналами ОС (SIGINT/SIGTERM) — это дело точки
входа (scripts/run.py), которая вызывает `request_stop()` из обработчика
сигнала. Такое разделение позволяет использовать EfiApp и в контексте, где
сигналы не нужны (тесты, встраивание в другой процесс).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Coroutine
from datetime import UTC, datetime, timedelta
from datetime import time as dt_time
from typing import Any

from pyrogram import Client

from efi.behavior.affinity import AffinityTracker
from efi.behavior.ambiguity import PendingClarifications
from efi.behavior.busy_engine import AnyBusyState, BusyEngine
from efi.behavior.collab_coding import CollabCodingDesk
from efi.behavior.conversation_lifecycle import ConversationLifecycle
from efi.behavior.curiosity import CuriosityTracker
from efi.behavior.initiative import InitiativeGate
from efi.behavior.life_engine import BackgroundLifeWorker
from efi.behavior.organic_ping import OrganicPingGenerator
from efi.behavior.ping_reason import PingReasonBuilder
from efi.behavior.reminders import ReminderScheduler, ReminderStore
from efi.behavior.researcher import BackgroundResearcher
from efi.behavior.scheduler import ScheduledJob, Scheduler, seconds_until_next
from efi.behavior.silence_monitor import SilenceMonitor
from efi.behavior.spontaneous_ping import SpontaneousPingScheduler
from efi.config.schema import Settings, TaskRole
from efi.dashboard.logbus import LogBuffer
from efi.dashboard.metrics import LLMMetricsCollector
from efi.dashboard.server import DashboardServer
from efi.dashboard.snapshot import DashboardContext
from efi.db.chat_directory import ChatDirectory
from efi.db.core import Database
from efi.db.history_repository import SqliteHistoryRepository
from efi.db.models import MIGRATIONS
from efi.dev.engine import DevEngine
from efi.dev.github_sync import GitHubSync
from efi.dev.maintenance import ProjectMaintainer
from efi.dev.qwen_client import QwenCoderClient
from efi.dev.reporter import DevReporter
from efi.dev.sandbox import CodeSandbox
from efi.dev.store import DevTaskStore
from efi.dev.worker import DevWorker
from efi.humanizer.anti_repeat import AntiRepeatTracker
from efi.media.stt_groq import GroqSTT
from efi.memory.beliefs import BeliefStore
from efi.memory.consolidation import DiaryConsolidator
from efi.memory.dedup import KnowledgeStore
from efi.memory.diary import Diary
from efi.memory.facts import FactStore
from efi.memory.ingest import MemoryIngestor
from efi.memory.knowledge_sink import EpisodeKnowledgeSink
from efi.memory.local_embeddings import LocalEmbeddingEngine
from efi.memory.parser import PerceptionParser
from efi.memory.people import PeopleStore
from efi.memory.pulse import MemoryPulse
from efi.memory.rag import RAGMemory
from efi.memory.social_memory import SocialInteractionStore
from efi.memory.tfidf_fallback import TfidfFallbackIndex
from efi.memory.validator import FactValidator
from efi.memory.working_memory import WorkingMemory
from efi.notifications.manager import NotificationManager
from efi.notifications.worker import Worker
from efi.prompts.builder import EfiSystemPromptBuilder
from efi.prompts.loader import PromptLoader
from efi.security.access_control import describe_access_policy
from efi.telegram.chat_orchestrator import ChatOrchestrator
from efi.telegram.client import TelegramClientWrapper
from efi.telegram.comments import (
    ChannelPostWatcher,
    RandomCommentEngager,
    ThreadStateStore,
    build_community_interests,
)
from efi.telegram.handlers import TelegramEventHandlers
from efi.telegram.typing_tracker import TypingTracker
from efi.tools.base import Tool
from efi.tools.chat_management.join_chat import JoinChatTool
from efi.tools.chat_management.leave_chat import LeaveChatTool
from efi.tools.chat_management.search_chats import SearchChatsTool
from efi.tools.dev_tools.project_status import DevProjectStatusTool
from efi.tools.dev_tools.start_project import StartDevProjectTool
from efi.tools.memory_tools.ask_diary import AskDiaryTool
from efi.tools.memory_tools.manage_belief import UpdateBeliefTool
from efi.tools.memory_tools.manage_promises import CompletePromiseTool, RememberPromiseTool
from efi.tools.memory_tools.recall_fact import RecallFactTool
from efi.tools.memory_tools.remember_diary_entry import RememberDiaryEntryTool
from efi.tools.memory_tools.remember_fact import RememberFactTool
from efi.tools.memory_tools.remember_person import RememberPersonTool
from efi.tools.memory_tools.update_self_state import UpdateSelfStateTool
from efi.tools.registry import ToolRegistry
from efi.tools.system_tools.device import GetBatteryStatusTool, TriggerVibrationTool
from efi.tools.telegram_actions.edit_message import EditMessageTool
from efi.tools.telegram_actions.forward_message import ForwardMessageTool
from efi.tools.telegram_actions.react_with_emoji import ReactWithEmojiTool
from efi.tools.telegram_actions.send_message import SendMessageTool
from efi.tools.telegram_actions.stickers import SendStickerTool
from efi.tools.web_tools.get_weather import GetWeatherTool
from efi.tools.web_tools.web_search import WebSearchTool

logger = logging.getLogger(__name__)

_DEFAULT_WORKER_COUNT = 3
_QUEUE_DRAIN_TIMEOUT_SECONDS = 30.0

#: Сколько раз подряд перезапускать упавшую фоновую службу и с какой паузой.
#: Потолок нужен, чтобы безнадёжно сломанная служба (нет файла, нет прав) не
#: перезапускалась вечно; шести попыток с удвоением хватает, чтобы пережить
#: любую временную неприятность и сдаться на постоянной.
_MAX_SERVICE_RESTARTS = 6
_SERVICE_RESTART_BASE_DELAY = 5.0
_SERVICE_RESTART_MAX_DELAY = 300.0
_CONSOLIDATION_TRIGGER_AT = dt_time(hour=3, minute=30)
#: "С начала времён" — для get_active_chat_ids(since=...) в _active_chat_candidates,
#: где нужны ВСЕ чаты с известной историей, а не только недавние.
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


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
        self._background_tasks: list[asyncio.Task[None]] = []
        self._worker_tasks: list[asyncio.Task[None]] = []
        self._started_at = datetime.now(UTC)

        # -- наблюдаемость ---------------------------------------------------
        # Строится ПЕРВОЙ: сборщик метрик нужен LLM-роутеру уже в конструкторе
        # (он оборачивает провайдеров при создании), а буфер логов должен
        # встать на корневой логгер до того, как подсистемы начнут писать.
        dashboard_settings = settings.dashboard
        self._log_buffer = LogBuffer(
            capacity=dashboard_settings.log_buffer_size, level=dashboard_settings.log_level_no
        )
        self._llm_metrics = LLMMetricsCollector(history=dashboard_settings.metrics_history)

        # -- инфраструктура ------------------------------------------------
        self._database = Database(settings.paths.db_path, migrations=MIGRATIONS)
        self._llm_router = settings.build_router(
            metrics_sink=self._llm_metrics.sink if dashboard_settings.enabled else None
        )

        # -- память ----------------------------------------------------------
        diary_dir = settings.memory.resolve_diary_dir(settings.paths)
        self._diary = Diary(diary_dir, plagiarism_threshold=settings.memory.plagiarism_threshold)
        self._tfidf = TfidfFallbackIndex()
        self._local_embeddings = (
            LocalEmbeddingEngine(settings.memory.local_embedding_model)
            if settings.memory.use_local_embeddings
            else None
        )
        self._rag = RAGMemory(self._diary, self._llm_router, self._tfidf, local_embeddings=self._local_embeddings)
        self._working_memory = WorkingMemory(
            settings.paths.data_dir / "working_memory.json", timezone=settings.timezone
        )
        self._facts = FactStore(self._database)
        # -- строгое хранилище знаний (границы доверия + домены C/P/H) --------
        # Собирается ЗДЕСЬ, сразу за RAG: дедупликации нужен тот же источник
        # эмбеддингов, что и дневнику (сравнивать факты вектором другого
        # движка — значит сравнивать несравнимое, см. efi/memory/dedup.py).
        self._knowledge = KnowledgeStore(self._database, embedder=self._rag)
        self._fact_validator = FactValidator(owner_id=settings.telegram.owner_id)
        self._perception = PerceptionParser(self._llm_router)
        self._pending_clarifications = PendingClarifications()
        self._memory_ingestor = MemoryIngestor(
            self._perception,
            self._fact_validator,
            self._knowledge,
            pending=self._pending_clarifications,
        )
        self._history = SqliteHistoryRepository(self._database)
        # Справочник чатов: личка это, группа или канал. Заполняется из
        # входящих сообщений, читается проактивным путём — единственным, у
        # которого своего Pyrogram-объекта чата нет (см. efi/db/chat_directory.py).
        self._chat_directory = ChatDirectory(self._database)
        # -- ремесло: свои проекты, код, GitHub ------------------------------
        # Хранилище задач и стол переговоров поднимаются ВСЕГДА, даже при
        # выключенной подсистеме: они дёшевы (таблица и словарь в памяти) и
        # нужны промпту с инструментами, чтобы Эфи знала, что у неё есть и
        # чего нет. Сам конвейер собирается ниже и только при dev.enabled.
        self._dev_store = DevTaskStore(self._database)
        self._collab_desk = CollabCodingDesk(self._dev_store)
        # -- субъектность (граф убеждений + близость/уважение + любопытство) ----
        # Все три — только Database как зависимость, поэтому конструируются
        # здесь, ДО EfiSystemPromptBuilder (которому нужны beliefs/affinity) и
        # ДО телеграм-обработчиков (которым нужны все три как *_recorder).
        self._beliefs = BeliefStore(self._database)
        self._affinity = AffinityTracker(self._database)
        self._curiosity = CuriosityTracker(self._database)
        # Социальная память по КОНКРЕТНЫМ людям (в группе chat_id общий,
        # а собеседники разные) — см. efi/memory/people.py. Зависит от
        # BeliefStore: устойчивое отношение к человеку перетекает в общую
        # матрицу убеждений, поэтому конструируется ПОСЛЕ него.
        self._people = PeopleStore(self._database, beliefs=self._beliefs)
        # Социальная память внешнего опыта: журнал в SQLite + индексация в
        # векторную память через RAG, поэтому конструируется ПОСЛЕ _rag.
        self._social_memory = SocialInteractionStore(self._database, rag=self._rag)
        # Консолидация памяти. Конструируется ПОСЛЕ PeopleStore: строгое
        # хранилище знаний, которое она наполняет, разрешает упоминания по
        # каталогу известных людей (см. efi/memory/catalog.py).
        self._consolidator = DiaryConsolidator(
            self._diary,
            self._llm_router,
            self._rag,
            novelization_char_limit=settings.memory.novelization_char_limit,
            novelization_max_output_tokens=settings.memory.novelization_max_output_tokens,
            character_name=settings.character_name,
            # Строгая память подключается ЗДЕСЬ, а не в пульсе: novelize_chat —
            # единственная точка, общая для частого пульса и ночного прохода,
            # и повесив разбор знаний на одну из них, мы получили бы память,
            # зависящую от того, каким путём эпизод дошёл до осмысления.
            knowledge=EpisodeKnowledgeSink(self._memory_ingestor, self._people),
        )
        # Жизненный цикл диалога с посторонними (владелец vs остальные).
        self._lifecycle = ConversationLifecycle(
            self._database,
            owner_id=settings.telegram.owner_id,
            # Куда вообще разрешено писать первой — тот же список, которым
            # владелец задаёт «свои» чаты. Личка владельца добавляется внутри.
            proactive_chats=settings.telegram.allowed_chats,
        )
        # Пульс памяти — превращает прожитое в воспоминания по ходу дня, а не
        # раз в сутки ночью. Зависит и от консолидатора, и от журнала внешнего
        # опыта (тот подмешивается в разбор эпизода), поэтому конструируется
        # после обоих. См. докстринг efi/memory/pulse.py.
        pulse_settings = settings.memory_pulse
        self._memory_pulse = MemoryPulse(
            self._consolidator,
            self._history,
            self._facts,
            experience=self._social_memory,
            check_interval_seconds=pulse_settings.check_interval_seconds,
            episode_idle_seconds=pulse_settings.episode_idle_seconds,
            max_messages_before_flush=pulse_settings.max_messages_before_flush,
            min_messages=pulse_settings.min_messages,
            lookback=timedelta(hours=pulse_settings.lookback_hours),
        )

        # -- инструменты, нужные фоновым исследователям (см. ниже) -------------
        # Вынесены выше "промптов"/"проактивности", т.к. BackgroundResearcher/
        # BackgroundLifeWorker должны быть готовы ДО SpontaneousPingScheduler
        # (тому нужен consume_incubated_thought как incubated_thought_provider)
        # и до телеграм-обработчиков (которым нужен GroqSTT).
        # journal=social_memory: каждый поход в интернет откладывается в
        # память сразу. Без этого веб-поиск внутри разговора не сохранялся
        # НИГДЕ — результаты приходят модели TOOL-сообщением, а оно в таблицу
        # `messages` не пишется (см. SocialInteractionKind.WEB_LOOKUP).
        self._web_search_tool = WebSearchTool(journal=self._social_memory)
        self._weather_tool = GetWeatherTool()
        # Ключ Groq для STT не дублируется отдельным полем в конфиге — берётся
        # из уже настроенных LLM-эндпоинтов, если явного переопределения нет
        # (см. Settings.resolve_groq_api_key и докстринг SttSettings).
        groq_api_key = settings.resolve_groq_api_key()
        self._stt = GroqSTT(groq_api_key.get_secret_value()) if groq_api_key is not None else None

        # -- промпты -----------------------------------------------------
        templates_dir = settings.paths.base_dir / "efi" / "prompts" / "templates"
        self._prompt_loader = PromptLoader(templates_dir)
        # Чем Эфи интересуется — worldview.json плюс семена любопытства из
        # разговоров. Нужны и участию в сообществе (efi/telegram/comments.py),
        # и замыслам собственных проектов (efi/dev/worker.py), поэтому
        # конструируются здесь, до обоих потребителей.
        self._community_interests = build_community_interests(self._database, templates_dir / "worldview.json")
        self._prompt_builder = EfiSystemPromptBuilder(
            self._prompt_loader,
            settings,
            self._rag,
            self._working_memory,
            self._beliefs,
            self._affinity,
            self._people,
            knowledge=self._knowledge,
            clarifications=self._pending_clarifications,
            # Своё ремесло в промпте: чем занята в коде и что уже выложила
            # (см. efi/prompts/builder.py, блоки «Твоё ремесло» и «Предложение
            # проекта»). Передаются всегда — блоки просто пусты, пока нечего
            # рассказывать.
            dev_store=self._dev_store,
            collab=self._collab_desk,
        )

        # -- humanizer / проактивность --------------------------------------
        self._anti_repeat = AntiRepeatTracker(settings.humanizer)
        # Право заговорить первой — ОДНО на все инициативные службы. Три
        # службы с тремя личными счётчиками дали бы ровно то, что было в
        # переписке: три «эй» подряд вместо одного (см. initiative.py).
        self._initiative = InitiativeGate(self._facts)
        self._notification_manager = NotificationManager(worker_count=worker_count)
        self._silence_monitor = SilenceMonitor(
            self._notification_manager,
            quiet_hours=settings.quiet_hours,
            timezone=settings.timezone,
            initiative=self._initiative,
        )
        # reasons проставляется ниже: PingReasonBuilder зависит от
        # BackgroundResearcher, который конструируется после монитора.
        # Отложенные напоминания («напиши мне через 10 минут»). Персистентные:
        # обещание со сроком обязано пережить перезапуск, иначе оно тихо
        # исчезает ровно тогда, когда человек на него рассчитывает.
        self._reminders = ReminderStore(self._database)
        self._reminder_scheduler = ReminderScheduler(self._notification_manager, self._reminders)
        self._scheduler = Scheduler(self._notification_manager, _build_scheduled_jobs())
        self._researcher = BackgroundResearcher(
            templates_dir / "worldview.json", self._web_search_tool, self._rag, self._llm_router, self._facts
        )
        # С чем именно она приходит, когда пишет первой. Без повода служба
        # молчит — раньше на его месте стояло «просто напомнить о себе», и из
        # этого получалось единственно возможное «эй, ты там живой?».
        self._ping_reasons = PingReasonBuilder(
            knowledge=self._knowledge,
            people=self._people,
            diary=self._diary,
            working_memory=self._working_memory,
            incubated_thought_provider=self._researcher.consume_incubated_thought,
        )
        self._silence_monitor.set_reasons(self._ping_reasons)
        self._spontaneous_ping = SpontaneousPingScheduler(
            self._notification_manager,
            self._active_chat_candidates,
            reasons=self._ping_reasons,
            quiet_hours=settings.quiet_hours,
            timezone=settings.timezone,
            initiative=self._initiative,
        )
        self._organic_ping = OrganicPingGenerator(
            self._notification_manager,
            self._affinity,
            importance_threshold=settings.life_engine.ping_importance_threshold,
            quiet_hours=settings.quiet_hours,
            timezone=settings.timezone,
            initiative=self._initiative,
        )
        self._life_engine = BackgroundLifeWorker(
            self._curiosity,
            self._web_search_tool,
            self._rag,
            self._llm_router,
            self._organic_ping,
            check_interval_seconds=settings.life_engine.check_interval_seconds,
        )
        # Конвейер разработки: кодер, песочница, GitHub и фоновый воркер.
        # Собирается только при dev.enabled и настроенном кодере, поэтому
        # может быть None — см. _build_dev_worker.
        self._dev_worker = self._build_dev_worker()

        self._busy_engine = BusyEngine(
            self._working_memory,
            self._affinity,
            # Занята она не только исследованием: пока пишется проект, «не
            # сразу увидела сообщение» — правда, а не симуляция.
            AnyBusyState(
                lambda: self._life_engine.is_researching,
                lambda: self._dev_worker is not None and self._dev_worker.is_coding,
            ),
            settings.busy_engine,
            last_message_source=self._history,
        )

        # -- telegram --------------------------------------------------------
        phone_number = (
            settings.telegram.phone_number.get_secret_value() if settings.telegram.phone_number else None
        )
        self._pyrogram_client = Client(
            settings.paths.session_name,
            api_id=settings.telegram.api_id,
            api_hash=settings.telegram.api_hash.get_secret_value(),
            # Номер нужен только при первой интерактивной авторизации; в
            # остальных запусках его нет, и Pyrogram это допускает.
            phone_number=phone_number,
            workdir=str(settings.paths.session_path.parent),
        )
        self._telegram_client = TelegramClientWrapper(self._pyrogram_client, settings.humanizer)
        # Оркестратор конструируется ДО обработчиков и воркеров: первым он
        # нужен буферу входящих (снять устаревшую генерацию в момент приёма
        # сообщения), вторым — чтобы обработка шла отменяемым таском.
        self._orchestrator = ChatOrchestrator()
        self._typing_tracker = TypingTracker(ttl_seconds=settings.humanizer.debounce_typing_ttl_seconds)
        # -- участие в сообществе (комментарии/треды) ------------------------
        self._thread_state = ThreadStateStore(self._database)
        self._channel_post_watcher = ChannelPostWatcher(
            self._notification_manager,
            settings.telegram,
            settings.community,
            self._community_interests,
            self._thread_state,
            self._social_memory,
        )
        self._random_comment_engager = RandomCommentEngager(
            self._notification_manager,
            self._pyrogram_client,
            settings.telegram,
            settings.community,
            self._community_interests,
            self._thread_state,
            self._social_memory,
        )

        self._telegram_handlers = TelegramEventHandlers(
            self._notification_manager,
            settings.telegram,
            self._llm_router,
            settings.paths.cache_dir,
            settings.humanizer,
            self._typing_tracker,
            activity_recorder=self._silence_monitor,
            affinity_recorder=self._affinity,
            curiosity_recorder=self._curiosity,
            organic_ping_recorder=self._organic_ping,
            people_recorder=self._people,
            chat_recorder=self._chat_directory,
            collab_recorder=self._collab_desk,
            stt=self._stt,
            orchestrator=self._orchestrator,
        )

        # -- реестр инструментов -------------------------------------------------
        self._tool_registry = ToolRegistry()
        self._tool_registry.register_all(self._build_tools())

        # -- дашборд ------------------------------------------------------------
        # Конструируется ПОСЛЕДНИМ: он смотрит на всё остальное. Списки задач
        # передаются вызываемыми, а не значениями — они наполняются в start(),
        # уже после сборки контекста.
        self._dashboard: DashboardServer | None = None
        if settings.dashboard.enabled:
            self._dashboard = DashboardServer(
                DashboardContext(
                    settings=settings,
                    logs=self._log_buffer,
                    metrics=self._llm_metrics,
                    started_at=self._started_at,
                    database=self._database,
                    diary=self._diary,
                    working_memory=self._working_memory,
                    history=self._history,
                    beliefs=self._beliefs,
                    affinity=self._affinity,
                    people=self._people,
                    lifecycle=self._lifecycle,
                    busy_engine=self._busy_engine,
                    life_engine=self._life_engine,
                    notifications=self._notification_manager,
                    orchestrator=self._orchestrator,
                    telegram=self._telegram_client,
                    tools=self._tool_registry,
                    llm_router=self._llm_router,
                    prompt_loader=self._prompt_loader,
                    dev_store=self._dev_store,
                    background_tasks=lambda: self._background_tasks,
                    worker_tasks=lambda: self._worker_tasks,
                ),
                settings.dashboard,
            )

    def _build_dev_worker(self) -> DevWorker | None:
        """
        Собирает конвейер разработки — или честно возвращает None.

        Две причины не собирать, и обе не ошибки: подсистема выключена
        (`dev.enabled = false`, дефолт) или не настроен кодер — ключа Groq
        нет ни явно, ни в llm_roles. Во втором случае об этом говорится в
        логе: конфиг с `enabled = true` и без ключа — это намерение, которое
        молча не сработало бы, а такое всегда должно быть слышно.

        GitHub-токена может не быть и при рабочей подсистеме: тогда проекты
        пишутся и коммитятся локально (см. efi/dev/github_sync.py).
        """
        dev_settings = self._settings.dev
        if not dev_settings.enabled:
            return None

        coder_endpoint = self._settings.resolve_coder_endpoint()
        if coder_endpoint is None:
            logger.warning(
                "app: dev.enabled = true, но кодер не настроен — нет ни dev.coder, ни ключа Groq "
                "в llm_roles. Разработка не поднимется"
            )
            return None

        workspace = dev_settings.workspace_dir(self._settings.paths)
        workspace.mkdir(parents=True, exist_ok=True)
        token = dev_settings.github_token.get_secret_value() if dev_settings.github_token else ""

        coder = QwenCoderClient(coder_endpoint)
        sandbox = CodeSandbox(enable_linter=dev_settings.lint_generated_code)
        engine = DevEngine(
            self._llm_router,
            coder,
            sandbox,
            # Замысел придумывает фоновая роль, а не MAIN: никто не ждёт
            # этого ответа в чате, и занимать им канал живого диалога нельзя
            # (регламент ролей — см. efi.config.schema.TaskRole).
            design_role=TaskRole.BACKGROUND,
            max_fix_iterations=dev_settings.max_fix_iterations,
        )
        github = GitHubSync(
            workspace,
            token=token,
            owner=dev_settings.github_owner,
            ssh_key_path=dev_settings.github_ssh_key_path,
            private=dev_settings.repo_private,
            push_enabled=dev_settings.push_enabled,
        )
        reporter = DevReporter(
            self._notification_manager,
            social_memory=self._social_memory,
            quiet_hours=self._settings.quiet_hours,
            timezone=self._settings.timezone,
            initiative=self._initiative,
            progress_probability=dev_settings.progress_probability,
        )
        # Возвращение к своим проектам: перечитать, поправить, изредка
        # спросить. Отдельный объект, а не метод воркера, потому что это
        # другая работа: там «сделать новое», здесь «пересмотреть сделанное».
        maintainer = ProjectMaintainer(
            self._dev_store,
            self._llm_router,
            coder,
            sandbox,
            github,
            reporter,
            workspace,
            review_interval=timedelta(days=dev_settings.review_interval_days),
            patch_threshold=dev_settings.patch_importance_threshold,
            discuss_threshold=dev_settings.discuss_importance_threshold,
            review_probability=dev_settings.review_probability,
        )
        logger.info(
            "app: разработка включена (кодер %s, %s)",
            coder_endpoint.model,
            "с пушем на GitHub" if github.can_publish else "локально, без пуша",
        )
        return DevWorker(
            self._dev_store,
            engine,
            github,
            reporter,
            maintainer=maintainer,
            interests=self._community_interests,
            # Своя затея рассказывается владельцу: чат для неё выбирается
            # здесь, а не воркером, — это единственное место, которое знает
            # про owner_id.
            owner_chat_id=self._settings.telegram.owner_id,
            check_interval_seconds=dev_settings.check_interval_seconds,
            self_initiated_probability=dev_settings.self_initiated_probability,
        )

    def _build_tools(self) -> list[Tool]:
        return [
            AskDiaryTool(self._rag, min_relatedness=self._settings.memory.min_relatedness),
            RememberFactTool(self._knowledge, self._fact_validator),
            RecallFactTool(self._knowledge, self._fact_validator),
            RememberDiaryEntryTool(self._rag),
            UpdateBeliefTool(self._beliefs),
            UpdateSelfStateTool(self._working_memory),
            RememberPromiseTool(
                self._working_memory,
                reminders=self._reminders,
                # Проверяем право написать первой В МОМЕНТ ОБЕЩАНИЯ: пообещать
                # и не смочь хуже, чем сразу честно предупредить.
                can_schedule=self._lifecycle.allows_proactive_ping_to_chat,
            ),
            CompletePromiseTool(self._working_memory, reminders=self._reminders),
            RememberPersonTool(self._people),
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
            StartDevProjectTool(self._collab_desk),
            DevProjectStatusTool(self._dev_store),
            self._web_search_tool,
            self._weather_tool,
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
        Список чатов-кандидатов для спонтанного пинга: личные переписки из
        allowed_chats, пересечённые с чатами, где реально было хоть одно
        сообщение (efi.db.history_repository.SqliteHistoryRepository.
        get_active_chat_ids).

        Группы и каналы отсекаются, даже если владелец перечислил их в
        allowed_chats: этот список отвечает на вопрос «где Эфи вправе
        говорить», а не «кому уместно написать первой». Написать первой
        можно человеку; в общем чате то же самое сообщение — объявление на
        весь чат, и выглядело оно ровно так: спонтанный пинг ушёл в группу,
        где у Эфи админка, обычным «как дела» (см. efi/telegram/chat_scope.py
        и дублирующую проверку в efi.notifications.worker.Worker —
        кандидатов эта служба отбирает не одна).

        Раньше отдавался «сырой» allowed_chats целиком. chat_id, который
        туда попал (например, руками в behavior.toml), но с которым этот
        Telegram-аккаунт ещё ни разу не обменивался сообщением, Pyrogram
        локально не резолвит (peer неизвестен его storage) — попытка
        send_message для такого чата падает изнутри Pyrogram KeyError'ом
        ("ID not found: ...", resolve_peer/get_peer_by_id) на КАЖДОЙ
        попытке пинга, без единого шанса на успех. Пересечение с историей —
        дешёвая гарантия, что peer уже засветился хотя бы раз и кэш есть.
        """
        # Личка владельца — кандидат всегда, как и в гейте воркера
        # (ConversationLifecycle._proactive_chats): в Telegram её chat_id
        # равен owner_id, и требовать от владельца вписать самого себя в
        # allowed_chats ради того, чтобы Эфи ему писала, незачем.
        allowed = {self._settings.telegram.owner_id, *self._settings.telegram.allowed_chats}
        active = await self._history.get_active_chat_ids(since=_EPOCH)
        candidates: list[int] = []
        for chat_id in active:
            if chat_id not in allowed:
                continue
            if not (await self._chat_directory.kind_of(chat_id)).is_one_on_one:
                logger.debug("app: chat_id=%s пропущен для спонтанного пинга — это не личка", chat_id)
                continue
            candidates.append(chat_id)
        return candidates

    async def start(self) -> None:
        """Поднимает все подсистемы: Telegram-клиент, обработчики, воркеры, проактивные сервисы."""
        # Буфер логов встаёт на корневой логгер ПЕРВЫМ делом: иначе ровно то,
        # что происходит на старте (а падает чаще всего именно там), в ленту
        # дашборда не попадёт.
        self._log_buffer.install()
        logger.info("app: starting")

        # Кто фактически может с ней говорить — одной строкой при старте.
        # «Почему она не отвечает» это вопрос про сочетание lockdown_mode,
        # allowed_chats и community_chats, и выяснять его по конфигу вручную
        # неудобно ровно тогда, когда что-то не работает.
        logger.info("app: отвечает — %s", describe_access_policy(self._settings.telegram))

        # Какие роли LLM не настроены и в чью модель они из-за этого пойдут.
        # Молча подставить MAIN вместо VISION нельзя: текстовая модель на
        # фотографию ответит ошибкой или выдумкой, и узнать об этом владелец
        # должен при запуске, а не когда ему пришлют картинку.
        for note in self._settings.llm_roles.describe_fallbacks():
            logger.warning("app: %s", note)

        self._telegram_handlers.register(self._pyrogram_client)
        self._channel_post_watcher.register(self._pyrogram_client)
        self._typing_tracker.register(self._pyrogram_client)
        await self._telegram_client.start()

        for worker_index in range(self._notification_manager.worker_count):
            worker = Worker(
                worker_index,
                self._notification_manager,
                llm_router=self._llm_router,
                tool_registry=self._tool_registry,
                history=self._history,
                system_prompt_builder=self._prompt_builder,
                busy_engine=self._busy_engine,
                main_role=TaskRole.MAIN,
                telegram=self._telegram_client,
                history_limit=self._settings.memory.history_limit,
                lifecycle=self._lifecycle,
                social_memory=self._social_memory,
                orchestrator=self._orchestrator,
                working_memory=self._working_memory,
                clarifications=self._pending_clarifications,
                initiative=self._initiative,
                chat_directory=self._chat_directory,
            )
            self._worker_tasks.append(asyncio.create_task(worker.run(), name=f"worker-{worker_index}"))

        self._background_tasks.extend(
            [
                self._spawn_supervised(lambda: self._scheduler.run(), name="scheduler"),
                self._spawn_supervised(lambda: self._silence_monitor.run(), name="silence_monitor"),
                self._spawn_supervised(lambda: self._spontaneous_ping.run(), name="spontaneous_ping"),
                self._spawn_supervised(lambda: self._researcher.run(), name="background_researcher"),
                self._spawn_supervised(lambda: self._life_engine.run(), name="life_engine"),
                self._spawn_supervised(lambda: self._prompt_loader.watch(), name="prompt_loader_watch"),
                self._spawn_supervised(lambda: self._run_consolidation_loop(), name="diary_consolidation"),
                self._spawn_supervised(lambda: self._random_comment_engager.run(), name="random_comment_engager"),
                self._spawn_supervised(lambda: self._reminder_scheduler.run(), name="reminders"),
            ]
        )

        if self._dev_worker is not None:
            dev_worker = self._dev_worker
            self._background_tasks.append(
                self._spawn_supervised(lambda: dev_worker.run(), name="dev_worker")
            )

        if self._settings.memory_pulse.enabled:
            self._background_tasks.append(self._spawn_supervised(lambda: self._memory_pulse.run(), name="memory_pulse"))

        if self._dashboard is not None:
            # Дашборд поднимается ПОСЛЕДНИМ и не через _spawn_supervised: он
            # не крутит свой цикл, а держит asyncio-сервер, и его падение при
            # старте (занятый порт) не должно остаться незамеченным — но и
            # ронять из-за него уже поднятую Эфи неправильно.
            try:
                await self._dashboard.start()
            except OSError as exc:
                logger.error("app: dashboard failed to start (%s), continuing without it", exc)
                self._dashboard = None

        logger.info(
            "app: started (%d workers, %d background services)",
            len(self._worker_tasks), len(self._background_tasks),
        )

    def _spawn_supervised(self, factory: Callable[[], Coroutine[Any, Any, None]], *, name: str) -> asyncio.Task[None]:
        """
        Фоновая служба, которая ПЕРЕЗАПУСКАЕТСЯ, если всё-таки упала.

        Голый asyncio.create_task() для долгоживущего сервиса — тихая дыра:
        упавшая корутина просто перестаёт существовать, а исключение всплывает
        либо на shutdown при gather(), либо вообще только сборщиком мусора как
        "Task exception was never retrieved". Служба молча переставала
        работать, и в логах на этот счёт не было ничего.

        Логирования, которое здесь стояло раньше, оказалось мало. Оно делает
        поломку видимой в логе — но Эфи от этого не начинает снова писать
        первой. А смотрят в лог обычно уже после того, как заметили странность
        поведения, то есть спустя дни.

        Поэтому принимается ФАБРИКА корутины, а не корутина: перезапуск —
        это новый вызов, а однажды исчерпанную корутину повторно запустить
        нельзя. Между попытками — растущая пауза: если служба падает сразу
        после старта (испорченный файл, недоступная БД), перезапуск в цикле
        только забьёт лог и посадит батарею.
        """

        async def _supervise() -> None:
            attempt = 0
            while True:
                try:
                    await factory()
                    logger.info("app: background task %r finished on its own", name)
                    return
                except asyncio.CancelledError:
                    raise
                except Exception:
                    attempt += 1
                    if attempt > _MAX_SERVICE_RESTARTS:
                        logger.error(
                            "app: background task %r упала %d раз подряд, больше не перезапускаю",
                            name, attempt, exc_info=True,
                        )
                        return
                    delay = min(_SERVICE_RESTART_BASE_DELAY * 2 ** (attempt - 1), _SERVICE_RESTART_MAX_DELAY)
                    logger.error(
                        "app: background task %r died, перезапуск через %.0fs (попытка %d/%d)",
                        name, delay, attempt, _MAX_SERVICE_RESTARTS, exc_info=True,
                    )
                    await asyncio.sleep(delay)

        return asyncio.create_task(_supervise(), name=name)

    async def _run_consolidation_loop(self) -> None:
        """
        Ночное обслуживание корпуса памяти: подбор хвостов новеллизации,
        dedup существующих записей + сжатие старых записей в мемуары через
        LLM (см. efi/memory/consolidation.py). Идёт своим отдельным
        ежедневным расписанием, НЕ через NotificationManager — это
        обслуживание данных, а не разговорный ответ модели. Дополняет, а не
        заменяет "nightly_consolidation" ScheduledJob (та даёт личности повод
        отрефлексировать день в разговорном формате).

        Новеллизация здесь БОЛЬШЕ НЕ ОСНОВНОЙ путь пополнения дневника: по
        ходу дня её делает efi.memory.pulse.MemoryPulse, эпизодами, сразу
        после того, как разговор закончился. Ночью остаётся только то, до
        чего пульс не добрался — чаты, где эпизод так и не закрылся, и, если
        пульс выключен настройкой, вообще всё за сутки (прежнее поведение).
        """
        try:
            while True:
                await asyncio.sleep(seconds_until_next(_CONSOLIDATION_TRIGGER_AT))

                # Новеллизация — ПЕРВОЙ: она создаёт новые записи из дня, а
                # dedup ниже заодно подчистит и их, если что-то похожее уже
                # было записано вручную через remember_diary_entry за день.
                # Под общим локом с пульсом: оба пути двигают одну и ту же
                # отметку last_novelized_at, и без взаимного исключения могли
                # бы прочитать её одновременно и разобрать одно окно дважды.
                async with self._memory_pulse.novelization_lock:
                    novelized = await self._consolidator.novelize_recent_history(
                        history=self._history,
                        facts=self._facts,
                        lookback=timedelta(days=self._settings.memory.novelization_lookback_days),
                        min_messages=self._settings.memory.novelization_min_messages,
                        experience=self._social_memory,
                    )
                logger.info("app: nightly novelization saved %d new diary entries", novelized)

                removed = await self._consolidator.deduplicate(
                    plagiarism_threshold=self._settings.memory.plagiarism_threshold
                )
                logger.info("app: nightly dedup removed %d duplicate diary entries", removed)

                merged = await self._consolidator.summarize_stale_entries()
                if merged is not None:
                    logger.info("app: nightly consolidation created memoir entry %s", merged.id)

                pruned_messages = await self._history.prune_old_messages()
                logger.info("app: nightly cleanup pruned %d old history messages", pruned_messages)
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

        # Дашборд гасится первым: он читает состояние всех подсистем, и его
        # запрос, пришедший посреди остановки, увидел бы полуразобранное
        # приложение. Логи при этом продолжают писаться в буфер до самого
        # конца — обработчик снимается уже после остановки всего остального.
        if self._dashboard is not None:
            await self._dashboard.stop()

        # Недописанные, ещё не отфлашенные из дебаунсера сообщения (человек
        # написал что-то за секунды до остановки) — сбрасываем в очередь,
        # а не молча теряем.
        await self._telegram_handlers.flush_pending()
        # Запланированные, но ещё не сработавшие комментарии — снимаем:
        # они спят минутами, и без отмены shutdown ждал бы их впустую.
        await self._channel_post_watcher.cancel_pending()
        # Активные генерации: недоговорённая серия бабблов не должна держать
        # остановку на своих паузах между сообщениями.
        await self._orchestrator.cancel_all()
        # Отложенные повторы недоставленных проактивных уведомлений — тоже
        # спят минутами, и ждать их на выключении незачем: повод протухнет
        # раньше, чем таймер сработает.
        await self._notification_manager.cancel_retries()

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
        await self._weather_tool.aclose()
        if self._stt is not None:
            await self._stt.aclose()
        # БД — последней: до этого момента остановка ещё может писать
        # (сохранение недоговорённых бабблов, закрытие обещаний). Закрыть
        # обязательно: aiosqlite держит под соединение НЕ-daemon-поток, и
        # незакрытое соединение не даёт процессу завершиться.
        await self._database.close()

        logger.info("app: stopped")
        self._log_buffer.uninstall()


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
