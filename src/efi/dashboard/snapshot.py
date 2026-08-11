"""
efi/dashboard/snapshot.py

`DashboardContext` — единственная точка, где дашборд знает про внутренности
приложения, и функции сборки снимков состояния для каждого раздела.

Почему контекст, а не ссылка на `EfiApp`: во-первых, обратный импорт
(`dashboard` -> `app` -> `dashboard`) — цикл; во-вторых, все поля здесь
необязательные, поэтому раздел дашборда можно проверить тестом, собрав
контекст из двух-трёх настоящих объектов и не поднимая ни Telegram, ни
LLM-роутер. `EfiApp` просто заполняет контекст тем, что у него уже есть.

Никаких LLM-вызовов на путях дашборда: снимки собираются из памяти
процесса, SQLite и файлов дневника. Открыть страницу состояния не должно
стоить ни одного запроса к модели — иначе наблюдение начало бы менять
наблюдаемое (и тратить лимиты бесплатных тиров).
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Protocol

from efi.behavior.affinity import AffinityTracker
from efi.behavior.busy_engine import BusyEngine
from efi.behavior.conversation_lifecycle import ConversationLifecycle
from efi.behavior.life_engine import BackgroundLifeWorker
from efi.behavior.quiet_hours import is_quiet_hours
from efi.config.schema import Settings, TaskRole
from efi.dashboard import queries
from efi.dashboard.logbus import LogBuffer
from efi.dashboard.metrics import LLMMetricsCollector
from efi.db.core import Database
from efi.db.history_repository import SqliteHistoryRepository
from efi.llm.router import LLMRouter
from efi.memory.beliefs import BeliefStore
from efi.memory.diary import Diary
from efi.memory.people import PeopleStore
from efi.memory.working_memory import WorkingMemory
from efi.notifications.manager import NotificationManager
from efi.notifications.schemas import Notification, NotificationType
from efi.prompts.loader import PromptLoader
from efi.tools.base import ToolContext
from efi.tools.registry import ToolRegistry
from efi.utils.text import looks_unfinished

logger = logging.getLogger(__name__)

_DIARY_PREVIEW_CHARS = 220


class GenerationSource(Protocol):
    """Что дашборду нужно от `efi.telegram.chat_orchestrator.ChatOrchestrator`."""

    def active_chat_ids(self) -> list[int]: ...


class TelegramStatusSource(Protocol):
    """Что дашборду нужно от `efi.telegram.client.TelegramClientWrapper`."""

    @property
    def is_connected(self) -> bool: ...


#: Человекочитаемые названия фоновых сервисов (ключ — имя asyncio-задачи,
#: которое `EfiApp._spawn_supervised` присваивает при запуске).
_SERVICE_TITLES: dict[str, tuple[str, str]] = {
    "scheduler": ("Планировщик", "Ночная консолидация и утреннее пробуждение по расписанию"),
    "silence_monitor": ("Монитор тишины", "Замечает затянувшееся молчание в чате и отложенные обещания вернуться"),
    "spontaneous_ping": ("Спонтанные пинги", "Решает написать первой в чат, где давно ничего не было"),
    "background_researcher": ("Фоновый исследователь", "Копает темы из картины мира, вынашивает мысли"),
    "life_engine": ("Движок жизни", "Разбирает семена любопытства и приносит находки"),
    "prompt_loader_watch": ("Слежение за шаблонами", "Горячая перезагрузка personality.md без перезапуска"),
    "diary_consolidation": ("Ночная консолидация", "Новеллизация хвостов, дедупликация и мемуары"),
    "random_comment_engager": ("Участие в тредах", "Заглядывает в обсуждения сообщества и иногда вписывается"),
    "memory_pulse": ("Пульс памяти", "Превращает законченный разговор в запись дневника по ходу дня"),
}


@dataclass(slots=True)
class DashboardContext:
    """
    Ссылки на подсистемы, состояние которых показывает дашборд.

    Всё, кроме настроек и буфера логов, необязательно: раздел, для которого
    зависимости не переданы, отдаёт пустой ответ вместо падения.
    """

    settings: Settings
    logs: LogBuffer
    metrics: LLMMetricsCollector = field(default_factory=LLMMetricsCollector)
    started_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    database: Database | None = None
    diary: Diary | None = None
    working_memory: WorkingMemory | None = None
    history: SqliteHistoryRepository | None = None
    beliefs: BeliefStore | None = None
    affinity: AffinityTracker | None = None
    people: PeopleStore | None = None
    lifecycle: ConversationLifecycle | None = None
    busy_engine: BusyEngine | None = None
    life_engine: BackgroundLifeWorker | None = None
    notifications: NotificationManager | None = None
    orchestrator: GenerationSource | None = None
    telegram: TelegramStatusSource | None = None
    tools: ToolRegistry | None = None
    llm_router: LLMRouter | None = None
    prompt_loader: PromptLoader | None = None

    #: Возвращают живые списки задач приложения. Именно вызываемые, а не
    #: списки: `EfiApp` наполняет их уже после `start()`, и сохранённая
    #: ссылка на пустой список навсегда осталась бы пустой.
    background_tasks: Callable[[], Sequence[asyncio.Task[Any]]] | None = None
    worker_tasks: Callable[[], Sequence[asyncio.Task[Any]]] | None = None

    @property
    def uptime_seconds(self) -> float:
        return max(0.0, (datetime.now(UTC) - self.started_at).total_seconds())


# ---------------------------------------------------------------------------
# Обзор
# ---------------------------------------------------------------------------


async def build_overview(context: DashboardContext) -> dict[str, Any]:
    """Верхний уровень: жива ли Эфи, чем занята прямо сейчас и что у неё накопилось."""
    settings = context.settings
    counts: dict[str, int] = {}
    activity: list[dict[str, Any]] = []
    if context.database is not None:
        counts, activity = await asyncio.gather(
            queries.table_counts(context.database),
            queries.messages_per_day(context.database, days=14),
        )
    if context.diary is not None:
        counts["diary"] = len(await context.diary.all_entries())

    working_memory: dict[str, Any] = {}
    if context.working_memory is not None:
        snapshot = await context.working_memory.load()
        working_memory = {
            "emotional_state": snapshot.emotional_state,
            "physical_state": snapshot.physical_state,
            "energy": round(snapshot.energy, 3),
            "open_items": sum(1 for item in snapshot.items if not item.done),
            "updated_at": snapshot.updated_at.isoformat(),
        }

    return {
        "character_name": settings.character_name,
        "environment": settings.environment.value,
        "debug": settings.debug,
        "now": datetime.now(UTC).isoformat(),
        "started_at": context.started_at.isoformat(),
        "uptime_seconds": round(context.uptime_seconds, 1),
        "telegram": _telegram_overview(context),
        "queue": _queue_overview(context),
        "generations": _generation_overview(context),
        "services": _services_overview(context),
        "quiet_hours": _quiet_hours_overview(context),
        "working_memory": working_memory,
        "llm": context.metrics.totals(),
        "logs": {
            "total": context.logs.total,
            "buffered": context.logs.buffered,
            "capacity": context.logs.capacity,
            "counts": context.logs.counts(),
            "subscribers": context.logs.subscriber_count,
        },
        "counts": counts,
        "activity": activity,
    }


def _telegram_overview(context: DashboardContext) -> dict[str, Any]:
    telegram = context.settings.telegram
    connected: bool | None = None
    if context.telegram is not None:
        try:
            connected = bool(context.telegram.is_connected)
        except Exception:
            # Состояние соединения читается из чужой библиотеки; страница
            # состояния не должна падать оттого, что оно недоступно.
            logger.debug("dashboard: unable to read telegram connection state", exc_info=True)
            connected = None
    return {
        "connected": connected,
        "owner_id": telegram.owner_id,
        "lockdown_mode": telegram.lockdown_mode.value,
        "allowed_chats": len(telegram.allowed_chats),
        "community_chats": len(telegram.community_chats),
        "can_join_chats": telegram.can_join_chats,
        "can_leave_chats": telegram.can_leave_chats,
    }


def _queue_overview(context: DashboardContext) -> dict[str, Any]:
    manager = context.notifications
    if manager is None:
        return {"total": 0, "worker_count": 0, "workers": []}

    tasks = list(context.worker_tasks()) if context.worker_tasks is not None else []
    workers: list[dict[str, Any]] = []
    for index in range(manager.worker_count):
        task = tasks[index] if index < len(tasks) else None
        state = _task_state(task)
        workers.append(
            {
                "index": index,
                "queued": manager.qsize(index),
                "state": state["state"],
                "detail": state["detail"],
            }
        )
    return {"total": manager.qsize(), "worker_count": manager.worker_count, "workers": workers}


def _generation_overview(context: DashboardContext) -> dict[str, Any]:
    if context.orchestrator is None:
        return {"active": 0, "chats": []}
    chat_ids = context.orchestrator.active_chat_ids()
    return {
        "active": len(chat_ids),
        "chats": [{"chat_id": chat_id, "label": _chat_label(context, chat_id)} for chat_id in chat_ids],
    }


def _services_overview(context: DashboardContext) -> list[dict[str, Any]]:
    """
    Состояние каждого фонового сервиса: запущен, остановлен или упал.

    Отдельно от `enabled` из конфига: сервис может быть включён настройками,
    но лежать — именно этот разрыв дашборд и должен показывать. До него
    упавшая фоновая задача была видна только строчкой `background task ... died`
    в логах, которую нужно было не пропустить в момент падения.
    """
    settings = context.settings
    enabled_by_config = {
        "memory_pulse": settings.memory_pulse.enabled,
        "random_comment_engager": settings.community.enabled and bool(settings.telegram.community_chats),
        "spontaneous_ping": True,
        "silence_monitor": True,
        "scheduler": True,
        "background_researcher": True,
        "life_engine": True,
        "prompt_loader_watch": True,
        "diary_consolidation": True,
    }

    tasks = list(context.background_tasks()) if context.background_tasks is not None else []
    running_by_name = {task.get_name(): task for task in tasks}

    services: list[dict[str, Any]] = []
    for name, (title, description) in _SERVICE_TITLES.items():
        task = running_by_name.get(name)
        state = _task_state(task)
        if task is None:
            state = {
                "state": "disabled" if not enabled_by_config.get(name, True) else "not_started",
                "detail": "выключен в конфигурации" if not enabled_by_config.get(name, True) else "",
            }
        services.append(
            {
                "name": name,
                "title": title,
                "description": description,
                "enabled": enabled_by_config.get(name, True),
                "state": state["state"],
                "detail": state["detail"],
            }
        )

    # Задачи, которых нет в справочнике (появятся в будущем), всё равно видны —
    # лучше строка без описания, чем незаметно не показанный сервис.
    for name, task in running_by_name.items():
        if name in _SERVICE_TITLES:
            continue
        state = _task_state(task)
        services.append(
            {
                "name": name,
                "title": name,
                "description": "",
                "enabled": True,
                "state": state["state"],
                "detail": state["detail"],
            }
        )
    return services


def _task_state(task: asyncio.Task[Any] | None) -> dict[str, str]:
    if task is None:
        return {"state": "not_started", "detail": ""}
    if not task.done():
        return {"state": "running", "detail": ""}
    if task.cancelled():
        return {"state": "stopped", "detail": "задача отменена"}
    exception = task.exception()
    if exception is not None:
        return {"state": "failed", "detail": f"{type(exception).__name__}: {exception}"}
    return {"state": "finished", "detail": "завершилась сама"}


def _quiet_hours_overview(context: DashboardContext) -> dict[str, Any]:
    quiet = context.settings.quiet_hours
    now = datetime.now()
    active = quiet.enabled and is_quiet_hours(now, start_hour=quiet.start_hour, end_hour=quiet.end_hour)
    return {
        "enabled": quiet.enabled,
        "start_hour": quiet.start_hour,
        "end_hour": quiet.end_hour,
        "active_now": active,
        "local_time": now.isoformat(timespec="seconds"),
    }


def _chat_label(context: DashboardContext, chat_id: int | None) -> str:
    if chat_id is None:
        return "без чата"
    label = context.settings.telegram.chat_labels.get(chat_id)
    if label:
        return label
    if chat_id == context.settings.telegram.owner_id:
        return context.settings.telegram.owner_display_name or "владелец"
    return str(chat_id)


# ---------------------------------------------------------------------------
# Состояние самой Эфи
# ---------------------------------------------------------------------------


async def build_self_state(context: DashboardContext) -> dict[str, Any]:
    """Рабочая память, убеждения, близость к чатам и текущая «занятость»."""
    payload: dict[str, Any] = {
        "working_memory": {"emotional_state": "", "physical_state": "", "energy": None, "items": []},
        "beliefs": [],
        "affinity": [],
        "busy": None,
        "quiet_hours": _quiet_hours_overview(context),
        "researching": None,
    }

    if context.working_memory is not None:
        snapshot = await context.working_memory.load()
        payload["working_memory"] = {
            "emotional_state": snapshot.emotional_state,
            "physical_state": snapshot.physical_state,
            "energy": round(snapshot.energy, 3),
            "updated_at": snapshot.updated_at.isoformat(),
            "items": [
                {
                    "text": item.text,
                    "created_at": item.created_at.isoformat(),
                    "last_updated": item.last_updated.isoformat(),
                    "done": item.done,
                }
                for item in snapshot.items
            ],
        }

    if context.beliefs is not None:
        payload["beliefs"] = [
            {
                "topic": belief.topic,
                "stance": belief.stance,
                "confidence_score": round(belief.confidence_score, 3),
                "origin_date": belief.origin_date.isoformat(),
            }
            for belief in await context.beliefs.all_beliefs()
        ]

    if context.database is not None:
        rows = await queries.chats(context.database, limit=50)
        payload["affinity"] = [
            {
                "chat_id": row["chat_id"],
                "label": _chat_label(context, row["chat_id"]),
                "affinity": row["affinity"],
                "respect_level": row["respect_level"],
                "message_count": row["message_count"],
                "last_at": row["last_at"],
            }
            for row in rows
            if row["affinity"] is not None
        ]

    if context.life_engine is not None:
        payload["researching"] = context.life_engine.is_researching

    if context.busy_engine is not None:
        # Оценка "сколько бы она сейчас тянула перед ответом" для события без
        # чата: расчёт содержит случайную составляющую, поэтому это именно
        # оценка текущей занятости, а не воспроизводимое число.
        decision = await context.busy_engine.decide(None)
        payload["busy"] = {
            "delay_seconds": round(decision.delay_seconds, 2),
            "is_active_conversation": decision.is_active_conversation,
        }

    return payload


# ---------------------------------------------------------------------------
# Функции: сервисы, инструменты, модели
# ---------------------------------------------------------------------------


async def build_functions(context: DashboardContext) -> dict[str, Any]:
    """Всё, что у Эфи «работает»: фоновые сервисы, инструменты модели и маршруты LLM."""
    return {
        "services": _services_overview(context),
        "queue": _queue_overview(context),
        "tools": _tools_overview(context),
        "llm_roles": _llm_roles_overview(context),
        "behavior": _behavior_overview(context),
    }


def _tools_overview(context: DashboardContext) -> list[dict[str, Any]]:
    if context.tools is None:
        return []
    # Нейтральный контекст: у события без чата инструменту нечего запрещать по
    # собеседнику, поэтому `is_available` покажет только его собственные
    # ограничения (например, выключенный настройкой join_chat).
    probe = ToolContext(
        notification=Notification(type=NotificationType.NIGHTLY_TASK, message="dashboard probe"),
    )
    available = {tool.name for tool in context.tools.available_tools(probe)}
    return [
        {
            "name": tool.name,
            "description": tool.description,
            "available": tool.name in available,
            "parameters": sorted((tool.parameters.get("properties") or {}).keys()),
        }
        for tool in sorted(context.tools.all_tools(), key=lambda tool: tool.name)
    ]


def _llm_roles_overview(context: DashboardContext) -> list[dict[str, Any]]:
    """Кандидаты каждой роли по порядку перебора и то, кто из них сейчас в cooldown."""
    routes = context.settings.llm_roles.as_routes()
    cooldowns = context.llm_router.cooldown_snapshot() if context.llm_router is not None else {}

    rows: list[dict[str, Any]] = []
    for role in TaskRole:
        route = routes[role]
        slots = [("primary", route.primary), ("fallback", route.fallback)]
        candidates = []
        for slot, endpoint in slots:
            if endpoint is None:
                continue
            key = (endpoint.base_url, endpoint.model)
            candidates.append(
                {
                    "slot": slot,
                    "base_url": endpoint.base_url,
                    "model": endpoint.model,
                    "timeout_seconds": endpoint.timeout_seconds,
                    "cooldown_seconds": round(cooldowns.get(key, 0.0), 1),
                }
            )
        rows.append(
            {
                "role": role.value,
                "degrade_to": route.degrade_to.value if route.degrade_to is not None else None,
                "candidates": candidates,
            }
        )
    return rows


def _behavior_overview(context: DashboardContext) -> dict[str, Any]:
    """Ключевые настройки поведения — чтобы не лезть в behavior.toml, чтобы понять, почему она молчит."""
    settings = context.settings
    return {
        "humanizer": {
            "typing_wpm": [settings.humanizer.typing_wpm_min, settings.humanizer.typing_wpm_max],
            "typo_probability": settings.humanizer.typo_probability,
            "typo_self_correct_probability": settings.humanizer.typo_self_correct_probability,
            "max_messages_per_burst": settings.humanizer.max_messages_per_burst,
            "debounce_window": [
                settings.humanizer.debounce_window_min_seconds,
                settings.humanizer.debounce_window_max_seconds,
            ],
            "anti_repeat_max_history": settings.humanizer.anti_repeat_max_history,
        },
        "memory": {
            "history_limit": settings.memory.history_limit,
            "min_relatedness": settings.memory.min_relatedness,
            "plagiarism_threshold": settings.memory.plagiarism_threshold,
            "use_local_embeddings": settings.memory.use_local_embeddings,
            "local_embedding_model": settings.memory.local_embedding_model,
        },
        "memory_pulse": {
            "enabled": settings.memory_pulse.enabled,
            "check_interval_seconds": settings.memory_pulse.check_interval_seconds,
            "episode_idle_seconds": settings.memory_pulse.episode_idle_seconds,
            "min_messages": settings.memory_pulse.min_messages,
        },
        "busy_engine": {
            "base_delay": [
                settings.busy_engine.base_delay_min_seconds,
                settings.busy_engine.base_delay_max_seconds,
            ],
            "max_delay_seconds": settings.busy_engine.max_delay_seconds,
            "active_conversation_window_seconds": settings.busy_engine.active_conversation_window_seconds,
        },
        "life_engine": {
            "check_interval_seconds": settings.life_engine.check_interval_seconds,
            "ping_importance_threshold": settings.life_engine.ping_importance_threshold,
        },
        "community": {
            "enabled": settings.community.enabled,
            "chats": len(settings.telegram.community_chats),
            "comment_probability": settings.community.comment_probability,
            "delay": [settings.community.min_delay_seconds, settings.community.max_delay_seconds],
            "topic_match_min_score": settings.community.topic_match_min_score,
        },
    }


# ---------------------------------------------------------------------------
# Дневник
# ---------------------------------------------------------------------------


async def build_diary_list(
    context: DashboardContext,
    *,
    query: str = "",
    limit: int = 50,
    offset: int = 0,
) -> dict[str, Any]:
    """
    Список записей дневника без эмбеддингов.

    Поиск здесь подстрочный, а не семантический: семантический потребовал бы
    эмбеддинга запроса, то есть сетевого вызова на каждое нажатие клавиши в
    поле поиска. Смысловой поиск по дневнику — это инструмент самой Эфи
    (`ask_diary`), а дашборду достаточно честного «найди по словам».
    """
    if context.diary is None:
        return {"total": 0, "shown": 0, "offset": offset, "entries": []}

    entries = await context.diary.all_entries()
    needle = query.strip().lower()
    if needle:
        entries = [entry for entry in entries if needle in entry.body.lower() or needle in entry.id.lower()]

    entries.sort(key=lambda entry: entry.metadata.created_at, reverse=True)
    total = len(entries)
    page = entries[offset : offset + max(1, limit)]

    return {
        "total": total,
        "shown": len(page),
        "offset": offset,
        "query": query,
        "entries": [
            {
                "id": entry.id,
                "preview": _preview(entry.body),
                "length": len(entry.body),
                "created_at": entry.metadata.created_at.isoformat(),
                "last_used": entry.metadata.last_used.isoformat() if entry.metadata.last_used else None,
                "usage_count": entry.metadata.usage_count,
                "confidence": entry.metadata.confidence,
                "score": entry.metadata.score,
                "embedding_dim": len(entry.metadata.embedding),
                "unfinished": looks_unfinished(entry.body),
            }
            for entry in page
        ],
    }


async def build_diary_entry(context: DashboardContext, entry_id: str) -> dict[str, Any] | None:
    """Полный текст одной записи. Эмбеддинг наружу не отдаётся — только его размерность."""
    if context.diary is None:
        return None
    entry = await context.diary.get(entry_id)
    if entry is None:
        return None
    return {
        "id": entry.id,
        "body": entry.body,
        "created_at": entry.metadata.created_at.isoformat(),
        "last_used": entry.metadata.last_used.isoformat() if entry.metadata.last_used else None,
        "usage_count": entry.metadata.usage_count,
        "confidence": entry.metadata.confidence,
        "score": entry.metadata.score,
        "embedding_dim": len(entry.metadata.embedding),
        "is_ground_truth": entry.metadata.is_ground_truth,
        "is_marked_false": entry.metadata.is_marked_false,
        # Записи, сохранённые до починки бюджетов вывода, так и остались
        # оборванными на полуслове. Удалять их за спиной у владельца — не
        # дело дашборда, но показать, какие именно пострадали, он обязан:
        # иначе их не отличить от целых.
        "unfinished": looks_unfinished(entry.body),
    }


def _preview(body: str) -> str:
    text = " ".join(body.split())
    if len(text) <= _DIARY_PREVIEW_CHARS:
        return text
    return text[:_DIARY_PREVIEW_CHARS].rstrip() + "…"


# ---------------------------------------------------------------------------
# Память и люди
# ---------------------------------------------------------------------------


async def build_memory(context: DashboardContext, *, limit: int = 100, query: str = "") -> dict[str, Any]:
    """Факты, семена любопытства, журнал внешнего опыта и состояния диалогов — одним снимком."""
    if context.database is None:
        return {
            "knowledge": [],
            "rejections": [],
            "facts": [],
            "seeds": [],
            "social": [],
            "conversations": [],
            "threads": [],
            "tasks": [],
        }

    knowledge, rejections, facts, seeds, social, conversations, threads, tasks = await asyncio.gather(
        queries.knowledge_facts(context.database, limit=limit, query=query),
        queries.knowledge_rejections(context.database, limit=limit),
        queries.facts(context.database, limit=limit, query=query),
        queries.curiosity_seeds(context.database, limit=limit),
        queries.social_interactions(context.database, limit=limit),
        queries.conversation_states(context.database, limit=limit),
        queries.thread_states(context.database, limit=limit),
        queries.proactive_tasks(context.database, limit=limit),
    )
    for row in conversations:
        row["chat_label"] = _chat_label(context, row["chat_id"])
    for row in tasks:
        row["chat_label"] = _chat_label(context, row["chat_id"])
    return {
        "knowledge": knowledge,
        "rejections": rejections,
        "facts": facts,
        "seeds": seeds,
        "social": social,
        "conversations": conversations,
        "threads": threads,
        "tasks": tasks,
    }


async def build_people(context: DashboardContext, *, limit: int = 100) -> dict[str, Any]:
    """Кого Эфи знает лично: объём общения, близость, уважение и сложившееся впечатление."""
    if context.database is None:
        return {"people": []}
    rows = await queries.people(context.database, limit=limit)
    owner_id = context.settings.telegram.owner_id
    for row in rows:
        row["is_owner"] = row["user_id"] == owner_id
    return {"people": rows}


# ---------------------------------------------------------------------------
# Чаты и переписка
# ---------------------------------------------------------------------------


async def build_chats(context: DashboardContext, *, limit: int = 100) -> dict[str, Any]:
    if context.database is None:
        return {"chats": []}
    rows = await queries.chats(context.database, limit=limit)
    allowed = set(context.settings.telegram.allowed_chats)
    community = set(context.settings.telegram.community_chats)
    for row in rows:
        chat_id = int(row["chat_id"])
        row["label"] = _chat_label(context, chat_id)
        row["is_owner"] = chat_id == context.settings.telegram.owner_id
        row["is_allowed"] = chat_id in allowed
        row["is_community"] = chat_id in community
        row["is_generating"] = (
            context.orchestrator is not None and chat_id in set(context.orchestrator.active_chat_ids())
        )
    return {"chats": rows}


async def build_chat_messages(
    context: DashboardContext,
    chat_id: int,
    *,
    limit: int = 100,
    before_id: int | None = None,
) -> dict[str, Any]:
    if context.database is None:
        return {"chat_id": chat_id, "messages": []}
    rows = await queries.messages(context.database, chat_id, limit=limit, before_id=before_id)
    return {
        "chat_id": chat_id,
        "label": _chat_label(context, chat_id),
        "messages": rows,
        "has_more": len(rows) >= limit,
    }


# ---------------------------------------------------------------------------
# Промпты
# ---------------------------------------------------------------------------


async def build_prompts(context: DashboardContext) -> dict[str, Any]:
    """
    Исходники личности: `personality.md` (с горячей перезагрузкой) и то, что
    задано настройками. Готовый системный промпт здесь НЕ собирается — его
    сборка ходит в RAG за эмбеддингами, то есть в сеть, а открытие страницы
    не должно стоить запроса к модели.
    """
    personality_template: str | None = None
    if context.prompt_loader is not None:
        try:
            personality_template = await context.prompt_loader.get("personality")
        except FileNotFoundError:
            personality_template = None

    return {
        "character_name": context.settings.character_name,
        "personality_prompt": context.settings.personality_prompt,
        "personality_template": personality_template,
        "sycophancy_protection": context.settings.state_vector.sycophancy_protection_text,
        "relevant_beliefs_limit": context.settings.state_vector.relevant_beliefs_limit,
        "owner_display_name": context.settings.telegram.owner_display_name,
    }


__all__ = [
    "DashboardContext",
    "build_chat_messages",
    "build_chats",
    "build_diary_entry",
    "build_diary_list",
    "build_functions",
    "build_memory",
    "build_overview",
    "build_people",
    "build_prompts",
    "build_self_state",
]
