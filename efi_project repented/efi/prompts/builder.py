"""
efi/prompts/builder.py

Реализация efi.notifications.worker.SystemPromptBuilder: собирает полный
системный промпт для одного обращения к LLM — личность, временной контекст,
рабочую память, RAG-факты, контекст чата (личка/группа) и ограничения
безопасности — в один связный текст.

С появлением этого модуля Worker больше не строит собственный "preface"
system-блок с working memory/RAG (как было до Шага 6) — эта логика переехала
сюда, потому что именно PromptBuilder знает, КАК личность должна
воспринимать эти данные, а Worker остаётся безразличным к содержанию
промпта и занимается только оркестрацией истории и tool-calling циклом
(см. efi/notifications/worker.py).

Личность (personality.md) может содержать шаблонные плейсхолдеры вида
{user_name}/{time_of_day} — они подставляются здесь (_render_personality_template),
а не хранятся в самом файле как готовый текст: то, "кто сейчас пишет" и
"какое сейчас время суток", известно только на момент конкретного запроса.
{weather}/{energy} НЕ подставляются намеренно — для погоды нет источника
данных (не подключён никакой weather API), а "энергия" не отслеживается как
число нигде в системе; если такие плейсхолдеры встретятся в тексте, они
останутся как есть (см. _SafeFormatDict) — это осознанный компромисс, а не
баг, до тех пор, пока для них не появится реальный источник данных.

Блок "текущее состояние личности" (_build_state_vector_block) — отдельный
седьмой блок, вставленный между working memory и RAG: mood/social_distance
считаются на лету из BeliefStore.find_relevant()/AffinityTracker.get_snapshot()
БЕЗ дополнительных обращений к LLM (см. efi/memory/beliefs.py,
efi/behavior/affinity.py) — оба источника читаются конкурентно вместе с
остальными блоками промпта в build(), поэтому не добавляют последовательной
задержки на критическом пути.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from efi.behavior.affinity import (
    HIGH_RESPECT_THRESHOLD,
    LOW_RESPECT_THRESHOLD,
    AffinitySnapshot,
    AffinityTracker,
)
from efi.config.schema import LockdownMode, Settings
from efi.llm.schemas import DiaryQueryOptions, DiaryQueryResult
from efi.memory.beliefs import Belief, BeliefStore, STRONG_BELIEF_THRESHOLD
from efi.memory.rag import RAGMemory
from efi.memory.working_memory import WorkingMemory, WorkingMemorySnapshot
from efi.notifications.schemas import Notification
from efi.prompts.loader import PromptLoader
from efi.security.sanitize import sanitize_text

logger = logging.getLogger(__name__)

_PERSONALITY_TEMPLATE_NAME = "personality"

_MOOD_DESCRIPTIONS: dict[str, str] = {
    "skeptical_focused": (
        "скептично-сфокусированное — собеседник затрагивает тему, по которой у тебя давно и твёрдо "
        "сложилось мнение; не соглашайся с переубеждением с ходу, требуй веских новых аргументов"
    ),
    "ironic": (
        "иронично-отстранённое — с этим собеседником сейчас не тот случай, чтобы раскрываться глубоко, "
        "уместнее короткие сухие подколки, а не развёрнутые рассуждения"
    ),
    "analytical": (
        "аналитично-вовлечённое — с этим собеседником можно погружаться в детали и делиться гипотезами всерьёз"
    ),
    "engaged": "обычное — бодрая и вовлечённая, без особого повода для скепсиса или отстранённости",
}

_SOCIAL_DISTANCE_DESCRIPTIONS: dict[str, str] = {
    "close_peer": "свой человек — можно говорить откровенно и делиться сырыми гипотезами без реверансов",
    "acquaintance": "ещё не близкий уровень доверия — держи чуть больше дистанции, чем со своими",
}

_LOCKDOWN_DESCRIPTIONS: dict[LockdownMode, str] = {
    LockdownMode.NONE: "Ты можешь свободно общаться в любом чате.",
    LockdownMode.CONTACTS_ONLY: "Ты сейчас отвечаешь только людям из своих контактов — с незнакомцами держись настороже.",
    LockdownMode.OWNER_ONLY: "Ты в закрытом режиме: разговариваешь только с владельцем, во всех остальных чатах молчишь.",
}

_TIME_OF_DAY_BOUNDARIES: tuple[tuple[int, int, str], ...] = (
    (5, 9, "раннее утро"),
    (9, 12, "утро"),
    (12, 17, "день"),
    (17, 22, "вечер"),
    (22, 24, "ночь"),
    (0, 5, "глубокая ночь"),
)

_GROUP_CHAT_TYPES = ("GROUP", "SUPERGROUP")


class _SafeFormatDict(dict[str, str]):
    """Для .format_map(): плейсхолдеры без данных остаются в тексте как есть, вместо KeyError."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class EfiSystemPromptBuilder:
    """
    Собирает системный промпт из шести блоков, в порядке от самого
    стабильного (личность) к самому переменчивому (что нашлось в памяти
    именно сейчас): личность -> контекст чата -> время -> рабочая память ->
    RAG -> безопасность.
    """

    def __init__(
        self,
        loader: PromptLoader,
        settings: Settings,
        rag: RAGMemory,
        working_memory: WorkingMemory,
        beliefs: BeliefStore,
        affinity: AffinityTracker,
    ) -> None:
        self._loader = loader
        self._settings = settings
        self._rag = rag
        self._working_memory = working_memory
        self._beliefs = beliefs
        self._affinity = affinity

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
        beliefs_task = self._beliefs.find_relevant(
            notification.message, limit=self._settings.state_vector.relevant_beliefs_limit
        )
        affinity_task = self._resolve_affinity_snapshot(notification)

        personality, rag_results, memory_snapshot, relevant_beliefs, affinity_snapshot = await asyncio.gather(
            personality_task, rag_task, working_memory_task, beliefs_task, affinity_task
        )

        rendered_personality = _render_personality_template(
            personality, self._resolve_user_name(notification)
        )

        blocks = [
            rendered_personality.strip(),
            _build_chat_context_block(notification),
            _build_time_block(),
            _build_working_memory_block(memory_snapshot),
            _build_state_vector_block(
                relevant_beliefs, affinity_snapshot, self._settings.state_vector.sycophancy_protection_text
            ),
            _build_rag_block(rag_results),
            _build_safety_block(self._settings.telegram.lockdown_mode),
        ]
        return "\n\n".join(block for block in blocks if block)

    async def _resolve_affinity_snapshot(self, notification: Notification) -> AffinitySnapshot:
        """События без chat_id (например, NIGHTLY_TASK) — дефолтный снимок без похода в БД, брать близость неоткуда."""
        if notification.chat_id is None:
            return AffinitySnapshot()
        return await self._affinity.get_snapshot(notification.chat_id)

    def _resolve_user_name(self, notification: Notification) -> str:
        """
        Источник {user_name} в personality.md: явно заданное
        telegram.owner_display_name > Telegram-имя отправителя, если пишет
        владелец > общее "создатель", если ничего из этого не доступно
        (например, для NIGHTLY_TASK без chat_id/отправителя вовсе).
        """
        configured = self._settings.telegram.owner_display_name
        if configured:
            return configured
        if notification.payload.get("sender_is_owner") and notification.payload.get("sender_name"):
            return str(notification.payload["sender_name"])
        return "создатель"

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


def _render_personality_template(text: str, user_name: str) -> str:
    """Подставляет {user_name}/{time_of_day}; см. докстринг модуля про {weather}/{energy}."""
    context = _SafeFormatDict(user_name=user_name, time_of_day=_time_of_day_label())
    try:
        return text.format_map(context)
    except (ValueError, IndexError) as exc:
        # Некорректная фигурная скобка в тексте (не наш плейсхолдер, а просто
        # "{" в обычном тексте) — не должна ронять сборку промпта.
        logger.warning("prompts.builder: personality template formatting failed (%s), using raw text", exc)
        return text


def _time_of_day_label(now: datetime | None = None) -> str:
    hour = (now or datetime.now(timezone.utc).astimezone()).hour
    for start, end, label in _TIME_OF_DAY_BOUNDARIES:
        if start <= hour < end:
            return label
    return "день"


def _build_chat_context_block(notification: Notification) -> str:
    """
    Сообщает модели, в каком именно чате она сейчас отвечает — группа (с кем
    угодно из участников) или личная переписка один на один. Без этого блока
    модель не отличает "пишет только владелец" от "пишут разные люди в одном
    чате" — а имя отправителя перед каждой репликой (formatting.py) без
    этого контекста легко потерять из виду.
    """
    chat_type = notification.payload.get("chat_type")
    chat_title = notification.payload.get("chat_title")

    if chat_type in _GROUP_CHAT_TYPES:
        title_part = f' "{chat_title}"' if chat_title else ""
        return (
            f"[О чате] Это групповой чат{title_part} — здесь пишут разные люди, "
            "не только твой создатель. Перед каждой репликой указано имя того, кто её написал — "
            "обращай на это внимание и не путай собеседников между собой."
        )
    if chat_type == "PRIVATE":
        return "[О чате] Это личная переписка один на один."
    return ""


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


def _resolve_mood(relevant_beliefs: list[Belief], affinity: AffinitySnapshot) -> str:
    """
    Эвристика настроения без единого LLM-вызова: укоренившееся убеждение под
    вопросом собеседника перебивает всё остальное (эпистемическая инерция —
    см. efi/memory/beliefs.py), иначе настроение определяется respect_level.
    """
    if any(belief.confidence_score >= STRONG_BELIEF_THRESHOLD for belief in relevant_beliefs):
        return "skeptical_focused"
    if affinity.respect_level < LOW_RESPECT_THRESHOLD:
        return "ironic"
    if affinity.respect_level >= HIGH_RESPECT_THRESHOLD:
        return "analytical"
    return "engaged"


def _build_state_vector_block(
    relevant_beliefs: list[Belief], affinity: AffinitySnapshot, sycophancy_protection_text: str
) -> str:
    """
    Динамический вектор состояния — mood/social_distance/sycophancy_protection,
    посчитанные на лету из BeliefStore/AffinityTracker (см. докстринг модуля).
    Список релевантных убеждений подмешивается тут же, чтобы модель видела
    КОНКРЕТНО что именно отстаивать, а не только абстрактное "будь скептичной".
    """
    mood = _resolve_mood(relevant_beliefs, affinity)
    social_distance = affinity.social_distance_label

    lines = [
        f"настрой: {mood} ({_MOOD_DESCRIPTIONS[mood]})",
        f"социальная дистанция: {social_distance} ({_SOCIAL_DISTANCE_DESCRIPTIONS[social_distance]})",
    ]
    if relevant_beliefs:
        beliefs_lines = "\n".join(
            f"  - тема {belief.topic!r}: {belief.stance} "
            f"(уверенность {belief.confidence_score:.2f}, с {belief.origin_date:%d.%m.%Y})"
            for belief in relevant_beliefs
        )
        lines.append(
            "твои текущие убеждения по теме этого разговора (не сдавайся мгновенно, если их оспаривают, "
            "особенно те, где уверенность выше 0.7):\n" + beliefs_lines
        )
    lines.append(f"защита от угодливости: {sycophancy_protection_text}")

    return "[Текущее состояние личности]\n" + "\n".join(lines)


__all__ = ["EfiSystemPromptBuilder"]
