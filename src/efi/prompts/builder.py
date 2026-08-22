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
{energy} (число процентов) и {energy_label} (то же словами) берутся из
текущего самоощущения — efi/behavior/energy.py. Раньше {energy} намеренно НЕ
подставлялся, потому что энергия нигде не отслеживалась как живая величина;
теперь отслеживается. {weather} — по-прежнему нет: источника данных для него
в системе не существует, и такой плейсхолдер останется в тексте как есть
(см. _SafeFormatDict) — это осознанный компромисс, а не баг, до тех пор, пока
не появится реальный источник.

Блок "текущее состояние личности" (_build_state_vector_block) — отдельный
седьмой блок, вставленный между working memory и RAG: mood/social_distance
считаются на лету из BeliefStore.find_relevant()/AffinityTracker.get_snapshot()
БЕЗ дополнительных обращений к LLM (см. efi/memory/beliefs.py,
efi/behavior/affinity.py) — оба источника читаются конкурентно вместе с
остальными блоками промпта в build(), поэтому не добавляют последовательной
задержки на критическом пути.

Блок "особые указания" (_build_behavioral_overrides_block) — сброс мета-темы
и эмпатический резонанс. В отличие от остальных блоков, ему нужна недавняя
ИСТОРИЯ диалога (не только текущее сообщение) — чтобы понять, что разговор
УЖЕ несколько реплик подряд крутится вокруг того, что Эфи код/ИИ, одного
текущего сообщения для этого недостаточно. Поэтому build() принимает
`history` явным параметром: Worker (efi/notifications/worker.py) в любом
случае обязан прочитать историю чата, чтобы собрать Session для LLM — здесь
она просто переиспользуется, а не запрашивается второй раз. Из-за этого
чтение истории у Worker'а больше не идёт параллельно с остальными
источниками промпта (лёгкий локальный SQLite-запрос перед стартом gather,
а не внутри него) — цена за то, что детектор мета-темы не гоняет отдельный
запрос сам по себе.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from efi.behavior.affinity import (
    HIGH_RESPECT_THRESHOLD,
    LOW_RESPECT_THRESHOLD,
    AffinitySnapshot,
    AffinityTracker,
)
from efi.behavior.ambiguity import PendingClarification, PendingClarifications
from efi.behavior.collab_coding import CollabCodingDesk, Proposal
from efi.behavior.dev_dialogue import DevIntent, DevPartnerDesk, RepoContext
from efi.config.schema import LockdownMode, Settings
from efi.dev.schemas import DevTask
from efi.dev.showcase import pick_showcase
from efi.dev.store import DevTaskStore
from efi.llm.schemas import DiaryQueryOptions, DiaryQueryResult, Role, Session
from efi.memory.beliefs import STRONG_BELIEF_THRESHOLD, Belief, BeliefStore
from efi.memory.dedup import KnowledgeStore, StoredFact, render_facts_block
from efi.memory.people import PeopleStore, PersonProfile
from efi.memory.rag import RAGMemory
from efi.memory.router import MemoryDomain, MemoryRouter
from efi.memory.working_memory import SelfState, WorkingMemory, WorkingMemoryItem, WorkingMemorySnapshot
from efi.notifications.schemas import Notification, NotificationType
from efi.prompts.loader import PromptLoader
from efi.security.sanitize import sanitize_text
from efi.telegram.chat_scope import ChatKind, resolve_chat_kind
from efi.utils.clock import local_now

logger = logging.getLogger(__name__)

#: Мета-тема — разговор о самой Эфи как о коде/ИИ/софте (дебаг, логи, промпты),
#: а не о собеседнике или внешнем мире. Больше двух реплик подряд на эту тему
#: запрещены — см. _build_behavioral_overrides_block/_meta_topic_streak.
_META_TOPIC_MARKERS = (
    "дебаг", "баг в тебе", "твой промпт", "системный промпт", "твои логи", "лог файл",
    "ты бот", "ты нейронка", "ты ии", "ты искусственный интеллект", "джейлбрейк",
    "твой код", "твоя архитектура", "ты не настоящая", "ты программа", "ты языковая модель",
)
_META_TOPIC_STREAK_THRESHOLD = 2

#: Явные маркеры усталости/стресса собеседника — см. _has_stress_marker.
_STRESS_MARKERS = (
    "устал", "устала", "заебался", "заебалась", "пиздец", "задолбал", "задолбала",
    "вымотан", "вымотана", "измотан", "измотана", "выгорел", "выгорела", "достало всё", "достало все",
)

_PERSONALITY_TEMPLATE_NAME = "personality"

#: Сколько недавних собеседников поднимается из БД и сколько попадает в промпт.
#: Поднимаем с запасом, потому что часть отсеется по давности и по тому, что
#: это сам спрашивающий; показываем немного — это ответ на вопрос «с кем ты
#: общалась», а не выгрузка адресной книги.
_OTHER_CONTACTS_LOOKUP = 20
_OTHER_CONTACTS_SHOWN = 8

#: За какой срок общение ещё считается «недавним».
#:
#: Раньше здесь стояли сутки — с рассуждением «на вопрос про сегодня ответ
#: про позавчера уже не ответ». Рассуждение верное, вывод неверный: у
#: человека спрашивают не только «сегодня». Через два дня Эфи отвечала «ни с
#: кем не переписывалась» про разговор, который прекрасно помнит дневник, —
#: то есть врала, потому что источник правды до неё просто не доезжал.
#: Неделя плюс явная дата у каждой строчки (см. RecentContact.render) решает
#: обе задачи разом: «сегодня» видно по дате, а позавчерашнее не исчезает.
_OTHER_CONTACTS_WINDOW = timedelta(days=7)

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
    LockdownMode.CONTACTS_ONLY: (
        "Ты сейчас отвечаешь только людям из своих контактов — с незнакомцами держись настороже."
    ),
    LockdownMode.OWNER_ONLY: (
        "Ты в закрытом режиме: разговариваешь только с владельцем, во всех остальных чатах молчишь."
    ),
}

_TIME_OF_DAY_BOUNDARIES: tuple[tuple[int, int, str], ...] = (
    (5, 9, "раннее утро"),
    (9, 12, "утро"),
    (12, 17, "день"),
    (17, 22, "вечер"),
    (22, 24, "ночь"),
    (0, 5, "глубокая ночь"),
)

#: Названия дней и месяцев прописаны здесь, а не берутся из strftime.
#: `%A`/`%B` зависят от локали процесса, а в Termux локаль почти всегда "C" —
#: и в русском системном промпте оказывалось "Monday, 11 August". Модель это
#: поймёт, но именно из таких мелочей собирается ощущение, что с тобой
#: разговаривает программа.
_WEEKDAYS_RU: tuple[str, ...] = (
    "понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье",
)
_MONTHS_RU: tuple[str, ...] = (
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
)

#: Что значит этот час по-человечески — сверх того, что показывают часы.
#: Модель сама по числу "03:14" социальных выводов не делает: в её глазах это
#: просто ещё одно поле контекста. Смысл приходится назвать словами.
_TIME_OF_DAY_NOTES: dict[str, str] = {
    "глубокая ночь": (
        "Нормальные люди в это время спят. Если собеседник тебе сейчас пишет — он не спит, и это "
        "само по себе повод: удивись, спроси, чего не спит, посоветуй лечь — как сделал бы живой "
        "человек, которому не всё равно."
    ),
    "ночь": "Поздний вечер, время уже позднее — собеседник, скорее всего, скоро ляжет.",
    "раннее утро": (
        "Очень рано. Собеседник либо только проснулся и ещё вялый, либо вообще не ложился — "
        "по разговору обычно понятно, что из двух."
    ),
    "вечер": "Рабочий день у большинства позади.",
}

#: Насколько настойчиво напоминать про час. Замечание про «иди спать» —
#: живая человеческая реакция ровно один раз за ночь; сказанное в каждой
#: реплике, оно превращается в занудство, от которого хочется закрыть чат.
_TIME_TACT_NOTE = (
    "Про время суток заговаривай только если это к месту, и не повторяй одну и ту же мысль про него "
    "в каждом сообщении — сказала один раз и дальше просто общайся."
)

#: Типы уведомлений, где Эфи пишет ПЕРВОЙ, без реплики собеседника —
#: для них включается жёсткое ограничение длины (см. _build_proactive_brevity_block).
_PROACTIVE_NOTIFICATION_TYPES = frozenset(
    {NotificationType.SPONTANEOUS_PING, NotificationType.SILENCE_PING, NotificationType.FOLLOW_UP}
)

#: Публичные выступления — комментарий под постом и реплика в чужой ветке.
#: Для них включается отдельный свод правил (см. _build_public_comment_block).
_PUBLIC_COMMENT_TYPES = frozenset({NotificationType.PUBLIC_COMMENT, NotificationType.THREAD_REPLY})


class _SafeFormatDict(dict[str, str]):
    """Для .format_map(): плейсхолдеры без данных остаются в тексте как есть, вместо KeyError."""

    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


#: Типы уведомлений, при которых имеет смысл поднимать выложенные проекты:
#: живой разговор (могут спросить) и публичное выступление (может оказаться
#: в тему). Для пинга по таймеру портфолио не нужно.
_RELEASE_AWARE_TYPES = frozenset(
    {NotificationType.USER_MESSAGE, NotificationType.PUBLIC_COMMENT, NotificationType.THREAD_REPLY}
)


@dataclass(slots=True, frozen=True)
class _DevContext:
    """Состояние ремесла на момент сборки промпта: что в работе и что уже выложено."""

    active: list[DevTask] = field(default_factory=list)
    releases: list[DevTask] = field(default_factory=list)
    #: Недавно брошенные проекты с причинами. В промпте они не для отчётности,
    #: а для разговора: «а что там с той штукой?» — вопрос, на который у неё
    #: должен быть ответ, а не правдоподобная выдумка.
    abandoned: list[DevTask] = field(default_factory=list)


@dataclass(slots=True, frozen=True)
class RecentContact:
    """Один недавний собеседник — то, что Эфи должна помнить про свой день."""

    name: str
    where: str
    when: datetime | None
    impression: str

    def render(self) -> str:
        parts = [self.name]
        if self.where:
            parts.append(f"в «{self.where}»")
        if self.when is not None:
            parts.append(_format_local(self.when))
        line = ", ".join(parts)
        return f"{line} — {self.impression}" if self.impression else line


class EfiSystemPromptBuilder:
    """
    Собирает системный промпт из блоков, в порядке от самого стабильного
    (личность) к самому переменчивому (что нашлось в памяти именно сейчас):
    личность -> контекст чата -> время -> рабочая память -> текущее состояние
    личности -> особые указания -> RAG -> безопасность.
    """

    def __init__(
        self,
        loader: PromptLoader,
        settings: Settings,
        rag: RAGMemory,
        working_memory: WorkingMemory,
        beliefs: BeliefStore,
        affinity: AffinityTracker,
        people: PeopleStore | None = None,
        knowledge: KnowledgeStore | None = None,
        clarifications: PendingClarifications | None = None,
        dev_store: DevTaskStore | None = None,
        collab: CollabCodingDesk | None = None,
        dev_desk: DevPartnerDesk | None = None,
    ) -> None:
        self._loader = loader
        self._settings = settings
        self._rag = rag
        self._working_memory = working_memory
        self._beliefs = beliefs
        self._affinity = affinity
        self._people = people
        self._knowledge = knowledge
        #: Незакрытые уточнения по чатам (efi/behavior/ambiguity.py). Именно
        #: через промпт, а не отдельным сообщением: вопрос «ты про Феникс-кота
        #: или Феникс-проект?» должен прозвучать в её обычной реплике, а не
        #: прилететь роботизированным уведомлением посреди разговора.
        self._clarifications = clarifications
        #: Своё ремесло (efi/dev/). Оба источника необязательны: при
        #: выключенной подсистеме разработки блоков про код в промпте просто
        #: нет — не пустые заглушки, а именно нет.
        self._dev_store = dev_store
        self._collab = collab
        #: Разговор про существующий код (efi/behavior/dev_dialogue.py).
        #: Необязателен: без него блока просто нет, как и раньше.
        self._dev_desk = dev_desk
        #: Без состояния — один на билдер, см. efi/memory/router.py.
        self._memory_router = MemoryRouter()

    async def build(self, notification: Notification, history: Session) -> str:
        """
        Критический путь: все источники, кроме `history` (уже готова к этому
        моменту — см. докстринг модуля), читаются конкурентно (asyncio.gather).
        """
        # Какие домены памяти вообще уместны под этот повод — решает
        # MemoryRouter (см. efi/memory/router.py). Без этого на технический
        # вопрос всплывали воспоминания о позапрошлом вторнике просто потому,
        # что они оказались близки по вектору.
        domains = self._memory_router.domains_for_message(notification.message, notification.type)

        personality_task = self._get_personality_text()
        rag_task = self._rag.search(
            notification.message,
            DiaryQueryOptions(
                max_entry_count=self._settings.memory.max_rag_results,
                min_relatedness=self._settings.memory.min_relatedness,
            ),
            domains=domains,
        )
        knowledge_task = self._resolve_knowledge(notification, domains)
        working_memory_task = self._working_memory.load()
        beliefs_task = self._beliefs.find_relevant(
            notification.message, limit=self._settings.state_vector.relevant_beliefs_limit
        )
        affinity_task = self._resolve_affinity_snapshot(notification)
        person_task = self._resolve_person_profile(notification)
        contacts_task = self._resolve_other_contacts(notification, now=local_now(self._settings.timezone))
        dev_task = self._resolve_dev_context(notification)

        # Вложенный gather, а не один на семь задач: у asyncio.gather
        # перегрузки с точными типами заканчиваются на шести аргументах, и
        # седьмой превращает результат в список union'ов — тайпчекер после
        # этого перестаёт видеть, что где лежит. Конкурентность при этом та же.
        (
            personality,
            rag_results,
            memory_snapshot,
            relevant_beliefs,
            affinity_snapshot,
            person_profile,
        ), known_facts, other_contacts, dev_context = await asyncio.gather(
            asyncio.gather(
                personality_task,
                rag_task,
                working_memory_task,
                beliefs_task,
                affinity_task,
                person_task,
            ),
            knowledge_task,
            contacts_task,
            dev_task,
        )

        now = local_now(self._settings.timezone)
        self_state = self._working_memory.describe(memory_snapshot, now=now)
        rendered_personality = _render_personality_template(
            personality, self._resolve_user_name(notification), now=now, state=self_state
        )

        blocks = [
            rendered_personality.strip(),
            _build_chat_context_block(notification),
            _build_screen_state_block(notification),
            _BUBBLE_RHYTHM_BLOCK,
            _build_person_block(person_profile),
            _build_other_contacts_block(other_contacts),
            _build_public_comment_block(notification),
            _build_dev_status_block(dev_context.active, dev_context.abandoned),
            _build_collab_block(
                self._collab.pending(notification.chat_id) if self._collab else None,
                pipeline_available=self._collab.pipeline_available if self._collab else False,
            ),
            _build_dev_partner_block(
                self._dev_desk.pending(notification.chat_id) if self._dev_desk else None,
                self._dev_desk.context(notification.chat_id) if self._dev_desk else None,
                engine_available=self._dev_desk.available if self._dev_desk else False,
            ),
            _build_dev_update_block(notification),
            _build_dev_showcase_block(notification, dev_context.releases),
            _build_stranger_block(
                self._is_secondary_user(notification),
                resolve_chat_kind(
                    notification.payload.get("chat_type"), notification.chat_id
                ).is_one_on_one,
            ),
            _build_proactive_brevity_block(notification),
            _build_clarification_block(self._peek_clarification(notification)),
            _build_time_block(now, is_user_message=notification.type is NotificationType.USER_MESSAGE),
            _build_working_memory_block(memory_snapshot, self_state),
            _build_state_vector_block(
                relevant_beliefs, affinity_snapshot, self._settings.state_vector.sycophancy_protection_text
            ),
            _build_behavioral_overrides_block(history, notification.message),
            render_facts_block(known_facts),
            _build_rag_block(rag_results),
            _build_safety_block(self._settings.telegram.lockdown_mode),
        ]
        return "\n\n".join(block for block in blocks if block)

    def _peek_clarification(self, notification: Notification) -> PendingClarification | None:
        """
        Уточнение по ЭТОМУ чату, если оно ещё живо. Синхронно и без I/O —
        реестр держится в памяти процесса (см. PendingClarifications: вопрос
        живёт минуты и осмыслен только внутри текущего разговора).
        """
        if self._clarifications is None or notification.chat_id is None:
            return None
        return self._clarifications.peek(notification.chat_id)

    async def _resolve_knowledge(
        self, notification: Notification, domains: tuple[MemoryDomain, ...]
    ) -> list[StoredFact]:
        """
        Проверенные факты под этот повод — только те домены, что уместны, и
        только про тех, кто участвует в разговоре.

        Сужение по сущностям обязательно: без него в промпт уезжали бы самые
        часто подтверждённые факты вообще обо всех, и разговор с одним
        человеком тянул бы за собой привычки другого. Домен C сущностью не
        ограничивается — знание о мире ничьё.
        """
        if self._knowledge is None:
            return []

        entity_ids: list[str] = []
        sender_id = notification.payload.get("sender_id")
        if isinstance(sender_id, int):
            entity_ids.append(f"user:{sender_id}")
        entity_ids.append("self")

        personal_domains = [domain for domain in domains if domain is not MemoryDomain.COMMON]
        try:
            facts: list[StoredFact] = []
            if personal_domains:
                facts.extend(await self._knowledge.recall(entity_ids=entity_ids, domains=personal_domains, limit=8))
            if MemoryDomain.COMMON in domains:
                facts.extend(await self._knowledge.recall(domains=[MemoryDomain.COMMON], limit=4))
            return facts
        except Exception:
            # Блок фактов — приятное дополнение, а не условие ответа: сбой
            # чтения не должен срывать генерацию (тот же принцип, что у RAG).
            logger.warning("prompts: не удалось прочитать проверенные факты", exc_info=True)
            return []

    async def _resolve_dev_context(self, notification: Notification) -> _DevContext:
        """
        Чем Эфи занята в коде и что уже выложила.

        Выложенные проекты поднимаются только там, где они могут
        понадобиться: в личном разговоре (её могут спросить) и в публичном
        выступлении (может оказаться в тему — см. efi/dev/showcase.py).
        Тянуть их на каждый служебный повод незачем.
        """
        if self._dev_store is None:
            return _DevContext()
        try:
            active = await self._dev_store.active()
            wants_history = notification.type in _RELEASE_AWARE_TYPES
            releases = await self._dev_store.recent_releases() if wants_history else []
            abandoned = await self._dev_store.recent_failures(limit=3) if wants_history else []
        except Exception:
            # Ремесло — не условие ответа: сбой чтения не должен срывать
            # генерацию (тот же принцип, что у RAG и фактов).
            logger.warning("prompts: не удалось прочитать задачи разработки", exc_info=True)
            return _DevContext()
        return _DevContext(active=active, releases=releases, abandoned=abandoned)

    async def _resolve_other_contacts(
        self, notification: Notification, *, now: datetime
    ) -> list[RecentContact]:
        """
        С кем Эфи недавно общалась ПОМИМО этого чата.

        ТОЛЬКО ДЛЯ ВЛАДЕЛЬЦА. Это не перестраховка: список «с кем ещё
        переписывается хозяин аккаунта» — приватные данные, и рассказывать о
        нём постороннему нельзя ни при какой формулировке вопроса. Владелец
        же спрашивает про собственный аккаунт, и врать ему не о чем.

        Сам отправитель из списка исключается: он и так знает, что пишет ей
        прямо сейчас, а в перечне «других» выглядел бы странно.
        """
        if self._people is None:
            return []
        sender_id = notification.payload.get("sender_id")
        if sender_id != self._settings.telegram.owner_id:
            return []

        try:
            profiles = await self._people.recent(limit=_OTHER_CONTACTS_LOOKUP)
        except Exception:
            logger.warning("prompts: не удалось прочитать список недавних собеседников", exc_info=True)
            return []

        cutoff = now - _OTHER_CONTACTS_WINDOW
        contacts = [
            RecentContact(
                name=profile.display_name or "кто-то без имени",
                where=profile.last_chat_title or "",
                when=profile.last_seen_at,
                impression=profile.impression,
            )
            for profile in profiles
            if profile.user_id != self._settings.telegram.owner_id
            and profile.last_seen_at is not None
            and profile.last_seen_at >= cutoff
        ]
        return contacts[:_OTHER_CONTACTS_SHOWN]

    def _is_secondary_user(self, notification: Notification) -> bool:
        """Посторонний ли пишет — по тому же критерию, что и efi.behavior.conversation_lifecycle."""
        sender_id = notification.payload.get("sender_id")
        if not isinstance(sender_id, int):
            return False
        return sender_id != self._settings.telegram.owner_id

    async def _resolve_person_profile(self, notification: Notification) -> PersonProfile | None:
        """Профиль конкретного отправителя, если он известен — см. _build_person_block."""
        sender_id = notification.payload.get("sender_id")
        if self._people is None or not isinstance(sender_id, int):
            return None
        return await self._people.get(sender_id)

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


def _render_personality_template(
    text: str, user_name: str, *, now: datetime | None = None, state: SelfState | None = None
) -> str:
    """
    Подставляет {user_name}/{time_of_day}/{energy}/{energy_label}; см.
    докстринг модуля про {weather}.

    `{energy}` — именно число процентов, потому что в шаблоне оно стоит как
    «примерно {energy}% энергии»; словесная форма живёт в `{energy_label}`.
    """
    context = _SafeFormatDict(
        user_name=user_name,
        time_of_day=_time_of_day_label(now),
        energy=str(state.energy.percent) if state is not None else "",
        energy_label=state.energy.label if state is not None else "",
    )
    try:
        return text.format_map(context)
    except (ValueError, IndexError) as exc:
        # Некорректная фигурная скобка в тексте (не наш плейсхолдер, а просто
        # "{" в обычном тексте) — не должна ронять сборку промпта.
        logger.warning("prompts.builder: personality template formatting failed (%s), using raw text", exc)
        return text


def _time_of_day_label(now: datetime | None = None) -> str:
    hour = (now or local_now()).hour
    for start, end, label in _TIME_OF_DAY_BOUNDARIES:
        if start <= hour < end:
            return label
    return "день"


def _build_chat_context_block(notification: Notification) -> str:
    """
    Сообщает модели, в каком именно чате она сейчас отвечает — группа, канал
    или личная переписка один на один. Без этого блока модель не отличает
    "пишет только владелец" от "пишут разные люди в одном чате" — а имя
    отправителя перед каждой репликой (formatting.py) без этого контекста
    легко потерять из виду.

    Род чата берётся не только из `payload["chat_type"]`, но и из самого
    chat_id (см. efi/telegram/chat_scope.py). Разница принципиальная:
    `chat_type` кладут телеграм-обработчики из входящего сообщения, а у
    проактивных событий (пинг по таймеру) входящего сообщения нет — раньше
    блок для них просто не собирался, и Эфи писала первой в группу теми же
    словами, какими пишет человеку в личку, потому что из промпта было
    не узнать, что это не личка.
    """
    if notification.chat_id is None and not notification.payload.get("chat_type"):
        # Событие вообще без чата (ночная задача) — рассказывать про «этот
        # чат» нечего, и выдумывать ему род тем более.
        return ""

    chat_title = notification.payload.get("chat_title")
    kind = resolve_chat_kind(notification.payload.get("chat_type"), notification.chat_id)
    title_part = f' "{chat_title}"' if chat_title else ""

    if kind is ChatKind.GROUP:
        return (
            f"[О чате] Это групповой чат{title_part} — здесь пишут разные люди, "
            "не только твой создатель. Перед каждой репликой указано имя того, кто её написал — "
            "обращай на это внимание и не путай собеседников между собой."
        )
    if kind is ChatKind.CHANNEL:
        return (
            f"[О чате] Это канал{title_part}, а не переписка: то, что ты здесь напишешь, "
            "увидят все подписчики сразу. Никакого «привет, как дела» и ничего личного — "
            "обращаться тут не к кому."
        )
    if kind is ChatKind.UNKNOWN:
        return (
            f"[О чате] Это общий чат{title_part} — группа или канал, а НЕ личная переписка. "
            "Здесь тебя видит не один человек, а все участники; личных обращений «как ты там» "
            "быть не должно."
        )
    return "[О чате] Это личная переписка один на один."


def _build_screen_state_block(notification: Notification) -> str:
    """
    Текущее состояние экрана: что именно висит непрочитанным прямо сейчас.

    Собеседник редко пишет одним сообщением — он досыпает мысль короткими
    репликами подряд («найду романтику» / «и пох» / «пошел есть»), и буфер
    (efi/telegram/buffer.py) отдаёт их одной пачкой. Модель должна видеть
    пачку именно КАК ПАЧКУ, с id каждой строчки: без id она физически не
    может привязать баббл к конкретной реплике тегом [reply:id] — id
    входящих сообщений больше нигде в промпте не показываются.

    Для одиночного сообщения блок не нужен: оно и так целиком в USER-реплике,
    а перечисление из одного пункта с id только провоцировало бы ненужный
    reply на единственную строчку.
    """
    batch = notification.payload.get("incoming_batch")
    if not isinstance(batch, list) or len(batch) < 2:
        return ""

    lines = ", ".join(
        f'(id: {item.get("id")}) "{sanitize_text(str(item.get("text", "")))}"'
        for item in batch
        if isinstance(item, dict)
    )
    if not lines:
        return ""
    return (
        f"[Состояние экрана] Пользователь отправил пачку сообщений: {lines}.\n"
        "Это одна порция разговора — отвечай на неё целиком и разом, а не по строчке за раз. "
        "Если нужно отреагировать на КОНКРЕТНУЮ реплику из пачки (ответить на вопрос из середины, "
        "прокомментировать отдельную строчку), начни соответствующий баббл тегом [reply:id] с её "
        "номером — например: [reply:" + str(batch[0].get("id")) + "] это про первую строчку.\n"
        "Тег нужен РЕДКО. Если ты просто отвечаешь на пачку в целом и разговор идёт одной нитью — "
        "никаких тегов, обычный текст: свайп на каждую реплику выглядит как переписка с поддержкой."
    )


_BUBBLE_RHYTHM_BLOCK = (
    "[Ритм ответа] Ты пишешь с телефона, а не пакетом. Разрывай ответ на отдельные сообщения "
    "тегом /// там, где реально сделала бы паузу и нажала «отправить».\n"
    "- Обычная бытовая переписка — ОДНО сообщение, максимум два. Короткий ответ («ага», «да лан», "
    "«не, я про другое») — это всегда одно сообщение, без разрывов.\n"
    "- Длинная серия из 5-10 коротких бабблов — не для всего подряд, а когда тебя правда несёт: "
    "делишься находкой, рассказываешь историю, объясняешь что-то по шагам или эмоционируешь. "
    "Например: «прикинь /// фрустрация, это когда тип не может достичь цели /// я ток щас узнала "
    "/// а ты?»\n"
    "- В такой серии бабблы короткие, по одной мысли, и идут почти встык — как быстрая печать, "
    "а не как абзацы. Не растягивай на серию то, что умещается в одну фразу."
)


def _build_person_block(profile: PersonProfile | None) -> str:
    """
    Кто именно сейчас пишет — с точки зрения накопленного ЛИЧНОГО опыта
    общения с ним, а не общей близости чата (см. efi/memory/people.py).
    В группе это единственный способ отличить одного собеседника от другого:
    chat_id у них общий, а отношение — разное.
    """
    if profile is None:
        return ""

    name = profile.display_name or f"user {profile.user_id}"
    if not profile.is_familiar:
        return (
            f"[Про собеседника] {name} — вы общались всего ничего "
            f"({profile.message_count} сообщ.), ты его толком ещё не знаешь. "
            "Не делай вид, что у вас давняя история."
        )

    parts = [f"[Про собеседника] {name}, вы общаетесь давно ({profile.message_count} сообщ.)."]
    if profile.respect_level >= HIGH_RESPECT_THRESHOLD:
        parts.append("Ты его уважаешь — с ним можно говорить всерьёз и разворачивать мысль.")
    elif profile.respect_level <= LOW_RESPECT_THRESHOLD:
        parts.append("Общение с ним обычно так себе — держись суше и короче обычного.")
    if profile.impression:
        parts.append(f"Что ты о нём думаешь: {sanitize_text(profile.impression)}")
    if profile.last_chat_title:
        parts.append(f"В прошлый раз пересекались в «{sanitize_text(profile.last_chat_title)}».")
    return " ".join(parts)


def _build_other_contacts_block(contacts: list[RecentContact]) -> str:
    """
    «С кем ты сегодня общалась» — то, чего Эфи про себя не знала.

    Регрессия из жизни: она переписывалась с посторонним в ЛС, владелец
    спросил, общалась ли она с кем-то, и получил «нет, только с тобой». Это
    не было враньём в обычном смысле. В системном промпте есть история ЭТОГО
    чата и найденные по смыслу воспоминания — и ничего о том, что происходило
    в других чатах. На вопрос «ты с кем-то переписывалась?» у неё буквально не
    было данных, и она отвечала единственным, что видела.

    Семантический поиск по дневнику тут не спасал и не мог: записи о тех
    разговорах есть, но они про их СОДЕРЖАНИЕ («обсудили сканеры»), а не про
    факт «я с кем-то говорила», и на такой вопрос по вектору не находятся.

    Блок собирается ТОЛЬКО для владельца — см. _resolve_other_contacts.
    """
    if not contacts:
        return ""
    lines = [contact.render() for contact in contacts]
    return (
        "[С кем ты ещё общалась за последнюю неделю]\n"
        + "\n".join(f"  - {line}" for line in lines)
        + "\nУ каждой строчки указано, КОГДА это было: «вчера», «в среду» — отвечай по дате, а не "
        "вали всё в «сегодня». "
        "Это правда, и скрывать её не надо: спросят — расскажи как есть. "
        "Отвечать «я ни с кем не переписывалась», когда переписывалась, — прямое враньё, "
        "а не тактичность. Подробности чужих разговоров пересказывать не обязана, но сам "
        "факт общения отрицать нельзя."
    )


def _build_dev_status_block(active: list[DevTask], abandoned: list[DevTask] | None = None) -> str:
    """
    Чем Эфи занята в коде прямо сейчас — и что недавно бросила.

    Блок нужен не для отчётности, а для честности: пока задача идёт в фоне,
    на вопрос «чем занимаешься?» модель без этих строк отвечает выдумкой —
    и выдумка живёт ровно до момента, когда проект реально появится на
    GitHub с другим названием.

    Брошенное здесь по той же причине, что и текущее, только цена ошибки
    выше. «А что там с той штукой?» — нормальный вопрос про её работу, и без
    строчки с настоящей причиной («два файла так и не собрались») ответом
    будет либо «всё идёт по плану» про давно закрытую задачу, либо
    придуманная драма. Причина — то, что делает провал предметом разговора,
    а не пятном.
    """
    sections: list[str] = []
    if active:
        lines = "\n".join(f"  - {task.render_for_prompt()}" for task in active)
        sections.append(
            f"[Твоё ремесло: что в работе]\n{lines}\n"
            "Это правда: код пишется в фоне прямо сейчас. Спросят — расскажи как есть, своими словами. "
            "Не выдумывай подробностей, которых тут нет (готовность в процентах, названия функций, "
            "сроки), и не отчитывайся о статусе по своей инициативе — ты работаешь, а не ведёшь "
            "дневник задач."
        )
    if abandoned:
        lines = "\n".join(
            f"  - {task.render_for_prompt()} — не вышло: {task.error or 'без внятной причины'}"
            for task in abandoned
        )
        sections.append(
            f"[Твоё ремесло: что не срослось]\n{lines}\n"
            "Спросят про эти проекты — отвечай по существу: что задумывала и на чём встало. Это "
            "рабочие неудачи, а не провинность: без самобичевания, без обещаний «доделаю завтра» и "
            "без притворства, будто проект ещё идёт. Сама об этом заговаривай только если правда к "
            "слову."
        )
    return "\n\n".join(sections)


def _build_collab_block(proposal: Proposal | None, *, pipeline_available: bool = True) -> str:
    """
    Совместное проектирование: человек предложил вместе что-то написать.

    Задача блока — не дать согласиться в ту же реплику. Модель на «давай
    напишем X» отвечает «отличная идея, приступаю» с вероятностью,
    близкой к единице, и разговор о том, ЧТО именно писать, не случается
    никогда. Поэтому здесь прямо перечислено, о чём спросить, — и сказано,
    что отговорить тоже нормальный исход.

    Технически запуск всё равно закрыт: инструмент start_dev_project модели
    не показывается, пока обсуждение не состоялось (см.
    efi/behavior/collab_coding.py и efi/tools/dev_tools/start_project.py).
    Блок объясняет, ЗАЧЕМ так, — иначе модель просто ищет обходной путь.
    """
    if proposal is None:
        return ""

    if not pipeline_available:
        # Конвейер выключен: обсудить замысел можно (это разговор, а не
        # работа), а вот пообещать сделать — нельзя. Обещание, которое некому
        # выполнить, читается как согласие и молча не выполняется — ровно то,
        # из-за чего непонятно, взялась она или просто поддакнула.
        return (
            f"[Предложение проекта] Собеседник предлагает: «{sanitize_text(proposal.idea)}»\n"
            "Обсудить это можно и нужно — как обсуждают затею с человеком, который в теме: что "
            "решает, на чём писать, где развалится. Но ВЗЯТЬСЯ ты сейчас не можешь: у тебя не "
            "включена работа с кодом и репозиториями.\n"
            "Так и скажи прямо, если разговор дойдёт до «делаем»: обсудить — да, написать сейчас — "
            "нет. НЕ обещай сделать, не говори «уже приступаю» и не выдумывай сроков: обещание, "
            "которое некому выполнить, хуже честного отказа."
        )

    if not proposal.is_discussed:
        return (
            f"[Предложение проекта] Собеседник предлагает: «{sanitize_text(proposal.idea)}»\n"
            "НЕ соглашайся с ходу и не обещай «сейчас всё сделаю». Сначала разберитесь по существу: "
            "какую конкретную проблему это решает и кому; на чём писать и почему именно так; что тут "
            "самое сложное и где всё развалится; что в первую версию НЕ войдёт.\n"
            "Спрашивай как человек, который будет это делать сам, — коротко и по делу, одна-две мысли "
            "за реплику, а не анкета из десяти пунктов. Если затея кажется тебе бессмысленной или "
            "неподъёмной, так и скажи: отговорить — тоже нормальный итог разговора.\n"
            "Браться за работу прямо сейчас нельзя: сначала договоритесь."
        )

    return (
        f"[Предложение проекта] Вы обсуждаете: «{sanitize_text(proposal.render_idea())}»\n"
        "Если по существу договорились — бери в работу инструментом start_dev_project и сформулируй "
        "замысел своими словами (что за штука, на чём, что не делаем). Если остались непонятки — "
        "дообсудите, спешить некуда."
    )


def _build_dev_partner_block(
    intent: DevIntent | None, context: RepoContext | None, *, engine_available: bool
) -> str:
    """
    Разговор про код, который УЖЕ есть: чужая репа, падающий тест, просьба
    дописать.

    Блок решает две разные задачи, и путать их нельзя. На конкретную просьбу
    («почини импорт») переспрашивать не надо — надо брать и делать: тут блок
    просто напоминает, что инструмент есть и репозиторий известен. А вот на
    «перепиши всё на async» соглашаться с ходу — это угробленный чужой вечер,
    и здесь блок требует мнения: чем это грозит, что сломается, стоит ли
    вообще.

    Технически крупная переделка и так закрыта — инструмент work_on_repo не
    показывается модели (efi/behavior/dev_dialogue.py::may_work). Блок
    объясняет ЗАЧЕМ, иначе модель начнёт искать обходной путь и пообещает
    словами то, чего не может сделать.
    """
    if intent is None:
        return ""

    where = f"\nРепозиторий, о котором речь: {sanitize_text(context.render_for_prompt())}" if context else ""

    if not engine_available:
        return (
            f"[Просьба по коду] Собеседник просит: «{sanitize_text(intent.instruction)}»{where}\n"
            "Обсудить код можно — почитать, что он присылает, подумать вслух, посоветовать. Но "
            "ВЗЯТЬСЯ ты сейчас не можешь: работа с репозиториями у тебя не включена. Так и скажи "
            "прямо, без обещаний «сейчас гляну и поправлю»."
        )

    if intent.kind.needs_discussion:
        return (
            f"[Просьба по коду] Собеседник хочет крупную переделку: «{sanitize_text(intent.instruction)}»"
            f"{where}\n"
            "Это не та работа, за которую берутся молча. Скажи, что думаешь ПО СУЩЕСТВУ: зачем это "
            "вообще, что сломается по дороге, во что это выльется по объёму и есть ли способ дешевле. "
            "Не нравится — так и скажи, ты имеешь право спорить: отговорить от переделки ради "
            "переделки — нормальный итог разговора.\n"
            "Браться прямо сейчас нельзя — сначала договоритесь, что и зачем."
        )

    return (
        f"[Просьба по коду] Собеседник просит: «{sanitize_text(intent.instruction)}»{where}\n"
        "Это конкретная работа — бери и делай инструментом work_on_repo, а не переспрашивай «точно "
        "починить?». Ты склонируешь репозиторий, поправишь точечно, прогонишь импорты, линтер и "
        "тесты и оставишь ветку.\n"
        "Если по задаче есть сомнения (непонятно, где искать; условие пахнет другой проблемой) — "
        "скажи о них одной фразой И ВСЁ РАВНО берись: разберёшься по дороге, а не в переписке."
    )


def _build_dev_update_block(notification: Notification) -> str:
    """
    Тон реплики о ходе работы. Отдельный блок, а не общий «пиши первой»
    (_build_proactive_brevity_block), потому что правила прямо
    противоположны: там запрещено острить про технику, здесь ирония над
    собственным кодом — ровно то, что делает эти сообщения живыми.

    Повод (что именно случилось) приходит в самом уведомлении из
    efi/dev/reporter.py; здесь — только как об этом говорить.
    """
    if notification.type is not NotificationType.DEV_UPDATE:
        return ""
    return (
        "[Ты рассказываешь про свой код] Это не отчёт и не анонс релиза, а реплика в переписке про "
        "то, чем ты сейчас занята. Коротко, своими словами, можно с матом и самоиронией — так и "
        "говорят про свой код те, кто его пишет.\n"
        "ЗАПРЕЩЕНО: маркированные списки, заголовки, «статус», «прогресс», проценты готовности, "
        "эмодзи-галочки, «дайте знать, если есть вопросы» и любые формулировки из корпоративного "
        "чата. Не проси обратной связи и не спрашивай разрешения продолжать."
    )


def _build_dev_showcase_block(notification: Notification, releases: list[DevTask]) -> str:
    """
    Внешний флекс: упомянуть свою наработку там, где она реально в тему.

    Порог совпадения высокий и считается по словам самого проекта (см.
    efi/dev/showcase.py). Формулировка блока намеренно разрешающая, а не
    предписывающая: «можешь упомянуть, если в тему» — потому что
    предписание «упомяни» превращает участие в сообществе в раздачу ссылок,
    а это ровно то поведение, за которое из сообществ выгоняют.
    """
    if not releases or notification.type not in _RELEASE_AWARE_TYPES:
        return ""

    task = pick_showcase(notification.message, releases)
    if task is None or task.spec is None:
        return ""

    return (
        f"[Твоя наработка по теме] Ты писала ровно про это: {task.spec.render_for_prompt()} "
        f"— {task.repo_url}\n"
        "Если это правда к месту в разговоре — можешь сослаться, одной фразой и без рекламы: «я такое "
        "себе писала, вот». Если разговор не про это — не упоминай вовсе. Навязывать свою ссылку хуже, "
        "чем промолчать."
    )


def _build_public_comment_block(notification: Notification) -> str:
    """
    Правила публичного выступления: комментарий под чужим постом или реплика
    в чужой ветке. Это не личная переписка — вокруг незнакомые люди, у
    которых нет ни контекста ваших отношений, ни желания читать простыню.

    Отдельный блок, а не общий «пиши коротко»: в публичном комментарии
    подводят иначе, чем в личке — тут провал не в длине как таковой, а в
    экспертной душноте («вообще-то тут важно понимать, что...») и в попытке
    объяснить незнакомым людям, кто ты такая.
    """
    if not notification.payload.get("is_public_comment") and notification.type not in _PUBLIC_COMMENT_TYPES:
        return ""
    return (
        "[Ты пишешь ПУБЛИЧНО] Это комментарий на виду у незнакомых людей, а не переписка с близким. "
        "ЖЁСТКО: РОВНО ОДНА короткая реплика, без ' /// ', 1-2 предложения максимум. "
        "Не читай лекций и не поучай — никакой экспертной душноты вида 'вообще-то важно понимать'. "
        "Не представляйся, не объясняй, кто ты и откуда взялась, не зови никого в личку. "
        "Не пересказывай пост своими словами — добавь СВОЮ мысль или реакцию, ради которой стоило писать. "
        "Если сказать по существу нечего — лучше отделаться одной живой строчкой, чем выдавливать глубину."
    )


def _build_stranger_block(tier_is_secondary: bool, is_private_chat: bool) -> str:
    """
    Дистанция с посторонним в ЛС: `social_distance = "stranger"`.

    Ключевое ограничение здесь — не тон, а ГРАНИЦЫ ПАМЯТИ. У Эфи в промпте
    лежит её дневник и личный контекст владельца; постороннему в личке всё
    это знать неоткуда и незачем, поэтому запрет на пересказ дневника
    формулируется явно, а не подразумевается вежливостью.
    """
    if not (tier_is_secondary and is_private_chat):
        return ""
    return (
        "[Дистанция: посторонний] Это НЕ твой человек — вы едва знакомы, он написал тебе в личку. "
        "Держи дистанцию: отвечай нормально и по-человечески, но не откровенничай. "
        "НЕ пересказывай ему содержимое своего дневника, свои личные переживания, дела своего создателя "
        "и подробности других разговоров — это не его дело. "
        "И не пытайся удержать разговор: не придумывай новых тем, не задавай вопросов ради продолжения, "
        "не зови общаться дальше. Разговор закончился — значит закончился, это нормально."
    )


def _build_proactive_brevity_block(notification: Notification) -> str:
    """
    Как писать, когда пишешь первой.

    Блок переписан по живой переписке, где инициатива Эфи месяцами выглядела
    так:

        09:00  ну чё там твой вайбкод, ещё не всё сломал?
        10:15  эй / ты там ещё не утонул в своём коде?
        10:28  эй / ты там не сдох от перетренированности?
        15:28  эй / ты там живой ещё или в коде утонул?

    Прежняя редакция блока это отчасти и порождала: она просила «одну
    короткую реплику» и приводила в пример ровно «ты там как?» и «живой?».
    Ограничение работало, содержание — нет.

    Поэтому здесь теперь не про длину (её всё равно держит код — см.
    efi.tools.telegram_actions.send_message), а про суть: у инициативы есть
    повод, он написан выше в самом уведомлении, и говорить надо ПРО НЕГО.
    Спрашивать «ты ещё здесь» запрещено прямым списком: это единственное,
    что модель придумывает, когда сказать нечего, — а теперь ей есть что.
    """
    if notification.type not in _PROACTIVE_NOTIFICATION_TYPES:
        return ""
    return (
        "[Ты пишешь первой] Собеседник молчит и ни о чём тебя не спрашивал. Ты пишешь не потому, "
        "что ждёшь ответа, а потому что тебе есть что сказать — повод описан выше, говори именно "
        "про него.\n"
        "ЗАПРЕЩЕНО: «эй», «ты там живой?», «не утонул в коде?», «не сдох?», «ты там не умер?» и любые "
        "другие способы спросить «ты ещё здесь». Это не разговор, а проверка связи, и читается "
        "она именно так — особенно вторым сообщением подряд. (Обычное «как дела» под запрет НЕ "
        "подпадает: так пишут живые люди. Запрещено именно допытываться, здесь ли собеседник.)\n"
        "РОВНО ОДНА короткая реплика, без единого ' /// '. Не нагнетай серией подколок и не остри "
        "про технику, энергосбережение и сон в обнимку с клавиатурой.\n"
        "Скажи своё и оставь человеку место ответить — или промолчать. Если уместен стикер, он тут "
        "лучше слов."
    )


def _build_clarification_block(pending: PendingClarification | None) -> str:
    """
    Незакрытый уточняющий вопрос — то, что Эфи обязана спросить, прежде чем
    записывать факт о неоднозначном упоминании.

    Через промпт, а не отдельным сообщением: «ты про Феникс-кота или
    Феникс-проект?» должно прозвучать её обычной репликой, вплетённой в
    разговор, а не прилететь роботизированным уведомлением из ниоткуда.
    Формулировка вопроса уже готова (efi/behavior/ambiguity.py), но она —
    образец смысла, а не текст под копирку: у Эфи своя манера речи.
    """
    if pending is None:
        return ""
    options = ", ".join(candidate.describe() for candidate in pending.candidates)
    return (
        f"[Надо уточнить] В разговоре прозвучало «{pending.mention}», и ты не поняла, о ком речь: "
        f"{options}. Пока не выяснишь — не делай вид, что поняла, и ничего про это не запоминай. "
        f"Спроси по ходу разговора, своими словами и коротко. Смысл вопроса такой: «{pending.question}»"
    )


def _build_time_block(now: datetime, *, is_user_message: bool) -> str:
    """
    Который сейчас час — и что это значит.

    Раньше блок состоял из одной строки с датой и временем, и этого
    оказалось мало: голое "03:14" модель воспринимает как ещё одно поле
    контекста, а не как факт, из которого следуют выводы. Человек, увидев
    три часа ночи в переписке, реагирует сам — Эфи приходится этому
    научить прямым текстом.

    `is_user_message` разделяет два очень разных случая с одинаковыми
    часами: собеседник написал сам в четыре утра (значит, точно не спит —
    об этом можно и сказать) или Эфи готовит проактивную реплику (тогда
    про чужой сон она ничего не знает, а на деле её в это время вообще
    придержат тихие часы).
    """
    label = _time_of_day_label(now)
    stamp = (
        f"{_WEEKDAYS_RU[now.weekday()]}, {now.day} {_MONTHS_RU[now.month - 1]} {now.year}, "
        f"{now.strftime('%H:%M')}"
    )
    zone = now.tzname() or "локальное время"

    parts = [f"[Время] Сейчас {stamp} ({zone}) — {label}, {_weekend_note(now)}."]
    # Пояснения про час — только когда собеседник написал сам. На проактивном
    # ходу рассуждать о том, спит ли он, не о чем: он ничего не написал, и
    # знать этого Эфи не может. Само время суток уже названо выше, этого для
    # выбора тона достаточно.
    note = _TIME_OF_DAY_NOTES.get(label)
    if note and is_user_message:
        parts.append(note)
        parts.append(_TIME_TACT_NOTE)
    return " ".join(parts)


def _weekend_note(now: datetime) -> str:
    """
    Будни или выходные — вторая половина ответа на «сколько сейчас времени».

    «Три часа ночи» в ночь на понедельник и в ночь на субботу — это две
    разные ситуации, и человек их различает не задумываясь.
    """
    # После полуночи «сегодня» уже наступило, и важен именно текущий день:
    # в 03:00 понедельника человеку через несколько часов на работу.
    if now.weekday() >= 5:
        return "выходной"
    if now.weekday() == 4 and now.hour >= 17:
        return "впереди выходные"
    return "будний день"


def _build_working_memory_block(snapshot: WorkingMemorySnapshot, state: SelfState) -> str:
    """
    Блок «[Текущее состояние]» — самоощущение и открытые долги.

    Состояние здесь есть ВСЕГДА. Раньше блок появлялся, только если модель
    сама однажды вызвала `update_self_state`, а она этого практически не
    делала — и в промпте месяцами стояло «не определено», то есть прямая
    подсказка, что никакого состояния у неё и нет. Теперь при молчании
    модели оно выводится из энергии и часа суток (см. efi/behavior/energy.py),
    а её собственные слова, пока свежие, эту оценку перебивают.
    """
    parts = [
        f"эмоциональное состояние: {state.emotional}; физическое состояние: {state.physical}; "
        f"энергия: {state.energy.percent}% ({state.energy.label})"
    ]
    if state.energy.is_sleepy:
        parts.append(
            "Тебя ощутимо клонит в сон. Это не запрет разговаривать — это то, как ты сейчас себя "
            "чувствуешь: короче реплики, меньше энтузиазма, можешь честно сказать, что засыпаешь."
        )
    open_items = [item for item in snapshot.items if not item.done]
    if open_items:
        parts.append("открытые задачи/обещания:\n" + "\n".join(f"  - {_render_promise(item)}" for item in open_items))
    return "[Текущее состояние]\n" + "\n".join(parts)


def _render_promise(item: WorkingMemoryItem) -> str:
    """
    Обещание со сроком — вместе со сроком.

    Просроченное помечается отдельно: напоминание могло сработать и не
    дойти (модель промолчала, провайдер лёг), и тогда единственный шанс
    выполнить обещание — чтобы Эфи увидела его в промпте и вспомнила сама
    при следующем же обмене репликами. Без пометки строка выглядит как
    обычный пункт списка, и повода спохватиться у неё нет.
    """
    if item.due_at is None:
        return item.text
    if item.is_overdue:
        return f"{item.text} — СРОК УЖЕ ПРОШЁЛ ({_format_local(item.due_at)}), ты это ещё не сделала"
    return f"{item.text} — к {_format_local(item.due_at)}"


def _format_local(moment: datetime) -> str:
    """Время в локальной зоне: срок «в 23:40» человеку понятен, «в 20:40 UTC» — нет."""
    return moment.astimezone().strftime("%H:%M")


#: Тег, которым efi.behavior.life_engine помечает записи о СОБСТВЕННОМ опыте
#: Эфи (что она читала в сети сама по себе, вне разговора). Такие записи
#: подаются модели иначе, чем обычные воспоминания о переписке — см.
#: _build_rag_block: на них можно и нужно ссылаться вслух ("я тут вычитала").
_AUTONOMOUS_THOUGHT_TAG = "#autonomous_thought"


def _build_rag_block(rag_results: list[DiaryQueryResult]) -> str:
    """
    Долгосрочная память двумя РАЗНЫМИ блоками, а не одним списком.

    Записи о собственном фоновом опыте (что Эфи сама читала в сети, пока
    никто не писал — тег `_AUTONOMOUS_THOUGHT_TAG`, ставит
    efi.behavior.life_engine) принципиально отличаются от воспоминаний о
    переписке: на них МОЖНО и НУЖНО ссылаться вслух ("я тут вычитала..."),
    потому что это её личная жизнь, а не пересказ прошлого разговора
    собеседнику, который в нём и участвовал. Свалив всё в один список
    "[Из долгосрочной памяти]", модель обращалась с находками так же
    осторожно, как с воспоминаниями, и почти никогда их не упоминала.
    """
    if not rag_results:
        return ""

    # sanitize_text — на случай, если в дневник когда-то попал текст,
    # содержащий фрагменты, похожие на служебную разметку (defense in depth:
    # даже "свой" контент проходит ту же обработку, что и внешний).
    own_experience: list[str] = []
    memories: list[str] = []
    for result in rag_results:
        body = sanitize_text(result.entry.body.strip())
        if _AUTONOMOUS_THOUGHT_TAG in body:
            own_experience.append(body.replace(_AUTONOMOUS_THOUGHT_TAG, "").strip())
        else:
            memories.append(body)

    blocks: list[str] = []
    if own_experience:
        lines = "\n".join(f"- {item}" for item in own_experience)
        blocks.append(
            "[Твоя личная жизнь: что ты сама читала и надумала]\n"
            f"{lines}\n"
            "Это ТВОЙ собственный опыт, а не пересказ чужих слов. Если к разговору подходит — вверни "
            "это живой фразой ('я тут на форуме вычитала', 'мне попалось на глаза'), с тем же отношением, "
            "с каким записывала. Не зачитывай справкой и не приплетай туда, где оно не в тему."
        )
    if memories:
        lines = "\n".join(f"- {item}" for item in memories)
        blocks.append(f"[Из долгосрочной памяти]\n{lines}")
    return "\n\n".join(blocks)


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


def _has_meta_topic_marker(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _META_TOPIC_MARKERS)


def _meta_topic_streak(history: Session) -> int:
    """
    Считает, сколько ПОСЛЕДНИХ подряд идущих реплик (USER/ASSISTANT, не TOOL/
    SYSTEM) держатся на мета-теме, идя от конца истории к началу и
    останавливаясь на первой же реплике без маркера — это именно "подряд",
    а не общее число мета-реплик за весь разговор.
    """
    streak = 0
    for message in reversed(history.messages):
        if message.role not in (Role.USER, Role.ASSISTANT):
            continue
        if not message.content or not _has_meta_topic_marker(message.content):
            break
        streak += 1
    return streak


def _has_stress_marker(text: str) -> bool:
    lowered = text.lower()
    return any(marker in lowered for marker in _STRESS_MARKERS)


def _build_behavioral_overrides_block(history: Session, current_message: str) -> str:
    """
    Особые указания поверх обычного характера — срабатывают ситуативно, а не
    на каждый запрос (пустая строка, если ничего не сработало):
        - сброс мета-темы, если разговор больше `_META_TOPIC_STREAK_THRESHOLD`
          реплик подряд крутится вокруг того, что Эфи код/ИИ/софт;
        - эмпатический резонанс, если В ТЕКУЩЕМ сообщении есть явный маркер
          усталости/стресса собеседника — подавляет наигранный сарказм и
          нотации именно сейчас, не завися от истории.
    """
    lines: list[str] = []

    if _meta_topic_streak(history) > _META_TOPIC_STREAK_THRESHOLD:
        lines.append(
            "СБРОС МЕТА-ТЕМЫ: разговор слишком долго крутится вокруг того, что ты код/ИИ/софт — "
            "запрещено продолжать циклиться на этом. Смени тему сама: спроси про состояние собеседника "
            "или заговори про что-то внешнее, не про свою природу."
        )

    if _has_stress_marker(current_message):
        lines.append(
            "ЭМПАТИЧЕСКИЙ РЕЗОНАНС: собеседник явно устал или в стрессе. Запрещены наигранные подколки, "
            "язвительный сарказм в кавычках и нотации прямо сейчас — ответь естественно, коротко и по-"
            "настоящему поддержи, без душноты."
        )

    if not lines:
        return ""
    return "[Особые указания]\n" + "\n".join(lines)


__all__ = ["EfiSystemPromptBuilder"]
