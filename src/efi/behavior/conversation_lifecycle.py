"""
efi/behavior/conversation_lifecycle.py

Жизненный цикл диалога с ПОСТОРОННИМИ — то, чего у персонального бота не
было вовсе: он общался со всеми одинаково и никогда не заканчивал разговор
сам.

Два разных статуса собеседника:
    PRIMARY   — владелец (`telegram.owner_id`). С ним диалог не
                «завершается», ему разрешены инициативные пинги, ему доступна
                вся глубина памяти.
    SECONDARY — все остальные. Разговор с ними имеет естественный конец, и
                Эфи вправе молча из него выйти.

Почему «молча»: реальный человек не объявляет «я завершаю диалог» — он
просто перестаёт отвечать, когда разговор исчерпан. Отсюда контракт
`evaluate()`: он возвращает решение, и при `should_disengage=True`
вызывающая сторона (efi.notifications.worker.Worker) НЕ генерирует ответ
вообще, а не генерирует прощание.

Три независимых повода закончить:
    - собеседник попрощался (`_FAREWELL_MARKERS`) — разговор логически
      закончен, отвечать «ага, пока» ещё раз незачем;
    - собеседник отвечает сухо и односложно несколько реплик подряд — ему
      явно не интересно, и продолжать значит навязываться;
    - накопилась навязчивость (`annoyance_score`) — он спамит, требует,
      грубит.

Навязчивость персистентна (`conversation_state`, efi.db.models): она
копится по паре (peer_user_id, chat_id) и переживает рестарт. Иначе
достаточно было бы перезапустить процесс, чтобы человек, которого Эфи уже
«закрыла», снова получил чистый лист.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum

from efi.db.core import Database

logger = logging.getLogger(__name__)

#: Порог накопленной навязчивости, после которого разговор закрывается.
ANNOYANCE_THRESHOLD = 1.0

#: Сколько сухих односложных реплик подряд читаются как «мне неинтересно».
TERSE_STREAK_THRESHOLD = 3

_TERSE_MAX_LENGTH = 12

_FAREWELL_MARKERS = (
    "пока", "бывай", "до связи", "спокойной ночи", "споки", "спок", "давай, пойду",
    "пойду я", "ладно, пошёл", "ладно, пошел", "всё, отбой", "все, отбой", "до завтра",
    "увидимся", "чао", "бай", "gn", "bye",
)

_TERSE_REPLIES = (
    "ок", "окей", "ага", "угу", "ясно", "понятно", "пон", "ладно", "лан", "ну ок",
    "да", "нет", "неа", "хз", "мгм", "угу.", "+", "ок.", "ясн",
)

#: Что именно повышает навязчивость и насколько. Значения подобраны так,
#: чтобы одна грубость не закрывала разговор мгновенно, но повторяющееся
#: давление упиралось в ANNOYANCE_THRESHOLD за несколько реплик.
_ANNOYANCE_DEMANDING = 0.25
_ANNOYANCE_HOSTILE = 0.4
_ANNOYANCE_SPAM = 0.3
_ANNOYANCE_DECAY = 0.15

_DEMANDING_MARKERS = (
    "ответь", "отвечай", "почему молчишь", "ты там живая", "ну же", "быстрее",
    "срочно", "немедленно", "я жду", "долго ещё", "долго еще",
)
_HOSTILE_MARKERS = (
    "тупая", "дура", "идиотка", "заткнись", "иди нахуй", "иди нах", "бесполезн",
    "ты просто бот", "тупой бот", "хуйню несешь", "хуйню несёшь",
)

_REPEATED_PUNCTUATION_RE = re.compile(r"[?!]{3,}")


class UserTier(StrEnum):
    """Статус собеседника — определяет и глубину доступа, и право на инициативу."""

    PRIMARY = "primary"
    SECONDARY = "secondary"


class ConversationStatus(StrEnum):
    ACTIVE = "active"
    CLOSED = "closed"


@dataclass(slots=True, frozen=True)
class ConversationState:
    """Персистентное состояние диалога с одним посторонним в одном чате."""

    peer_user_id: int
    chat_id: int
    annoyance_score: float = 0.0
    status: ConversationStatus = ConversationStatus.ACTIVE
    closed_reason: str = ""


@dataclass(slots=True, frozen=True)
class LifecycleDecision:
    """
    Решение по одной входящей реплике.

    `should_disengage=True` означает «не отвечай вообще» — не «ответь
    прощанием». Причина (`reason`) идёт только в лог, собеседник её не видит.
    """

    should_disengage: bool
    reason: str = ""
    annoyance_score: float = 0.0
    tier: UserTier = UserTier.SECONDARY


def is_farewell(text: str) -> bool:
    """Чистая функция: похоже ли сообщение на прощание."""
    lowered = text.strip().lower()
    if not lowered:
        return False
    return any(marker in lowered for marker in _FAREWELL_MARKERS)


def is_terse(text: str) -> bool:
    """Чистая функция: сухая односложная отписка («ок», «ага», «ясно»)."""
    stripped = text.strip().lower().rstrip(".!)")
    return bool(stripped) and len(stripped) <= _TERSE_MAX_LENGTH and stripped in _TERSE_REPLIES


def score_annoyance(text: str) -> float:
    """
    Насколько эта конкретная реплика «давит». Чистая функция без сети и
    состояния — накопление и затухание живут в ConversationLifecycle.
    """
    lowered = text.strip().lower()
    if not lowered:
        return 0.0

    score = 0.0
    if any(marker in lowered for marker in _HOSTILE_MARKERS):
        score += _ANNOYANCE_HOSTILE
    if any(marker in lowered for marker in _DEMANDING_MARKERS):
        score += _ANNOYANCE_DEMANDING
    if _REPEATED_PUNCTUATION_RE.search(lowered):
        score += _ANNOYANCE_SPAM
    return score


class ConversationLifecycle:
    """
    Решает, продолжать ли разговор с посторонним, и хранит накопленную
    навязчивость между перезапусками.

    Кэш в памяти обслуживает горячий путь (решение принимается перед каждым
    ответом), запись в БД идёт следом — тот же приём, что у
    efi.behavior.affinity.AffinityTracker.
    """

    def __init__(self, database: Database, *, owner_id: int, proactive_chats: Iterable[int] = ()) -> None:
        self._database = database
        self._owner_id = owner_id
        #: Чаты, куда владелец сам разрешил писать первой (telegram.allowed_chats).
        #: Личка владельца входит сюда всегда: в Telegram id приватного чата
        #: совпадает с user_id собеседника.
        self._proactive_chats = {owner_id, *proactive_chats}
        self._cache: dict[tuple[int, int], ConversationState] = {}
        self._terse_streak: dict[tuple[int, int], int] = {}

    def classify(self, user_id: int | None) -> UserTier:
        """Владелец или посторонний. `None` (нет отправителя) трактуется как посторонний — безопасный дефолт."""
        return UserTier.PRIMARY if user_id == self._owner_id else UserTier.SECONDARY

    def allows_proactive_ping(self, user_id: int | None) -> bool:
        """
        Разрешён ли инициативный пинг КОНКРЕТНОМУ человеку. Только владельцу:
        писать первой постороннему — навязчивость по определению, он не
        просил о себе напоминать.
        """
        return self.classify(user_id) is UserTier.PRIMARY

    def allows_proactive_ping_to_chat(self, chat_id: int | None, sender_id: int | None = None) -> bool:
        """
        Разрешён ли инициативный пинг в ЭТОТ ЧАТ.

        Отдельный метод, а не `allows_proactive_ping(sender_id)`, потому что у
        инициативы отправителя нет по определению: спонтанный пинг, пинг по
        затишью и follow-up рождаются не из чужой реплики, а из таймера, и
        `sender_id` в их payload взять неоткуда. Раньше Worker всё равно
        спрашивал именно про отправителя — получал None, None трактовался как
        «посторонний», и КАЖДЫЙ инициативный пинг отбрасывался ещё до
        обращения к LLM. Внешне это выглядело так, будто Эфи просто никогда
        не пишет первой: в логах «queued for chat_id=...» есть, а дальше
        тишина и ни одной ошибки.

        Правило: явный отправитель решает всё (он либо владелец, либо нет);
        если отправителя нет — решает чат. Разрешены личка владельца и то,
        что он сам перечислил в `telegram.allowed_chats`. Каналы сообщества
        сюда НЕ входят: участие в них — это ответ на чужой пост (см.
        efi/telegram/comments.py), а не право заговорить первой.
        """
        if sender_id is not None:
            return self.allows_proactive_ping(sender_id)
        return chat_id is not None and chat_id in self._proactive_chats

    async def get_state(self, peer_user_id: int, chat_id: int) -> ConversationState:
        """Состояние диалога; при первом обращении поднимается из БД (переживает рестарт) либо создаётся чистым."""
        key = (peer_user_id, chat_id)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        row = await self._database.fetch_one(
            "SELECT annoyance_score, status, closed_reason FROM conversation_state "
            "WHERE peer_user_id = ? AND chat_id = ?",
            (peer_user_id, chat_id),
        )
        state = (
            ConversationState(
                peer_user_id=peer_user_id,
                chat_id=chat_id,
                annoyance_score=row["annoyance_score"],
                status=ConversationStatus(row["status"]),
                closed_reason=row["closed_reason"],
            )
            if row is not None
            else ConversationState(peer_user_id=peer_user_id, chat_id=chat_id)
        )
        self._cache[key] = state
        return state

    async def evaluate(self, peer_user_id: int | None, chat_id: int, text: str) -> LifecycleDecision:
        """
        Главная точка входа: принять решение по входящей реплике и обновить
        состояние. Для владельца всегда «продолжаем» — его разговор не
        завершается и навязчивость по нему не копится вовсе.
        """
        tier = self.classify(peer_user_id)
        if tier is UserTier.PRIMARY or peer_user_id is None:
            return LifecycleDecision(should_disengage=False, tier=tier)

        state = await self.get_state(peer_user_id, chat_id)

        # Разговор уже закрыт: заново он открывается только содержательной
        # репликой. Если человек продолжает давить теми же «ответь!!!» —
        # молчание сохраняется, иначе закрытие ничего бы не значило.
        if state.status is ConversationStatus.CLOSED:
            if _is_substantive(text):
                state = await self._reopen(state)
            else:
                return LifecycleDecision(
                    should_disengage=True, reason="conversation already closed",
                    annoyance_score=state.annoyance_score, tier=tier,
                )

        key = (peer_user_id, chat_id)
        delta = score_annoyance(text)
        # Затухание: спокойная реплика гасит накопленное раздражение, иначе
        # один плохой день человека закрывал бы диалог с ним навсегда.
        annoyance = _clamp(state.annoyance_score + delta if delta > 0 else state.annoyance_score - _ANNOYANCE_DECAY)

        if is_terse(text):
            self._terse_streak[key] = self._terse_streak.get(key, 0) + 1
        else:
            self._terse_streak[key] = 0

        reason = _pick_disengage_reason(
            text, annoyance=annoyance, terse_streak=self._terse_streak.get(key, 0)
        )
        state = await self._persist(state, annoyance=annoyance, closed_reason=reason)

        if reason:
            logger.info(
                "conversation_lifecycle: disengaging from user_id=%s in chat_id=%s (%s, annoyance=%.2f)",
                peer_user_id, chat_id, reason, annoyance,
            )
        return LifecycleDecision(
            should_disengage=bool(reason), reason=reason, annoyance_score=annoyance, tier=tier
        )

    async def _reopen(self, state: ConversationState) -> ConversationState:
        reopened = ConversationState(
            peer_user_id=state.peer_user_id,
            chat_id=state.chat_id,
            # Навязчивость не обнуляем: человек вернулся с содержательной
            # репликой, но история давления никуда не делась.
            annoyance_score=state.annoyance_score,
            status=ConversationStatus.ACTIVE,
        )
        self._cache[(state.peer_user_id, state.chat_id)] = reopened
        return reopened

    async def _persist(
        self, state: ConversationState, *, annoyance: float, closed_reason: str
    ) -> ConversationState:
        updated = ConversationState(
            peer_user_id=state.peer_user_id,
            chat_id=state.chat_id,
            annoyance_score=annoyance,
            status=ConversationStatus.CLOSED if closed_reason else ConversationStatus.ACTIVE,
            closed_reason=closed_reason,
        )
        self._cache[(state.peer_user_id, state.chat_id)] = updated
        await self._database.execute(
            """
            INSERT INTO conversation_state
                (peer_user_id, chat_id, annoyance_score, status, closed_reason, updated_at)
            VALUES (?, ?, ?, ?, ?, ?)
            ON CONFLICT (peer_user_id, chat_id) DO UPDATE SET
                annoyance_score = excluded.annoyance_score,
                status = excluded.status,
                closed_reason = excluded.closed_reason,
                updated_at = excluded.updated_at
            """,
            (
                updated.peer_user_id,
                updated.chat_id,
                updated.annoyance_score,
                updated.status.value,
                updated.closed_reason,
                datetime.now(UTC).isoformat(),
            ),
        )
        return updated


def _pick_disengage_reason(text: str, *, annoyance: float, terse_streak: int) -> str:
    """Пустая строка — продолжаем разговор. Непустая — молча выходим, причина уходит в лог."""
    if is_farewell(text):
        return "farewell"
    if annoyance >= ANNOYANCE_THRESHOLD:
        return "annoyance threshold reached"
    if terse_streak >= TERSE_STREAK_THRESHOLD:
        return "interlocutor is disengaged (terse replies)"
    return ""


def _is_substantive(text: str) -> bool:
    """Содержательная реплика — не отписка и не прощание; ею закрытый разговор может открыться заново."""
    return not is_terse(text) and not is_farewell(text) and score_annoyance(text) == 0.0


def _clamp(value: float) -> float:
    return max(0.0, min(value, 1.0))


__all__ = [
    "ANNOYANCE_THRESHOLD",
    "ConversationLifecycle",
    "ConversationState",
    "ConversationStatus",
    "LifecycleDecision",
    "TERSE_STREAK_THRESHOLD",
    "UserTier",
    "is_farewell",
    "is_terse",
    "score_annoyance",
]
