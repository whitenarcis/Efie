"""
efi/memory/people.py

Социальная память по КОНКРЕТНЫМ ЛЮДЯМ, а не по чатам.

Зачем отдельно от efi.behavior.affinity.AffinityTracker: тот считает близость
и уважение на chat_id. В личке это одно и то же, но в группе за одним chat_id
стоят РАЗНЫЕ люди — и их вклад сваливался в один общий счётчик. Эфи не могла
ни отличить, кто именно её троллил, ни перенести сложившееся отношение к
человеку в другой чат, где он тоже присутствует. Здесь единица учёта —
Telegram user_id, поэтому отношение к человеку следует за ним по всем чатам.

Что хранится (таблица `people`, efi.db.models):
    - кто это (user_id + отображаемое имя);
    - affinity/respect_level ЛИЧНО к нему, по той же шкале [0..1] и той же
      дешёвой эвристике efi.behavior.affinity.classify_message (общая
      классификация, чтобы поведение было согласованным между двумя слоями);
    - контекст последней встречи: где виделись (chat_id + название группы) и
      когда, плюс сколько всего было сообщений;
    - `impression` — свободный текст сформированного отношения; его пишет
      сама модель, когда у неё складывается мнение о человеке.

Связь с матрицей убеждений (efi.memory.beliefs.BeliefStore): социальный опыт
не остаётся изолированным счётчиком. Когда отношение к человеку уходит в
явную крайность (устойчиво высокое уважение или устойчиво низкое), это
записывается убеждением о нём — и дальше работает как любое другое убеждение
Эфи, попадая в блок текущего состояния системного промпта через
BeliefStore.find_relevant() и влияя на тон общения.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from efi.behavior.affinity import classify_message
from efi.db.core import Database
from efi.memory.beliefs import BeliefStore

logger = logging.getLogger(__name__)

_DEFAULT_AFFINITY = 0.5
_DEFAULT_RESPECT = 0.5

#: Те же шаги, что и у чатовой близости (efi.behavior.affinity._DELTAS), но
#: применяются к персональному счётчику. Дублирование намеренное: у слоёв
#: разная единица учёта, и их шкалы имеют право разъехаться со временем.
_DELTAS: dict[str, tuple[float, float]] = {
    "trolling": (-0.05, -0.08),
    "deep_tech": (0.02, 0.06),
    "phatic": (0.01, -0.01),
    "neutral": (0.015, 0.005),
}

#: Начиная со скольких сообщений от человека имеет смысл делать выводы о нём:
#: по двум репликам мнение не формируют, это был бы шум, а не социальный опыт.
_MIN_MESSAGES_FOR_BELIEF = 12

#: Границы respect_level, при выходе за которые складывается устойчивое мнение
#: о человеке и переносится в общую матрицу убеждений (см. докстринг модуля).
_BELIEF_HIGH_RESPECT = 0.75
_BELIEF_LOW_RESPECT = 0.25


@dataclass(slots=True, frozen=True)
class PersonProfile:
    """Всё, что Эфи помнит о конкретном человеке. Именно это видит системный промпт."""

    user_id: int
    display_name: str = ""
    affinity: float = _DEFAULT_AFFINITY
    respect_level: float = _DEFAULT_RESPECT
    message_count: int = 0
    last_chat_id: int | None = None
    last_chat_title: str | None = None
    impression: str = ""

    @property
    def is_familiar(self) -> bool:
        """Знакомый человек, а не случайный прохожий — по накопленному объёму общения."""
        return self.message_count >= _MIN_MESSAGES_FOR_BELIEF


class PeopleStore:
    """
    Учёт персоналий: кто, сколько общались, в каком контексте и какое
    отношение сложилось.

    `beliefs` необязателен — без него класс остаётся чисто учётным, с ним
    социальный опыт дополнительно перетекает в общую матрицу убеждений (см.
    докстринг модуля). Ошибка записи убеждения не срывает учёт человека:
    отношение к собеседнику важнее, чем производный от него вывод.
    """

    def __init__(self, database: Database, *, beliefs: BeliefStore | None = None) -> None:
        self._database = database
        self._beliefs = beliefs

    async def get(self, user_id: int) -> PersonProfile | None:
        """Профиль человека, либо None, если Эфи с ним ещё не пересекалась."""
        row = await self._database.fetch_one(
            """
            SELECT user_id, display_name, affinity, respect_level, message_count,
                   last_chat_id, last_chat_title, impression
            FROM people WHERE user_id = ?
            """,
            (user_id,),
        )
        if row is None:
            return None
        return PersonProfile(
            user_id=row["user_id"],
            display_name=row["display_name"],
            affinity=row["affinity"],
            respect_level=row["respect_level"],
            message_count=row["message_count"],
            last_chat_id=row["last_chat_id"],
            last_chat_title=row["last_chat_title"],
            impression=row["impression"],
        )

    async def record_message(
        self,
        user_id: int,
        text: str,
        *,
        display_name: str = "",
        chat_id: int | None = None,
        chat_title: str | None = None,
    ) -> PersonProfile:
        """
        Регистрирует одно сообщение от человека: создаёт профиль при первой
        встрече, сдвигает персональные affinity/respect_level по той же
        эвристике, что и чатовая близость, и обновляет контекст последней
        встречи. Вызывается на КАЖДУЮ входящую реплику (см.
        efi.telegram.handlers.TelegramEventHandlers), поэтому не делает ничего
        дорогого — только один UPSERT.
        """
        current = await self.get(user_id)
        kind = classify_message(text)
        affinity_delta, respect_delta = _DELTAS[kind.value]

        base_affinity = current.affinity if current is not None else _DEFAULT_AFFINITY
        base_respect = current.respect_level if current is not None else _DEFAULT_RESPECT
        updated = PersonProfile(
            user_id=user_id,
            # Имя может прийти пустым (нет доступа к профилю) — тогда не
            # затираем уже известное им пустой строкой.
            display_name=display_name or (current.display_name if current is not None else ""),
            affinity=_clamp(base_affinity + affinity_delta),
            respect_level=_clamp(base_respect + respect_delta),
            message_count=(current.message_count if current is not None else 0) + 1,
            last_chat_id=chat_id,
            last_chat_title=chat_title,
            impression=current.impression if current is not None else "",
        )
        await self._persist(updated, is_new=current is None)
        await self._maybe_record_belief(updated)
        return updated

    async def set_impression(self, user_id: int, impression: str) -> PersonProfile | None:
        """
        Записывает сформированное отношение к человеку свободным текстом —
        то, что модель сама решила про него запомнить. Возвращает None, если
        такого человека Эфи ещё не встречала (запоминать не о ком).
        """
        current = await self.get(user_id)
        if current is None:
            return None
        await self._database.execute(
            "UPDATE people SET impression = ?, last_seen_at = ? WHERE user_id = ?",
            (impression, _now(), user_id),
        )
        logger.info("people: impression updated for user_id=%s", user_id)
        return PersonProfile(
            user_id=current.user_id,
            display_name=current.display_name,
            affinity=current.affinity,
            respect_level=current.respect_level,
            message_count=current.message_count,
            last_chat_id=current.last_chat_id,
            last_chat_title=current.last_chat_title,
            impression=impression,
        )

    async def recent(self, *, limit: int = 5) -> list[PersonProfile]:
        """Люди, с которыми Эфи общалась недавно — от самых свежих. Для блока «с кем ты общалась» в промпте."""
        rows = await self._database.fetch_all(
            """
            SELECT user_id, display_name, affinity, respect_level, message_count,
                   last_chat_id, last_chat_title, impression
            FROM people ORDER BY last_seen_at DESC LIMIT ?
            """,
            (limit,),
        )
        return [
            PersonProfile(
                user_id=row["user_id"],
                display_name=row["display_name"],
                affinity=row["affinity"],
                respect_level=row["respect_level"],
                message_count=row["message_count"],
                last_chat_id=row["last_chat_id"],
                last_chat_title=row["last_chat_title"],
                impression=row["impression"],
            )
            for row in rows
        ]

    async def _persist(self, profile: PersonProfile, *, is_new: bool) -> None:
        now = _now()
        await self._database.execute(
            """
            INSERT INTO people (
                user_id, display_name, affinity, respect_level, message_count,
                first_seen_at, last_seen_at, last_chat_id, last_chat_title, impression
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT (user_id) DO UPDATE SET
                display_name = excluded.display_name,
                affinity = excluded.affinity,
                respect_level = excluded.respect_level,
                message_count = excluded.message_count,
                last_seen_at = excluded.last_seen_at,
                last_chat_id = excluded.last_chat_id,
                last_chat_title = excluded.last_chat_title
            """,
            (
                profile.user_id,
                profile.display_name,
                profile.affinity,
                profile.respect_level,
                profile.message_count,
                now,
                now,
                profile.last_chat_id,
                profile.last_chat_title,
                profile.impression,
            ),
        )
        if is_new:
            logger.info("people: met a new person user_id=%s (%r)", profile.user_id, profile.display_name)

    async def _maybe_record_belief(self, profile: PersonProfile) -> None:
        """
        Переносит устойчивое отношение к человеку в общую матрицу убеждений —
        см. докстринг модуля. Мнение формируется только по накопленному опыту
        (`is_familiar`) и только на явных краях шкалы: на середине шкалы
        никакого вывода о человеке ещё нет, и записывать туда нечего.
        """
        if self._beliefs is None or not profile.is_familiar:
            return

        name = profile.display_name or f"user {profile.user_id}"
        if profile.respect_level >= _BELIEF_HIGH_RESPECT:
            stance = f"{name} — человек, которого я уважаю: с ним можно говорить всерьёз и по существу"
        elif profile.respect_level <= _BELIEF_LOW_RESPECT:
            stance = f"{name} — общаться с ним трудно, разговор чаще выходит пустой или неприятный"
        else:
            return

        try:
            await self._beliefs.upsert(
                f"человек:{profile.user_id}", stance, confidence_score=profile.respect_level
            )
        except Exception:
            # Учёт человека уже сохранён — производный вывод не настолько
            # важен, чтобы ронять из-за него обработку входящего сообщения.
            logger.warning("people: failed to record belief about user_id=%s", profile.user_id, exc_info=True)


def _clamp(value: float) -> float:
    return max(0.0, min(value, 1.0))


def _now() -> str:
    return datetime.now(UTC).isoformat()


__all__ = ["PeopleStore", "PersonProfile"]
