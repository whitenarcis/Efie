"""
efi/behavior/organic_ping.py

Органический пинг-генератор — превращает готовую находку фонового
исследования (efi.behavior.life_engine.InformedThought) в спонтанный пинг
собеседнику, НО только если находка достаточно важна (`weight` семени выше
`importance_threshold`) — не каждое фоновое исследование заслуживает того,
чтобы Эфи писала первой, часть находок просто оседает в дневнике как
пассивная память (уже сохранена туда самим BackgroundLifeWorker).

Причина пинга — не дежурное "как дела", а конкретная находка: payload несёт
её текстом ("Нашла интересное по нашей теме X"), чтобы SystemPromptBuilder/
LLM внутри Worker'а видели повод, а не додумывали его сами (тот же принцип,
что у efi.behavior.spontaneous_ping — этот модуль сам формулировать финальный
ответ не пытается, только ставит Notification с осмысленным поводом).

Второе назначение модуля — закрыть петлю обратной связи: когда собеседник
ОТВЕЧАЕТ в чат, где недавно был органический пинг, это более сильный сигнал
вовлечённости, чем обычная реплика, — see `handle_reply`, которая применяет
дополнительный (сверх обычной классификации efi.behavior.affinity.classify_message)
буст affinity/respect_level через AffinityTracker.apply_boost.
"""

from __future__ import annotations

import logging

from efi.behavior.affinity import AffinityTracker
from efi.behavior.life_engine import InformedThought
from efi.notifications.manager import NotificationManager
from efi.notifications.schemas import Notification, NotificationType

logger = logging.getLogger(__name__)

#: Дополнительный (сверх обычной классификации реплики) буст affinity/respect_level
#: за то, что собеседник отреагировал на находку, которую Эфи принесла сама, по
#: собственной инициативе — более сильный сигнал вовлечённости, чем рядовая реплика.
_ENGAGEMENT_AFFINITY_BOOST = 0.05
_ENGAGEMENT_RESPECT_BOOST = 0.05

_DEFAULT_IMPORTANCE_THRESHOLD = 0.6


class OrganicPingGenerator:
    """
    Держит per-chat реестр "недавно был органический пинг, ответа ещё не
    было" (аналог `_PendingFollowUp` в efi.behavior.silence_monitor —
    намеренно простой in-memory словарь, а не БД: сам факт "жду реакции на
    последний органический пинг" нужен только пока процесс жив, переживать
    рестарт ему незачем).
    """

    def __init__(
        self,
        manager: NotificationManager,
        affinity: AffinityTracker,
        *,
        importance_threshold: float = _DEFAULT_IMPORTANCE_THRESHOLD,
    ) -> None:
        self._manager = manager
        self._affinity = affinity
        self._importance_threshold = importance_threshold
        self._pending_by_chat: dict[int, int] = {}  # chat_id -> seed_id

    async def notify(self, thought: InformedThought) -> None:
        """
        Ставит SPONTANEOUS_PING по находке, если она достаточно важна и есть
        куда её нести (семя без source_chat_id — например, гипотетическое
        будущее семя, рождённое не из конкретного разговора, — просто
        остаётся тихой записью в дневнике, пинговать о ней некого).
        """
        if thought.source_chat_id is None:
            return
        if thought.weight < self._importance_threshold:
            logger.debug(
                "organic_ping: seed #%s (%r) below importance threshold (%.2f < %.2f), staying in diary only",
                thought.seed_id, thought.topic, thought.weight, self._importance_threshold,
            )
            return

        reason = f"Нашла интересное по нашей теме {thought.topic!r}"
        notification = Notification(
            type=NotificationType.SPONTANEOUS_PING,
            priority=6,
            chat_id=thought.source_chat_id,
            message=(
                f"{reason}: {thought.finding} Поделись этим с собеседником как своей находкой — "
                "он сам поднимал эту тему раньше, так что это прямое продолжение вашего разговора, "
                "а не дежурное 'привет'."
            ),
            payload={"reason": reason, "seed_id": thought.seed_id, "topic": thought.topic},
        )
        await self._manager.put(notification)
        self._pending_by_chat[thought.source_chat_id] = thought.seed_id
        logger.info(
            "organic_ping: queued for chat_id=%s (seed #%s, %r)", thought.source_chat_id, thought.seed_id, thought.topic
        )

    async def handle_reply(self, chat_id: int) -> None:
        """
        Вызывается на КАЖДОЕ входящее сообщение чата (см. `organic_ping_recorder`
        в efi.telegram.handlers.TelegramEventHandlers) — no-op, если в этом
        чате не было неотвеченного органического пинга. Одноразово: снимает
        отметку, следующий вызов для того же chat_id ничего не сделает, пока
        не появится новый органический пинг.
        """
        seed_id = self._pending_by_chat.pop(chat_id, None)
        if seed_id is None:
            return
        await self._affinity.apply_boost(
            chat_id, affinity_delta=_ENGAGEMENT_AFFINITY_BOOST, respect_delta=_ENGAGEMENT_RESPECT_BOOST
        )
        logger.debug("organic_ping: chat_id=%s engaged with seed #%s, affinity boosted", chat_id, seed_id)


__all__ = ["OrganicPingGenerator"]
