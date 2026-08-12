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
from efi.behavior.initiative import InitiativeGate
from efi.behavior.life_engine import InformedThought
from efi.behavior.quiet_hours import is_quiet_now
from efi.config.schema import QuietHoursSettings
from efi.notifications.manager import NotificationManager
from efi.notifications.schemas import Notification, NotificationType

logger = logging.getLogger(__name__)

#: Дополнительный (сверх обычной классификации реплики) буст affinity/respect_level
#: за то, что собеседник отреагировал на находку, которую Эфи принесла сама, по
#: собственной инициативе — более сильный сигнал вовлечённости, чем рядовая реплика.
_ENGAGEMENT_AFFINITY_BOOST = 0.05
_ENGAGEMENT_RESPECT_BOOST = 0.05

_DEFAULT_IMPORTANCE_THRESHOLD = 0.6

#: Разные ПОВОДЫ заговорить об одной и той же находке. Раньше здесь был один
#: жёстко зашитый шаблон ("Нашла интересное по нашей теме X: ... Поделись этим
#: как своей находкой"), из-за чего каждый органический пинг начинался
#: одинаково — по этой рамке они и читались как автоматическая рассылка, а не
#: как живое "о, вспомнила". Формулировку финальной реплики всё так же
#: выбирает сама модель (этот текст — ПОВОД в системном промпте, а не готовое
#: сообщение), но угол захода теперь каждый раз разный.
_PING_ANGLES: tuple[str, ...] = (
    "Ты только что докопалась до сути темы {topic!r}, и вот что выяснила: {finding} "
    "Тебе прямо не терпится этим поделиться — вы про это уже говорили раньше.",
    "Тебе попалось кое-что про {topic!r}, пока ты залипала в интернете: {finding} "
    "Расскажи об этом так, будто только что наткнулась и сразу вспомнила про собеседника.",
    "Ты обдумывала {topic!r} и пришла вот к чему: {finding} "
    "Поделись именно СВОИМ выводом — не пересказом статьи, а тем, что ты сама об этом думаешь.",
    "Помнишь, вы обсуждали {topic!r}? Ты с тех пор покопалась и нашла: {finding} "
    "Вернись к тому разговору с этой находкой, как возвращаются к недосказанному.",
    "Вот что ты нарыла по {topic!r}: {finding} "
    "Начни с сути находки, без разгона и без 'привет, как дела'.",
)


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
        quiet_hours: QuietHoursSettings | None = None,
        timezone: str = "",
        initiative: InitiativeGate | None = None,
    ) -> None:
        self._manager = manager
        self._affinity = affinity
        self._importance_threshold = importance_threshold
        self._quiet_hours = quiet_hours
        self._timezone = timezone
        #: Право заговорить первой — общее на все инициативные службы, см.
        #: efi/behavior/initiative.py.
        self._initiative = initiative
        self._pending_by_chat: dict[int, int] = {}  # chat_id -> seed_id

    async def notify(self, thought: InformedThought) -> None:
        """
        Ставит SPONTANEOUS_PING по находке, если она достаточно важна и есть
        куда её нести (семя без source_chat_id — например, гипотетическое
        будущее семя, рождённое не из конкретного разговора, — просто
        остаётся тихой записью в дневнике, пинговать о ней некого).

        В тихие часы (см. efi.behavior.quiet_hours) находка НЕ теряется —
        BackgroundLifeWorker её уже сохранил в дневник до вызова notify(),
        просто пинг о ней сейчас не ставится; следующий цикл life_engine
        возьмёт уже следующее семя, а не повторит попытку для этого же.
        """
        if thought.source_chat_id is None:
            return
        if is_quiet_now(self._quiet_hours, self._timezone):
            logger.debug("organic_ping: skipping seed #%s — quiet hours", thought.seed_id)
            return
        if thought.weight < self._importance_threshold:
            logger.debug(
                "organic_ping: seed #%s (%r) below importance threshold (%.2f < %.2f), staying in diary only",
                thought.seed_id, thought.topic, thought.weight, self._importance_threshold,
            )
            return

        if (
            self._initiative is not None
            and thought.source_chat_id is not None
            and not await self._initiative.may_initiate(thought.source_chat_id)
        ):
            logger.info(
                "organic_ping: в chat_id=%s висит неотвеченное сообщение — находка подождёт",
                thought.source_chat_id,
            )
            return

        reason = f"Нашла интересное по нашей теме {thought.topic!r}"
        notification = Notification(
            type=NotificationType.SPONTANEOUS_PING,
            priority=6,
            chat_id=thought.source_chat_id,
            message=_render_ping_message(thought),
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


def _render_ping_message(thought: InformedThought) -> str:
    """
    Повод заговорить — под каждую находку свой угол захода (см. _PING_ANGLES).
    Выбор случайный, но детерминированно "привязан" к семени: один и тот же
    seed_id всегда даёт один и тот же угол, поэтому повторная постановка
    пинга по той же находке не выглядит как вторая, чуть иначе
    сформулированная попытка достучаться.
    """
    angle = _PING_ANGLES[thought.seed_id % len(_PING_ANGLES)]
    return angle.format(topic=thought.topic, finding=thought.finding)


__all__ = ["OrganicPingGenerator"]
