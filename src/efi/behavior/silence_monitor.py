"""
efi/behavior/silence_monitor.py

SILENCE_PING: реакция на длительное затишье в чате (аналог
silence_monitor_lifecycle) — если ни от пользователя, ни от Эфи давно не
было сообщений, возможно, стоит написать первой. Модуль просто кладёт
Notification в NotificationManager; что конкретно будет сказано — решает
личность внутри Worker'а.

Здесь же раньше жила вторая, in-memory очередь отложенных FOLLOW_UP'ов
(`schedule_follow_up`). Она удалена: её никто никогда не вызывал, а
пережить перезапуск она не могла по построению — обещание «напиши через 10
минут» исчезало вместе с процессом. Отложенные возвращения к теме теперь
целиком в efi/behavior/reminders.py, поверх SQLite.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from efi.behavior.initiative import InitiativeGate
from efi.behavior.ping_reason import PingReasonBuilder
from efi.behavior.quiet_hours import is_quiet_now
from efi.config.schema import QuietHoursSettings
from efi.notifications.manager import NotificationManager
from efi.notifications.schemas import Notification, NotificationType
from efi.utils.bounded import BoundedDict
from efi.utils.loops import run_periodically

logger = logging.getLogger(__name__)

#: Потолок числа отслеживаемых чатов и срок жизни записи о чате.
#: Неделя: чат, где неделю ничего не происходило, всё равно за любым порогом
#: тишины, и помнить точную дату его последней активности незачем.
_MAX_TRACKED_CHATS = 512
_TRACKING_TTL_SECONDS = 7 * 24 * 3600.0


class SilenceMonitor:
    """
    Отслеживает последнюю активность (в любую сторону) по чату и очередь
    отложенных follow-up'ов.

    `record_activity()` должна вызываться на КАЖДОЕ сообщение — и входящее
    (efi.telegram.handlers.TelegramEventHandlers), и исходящее
    (efi.tools.telegram_actions.send_message.SendMessageTool) — иначе Эфи
    будет считать чат тихим сразу после того, как сама в нём написала.
    """

    def __init__(
        self,
        manager: NotificationManager,
        *,
        check_interval_seconds: float = 900.0,  # 15 минут
        silence_threshold: timedelta = timedelta(hours=6),
        quiet_hours: QuietHoursSettings | None = None,
        timezone: str = "",
        initiative: InitiativeGate | None = None,
        reasons: PingReasonBuilder | None = None,
    ) -> None:
        self._manager = manager
        self._check_interval_seconds = check_interval_seconds
        self._silence_threshold = silence_threshold
        self._quiet_hours = quiet_hours
        self._timezone = timezone
        #: Право заговорить первой — общее на все инициативные службы, см.
        #: efi/behavior/initiative.py.
        self._initiative = initiative
        #: Повод написать, если он есть (efi/behavior/ping_reason.py). Само
        #: затишье — тоже повод, но самый бедный из возможных, поэтому
        #: конкретный повод всегда предпочтительнее.
        self._reasons = reasons
        #: Оба словаря ограничены: запись про чат, где ничего не было
        #: неделю, ничего не решает — тишина там и так за любым порогом.
        self._last_activity: BoundedDict[int, datetime] = BoundedDict(
            max_entries=_MAX_TRACKED_CHATS, ttl=_TRACKING_TTL_SECONDS
        )
        self._last_silence_ping: BoundedDict[int, datetime] = BoundedDict(
            max_entries=_MAX_TRACKED_CHATS, ttl=_TRACKING_TTL_SECONDS
        )

    def set_reasons(self, reasons: PingReasonBuilder) -> None:
        """
        Поздняя привязка источника поводов. Нужна порядку сборки: builder
        зависит от фонового исследователя, а тот конструируется позже
        монитора (см. efi/app.py). Тащить монитор вниз по файлу ради этого
        значило бы перетасовать половину сборки.
        """
        self._reasons = reasons

    def record_activity(self, chat_id: int) -> None:
        """Отмечает, что в чате только что что-то произошло (сообщение в любую сторону). Синхронный, дешёвый вызов."""
        self._last_activity[chat_id] = datetime.now(UTC)

    async def run(self) -> None:
        """Основной цикл. Останавливается по отмене задачи (CancelledError) — см. efi/app.py graceful shutdown."""
        logger.info("silence_monitor: threshold=%s", self._silence_threshold)
        await run_periodically(
            self._check_silence, interval_seconds=self._check_interval_seconds, name="silence_monitor"
        )

    async def _check_silence(self) -> None:
        if is_quiet_now(self._quiet_hours, self._timezone):
            # Не трогаем _last_silence_ping — намеренно: как только тихие часы
            # закончатся, следующий проход снова увидит то же затишье и
            # запингует нормально, а не будет молчать ещё один полный
            # silence_threshold из-за пропущенного окна.
            logger.debug("silence_monitor: skipping silence check — quiet hours")
            return

        now = datetime.now(UTC)
        for chat_id, last_activity in list(self._last_activity.items()):
            if now - last_activity < self._silence_threshold:
                continue
            last_ping = self._last_silence_ping.get(chat_id)
            if last_ping is not None and now - last_ping < self._silence_threshold:
                continue  # уже пинговали по этому периоду тишины — не спамим повторно каждые check_interval_seconds
            if self._initiative is not None and not await self._initiative.may_initiate(chat_id, now=now):
                # Тишина в чате И неотвеченное сообщение — это не «повод
                # написать», а ответ на вопрос, почему тихо.
                continue
            reason = await self._reasons.reason_for(chat_id) if self._reasons is not None else None
            await self._manager.put(
                Notification(
                    type=NotificationType.SILENCE_PING,
                    priority=7,
                    chat_id=chat_id,
                    message=reason or _render_plain_silence(self._silence_threshold),
                    payload={},
                )
            )
            self._last_silence_ping[chat_id] = now
            logger.info("silence_monitor: queued SILENCE_PING for chat_id=%s", chat_id)


def _render_plain_silence(threshold: timedelta) -> str:
    """
    Затишье как повод — самый бедный из возможных, поэтому формулировка
    вынуждена делать за модель половину работы.

    Прямые запреты здесь не перестраховка, а разбор того, что получалось без
    них: «эй», «ты там живой?», «не утонул в коде?» — ровно эти фразы модель
    выдаёт, когда ей сказали «напиши первой» и не сказали, о чём.
    """
    hours = max(1, int(threshold.total_seconds() // 3600))
    return (
        f"В этом чате тихо уже часов {hours}. Ты просто вспомнила о собеседнике — без тревоги и без "
        "проверки связи.\n"
        "ЗАПРЕЩЕНО: «эй», «ты там живой?», «не утонул в коде?», «ты там не умер?» и любые другие "
        "способы спросить «ты ещё здесь». Это не общение, а пинг сервера — и человек читает их "
        "именно так. Обычное «как дела» — можно, так пишут живые люди.\n"
        "Вместо этого скажи что-нибудь СВОЁ: чем сама занималась, что заметила, что вспомнила. "
        "Одна короткая фраза, из которой видно, что у тебя своя жизнь, а не что ты ждала под дверью."
    )


__all__ = ["SilenceMonitor"]
