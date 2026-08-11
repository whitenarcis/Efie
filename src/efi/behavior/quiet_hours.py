"""
efi/behavior/quiet_hours.py

Общая проверка "сейчас тихие часы" для всех проактивных механизмов
(efi.behavior.spontaneous_ping.SpontaneousPingScheduler, efi.behavior.
organic_ping.OrganicPingGenerator, efi.behavior.silence_monitor.SilenceMonitor).

Без неё Эфи одинаково легко готова написать первой что в обед, что в 5 утра —
формально ей ничего не мешает; а "личная жизнь, ощущается как настоящий
человек" (см. постановку задачи) подразумевает, что нормальный человек не
строчит собеседнику по ночам, пока тот спит, даже если сам не спит. Общий
модуль, а не три копии одной и той же проверки внутри каждого планировщика —
чтобы поменять порог тихих часов в одном месте и все проактивные пути
согласованно замолчали на ночь.

Не влияет на РЕАКТИВНЫЙ путь (efi.notifications.worker.Worker для
USER_MESSAGE) — если собеседник сам написал среди ночи, Эфи всё равно
отвечает, просто сама первой не пишет.
"""

from __future__ import annotations

from datetime import datetime

from efi.config.schema import QuietHoursSettings
from efi.utils.clock import local_now


def is_quiet_hours(now: datetime, *, start_hour: int, end_hour: int) -> bool:
    """
    True, если `now.hour` попадает в промежуток тихих часов [start_hour, end_hour).

    Поддерживает промежуток, оборачивающийся через полночь (start_hour >
    end_hour, обычный случай ночи — например 23..8), а не только "дневной"
    порядок. `start_hour == end_hour` трактуется как "тихих часов нет"
    (нулевой интервал, а не сутки целиком — иначе выключить фичу значением
    "0 0" было бы невозможно отличить от "всегда тихо").
    """
    if start_hour == end_hour:
        return False
    hour = now.hour
    if start_hour < end_hour:
        return start_hour <= hour < end_hour
    return hour >= start_hour or hour < end_hour


def is_quiet_now(settings: QuietHoursSettings | None, timezone: str = "", *, now: datetime | None = None) -> bool:
    """
    «Сейчас тихие часы?» одним вызовом — с учётом настроенного пояса Эфи.

    Раньше каждый планировщик собирал это условие сам из трёх частей
    (`is not None`, `.enabled`, `is_quiet_hours(datetime.now(), ...)`), и все
    три копии одинаково полагались на часовой пояс процесса. Если TZ не
    настроен (proot, cron, VPS), процесс живёт по UTC — и ночная тишина
    наступала не в те часы, причём молча. См. efi/utils/clock.py.
    """
    if settings is None or not settings.enabled:
        return False
    moment = local_now(timezone, now=now)
    return is_quiet_hours(moment, start_hour=settings.start_hour, end_hour=settings.end_hour)


__all__ = ["is_quiet_hours", "is_quiet_now"]
