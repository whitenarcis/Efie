"""
efi/utils/clock.py

Единственный источник ответа на вопрос «который сейчас час у Эфи».

Зачем отдельный модуль. Время суток у Эфи решает больше, чем кажется: по нему
молчат тихие часы, по нему собирается блок [Время] в системном промпте, по
нему она понимает, что собеседник пишет ей в четыре утра. Раньше каждое из
этих мест звало `datetime.now()` самостоятельно, то есть молча полагалось на
часовой пояс процесса. В Termux это обычно верно — телефон живёт в поясе
владельца, — но ровно обычно: под `proot`, в сервисе, запущенном из cron, или
на VPS переменная TZ нередко пуста, и процесс живёт по UTC. Эфи в таком
запуске уверенно считает четыре утра полуднем: тихие часы не наступают, а в
промпте написано «день» — и никакого признака поломки, кроме странного
поведения.

Поэтому пояс задаётся явно (`Settings.timezone`), а пустое значение
по-прежнему означает «доверять системе»: это не регресс, а осознанный
дефолт для тех, у кого TZ настроен верно.
"""

from __future__ import annotations

import logging
from datetime import datetime, tzinfo
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

logger = logging.getLogger(__name__)

#: Пояса, которые не удалось разобрать, — чтобы не сыпать одинаковым
#: предупреждением в лог на каждый вызов (а зовут нас часто: перед каждым
#: ответом и на каждом тике проактивных служб).
_WARNED_TIMEZONES: set[str] = set()


def resolve_timezone(name: str | None) -> tzinfo | None:
    """
    Разбирает имя пояса в tzinfo. `None` означает «системный локальный пояс».

    Неизвестное имя не считается фатальным: свалиться на старте из-за опечатки
    в `timezone = "Europe/Moscw"` хуже, чем отработать по системному поясу с
    предупреждением в логе — второе оставляет Эфи живой.
    """
    if not name or not name.strip():
        return None
    key = name.strip()
    try:
        return ZoneInfo(key)
    except (ZoneInfoNotFoundError, ValueError, OSError):
        if key not in _WARNED_TIMEZONES:
            _WARNED_TIMEZONES.add(key)
            logger.warning(
                "clock: неизвестный часовой пояс %r, продолжаю по системному времени "
                "(проверьте написание, например Europe/Moscow)",
                key,
            )
        return None


def local_now(timezone: str | tzinfo | None = None, *, now: datetime | None = None) -> datetime:
    """
    Текущее время как aware-datetime в поясе Эфи.

    `now` позволяет подставить момент в тестах, не подменяя системные часы.
    Наивный `now` трактуется как уже локальный (именно так его отдаёт
    `datetime.now()`), aware — переводится в нужный пояс.
    """
    tz = resolve_timezone(timezone) if timezone is None or isinstance(timezone, str) else timezone
    moment = now if now is not None else datetime.now()
    # astimezone(None) — это «привести к системному локальному поясу», а
    # наивный момент он трактует как локальный. Ровно то поведение, которое
    # нужно в обеих ветках, поэтому отдельного if здесь нет.
    return moment.astimezone(tz)


__all__ = ["local_now", "resolve_timezone"]
