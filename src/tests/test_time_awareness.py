"""
Тесты «Эфи знает, который час» — efi.utils.clock, блок [Время] и тихие часы.

Раньше время у Эфи было техническим полем: в системный промпт уходила одна
строка вида «Monday, 11 August 2026, 03:14», и на этом всё. Голое число часов
модель не превращает в социальный вывод сама — на «03:14» она отвечает ровно
так же, как на «13:14». Человек так не умеет: увидев три часа ночи в
переписке, он про это скажет.

Второй, менее заметный слой — сам часовой пояс. Все места, где время значило
что-то (тихие часы в трёх планировщиках, блок промпта, дашборд), звали
`datetime.now()` и молча полагались на TZ процесса. В Termux на телефоне это
обычно верно, но под proot, из cron или на VPS переменная нередко пуста — и
процесс живёт по UTC. Эфи в таком запуске считает четыре утра полуднем: тихие
часы не наступают, а в промпте написано «день». Признаков поломки при этом
нет никаких, кроме странного поведения.
"""

from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from efi.behavior.quiet_hours import is_quiet_now
from efi.config.schema import QuietHoursSettings
from efi.prompts.builder import _build_time_block, _time_of_day_label
from efi.utils.clock import local_now, resolve_timezone

_MSK = ZoneInfo("Europe/Moscow")


def _at(year: int, month: int, day: int, hour: int, minute: int = 0) -> datetime:
    return datetime(year, month, day, hour, minute, tzinfo=_MSK)


# -- часовой пояс -------------------------------------------------------------


def test_named_timezone_is_resolved() -> None:
    assert resolve_timezone("Europe/Moscow") == _MSK


@pytest.mark.parametrize("name", ["", "   ", None])
def test_empty_timezone_means_system_local(name: str | None) -> None:
    assert resolve_timezone(name) is None


def test_typo_in_timezone_does_not_kill_the_app() -> None:
    """
    Свалиться на старте из-за опечатки в конфиге хуже, чем отработать по
    системному времени с предупреждением: второе оставляет Эфи живой.
    """
    assert resolve_timezone("Europe/Moscw") is None


def test_utc_process_still_sees_the_configured_zone() -> None:
    """Ровно тот случай, ради которого пояс задаётся явно: процесс живёт по UTC."""
    utc_moment = datetime(2026, 8, 11, 0, 30, tzinfo=UTC)

    moscow = local_now("Europe/Moscow", now=utc_moment)

    assert moscow.hour == 3, "полночь по UTC — это уже три часа ночи в Москве"


def test_naive_moment_is_treated_as_local() -> None:
    """`datetime.now()` отдаёт наивный момент — он должен читаться как локальный, а не как UTC."""
    naive = datetime(2026, 8, 11, 14, 0)

    assert local_now(now=naive).hour == 14


# -- блок [Время] -------------------------------------------------------------


def test_deep_night_tells_her_the_person_is_not_asleep() -> None:
    """
    Главное, чего не хватало: из «03:14» должен следовать вывод, а не просто
    число в контексте.
    """
    block = _build_time_block(_at(2026, 8, 10, 3, 14), is_user_message=True)

    assert "глубокая ночь" in block
    assert "не спит" in block
    assert "посоветуй лечь" in block


def test_deep_night_is_not_lectured_about_on_a_proactive_turn() -> None:
    """
    Собеседник ничего не написал — знать, спит он или нет, Эфи не может, и
    строить на этом реплику ей не с чего.
    """
    block = _build_time_block(_at(2026, 8, 10, 3, 14), is_user_message=False)

    assert "глубокая ночь" in block, "само время суток знать всё равно полезно — от него зависит тон"
    assert "не спит" not in block


def test_she_is_told_not_to_nag_about_the_hour() -> None:
    """
    Замечание «иди спать» — живая реакция ровно один раз за ночь. В каждой
    реплике оно превращается в занудство, от которого закрывают чат.
    """
    block = _build_time_block(_at(2026, 8, 10, 3, 14), is_user_message=True)

    assert "не повторяй" in block


def test_date_is_written_in_russian_regardless_of_process_locale() -> None:
    """
    `%A`/`%B` зависят от локали, а в Termux она почти всегда "C" — и в
    русском промпте оказывалось «Monday, 11 August».
    """
    block = _build_time_block(_at(2026, 8, 10, 3, 14), is_user_message=True)

    assert "понедельник, 10 августа 2026" in block
    assert "Monday" not in block and "August" not in block


def test_time_is_shown_with_its_zone() -> None:
    block = _build_time_block(_at(2026, 8, 10, 3, 14), is_user_message=True)

    assert "03:14" in block
    assert "MSK" in block


@pytest.mark.parametrize(
    ("moment", "expected"),
    [
        (_at(2026, 8, 10, 12, 0), "будний день"),  # понедельник
        (_at(2026, 8, 14, 12, 0), "будний день"),  # пятница до вечера
        (_at(2026, 8, 14, 19, 0), "впереди выходные"),  # вечер пятницы
        (_at(2026, 8, 15, 12, 0), "выходной"),  # суббота
        (_at(2026, 8, 16, 12, 0), "выходной"),  # воскресенье
    ],
)
def test_weekday_or_weekend_is_part_of_knowing_the_time(moment: datetime, expected: str) -> None:
    """«Три часа ночи» в ночь на понедельник и в ночь на субботу — разные ситуации."""
    assert expected in _build_time_block(moment, is_user_message=True)


def test_night_before_a_workday_is_a_workday() -> None:
    """В 03:00 понедельника человеку через несколько часов на работу — «сегодня» уже наступило."""
    assert "будний день" in _build_time_block(_at(2026, 8, 10, 3, 0), is_user_message=True)


@pytest.mark.parametrize(
    ("hour", "label"),
    [(3, "глубокая ночь"), (7, "раннее утро"), (10, "утро"), (14, "день"), (19, "вечер"), (23, "ночь")],
)
def test_every_hour_lands_in_a_named_part_of_the_day(hour: int, label: str) -> None:
    assert _time_of_day_label(_at(2026, 8, 10, hour)) == label


# -- тихие часы в том же поясе ------------------------------------------------


def test_quiet_hours_follow_the_configured_zone_not_the_process() -> None:
    """
    Регрессия: планировщики звали `datetime.now()`. На машине с TZ=UTC
    полночь в Москве выглядела как 21:00 — до тихих часов «ещё далеко», и
    Эфи спокойно писала первой в час ночи.
    """
    settings = QuietHoursSettings(start_hour=23, end_hour=8)
    utc_midnight_msk = datetime(2026, 8, 10, 21, 30, tzinfo=UTC)  # 00:30 по Москве

    assert is_quiet_now(settings, "Europe/Moscow", now=utc_midnight_msk) is True


def test_quiet_hours_off_when_disabled_or_unset() -> None:
    moment = _at(2026, 8, 10, 2, 0)

    assert is_quiet_now(None, "Europe/Moscow", now=moment) is False
    assert is_quiet_now(QuietHoursSettings(enabled=False), "Europe/Moscow", now=moment) is False


def test_quiet_hours_are_off_during_the_day() -> None:
    settings = QuietHoursSettings(start_hour=23, end_hour=8)

    assert is_quiet_now(settings, "Europe/Moscow", now=_at(2026, 8, 10, 14, 0)) is False
