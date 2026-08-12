"""
Тесты энергии и самоощущения (efi.behavior.energy + WorkingMemory.describe).

Регрессия из жизни: энергия Эфи стабильно показывала 70%, а состояние —
«не записано». Оба поля меняло ровно одно место: добровольный вызов
инструмента `update_self_state`, который модель не делала практически
никогда — у неё нет причины тратить ход хода на служебное действие, которого
от неё в этой реплике никто не ждал.

Последствия шире, чем «некрасиво на дашборде». Энергия — единственный вход
efi.behavior.busy_engine, отвечающий за «Эфи устала и отвечает медленнее»:
константа означала, что подсистема работает вхолостую. А «состояние: не
определено» в системном промпте — прямая подсказка модели, что состояния у
неё и нет.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from efi.behavior import energy as E
from efi.memory.working_memory import STATE_TTL, WorkingMemory, WorkingMemorySnapshot

_MSK = ZoneInfo("Europe/Moscow")


def _at(hour: int, minute: int = 0, day: int = 11) -> datetime:
    return datetime(2026, 8, day, hour, minute, tzinfo=_MSK)


def _memory(tmp_path: Path) -> WorkingMemory:
    return WorkingMemory(tmp_path / "wm.json", timezone="Europe/Moscow")


# -- энергия перестала быть константой ----------------------------------------


def test_energy_follows_the_hour_not_a_frozen_default() -> None:
    """
    Главная регрессия: в четыре утра и в полдень энергия обязана отличаться.
    """
    night = E.project(anchor=0.7, anchor_at=None, now=_at(4))
    noon = E.project(anchor=0.7, anchor_at=None, now=_at(12))

    assert night.level < 0.35
    assert noon.level > 0.7
    assert night.level != noon.level


def test_a_snapshot_without_a_timestamp_does_not_freeze_the_old_default() -> None:
    """
    Файл рабочей памяти из версии до этой модели содержит ровно то самое
    зависшее навсегда значение. Считать такой якорь свежим значило бы
    законсервировать починенный баг: обновились, а всё те же 70% в любой час.
    """
    stale = E.project(anchor=0.7, anchor_at=None, now=_at(3))

    assert stale.level == pytest.approx(E.circadian_baseline(3))


def test_talking_costs_energy() -> None:
    assert E.spend(0.8, turns=1) < 0.8
    assert E.spend(0.8, turns=10) < E.spend(0.8, turns=1)


def test_energy_never_leaves_the_range() -> None:
    assert E.spend(0.01, turns=100) == 0.0
    assert 0.0 <= E.project(anchor=0.0, anchor_at=_at(3), now=_at(12)).level <= 1.0


def test_rest_restores_energy_towards_the_hourly_norm() -> None:
    """Выжатая под ночь Эфи к полудню должна быть бодрой — иначе усталость копилась бы вечно."""
    exhausted_at_night = 0.15

    morning = E.project(anchor=exhausted_at_night, anchor_at=_at(23), now=_at(11, day=12))

    assert morning.level > 0.7


def test_a_short_break_does_not_fully_restore() -> None:
    """Иначе усталость от разговора не значила бы ничего: пять минут — и снова как новая."""
    after_talking = E.project(anchor=0.4, anchor_at=_at(20), now=_at(20, 5))

    assert after_talking.level < 0.5


def test_an_evening_of_talking_tires_her_but_does_not_flatten_her() -> None:
    """
    Проверка порядка величины на реальном ритме переписки: два часа подряд
    заметно сажают энергию, но не в ноль. При первом подборе TURN_COST тот же
    вечер давал 9% — один разговор выматывал сильнее, чем целые сутки.
    """
    anchor, moment = E.circadian_baseline(20), _at(20)
    for step in range(1, 61):  # реплика раз в две минуты, два часа
        now = _at(20) + timedelta(minutes=2 * step)
        anchor = E.spend(E.project(anchor=anchor, anchor_at=moment, now=now).level)
        moment = now

    assert 0.4 < anchor < 0.65, f"после двух часов разговора получилось {anchor:.2f}"


# -- состояние больше не «не записано» -----------------------------------------


async def test_state_is_never_undefined(tmp_path: Path) -> None:
    """Главная регрессия номер два: пустых строк в самоощущении быть не может."""
    memory = _memory(tmp_path)

    state = memory.describe(await memory.load(), now=_at(15))

    assert state.emotional
    assert state.physical
    assert state.is_derived is True


async def test_her_own_words_win_over_the_derived_state(tmp_path: Path) -> None:
    memory = _memory(tmp_path)
    await memory.update_state(emotional_state="злая как чёрт", now=_at(15))

    state = memory.describe(await memory.load(), now=_at(16))

    assert state.emotional == "злая как чёрт"
    assert state.is_derived is False


async def test_a_mood_expires_instead_of_lasting_forever(tmp_path: Path) -> None:
    """
    Настроение — состояние на сейчас, а не свойство характера. «Злая как
    чёрт», сказанное во вторник, к пятнице неправда.
    """
    memory = _memory(tmp_path)
    await memory.update_state(emotional_state="злая как чёрт", now=_at(15))

    later = memory.describe(await memory.load(), now=_at(15) + STATE_TTL + timedelta(minutes=1))

    assert later.emotional != "злая как чёрт"
    assert later.is_derived is True


async def test_legacy_state_without_a_timestamp_is_treated_as_stale(tmp_path: Path) -> None:
    """Именно эти зависшие навсегда строки и были проблемой — вечными они остаться не могут."""
    memory = _memory(tmp_path)
    snapshot = WorkingMemorySnapshot(emotional_state="бодрая", state_updated_at=None)

    assert memory.describe(snapshot, now=_at(3)).is_derived is True


async def test_explicit_energy_becomes_the_new_anchor(tmp_path: Path) -> None:
    """Слова Эфи о себе сильнее модели — но дальше релаксируют как обычно."""
    memory = _memory(tmp_path)

    await memory.update_state(energy=0.1, now=_at(12))
    snapshot = await memory.load()

    assert memory.describe(snapshot, now=_at(12)).energy.level == pytest.approx(0.1)
    assert memory.describe(snapshot, now=_at(18)).energy.level > 0.5, "к вечеру должна отойти"


async def test_spending_starts_from_the_projected_level_not_the_old_anchor(tmp_path: Path) -> None:
    """
    Иначе долгий перерыв, за который Эфи отдохнула, при первой же реплике
    откатывался бы к позавчерашней усталости.
    """
    memory = _memory(tmp_path)
    await memory.update_state(energy=0.1, now=_at(23))

    await memory.spend_energy(now=_at(12, day=12))
    snapshot = await memory.load()

    assert snapshot.energy > 0.7


async def test_energy_survives_a_restart(tmp_path: Path) -> None:
    """Якорь и его отметка обязаны переживать перезапуск — иначе усталость обнулялась бы каждым рестартом."""
    first = _memory(tmp_path)
    await first.update_state(energy=0.2, now=_at(20))

    second = _memory(tmp_path)
    snapshot = await second.load()

    assert snapshot.energy == pytest.approx(0.2)
    assert snapshot.energy_updated_at is not None


# -- описание словами ----------------------------------------------------------


@pytest.mark.parametrize(
    ("level", "hour"),
    [(0.1, 14), (0.4, 14), (0.6, 14), (0.9, 14), (0.1, 3), (0.4, 3)],
)
def test_every_level_has_words_for_it(level: float, hour: int) -> None:
    assert E.describe(level, hour).strip()


def test_the_same_number_reads_differently_at_night() -> None:
    """0.35 в девять вечера — «выдохлась за день», в четыре утра — «нормально для такого часа»."""
    assert E.describe(0.35, hour=21) != E.describe(0.35, hour=4)


def test_deep_night_counts_as_sleepy_even_at_full_energy() -> None:
    assert E.project(anchor=1.0, anchor_at=_at(3), now=_at(3)).is_sleepy is True


def test_midday_at_normal_energy_is_not_sleepy() -> None:
    assert E.project(anchor=0.8, anchor_at=_at(13), now=_at(13)).is_sleepy is False


# -- то, что видит модель ------------------------------------------------------


async def test_the_prompt_block_always_carries_a_state(tmp_path: Path) -> None:
    from efi.prompts.builder import _build_working_memory_block

    memory = _memory(tmp_path)
    snapshot = await memory.load()

    block = _build_working_memory_block(snapshot, memory.describe(snapshot, now=_at(14)))

    assert "[Текущее состояние]" in block
    assert "не определено" not in block
    assert "энергия:" in block


async def test_the_prompt_says_out_loud_when_she_is_sleepy(tmp_path: Path) -> None:
    from efi.prompts.builder import _build_working_memory_block

    memory = _memory(tmp_path)
    snapshot = await memory.load()

    block = _build_working_memory_block(snapshot, memory.describe(snapshot, now=_at(3)))

    assert "клонит в сон" in block


async def test_utc_process_does_not_shift_her_body_clock(tmp_path: Path) -> None:
    """Та же ловушка, что и с тихими часами: без пояса «глубокая ночь» уехала бы на девять вечера."""
    memory = _memory(tmp_path)
    snapshot = await memory.load()

    # 00:30 UTC — это 03:30 по Москве, то есть глубокая ночь.
    state = memory.describe(snapshot, now=datetime(2026, 8, 11, 0, 30, tzinfo=UTC))

    assert state.energy.level < 0.35
    assert state.energy.is_sleepy is True
