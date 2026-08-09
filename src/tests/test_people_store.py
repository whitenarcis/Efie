"""
Тесты для efi.memory.people.PeopleStore — социальная память по КОНКРЕТНЫМ
людям (user_id), в отличие от близости по чату (efi.behavior.affinity).

Ключевой сценарий: в группе за одним chat_id стоят разные люди, и их вклад
не должен сваливаться в общий счётчик — отношение к человеку следует за ним
по всем чатам.
"""

from __future__ import annotations

from pathlib import Path

from efi.db.core import Database
from efi.db.models import MIGRATIONS
from efi.memory.beliefs import BeliefStore
from efi.memory.people import PeopleStore

_TROLLING = "ты тупая нейронка, заткнись"
_DEEP_TECH = (
    "смотри, тут race condition в async коде: две корутины пишут в один словарь без лока, "
    "нужен рефактор архитектуры и нормальный паттерн проектирования вокруг этого"
)


def _store(tmp_path: Path, *, with_beliefs: bool = False) -> tuple[PeopleStore, BeliefStore]:
    database = Database(tmp_path / "test.db", migrations=MIGRATIONS)
    beliefs = BeliefStore(database)
    return PeopleStore(database, beliefs=beliefs if with_beliefs else None), beliefs


async def test_unknown_person_is_none(tmp_path: Path) -> None:
    people, _ = _store(tmp_path)
    assert await people.get(555) is None


async def test_first_message_creates_profile(tmp_path: Path) -> None:
    people, _ = _store(tmp_path)
    profile = await people.record_message(555, "привет", display_name="Рома", chat_id=-100, chat_title="Флудилка")

    assert profile.user_id == 555
    assert profile.display_name == "Рома"
    assert profile.message_count == 1
    assert profile.last_chat_id == -100
    assert profile.last_chat_title == "Флудилка"
    assert await people.get(555) is not None


async def test_two_people_in_one_chat_are_tracked_separately(tmp_path: Path) -> None:
    """Главный сценарий: общий chat_id, разные люди — счётчики не должны смешиваться."""
    people, _ = _store(tmp_path)
    for _ in range(5):
        await people.record_message(1, _DEEP_TECH, display_name="Умный", chat_id=-100)
        await people.record_message(2, _TROLLING, display_name="Тролль", chat_id=-100)

    smart = await people.get(1)
    troll = await people.get(2)
    assert smart is not None and troll is not None
    assert smart.respect_level > troll.respect_level
    assert smart.message_count == troll.message_count == 5


async def test_empty_display_name_does_not_erase_known_one(tmp_path: Path) -> None:
    people, _ = _store(tmp_path)
    await people.record_message(7, "привет", display_name="Рихтер")
    profile = await people.record_message(7, "ещё сообщение", display_name="")
    assert profile.display_name == "Рихтер"


async def test_impression_round_trip(tmp_path: Path) -> None:
    people, _ = _store(tmp_path)
    await people.record_message(9, "привет", display_name="Джек")

    updated = await people.set_impression(9, "вечно спорит по мелочам, но по делу")
    assert updated is not None
    assert updated.impression == "вечно спорит по мелочам, но по делу"

    reloaded = await people.get(9)
    assert reloaded is not None
    assert reloaded.impression == "вечно спорит по мелочам, но по делу"


async def test_impression_for_unknown_person_returns_none(tmp_path: Path) -> None:
    people, _ = _store(tmp_path)
    assert await people.set_impression(404, "кто это вообще") is None


async def test_recent_orders_by_last_seen(tmp_path: Path) -> None:
    people, _ = _store(tmp_path)
    await people.record_message(1, "первый", display_name="Первый")
    await people.record_message(2, "второй", display_name="Второй")

    recent = await people.recent(limit=5)
    assert [profile.user_id for profile in recent][:2] == [2, 1]


async def test_social_experience_feeds_the_belief_matrix(tmp_path: Path) -> None:
    """
    Опыт общения не остаётся изолированным счётчиком: устойчивое уважение к
    человеку переносится в общую матрицу убеждений и оттуда влияет на тон.
    """
    people, beliefs = _store(tmp_path, with_beliefs=True)
    for _ in range(20):
        await people.record_message(42, _DEEP_TECH, display_name="Рихтер")

    profile = await people.get(42)
    assert profile is not None and profile.is_familiar

    belief = await beliefs.get("человек:42")
    assert belief is not None
    assert "Рихтер" in belief.stance


async def test_no_belief_until_there_is_enough_experience(tmp_path: Path) -> None:
    """По паре реплик мнение о человеке не формируют — это был бы шум, а не социальный опыт."""
    people, beliefs = _store(tmp_path, with_beliefs=True)
    for _ in range(3):
        await people.record_message(43, _DEEP_TECH, display_name="Новенький")

    assert await beliefs.get("человек:43") is None
