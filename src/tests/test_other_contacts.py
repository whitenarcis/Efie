"""
Тесты честности про собственный день (блок «С кем ты ещё общалась»).

Регрессия из жизни: Эфи переписывалась с посторонним в ЛС, владелец спросил,
общалась ли она с кем-то, и получил «нет, только с тобой».

Это не было враньём в обычном смысле — врать было нечем. В системный промпт
попадают история ЭТОГО чата и найденные по смыслу воспоминания; про то, что
происходило в других чатах, там не было ничего. На вопрос «ты с кем-то
переписывалась?» у неё буквально не было данных, и она отвечала единственным,
что видела перед собой.

Семантический поиск по дневнику тут не спасал и не мог: записи о тех
разговорах есть, но они про их СОДЕРЖАНИЕ («обсудили сканеры»), а не про факт
«я с кем-то говорила», и по такому вопросу вектором не находятся.

Отдельная половина задачи — приватность. Список «с кем ещё переписывается
владелец аккаунта» посторонним не показывается ни при какой формулировке
вопроса.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from efi.db.core import Database
from efi.db.models import MIGRATIONS
from efi.memory.people import PeopleStore
from efi.notifications.schemas import Notification, NotificationType
from efi.prompts.builder import RecentContact, _build_other_contacts_block

_OWNER_ID = 2129889949
_STRANGER_ID = 777123
_OWNER_CHAT = _OWNER_ID
_STRANGER_CHAT = _STRANGER_ID


def _contact(name: str = "Рихтер", **overrides: object) -> RecentContact:
    defaults: dict[str, object] = {
        "name": name,
        "where": "",
        "when": datetime.now(UTC),
        "impression": "",
    }
    defaults.update(overrides)
    return RecentContact(**defaults)  # type: ignore[arg-type]


# -- сам блок ------------------------------------------------------------------


def test_the_block_names_who_she_talked_to() -> None:
    block = _build_other_contacts_block([_contact("Рихтер"), _contact("Аня")])

    assert "Рихтер" in block
    assert "Аня" in block


def test_the_block_forbids_denying_the_fact() -> None:
    """
    Главное содержание блока. Данных мало — надо ещё сказать, что отрицать
    их нельзя: «я ни с кем не переписывалась» при живой переписке это уже не
    тактичность, а враньё.
    """
    block = _build_other_contacts_block([_contact()])

    assert "скрывать её не надо" in block
    assert "враньё" in block


def test_details_of_other_conversations_stay_optional() -> None:
    """
    Разница между «не отрицай факт» и «выкладывай содержание». Первое —
    честность, второе — болтливость про чужие разговоры.
    """
    block = _build_other_contacts_block([_contact()])

    assert "не обязана" in block


def test_no_contacts_means_no_block() -> None:
    """Пустой блок вместо «сегодня ни с кем» — незачем занимать промпт отсутствием событий."""
    assert _build_other_contacts_block([]) == ""


def test_a_contact_renders_with_place_and_impression() -> None:
    contact = _contact("Рихтер", where="Плёнка и цифра", impression="душный, но по делу")

    rendered = contact.render()

    assert "Рихтер" in rendered
    assert "Плёнка и цифра" in rendered
    assert "душный, но по делу" in rendered


def test_a_contact_without_extras_renders_to_just_a_name() -> None:
    assert _contact("Аня", where="", when=None, impression="").render() == "Аня"


# -- кто это видит --------------------------------------------------------------


def _builder(tmp_path: Path, people: PeopleStore):  # type: ignore[no-untyped-def]
    from efi.config.schema import LLMRolesSettings, RoleRoute, Settings
    from efi.prompts.builder import EfiSystemPromptBuilder
    from efi.prompts.loader import PromptLoader

    settings = Settings(
        telegram={"api_id": 1, "api_hash": "x", "owner_id": _OWNER_ID},  # type: ignore[arg-type]
        llm_roles=LLMRolesSettings(
            main=RoleRoute(primary={"base_url": "https://x/v1", "api_key": "k", "model": "m"})  # type: ignore[arg-type]
        ),
    )
    return EfiSystemPromptBuilder(
        PromptLoader(tmp_path / "templates"),
        settings,
        rag=None,  # type: ignore[arg-type]
        working_memory=None,  # type: ignore[arg-type]
        beliefs=None,  # type: ignore[arg-type]
        affinity=None,  # type: ignore[arg-type]
        people=people,
    )


def _notification(sender_id: int, chat_id: int) -> Notification:
    return Notification(
        type=NotificationType.USER_MESSAGE,
        chat_id=chat_id,
        message="ты сегодня с кем-нибудь переписывалась?",
        payload={"sender_id": sender_id, "chat_type": "PRIVATE"},
    )


async def test_the_owner_sees_who_she_talked_to(tmp_path: Path) -> None:
    """Ровно тот случай из переписки: владелец спрашивает, а Эфи говорит «только с тобой»."""
    database = Database(tmp_path / "efi.db", migrations=MIGRATIONS)
    people = PeopleStore(database)
    await people.record_message(_STRANGER_ID, "привет", display_name="Рихтер", chat_id=_STRANGER_CHAT)

    contacts = await _builder(tmp_path, people)._resolve_other_contacts(
        _notification(_OWNER_ID, _OWNER_CHAT), now=datetime.now(UTC)
    )

    assert [contact.name for contact in contacts] == ["Рихтер"]


async def test_a_stranger_learns_nothing_about_other_chats(tmp_path: Path) -> None:
    """
    Приватность. Список «с кем ещё переписывается владелец аккаунта» —
    приватные данные, и никакая формулировка вопроса не должна их открывать.
    """
    database = Database(tmp_path / "efi.db", migrations=MIGRATIONS)
    people = PeopleStore(database)
    await people.record_message(_OWNER_ID, "привет", display_name="Рома", chat_id=_OWNER_CHAT)
    await people.record_message(999, "здоров", display_name="Кто-то ещё", chat_id=-500)

    contacts = await _builder(tmp_path, people)._resolve_other_contacts(
        _notification(_STRANGER_ID, _STRANGER_CHAT), now=datetime.now(UTC)
    )

    assert contacts == []


async def test_the_asker_is_not_listed_among_the_others(tmp_path: Path) -> None:
    """Владелец и так знает, что пишет ей прямо сейчас; в перечне «других» он выглядел бы странно."""
    database = Database(tmp_path / "efi.db", migrations=MIGRATIONS)
    people = PeopleStore(database)
    await people.record_message(_OWNER_ID, "привет", display_name="Рома", chat_id=_OWNER_CHAT)

    contacts = await _builder(tmp_path, people)._resolve_other_contacts(
        _notification(_OWNER_ID, _OWNER_CHAT), now=datetime.now(UTC)
    )

    assert contacts == []


async def test_conversations_are_remembered_for_a_week_not_a_day(tmp_path: Path) -> None:
    """
    Регрессия на «через пару дней забывает, с кем говорила». Окно было
    ровно сутки, и на третий день Эфи отвечала «ни с кем не переписывалась»
    про разговор, который прекрасно помнит дневник, — то есть врала, потому
    что источник правды до неё просто не доезжал.
    """
    database = Database(tmp_path / "efi.db", migrations=MIGRATIONS)
    people = PeopleStore(database)
    await people.record_message(_STRANGER_ID, "привет", display_name="Рихтер", chat_id=_STRANGER_CHAT)
    builder = _builder(tmp_path, people)

    three_days_later = await builder._resolve_other_contacts(
        _notification(_OWNER_ID, _OWNER_CHAT), now=datetime.now(UTC) + timedelta(days=3)
    )
    two_weeks_later = await builder._resolve_other_contacts(
        _notification(_OWNER_ID, _OWNER_CHAT), now=datetime.now(UTC) + timedelta(days=14)
    )

    assert [contact.name for contact in three_days_later] == ["Рихтер"]
    assert three_days_later[0].when is not None, "дата обязана быть: по ней отличают «сегодня» от «в среду»"
    assert two_weeks_later == [], "две недели — это уже не «недавно», а архив"


async def test_the_list_does_not_turn_into_an_address_book(tmp_path: Path) -> None:
    """Это ответ на вопрос «с кем ты общалась», а не выгрузка всех знакомых."""
    from efi.prompts.builder import _OTHER_CONTACTS_SHOWN

    database = Database(tmp_path / "efi.db", migrations=MIGRATIONS)
    people = PeopleStore(database)
    for index in range(_OTHER_CONTACTS_SHOWN + 4):
        await people.record_message(1000 + index, "привет", display_name=f"Гость {index}", chat_id=-index - 1)

    contacts = await _builder(tmp_path, people)._resolve_other_contacts(
        _notification(_OWNER_ID, _OWNER_CHAT), now=datetime.now(UTC)
    )

    assert len(contacts) == _OTHER_CONTACTS_SHOWN


# -- данные, на которых всё держится ---------------------------------------------


async def test_people_store_remembers_when_it_last_saw_someone(tmp_path: Path) -> None:
    """
    Без даты последней встречи список знакомых неотличим от списка «с кем
    говорила сегодня» — а вопрос был именно про сегодня.
    """
    database = Database(tmp_path / "efi.db", migrations=MIGRATIONS)
    people = PeopleStore(database)
    await people.record_message(_STRANGER_ID, "привет", display_name="Рихтер", chat_id=_STRANGER_CHAT)

    profile = await people.get(_STRANGER_ID)

    assert profile is not None
    assert profile.last_seen_at is not None
    assert profile.last_seen_at.tzinfo is not None, "наивная дата сломала бы сравнение с now(UTC)"
    assert (datetime.now(UTC) - profile.last_seen_at) < timedelta(minutes=1)
