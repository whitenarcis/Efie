"""
Тесты доменной маршрутизации памяти (efi.memory.router).

Смысл проверок: на технический вопрос не должны всплывать бытовые
воспоминания, а на «как дела» — справка про SQLite. До разделения на домены
RAG подмешивал и то и другое, потому что решал только по близости векторов,
и рассеянность выглядела как черта характера, а не как отсутствие фильтра.
"""

from __future__ import annotations

import pytest

from efi.llm.schemas import DiaryEntry, DiaryEntryMetadata
from efi.memory.router import ALL_DOMAINS, MemoryDomain, MemoryRouter, entry_domain
from efi.notifications.schemas import NotificationType


def _entry(domain: str) -> DiaryEntry:
    return DiaryEntry(id="e1", metadata=DiaryEntryMetadata(domain=domain), body="текст")


@pytest.fixture
def router() -> MemoryRouter:
    return MemoryRouter()


# -- разбор домена -----------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("C", MemoryDomain.COMMON),
        ("c", MemoryDomain.COMMON),
        ("common", MemoryDomain.COMMON),
        ("P", MemoryDomain.PERSONAL),
        ("personal", MemoryDomain.PERSONAL),
        ("H", MemoryDomain.HISTORY),
        ("history", MemoryDomain.HISTORY),
    ],
)
def test_domain_accepts_letter_and_full_name(raw: str, expected: MemoryDomain) -> None:
    """Модель пишет то букву, то слово; отбраковывать факт за формат имени домена — терять его зря."""
    assert MemoryDomain.parse(raw) == expected


def test_unknown_domain_without_default_raises() -> None:
    with pytest.raises(ValueError):
        MemoryDomain.parse("Q")


def test_unknown_domain_with_default_falls_back() -> None:
    assert MemoryDomain.parse("Q", default=MemoryDomain.HISTORY) is MemoryDomain.HISTORY


# -- маршрутизация запроса ---------------------------------------------------


@pytest.mark.parametrize(
    "text",
    ["что такое WAL в sqlite", "как работает asyncio", "объясни разницу между потоками и корутинами"],
)
def test_factual_question_takes_only_common(router: MemoryRouter, text: str) -> None:
    assert router.domains_for_message(text) == (MemoryDomain.COMMON,)


@pytest.mark.parametrize("text", ["как ты?", "я сегодня устал", "мне нравится этот фильм"])
def test_personal_talk_takes_person_and_history(router: MemoryRouter, text: str) -> None:
    domains = router.domains_for_message(text)

    assert MemoryDomain.COMMON not in domains
    assert set(domains) == {MemoryDomain.PERSONAL, MemoryDomain.HISTORY}


def test_recall_wins_over_technical_token(router: MemoryRouter) -> None:
    """«помнишь, ты ругался на asyncio» — вопрос про разговор, а не про библиотеку."""
    domains = router.domains_for_message("помнишь, ты ругалась на asyncio")

    assert domains[0] is MemoryDomain.HISTORY
    assert MemoryDomain.COMMON not in domains


def test_mixed_question_widens_instead_of_guessing(router: MemoryRouter) -> None:
    """Сомнение разрешается расширением выборки: потерять нужное дороже, чем подмешать лишнее."""
    assert set(router.domains_for_message("а ты умеешь в python?")) == set(ALL_DOMAINS)


def test_proactive_notification_never_pulls_world_knowledge(router: MemoryRouter) -> None:
    domains = router.domains_for_message(
        "Нашла интересное про sqlite", NotificationType.SPONTANEOUS_PING
    )

    assert MemoryDomain.COMMON not in domains


def test_empty_message_is_conversational(router: MemoryRouter) -> None:
    assert set(router.domains_for_message("   ")) == {MemoryDomain.PERSONAL, MemoryDomain.HISTORY}


# -- фильтр по дневнику ------------------------------------------------------


def test_diary_filter_keeps_only_requested_domains(router: MemoryRouter) -> None:
    predicate = router.diary_filter([MemoryDomain.COMMON])

    assert predicate(_entry("C")) is True
    assert predicate(_entry("P")) is False
    assert predicate(_entry("H")) is False


def test_legacy_entries_without_domain_read_as_history(router: MemoryRouter) -> None:
    """
    Записи, созданные до разделения, — это весь накопленный дневник. Выкинуть
    их из выдачи значило бы стереть человеку память об общении.
    """
    legacy = DiaryEntry(id="old", metadata=DiaryEntryMetadata.model_validate({}), body="старая запись")

    assert entry_domain(legacy) is MemoryDomain.HISTORY
    assert router.diary_filter([MemoryDomain.HISTORY])(legacy) is True


def test_empty_domain_set_disables_filtering(router: MemoryRouter) -> None:
    predicate = router.diary_filter([])

    assert predicate(_entry("C")) is True
    assert predicate(_entry("P")) is True
