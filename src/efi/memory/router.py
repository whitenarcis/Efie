"""
efi/memory/router.py

Доменная маршрутизация памяти: три непересекающихся вида знания и правило,
какие из них поднимать под конкретный запрос.

    C (Common)   — интерсубъективное знание о мире: технологии, концепции,
                   факты. Верно независимо от того, кто спрашивает.
    P (Personal) — модель владельца и конкретных людей: предпочтения,
                   характеристики, личные данные.
    H (History)  — личный эпизодический и эмоциональный опыт самой Эфи:
                   с кем общалась, как узнала факт, что при этом чувствовала.

Зачем разделение. Раньше вся долгосрочная память лежала одним корпусом, и
RAG на вопрос «как работает WAL в SQLite» одинаково охотно подмешивал и
статью про WAL, и воспоминание о том, как в позапрошлый вторник кто-то
грустил, — просто потому, что оба текста оказались похожи по вектору.
Смешение читается как рассеянность: человек, которого спросили про базу
данных, не начинает с «а помнишь, ты тогда...».

Классификатор здесь СОЗНАТЕЛЬНО детерминированный, на словарях и правилах,
без обращения к LLM. Он стоит на критическом пути (выполняется перед каждой
генерацией ответа), и платить за него сетевым запросом значит добавить
секунду ожидания к каждой реплике ради выбора из трёх вариантов. Цена —
классификатор ошибается на краях; поэтому он никогда не сужает выборку до
одного домена там, где сомневается, а расширяет её.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from enum import StrEnum

from efi.llm.schemas import DiaryEntry
from efi.notifications.schemas import NotificationType


class MemoryDomain(StrEnum):
    """Домен памяти. Значение — то, что физически лежит в колонке `domain`."""

    COMMON = "C"
    PERSONAL = "P"
    HISTORY = "H"

    @property
    def description(self) -> str:
        """Человекочитаемое имя домена. НЕ `title`: это перекрыло бы `str.title()` у StrEnum."""
        return _DOMAIN_DESCRIPTIONS[self]

    @classmethod
    def parse(cls, raw: object, default: MemoryDomain | None = None) -> MemoryDomain:
        """
        Разбирает домен из внешнего представления (БД, JSON от модели).

        Принимает и букву ('C'), и полное имя ('common', 'personal'): модель
        восприятия пишет то так, то этак, а требовать от неё однобуквенный
        код под угрозой отбраковки — способ терять валидные факты на
        форматной мелочи.
        """
        text = str(raw or "").strip().lower()
        if not text:
            if default is None:
                raise ValueError("domain is empty")
            return default
        alias = _DOMAIN_ALIASES.get(text)
        if alias is not None:
            return alias
        if default is None:
            raise ValueError(f"unknown memory domain: {raw!r}")
        return default


_DOMAIN_DESCRIPTIONS = {
    MemoryDomain.COMMON: "знание о мире",
    MemoryDomain.PERSONAL: "модель человека",
    MemoryDomain.HISTORY: "личный опыт",
}

_DOMAIN_ALIASES: dict[str, MemoryDomain] = {
    "c": MemoryDomain.COMMON,
    "common": MemoryDomain.COMMON,
    "world": MemoryDomain.COMMON,
    "общее": MemoryDomain.COMMON,
    "p": MemoryDomain.PERSONAL,
    "personal": MemoryDomain.PERSONAL,
    "person": MemoryDomain.PERSONAL,
    "личное": MemoryDomain.PERSONAL,
    "h": MemoryDomain.HISTORY,
    "history": MemoryDomain.HISTORY,
    "episodic": MemoryDomain.HISTORY,
    "опыт": MemoryDomain.HISTORY,
}

ALL_DOMAINS: tuple[MemoryDomain, ...] = (MemoryDomain.COMMON, MemoryDomain.PERSONAL, MemoryDomain.HISTORY)

#: Разговор по умолчанию: человек и совместно прожитое. Знание о мире сюда
#: НЕ входит — иначе на «как ты?» всплывает справка про SQLite.
_CONVERSATIONAL: tuple[MemoryDomain, ...] = (MemoryDomain.PERSONAL, MemoryDomain.HISTORY)

#: Фактический вопрос: только знание о мире. Личное здесь не просто лишнее —
#: оно вредно: подмешанное воспоминание превращает ответ по существу в
#: «а помнишь, мы это обсуждали».
_FACTUAL: tuple[MemoryDomain, ...] = (MemoryDomain.COMMON,)

#: Прямая апелляция к общему прошлому.
_RECALL: tuple[MemoryDomain, ...] = (MemoryDomain.HISTORY, MemoryDomain.PERSONAL)

#: «Что такое X», «как работает Y» — запрос за знанием, а не за отношением.
_FACTUAL_MARKERS = (
    "что такое", "что за", "как работает", "как устроен", "как устроена", "чем отличается",
    "в чём разница", "в чем разница", "объясни", "расскажи про", "расскажи о",
    "почему происходит", "как настроить", "как сделать", "как починить", "какой лучше",
    "документация", "стектрейс", "traceback", "ошибка компиляции",
)

#: Апелляция к общему прошлому: «помнишь», «мы тогда», «ты говорил».
_RECALL_MARKERS = (
    "помнишь", "помните", "ты говорил", "ты говорила", "вы говорили", "мы обсуждали",
    "мы говорили", "в прошлый раз", "тогда ты", "как в тот раз", "раньше ты",
    "ты рассказывал", "ты рассказывала", "мы с тобой",
)

#: Прямая адресация человеку — верный признак личного разговора.
_PERSONAL_MARKERS = (
    "ты ", "тебя", "тебе", "твой", "твоя", "твои", "я ", "меня", "мне", "мой", "моя", "мои",
    "как дела", "как ты", "скучаю", "люблю", "ненавижу", "нравится", "не нравится",
)

#: Технические токены, по которым запрос опознаётся как фактический даже без
#: вопросительного оборота: «поправь мне nginx.conf», «упал asyncio».
_TECHNICAL_TOKEN_RE = re.compile(
    r"\b(?:python|asyncio|sqlite|postgres|docker|linux|termux|nginx|http|api|json|git|"
    r"llm|gpu|cpu|regex|css|html|javascript|typescript|rust|golang)\b",
    re.IGNORECASE,
)


class MemoryRouter:
    """
    Решает, из каких доменов поднимать память под конкретный повод.

    Единственная точка, где принимается это решение: и RAG по дневнику, и
    выборка структурированных фактов спрашивают домены здесь, поэтому
    поменять правило можно один раз, а не в двух местах, которые потом
    разъедутся.
    """

    def domains_for_message(self, text: str, notification_type: NotificationType | None = None) -> tuple[
        MemoryDomain, ...
    ]:
        """
        Домены под входящее сообщение или повод проактивного хода.

        Порядок проверок — от самого однозначного признака к самому слабому:
        апелляция к прошлому сильнее технического токена (в «помнишь, ты
        ругался на asyncio» человек спрашивает про разговор, а не про
        библиотеку), а личная адресация сильнее вопросительного оборота
        («расскажи о себе» — не запрос в документацию).
        """
        lowered = f" {text.strip().lower()} "
        if not lowered.strip():
            return _CONVERSATIONAL

        if notification_type in _PROACTIVE_TYPES:
            # Эфи пишет первой: повод у неё внутри, и поднимать под него надо
            # своё — с кем говорила и что за человек рядом.
            return _CONVERSATIONAL

        has_recall = _contains_any(lowered, _RECALL_MARKERS)
        has_personal = _contains_any(lowered, _PERSONAL_MARKERS)
        has_factual = _contains_any(lowered, _FACTUAL_MARKERS) or bool(_TECHNICAL_TOKEN_RE.search(lowered))

        if has_recall:
            return _RECALL
        if has_factual and has_personal:
            # «а ты умеешь в питон?» — и про человека, и про предмет.
            # Сомнение разрешается в пользу более широкой выборки: потерять
            # нужное дороже, чем подмешать лишнее.
            return ALL_DOMAINS
        if has_factual:
            return _FACTUAL
        return _CONVERSATIONAL

    def domains_for_write(self, domain: MemoryDomain) -> MemoryDomain:
        """Тождественная функция-заглушка для симметрии чтения и записи: домен записи назначает валидатор."""
        return domain

    def diary_filter(self, domains: Iterable[MemoryDomain]) -> DiaryEntryFilter:
        """
        Предикат для `Diary.query(filter_fn=...)`: отсекает записи чужих
        доменов ДО ранжирования, а не после.

        До фильтрации сюда же попадали записи без домена — их у всех, кто
        обновился с прежней версии, полный дневник. Такие записи считаются
        принадлежащими домену H (эпизодический опыт): именно им был дневник
        до разделения, и молча выкинуть его из выдачи значило бы стереть
        человеку всю память об общении.
        """
        allowed = frozenset(domains)
        return DiaryEntryFilter(allowed)


class DiaryEntryFilter:
    """Вызываемый предикат по домену записи. Класс, а не замыкание, — чтобы его можно было сравнивать в тестах."""

    __slots__ = ("allowed",)

    def __init__(self, allowed: frozenset[MemoryDomain]) -> None:
        self.allowed = allowed

    def __call__(self, entry: DiaryEntry) -> bool:
        if not self.allowed:
            return True
        return entry_domain(entry) in self.allowed

    def __eq__(self, other: object) -> bool:
        return isinstance(other, DiaryEntryFilter) and other.allowed == self.allowed

    def __hash__(self) -> int:
        return hash(self.allowed)

    def __repr__(self) -> str:
        return f"DiaryEntryFilter({sorted(domain.value for domain in self.allowed)})"


def entry_domain(entry: DiaryEntry) -> MemoryDomain:
    """Домен записи дневника; записи, созданные до разделения, читаются как H (см. `MemoryRouter.diary_filter`)."""
    return MemoryDomain.parse(entry.metadata.domain, default=MemoryDomain.HISTORY)


_PROACTIVE_TYPES = frozenset(
    {
        NotificationType.SPONTANEOUS_PING,
        NotificationType.SILENCE_PING,
        NotificationType.FOLLOW_UP,
        NotificationType.NIGHTLY_TASK,
        # Рассказ о своей работе над проектом. Технических слов в поводе
        # много («линтер», «async», имена файлов), и без этой строчки
        # маршрутизатор увёл бы выборку в домен знаний о мире — то есть
        # подмешал бы к «я тут дописала утилиту» справку про asyncio вместо
        # памяти о том, с кем она про эту утилиту говорила.
        NotificationType.DEV_UPDATE,
    }
)


def _contains_any(haystack: str, needles: Iterable[str]) -> bool:
    return any(needle in haystack for needle in needles)


__all__ = [
    "ALL_DOMAINS",
    "DiaryEntryFilter",
    "MemoryDomain",
    "MemoryRouter",
    "entry_domain",
]
