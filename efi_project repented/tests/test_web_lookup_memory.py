"""
Тесты на то, что поход Эфи в интернет вообще остаётся в памяти.

Регрессия: результаты web_search приходят модели TOOL-сообщением, а в таблицу
`messages` пишутся только реплика собеседника и итоговый ответ Эфи (см.
efi/notifications/worker.py). Значит, ни история, ни новеллизация этого следа
не видели: разговор строился вокруг найденного, а через день Эфи не помнила,
что вообще что-то искала.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import httpx

from efi.db.core import Database
from efi.db.models import MIGRATIONS
from efi.llm.schemas import DiaryEntry, Message, Role, Session
from efi.memory.consolidation import _render_conversation
from efi.memory.social_memory import (
    TAG_WEB_LOOKUP,
    SocialInteraction,
    SocialInteractionKind,
    SocialInteractionStore,
)
from efi.notifications.schemas import Notification, NotificationType
from efi.tools.base import ToolContext
from efi.tools.web_tools.web_search import WebSearchTool

_SEARCH_HTML = """
<html><body><table>
<tr><td><a class="result-link" href="https://example.org/lazy">Ленивые импорты в Python</a></td></tr>
<tr><td>PEP 690 предлагал отложенный импорт, но был отклонён в 2023 году.</td></tr>
</table></body></html>
"""


class _RecordingRAG:
    def __init__(self) -> None:
        self.remembered: list[str] = []

    async def remember(self, body: str, *, confidence: float = 0.0) -> DiaryEntry | None:
        self.remembered.append(body)
        return DiaryEntry(id="entry_1", body=body)


class _RecordingJournal:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    async def record_web_lookup(
        self, *, query: str, digest: str, chat_id: int | None = None, thread_id: int | None = None
    ) -> int:
        self.calls.append({"query": query, "digest": digest, "chat_id": chat_id, "thread_id": thread_id})
        return len(self.calls)


def _tool(handler: Any, *, journal: Any = None) -> WebSearchTool:
    transport = httpx.MockTransport(handler)
    return WebSearchTool(client=httpx.AsyncClient(transport=transport), journal=journal)


def _context(chat_id: int | None = 42) -> ToolContext:
    return ToolContext(
        notification=Notification(type=NotificationType.USER_MESSAGE, message="погугли", chat_id=chat_id)
    )


# -- инструмент откладывает поиск в память ----------------------------------------


async def test_successful_search_is_journalled(tmp_path: Path) -> None:
    journal = _RecordingJournal()
    tool = _tool(lambda request: httpx.Response(200, text=_SEARCH_HTML), journal=journal)

    result = await tool.execute({"query": "ленивые импорты python"}, _context())

    assert "Ленивые импорты в Python" in result
    assert len(journal.calls) == 1
    assert journal.calls[0]["query"] == "ленивые импорты python"
    assert "PEP 690" in journal.calls[0]["digest"], "в память должна уходить фактура, а не только сам запрос"
    assert journal.calls[0]["chat_id"] == 42


async def test_empty_search_is_not_an_experience(tmp_path: Path) -> None:
    """«Ничего не нашлось» запоминать нечего — это не поход в интернет, а пустой звук."""
    journal = _RecordingJournal()
    tool = _tool(lambda request: httpx.Response(200, text="<html><body></body></html>"), journal=journal)

    await tool.execute({"query": "асдфасдф"}, _context())
    assert journal.calls == []


async def test_journal_failure_does_not_lose_the_search_results(tmp_path: Path) -> None:
    """Модель уже получила результаты — терять их из-за проблемы с записью в память недопустимо."""

    class _BrokenJournal:
        async def record_web_lookup(self, **kwargs: Any) -> int:
            raise RuntimeError("database is gone")

    tool = _tool(lambda request: httpx.Response(200, text=_SEARCH_HTML), journal=_BrokenJournal())
    result = await tool.execute({"query": "ленивые импорты python"}, _context())

    assert "Ленивые импорты в Python" in result


async def test_tool_works_without_a_journal_at_all(tmp_path: Path) -> None:
    tool = _tool(lambda request: httpx.Response(200, text=_SEARCH_HTML))
    assert "Ленивые импорты в Python" in await tool.execute({"query": "x"}, _context())


# -- журнал сохраняет это как полноценный опыт --------------------------------------


async def test_store_writes_lookup_to_both_layers(tmp_path: Path) -> None:
    rag = _RecordingRAG()
    store = SocialInteractionStore(Database(tmp_path / "test.db", migrations=MIGRATIONS), rag=rag)  # type: ignore[arg-type]

    await store.record_web_lookup(query="ленивые импорты", digest="PEP 690 отклонён", chat_id=42)

    recorded = await store.recent()
    assert len(recorded) == 1
    assert recorded[0].kind is SocialInteractionKind.WEB_LOOKUP
    assert TAG_WEB_LOOKUP in rag.remembered[0]
    assert "PEP 690" in rag.remembered[0]


async def test_lookup_keeps_more_text_than_a_chat_message(tmp_path: Path) -> None:
    """
    Сниппеты — это фактура (числа, названия, ссылки), ради которой запись и
    делается. Обрезанный до пары фраз результат поиска в дневнике бесполезен.
    """
    digest = "деталь " * 200
    body = SocialInteraction(
        kind=SocialInteractionKind.WEB_LOOKUP, text=f"запрос — {digest}"
    ).render_for_prompt()
    assert len(body) > 1000


async def test_context_lines_gather_the_episode_in_chronological_order(tmp_path: Path) -> None:
    store = SocialInteractionStore(Database(tmp_path / "test.db", migrations=MIGRATIONS))
    now = datetime.now(timezone.utc)

    await store.record(
        SocialInteraction(
            kind=SocialInteractionKind.WEB_LOOKUP, text="раньше", chat_id=42, created_at=now - timedelta(minutes=10)
        )
    )
    await store.record(
        SocialInteraction(
            kind=SocialInteractionKind.PUBLIC_COMMENT, text="позже", chat_id=42, created_at=now - timedelta(minutes=1)
        )
    )
    await store.record(
        SocialInteraction(
            kind=SocialInteractionKind.WEB_LOOKUP, text="в другом чате", chat_id=99, created_at=now
        )
    )

    lines = await store.context_lines_for_chat(42, since=now - timedelta(minutes=30))

    assert len(lines) == 2
    assert "раньше" in lines[0] and "позже" in lines[1]
    assert not any("в другом чате" in line for line in lines)


async def test_lines_before_the_window_are_not_replayed(tmp_path: Path) -> None:
    """Иначе один и тот же комментарий подмешивался бы в каждый следующий эпизод."""
    store = SocialInteractionStore(Database(tmp_path / "test.db", migrations=MIGRATIONS))
    now = datetime.now(timezone.utc)
    await store.record(
        SocialInteraction(
            kind=SocialInteractionKind.WEB_LOOKUP, text="вчерашнее", chat_id=42, created_at=now - timedelta(days=1)
        )
    )
    assert await store.context_lines_for_chat(42, since=now - timedelta(hours=1)) == []


# -- обрезание длинного разговора ----------------------------------------------------


def test_overflowing_conversation_keeps_the_end_not_the_beginning() -> None:
    """
    Раньше обрезался конец, и на длинном окне терялась самая свежая, ещё ни
    разу не осмысленная часть разговора — а на следующем проходе она уже была
    за отметкой last_novelized_at, то есть терялась навсегда.
    """
    session = Session(
        messages=[Message(role=Role.USER, content=f"реплика номер {index}") for index in range(200)]
    )
    rendered = _render_conversation(session, char_limit=200)

    assert "реплика номер 199" in rendered
    assert "реплика номер 0\n" not in rendered
    assert rendered.startswith("[...начало разговора опущено...]")


def test_short_conversation_is_not_marked_as_truncated() -> None:
    session = Session(messages=[Message(role=Role.USER, content="короткий разговор")])
    assert _render_conversation(session, char_limit=10_000) == "user: короткий разговор"
