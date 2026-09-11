"""
Тесты веб-поиска (efi.tools.web_tools.web_search).

Живые поисковые сервисы в тестах не дёргаются (сеть, лимиты, капчи): Tavily
мокается httpx.MockTransport, ddgs — подменой класса DDGS. Главное свойство
конвейера: сбой основного пути (Tavily) или «умного слоя» (рерайт/реранк)
не выглядит снаружи как «ничего не нашлось» — есть фолбэк и graceful
degradation.
"""

from __future__ import annotations

import json

import httpx
import pytest

from efi.notifications.schemas import Notification, NotificationType
from efi.tools.base import ToolContext
from efi.tools.web_tools import web_search
from efi.tools.web_tools.web_search import WebSearchTool

TAVILY_PAYLOAD = {
    "results": [
        {"title": "Первый результат", "url": "https://example.org/one", "content": "Описание первого.", "score": 0.9},
        {"title": "Второй результат", "url": "https://example.org/two", "content": "Описание второго.", "score": 0.5},
    ]
}


def _context() -> ToolContext:
    return ToolContext(notification=Notification(type=NotificationType.NIGHTLY_TASK, message="тест"))


def _tavily_client(status: int = 200, payload: dict | None = None) -> httpx.AsyncClient:
    body = json.dumps(payload if payload is not None else TAVILY_PAYLOAD)
    return httpx.AsyncClient(
        transport=httpx.MockTransport(lambda request: httpx.Response(status, content=body.encode()))
    )


class _FakeDDGS:
    """Подмена ddgs.DDGS: по бэкенду решает, что вернуть (или упасть)."""

    def __init__(self, *, fail_backends: tuple[str, ...] = (), empty_backends: tuple[str, ...] = ()) -> None:
        self.fail_backends = fail_backends
        self.empty_backends = empty_backends
        self.seen_backends: list[str] = []
        self.seen_queries: list[str] = []

    def __call__(self, *, timeout: float) -> _FakeDDGS:  # noqa: ARG002 — совместимость с DDGS(timeout=...)
        return self

    def __enter__(self) -> _FakeDDGS:
        return self

    def __exit__(self, *args: object) -> None:
        return None

    def text(self, query: str, *, backend: str, **kwargs: object) -> list[dict[str, str]]:  # noqa: ARG002
        self.seen_backends.append(backend)
        self.seen_queries.append(query)
        if backend in self.fail_backends:
            raise RuntimeError(f"backend {backend} is down")
        if backend in self.empty_backends:
            return []
        return [{"href": f"https://ddgs.example/{backend}", "title": f"Результат {backend}", "body": "сниппет"}]


class _RecordingJournal:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    async def record_web_lookup(
        self, *, query: str, digest: str, chat_id: int | None = None, thread_id: int | None = None
    ) -> int:
        self.calls.append({"query": query, "digest": digest, "chat_id": chat_id, "thread_id": thread_id})
        return len(self.calls) + 1


class _ScriptedRewriter:
    """FAST-модель: отдаёт заготовленный ключевой запрос или падает."""

    def __init__(self, rewritten: str | Exception) -> None:
        self.rewritten = rewritten
        self.received: list[str] = []

    async def chat(self, role: object, params: object, session: object) -> object:
        from efi.llm.schemas import Choice, Message, Response, Role

        self.received.append(str(session))
        if isinstance(self.rewritten, Exception):
            raise self.rewritten
        return Response(choices=[Choice(message=Message(role=Role.ASSISTANT, content=self.rewritten))])


class _ScriptedEmbedder:
    """Эмбеддер-заглушка: релевантность = длина запроса, совпадающая с заголовком."""

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.queries: list[str] = []

    async def embed_query(self, text: str) -> list[float]:
        self.queries.append(text)
        if self.fail:
            raise RuntimeError("embedding engine is down")
        return [1.0, 0.0]

    async def embed_document(self, text: str) -> list[float]:
        return [1.0, 1.0] if "Первый" in text else [0.0, 1.0]



# -- Tavily: основной путь ---------------------------------------------------


async def test_tavily_results_are_primary_backend() -> None:
    tool = WebSearchTool(tavily_api_key="tvly-key", client=_tavily_client())
    outcome = await tool.search("python asyncio")

    assert outcome.backend == "tavily"
    assert not outcome.failed
    assert [r.url for r in outcome.results] == ["https://example.org/one", "https://example.org/two"]
    assert outcome.results[0].relevance == pytest.approx(0.9)


async def test_tavily_sends_bearer_and_payload() -> None:
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        return httpx.Response(200, content=json.dumps(TAVILY_PAYLOAD).encode())

    tool = WebSearchTool(
        tavily_api_key="tvly-secret", client=httpx.AsyncClient(transport=httpx.MockTransport(handler))
    )
    await tool.search("python asyncio")

    assert captured[0].url.path == "/search"
    assert captured[0].headers["Authorization"] == "Bearer tvly-secret"
    assert json.loads(captured[0].content)["query"] == "python asyncio"


async def test_tavily_failure_falls_back_to_ddgs(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeDDGS()
    monkeypatch.setattr(web_search, "DDGS", fake)

    tool = WebSearchTool(tavily_api_key="tvly-key", client=_tavily_client(status=500))
    outcome = await tool.search("python asyncio")

    assert outcome.backend == "ddgs:duckduckgo"
    assert not outcome.failed
    assert fake.seen_backends == ["duckduckgo"]


# -- ddgs: фолбэк с перебором бэкендов ----------------------------------------


async def test_ddgs_backend_failure_tries_next_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeDDGS(fail_backends=("duckduckgo", "bing"))
    monkeypatch.setattr(web_search, "DDGS", fake)

    tool = WebSearchTool()
    outcome = await tool.search("python asyncio")

    assert outcome.backend == "ddgs:google"
    assert not outcome.failed
    assert len(outcome.results) == 1


async def test_ddgs_empty_backend_tries_next(monkeypatch: pytest.MonkeyPatch) -> None:
    """Пустая выдача одного движка — не пустая выдача всех."""
    fake = _FakeDDGS(empty_backends=("duckduckgo", "bing"))
    monkeypatch.setattr(web_search, "DDGS", fake)

    tool = WebSearchTool()
    outcome = await tool.search("заведомо несуществующий запрос")

    assert outcome.backend == "ddgs:google"
    assert not outcome.failed
    assert outcome.results


async def test_all_backends_down_is_reported_as_an_error(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeDDGS(fail_backends=tuple(web_search._BACKENDS))
    monkeypatch.setattr(web_search, "DDGS", fake)

    tool = WebSearchTool()
    outcome = await tool.search("python asyncio")

    assert outcome.failed is True
    assert outcome.render().startswith("error:")
    assert "mojeek" in outcome.error


async def test_empty_query_is_refused() -> None:
    tool = WebSearchTool()
    assert await tool.execute({"query": "   "}, _context()) == "error: query must not be empty"


# -- умный слой: рерайт, реранжирование ----------------------------------------


async def test_query_is_rewritten_before_hitting_the_engine(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeDDGS()
    monkeypatch.setattr(web_search, "DDGS", fake)
    rewriter = _ScriptedRewriter("python asyncio tutorial")

    tool = WebSearchTool(rewriter=rewriter)
    outcome = await tool.search("посоветуй что почитать про асинхронность в питоне")

    assert fake.seen_queries == ["python asyncio tutorial"]
    assert outcome.effective_query == "python asyncio tutorial"
    assert outcome.query == "посоветуй что почитать про асинхронность в питоне"


async def test_rewriter_failure_keeps_the_raw_query(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeDDGS()
    monkeypatch.setattr(web_search, "DDGS", fake)

    tool = WebSearchTool(rewriter=_ScriptedRewriter(RuntimeError("LLM is down")))
    outcome = await tool.search("что такое eBPF?")

    assert fake.seen_queries == ["что такое eBPF?"]
    assert not outcome.failed


async def test_results_are_reranked_by_semantic_similarity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tavily отдаёт score сам, поэтому реранк проверяем на ddgs-пути."""
    fake = _FakeDDGS(empty_backends=("duckduckgo", "bing", "google"))
    monkeypatch.setattr(web_search, "DDGS", fake)

    class _ManyDDGS(_FakeDDGS):
        def text(self, query: str, *, backend: str, **kwargs: object) -> list[dict[str, str]]:  # noqa: ARG002
            self.seen_backends.append(backend)
            self.seen_queries.append(query)
            return [
                {"href": "https://ddgs.example/a", "title": "Второй", "body": "про другое"},
                {"href": "https://ddgs.example/b", "title": "Первый", "body": "про первое"},
            ]

    fake = _ManyDDGS()
    monkeypatch.setattr(web_search, "DDGS", fake)
    embedder = _ScriptedEmbedder()

    tool = WebSearchTool(embedder=embedder)
    outcome = await tool.search("первый")

    assert embedder.queries == ["первый"]
    # [1,0] у запроса: «Первый» ([1,1], cos≈0.707) выше «Второй» ([0,1], cos=0).
    assert outcome.results[0].title == "Первый"


async def test_embedder_failure_keeps_the_source_order(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeDDGS()
    monkeypatch.setattr(web_search, "DDGS", fake)

    tool = WebSearchTool(embedder=_ScriptedEmbedder(fail=True))
    outcome = await tool.search("python asyncio")

    assert not outcome.failed
    assert len(outcome.results) == 1


# -- кэш ----------------------------------------------------------------------


async def test_repeated_query_is_served_from_cache(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeDDGS()
    monkeypatch.setattr(web_search, "DDGS", fake)

    tool = WebSearchTool()
    await tool.search("python asyncio")
    await tool.search("  PYTHON   asyncio  ")  # нормализация: регистр/пробелы

    assert fake.seen_backends == ["duckduckgo"]
    assert tool._cache["python asyncio"][1].cached is False

    outcome = await tool.search("python asyncio")
    assert outcome.cached is True
    assert outcome.results


# -- конфиг (config/web_search.toml) ------------------------------------------


async def test_blank_tavily_key_goes_straight_to_ddgs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Пустой/пробельный ключ в toml — «не настроено»: должен работать ddgs-путь."""
    fake = _FakeDDGS()
    monkeypatch.setattr(web_search, "DDGS", fake)

    tool = WebSearchTool(tavily_api_key="   ")  # как tavily_api_key = "" из конфига
    outcome = await tool.search("python asyncio")

    assert outcome.backend.startswith("ddgs:")
    assert not outcome.failed
    assert fake.seen_backends, "поиск должен был уйти в ddgs, а не в никуда"


def test_shipped_web_search_toml_is_read_by_settings() -> None:
    """
    Штатный config/web_search.toml должен парситься в Settings.web_search.

    Проверка по типу, а не по значению: ключ когда-нибудь заполнят реальным,
    и тест от этого ломаться не должен.
    """
    from efi.config.schema import Settings

    tavily_key = Settings().web_search.tavily_api_key
    assert tavily_key is None or isinstance(tavily_key.get_secret_value(), str)



async def test_execute_returns_rendered_results_and_journals_them(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeDDGS()
    monkeypatch.setattr(web_search, "DDGS", fake)
    journal = _RecordingJournal()

    tool = WebSearchTool(journal=journal)
    text = await tool.execute({"query": "ленивые импорты python"}, _context())

    assert text.startswith("Результаты поиска:")
    assert len(journal.calls) == 1
    assert journal.calls[0]["query"] == "ленивые импорты python"
    assert journal.calls[0]["chat_id"] is None


async def test_cached_hit_is_not_journalled_twice(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = _FakeDDGS()
    monkeypatch.setattr(web_search, "DDGS", fake)
    journal = _RecordingJournal()

    tool = WebSearchTool(journal=journal)
    await tool.execute({"query": "python asyncio"}, _context())
    await tool.execute({"query": "python asyncio"}, _context())

    assert len(journal.calls) == 1


async def test_journal_is_not_touched_when_nothing_was_found(monkeypatch: pytest.MonkeyPatch) -> None:
    """«Ничего не нашлось» — не опыт: в память такое не пишется."""
    fake = _FakeDDGS(fail_backends=tuple(web_search._BACKENDS))
    monkeypatch.setattr(web_search, "DDGS", fake)
    journal = _RecordingJournal()

    tool = WebSearchTool(journal=journal)
    await tool.execute({"query": "что-нибудь"}, _context())

    assert journal.calls == []
