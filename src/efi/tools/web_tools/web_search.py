"""
efi/tools/web_tools/web_search.py

Веб-поиск для Efie. Основной путь — Tavily Search API (поисковик, заточенный
под LLM-агентов: сам переформулирует и переранжирует выдачу, отдаёт чистый
JSON). Бесплатный тир — 1000 кредитов/мес; когда ключ не настроен или Tavily
недоступен, поиск прозрачно падает обратно на локальный метапоисковик ddgs с
цепочкой бэкендов (duckduckgo -> bing -> google -> mojeek): капча или лимит
одного движка не выглядит снаружи как «в интернете пусто».

Поверх источника работает общий конвейер:
    1. Рерайтинг запроса (опционально): разговорный вопрос («а что там с
       ценами на эфир?») превращается FAST-моделью в короткий ключевой
       запрос. Любая ошибка/таймаут LLM — тихо работаем с сырым текстом:
       поиск не должен зависеть от того, жив ли роутер.
    2. Реранжирование (опционально): если есть LocalEmbeddingEngine,
       сниппеты сортируются по косинусной близости к запросу. Сбой
       эмбеддингов — не повод терять выдачу: остаётся порядок источника.
    3. TTL-кэш: повторный запрос в пределах окна отдаётся из кэша без
       похода в сеть — это ещё и экономия кредитов Tavily.

Синхронный клиент ddgs уходит в пул потоков (asyncio.to_thread), чтобы не
блокировать асинхронный event loop бота.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

import httpx
from ddgs import DDGS

from efi.tools.base import Tool, ToolContext

if TYPE_CHECKING:
    from efi.llm.schemas import Response

logger = logging.getLogger(__name__)


class SearchJournal(Protocol):
    """Куда откладывается сам факт похода в интернет."""

    async def record_web_lookup(
        self, *, query: str, digest: str, chat_id: int | None = None, thread_id: int | None = None
    ) -> int: ...


class SearchQueryRewriter(Protocol):
    """Сужение LLMRouter до того, что нужно для рерайтинга запроса."""

    async def chat(self, role: Any, params: Any, session: Any) -> Response: ...


class SearchEmbedder(Protocol):
    """Сужение LocalEmbeddingEngine до того, что нужно для реранжирования."""

    async def embed_query(self, text: str) -> list[float]: ...

    async def embed_document(self, text: str) -> list[float]: ...


_TAVILY_SEARCH_URL = "https://api.tavily.com/search"
_TAVILY_TIMEOUT_SECONDS = 15.0

_REQUEST_TIMEOUT_SECONDS = 12.0
_REWRITE_TIMEOUT_SECONDS = 6.0
_MAX_RESULTS = 5
_SNIPPET_MAX_LENGTH = 220

#: Цепочка бэкендов ddgs: пустой/за капчей первый движок не должен выглядеть
#: снаружи как «в интернете пусто». Список живых бэкендов зависит от версии
#: пакета (9.x: brave, duckduckgo, google, grokipedia, mojeek, startpage,
#: wikipedia, yahoo) — несуществующий бэкенд ddgs пишет warning и игнорирует.
_BACKENDS = ("duckduckgo", "google", "brave", "mojeek")

#: Результаты с релевантностью ниже порога отсекаются только тогда, когда
#: после фильтра что-то остаётся: на экзотическом запросе порог может
#: зарезать и единственные живые результаты.
_RERANK_MIN_SIMILARITY = 0.25

_CACHE_TTL_SECONDS = 900.0
_CACHE_MAX_ENTRIES = 128

_REWRITE_SYSTEM_PROMPT = (
    "Ты превращаешь вопросы в поисковые запросы. "
    "Перепиши вопрос пользователя в короткую строку из ключевых слов для поисковика: "
    "без обращений, без вопросительных слов, без пунктуации в конце. "
    "Если вопрос на русском, оставь русский; если про англоязычную тему — переведи в английские ключевые слова. "
    "Ответ — ТОЛЬКО строка запроса, без пояснений и кавычек."
)



@dataclass(slots=True, frozen=True)
class SearchResult:
    """Один результат выдачи."""

    title: str
    snippet: str
    url: str
    relevance: float = 0.0

    def render(self) -> str:
        return f"- {self.title}: {self.snippet} ({self.url})"


@dataclass(slots=True, frozen=True)
class SearchOutcome:
    """Итог поиска в структурированном виде."""

    query: str
    results: list[SearchResult] = field(default_factory=list)
    error: str = ""
    backend: str = ""
    #: Истинный запрос, ушедший в поисковик, если рерайтинг его изменил.
    effective_query: str = ""
    #: Результат выдан из кэша — в журнал его записывать второй раз не нужно.
    cached: bool = False

    @property
    def failed(self) -> bool:
        return bool(self.error)

    def render(self) -> str:
        if self.failed:
            return f"error: поиск не удался: {self.error}"
        if not self.results:
            return f"По запросу {self.query!r} ничего не нашлось."
        return "Результаты поиска:\n" + "\n".join(result.render() for result in self.results)

    def digest(self) -> str:
        return "\n".join(result.render() for result in self.results)


class WebSearchTool(Tool):
    """Ищет в интернете: Tavily (если есть ключ) -> ddgs с цепочкой бэкендов."""

    name = "web_search"
    description = (
        "Ищет в интернете по запросу и возвращает список результатов (заголовок + краткое описание + ссылка). "
        "Подходит для общих фактических вопросов, новостей, 'погугли X'. НЕ подходит для точных live-данных "
        "вроде текущей погоды (для погоды используй get_weather) или курсов валют — сниппеты поиска могут "
        "быть устаревшими."
    )
    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "Поисковый запрос"},
        },
        "required": ["query"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        journal: SearchJournal | None = None,
        rewriter: SearchQueryRewriter | None = None,
        embedder: SearchEmbedder | None = None,
        tavily_api_key: str = "",
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._journal = journal
        self._rewriter = rewriter
        self._embedder = embedder
        self._tavily_api_key = tavily_api_key.strip()
        self._client = client
        #: Нормализованный запрос -> (монотонное время, SearchOutcome).
        self._cache: dict[str, tuple[float, SearchOutcome]] = {}

    async def aclose(self) -> None:
        self._cache.clear()
        if self._client is not None:
            await self._client.aclose()

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        query = str(arguments.get("query", "")).strip()
        if not query:
            return "error: query must not be empty"

        return (await self.search(query, journal_context=context)).render()

    async def search(self, query: str, *, journal_context: ToolContext | None = None) -> SearchOutcome:
        cache_key = " ".join(query.lower().split())
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        effective_query = await self._rewrite(query)

        outcome: SearchOutcome | None = None
        if self._tavily_api_key:
            outcome = await self._search_tavily(query, effective_query)
        if outcome is None or outcome.failed or not outcome.results:
            if outcome is not None and outcome.failed:
                logger.warning("web_search: Tavily failed (%s), falling back to ddgs", outcome.error)
            outcome = await self._search_ddgs(query, effective_query, previous=outcome)

        if outcome.results and self._embedder is not None:
            outcome = await self._rerank(outcome)

        if not outcome.failed:
            self._cache_put(cache_key, outcome)

        if journal_context is not None:
            await self._journal_lookup(outcome, journal_context)
        return outcome

    # -- источники ----------------------------------------------------------

    async def _search_tavily(self, query: str, effective_query: str) -> SearchOutcome:
        """Tavily Search API. Любая ошибка — фолбэк на ddgs, не пустая выдача."""
        client = self._client or httpx.AsyncClient()
        try:
            response = await client.post(
                _TAVILY_SEARCH_URL,
                headers={"Authorization": f"Bearer {self._tavily_api_key}"},
                json={
                    "query": effective_query,
                    "max_results": _MAX_RESULTS,
                    "search_depth": "basic",
                    "topic": "general",
                },
                timeout=_TAVILY_TIMEOUT_SECONDS,
            )
            response.raise_for_status()
            payload = response.json()
        except Exception as exc:
            return SearchOutcome(query=query, error=str(exc), backend="tavily", effective_query=effective_query)

        results: list[SearchResult] = []
        for item in payload.get("results", []):
            title = str(item.get("title", "")).strip()
            url = str(item.get("url", "")).strip()
            snippet = str(item.get("content", "")).strip()[:_SNIPPET_MAX_LENGTH]
            if title and url:
                try:
                    relevance = float(item.get("score", 0.0))
                except (TypeError, ValueError):
                    relevance = 0.0
                results.append(SearchResult(title=title, snippet=snippet, url=url, relevance=relevance))

        return SearchOutcome(
            query=query,
            results=results,
            backend="tavily",
            effective_query=effective_query,
            error="" if results else "tavily вернул пустую выдачу",
        )

    async def _search_ddgs(
        self, query: str, effective_query: str, *, previous: SearchOutcome | None = None
    ) -> SearchOutcome:
        """Бесплатный фолбэк: ddgs с перебором бэкендов."""

        def _fetch(backend: str) -> list[dict[str, Any]]:
            with DDGS(timeout=int(_REQUEST_TIMEOUT_SECONDS)) as ddgs:
                return list(
                    ddgs.text(
                        effective_query,
                        region="ru-ru",
                        safesearch="moderate",
                        max_results=_MAX_RESULTS,
                        backend=backend,
                    )
                )

        error = previous.error if previous is not None else ""
        results: list[SearchResult] = []
        last_backend = _BACKENDS[0]
        for backend in _BACKENDS:
            last_backend = backend
            try:
                raw_results = await asyncio.to_thread(_fetch, backend)
            except Exception as exc:
                logger.debug("web_search: ddgs backend %s failed for %r: %s", backend, effective_query, exc)
                error = str(exc)
                continue

            error = ""
            results = self._parse_ddgs(raw_results)
            if results:
                break
            # Пустая выдача одного движка — не пустая выдача всех: пробуем следующий.

        return SearchOutcome(
            query=query,
            results=results,
            error=error if not results else "",
            backend=f"ddgs:{last_backend}",
            effective_query=effective_query,
        )

    @staticmethod
    def _parse_ddgs(raw_results: list[dict[str, Any]]) -> list[SearchResult]:
        results: list[SearchResult] = []
        for item in raw_results:
            # Ключи варьируются между бэкендами ddgs — берём первое встреченное.
            url = next((str(item[key]).strip() for key in ("href", "url", "link") if item.get(key)), "")
            title = str(item.get("title", "")).strip()
            snippet = next(
                (str(item[key]).strip() for key in ("body", "description", "snippet") if item.get(key)), ""
            )[:_SNIPPET_MAX_LENGTH]

            if url and title:
                results.append(SearchResult(title=title, snippet=snippet, url=url))
        return results

    # -- умный слой ----------------------------------------------------------

    async def _rewrite(self, query: str) -> str:
        """Разговорный вопрос -> ключевой поисковый запрос. Без LLM — сам запрос."""
        if self._rewriter is None:
            return query
        from efi.config.schema import TaskRole
        from efi.llm.schemas import LLMParams, Message, Role, Session

        params = LLMParams(model="", system_prompt=_REWRITE_SYSTEM_PROMPT, max_output_tokens=100)
        session = Session(messages=[Message(role=Role.USER, content=query)])
        try:
            response = await asyncio.wait_for(
                self._rewriter.chat(TaskRole.FAST, params, session), timeout=_REWRITE_TIMEOUT_SECONDS
            )
        except Exception as exc:  # любой сбой рерайта не должен ронять поиск
            logger.debug("web_search: query rewrite failed for %r (%s), using raw query", query, exc)
            return query

        rewritten = response.choices[0].message.content.strip().strip('"') if response.choices else ""
        if not rewritten or len(rewritten) > 300:
            return query
        return rewritten

    async def _rerank(self, outcome: SearchOutcome) -> SearchOutcome:
        """Сортировка сниппетов по семантической близости к запросу."""
        assert self._embedder is not None
        try:
            query_vector = await self._embedder.embed_query(outcome.effective_query or outcome.query)
            document_vectors = await asyncio.gather(
                *(self._embedder.embed_document(f"{r.title} {r.snippet}") for r in outcome.results)
            )
        except Exception as exc:
            logger.debug("web_search: rerank failed for %r (%s), keeping source order", outcome.query, exc)
            return outcome

        scored = sorted(
            (
                (result, _cosine_similarity(query_vector, vector) if vector else 0.0)
                for result, vector in zip(outcome.results, document_vectors, strict=True)
            ),
            key=lambda pair: pair[1],
            reverse=True,
        )
        # На экзотическом запросе порог может зарезать всё — тогда лучше
        # отдать выдачу как есть, в порядке источника, чем «ничего не нашлось».
        kept = [(r, s) for r, s in scored if s >= _RERANK_MIN_SIMILARITY] or scored
        return SearchOutcome(
            query=outcome.query,
            results=[
                SearchResult(title=r.title, snippet=r.snippet, url=r.url, relevance=round(score, 4))
                for r, score in kept[:_MAX_RESULTS]
            ],
            backend=outcome.backend,
            effective_query=outcome.effective_query,
        )

    # -- кэш -----------------------------------------------------------------

    def _cache_get(self, key: str) -> SearchOutcome | None:
        entry = self._cache.get(key)
        if entry is None:
            return None
        stored_at, outcome = entry
        if time.monotonic() - stored_at > _CACHE_TTL_SECONDS:
            self._cache.pop(key, None)
            return None
        return SearchOutcome(
            query=outcome.query,
            results=outcome.results,
            backend=outcome.backend,
            effective_query=outcome.effective_query,
            cached=True,
        )

    def _cache_put(self, key: str, outcome: SearchOutcome) -> None:
        if len(self._cache) >= _CACHE_MAX_ENTRIES:
            now = time.monotonic()
            self._cache = {k: v for k, v in self._cache.items() if now - v[0] <= _CACHE_TTL_SECONDS}
            if len(self._cache) >= _CACHE_MAX_ENTRIES:
                self._cache.clear()
        self._cache[key] = (time.monotonic(), outcome)

    async def _journal_lookup(self, outcome: SearchOutcome, context: ToolContext) -> None:
        if self._journal is None:
            return
        if outcome.cached:
            return  # факт этого поиска в памяти уже есть — дублировать нечего
        if not outcome.results:
            return  # «Ничего не нашлось» — не опыт: в память такое не пишется
        thread_id = context.notification.payload.get("thread_id")
        try:
            await self._journal.record_web_lookup(
                query=outcome.query,
                digest=outcome.digest(),
                chat_id=context.chat_id,
                thread_id=thread_id if isinstance(thread_id, int) else None,
            )
        except Exception:
            logger.warning("web_search: failed to journal lookup for %r", outcome.query, exc_info=True)


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b, strict=False))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return float(dot / (norm_a * norm_b))


__all__ = ["SearchJournal", "SearchOutcome", "SearchResult", "WebSearchTool"]