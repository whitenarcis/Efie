"""
Тесты веб-поиска (efi.tools.web_tools.web_search).

Главный из них — регрессия на баг, из-за которого поиск не работал ВООБЩЕ:
DuckDuckGo отдаёт результаты редиректом `//duckduckgo.com/l/?uddg=<адрес>`,
а парсер выбрасывал любую ссылку со словом "duckduckgo.com" в адресе как
внутреннюю навигацию — то есть все результаты подряд. Наружу это выглядело
как вечное «ничего не нашлось» и «yielded nothing useful» в логах фоновых
исследователей.

Вёрстка движков здесь зафиксирована образцами: живьём их в тестах дёргать
нельзя (сеть, лимиты, капчи), а именно разбор разметки и ломался.
"""

from __future__ import annotations

from urllib.parse import quote

import httpx
import pytest

from efi.notifications.schemas import Notification, NotificationType
from efi.tools.base import ToolContext
from efi.tools.web_tools.web_search import (
    WebSearchTool,
    parse_duckduckgo_html,
    parse_duckduckgo_lite,
    parse_mojeek,
)


def _redirect(target: str) -> str:
    return f"//duckduckgo.com/l/?uddg={quote(target, safe='')}&rut=abc123"


DDG_LITE_HTML = f"""
<html><body>
<table>
  <tr><td valign="top">1.&nbsp;</td>
      <td><a rel="nofollow" href="{_redirect('https://docs.python.org/3/library/asyncio.html')}"
             class="result-link">asyncio — Asynchronous I/O</a></td></tr>
  <tr><td>&nbsp;</td><td class="result-snippet">asyncio is a library to write concurrent code
      using the async/await syntax.</td></tr>
  <tr><td>&nbsp;</td><td class="link-text">docs.python.org/3/library/asyncio.html</td></tr>
  <tr class="result-sep"></tr>
  <tr><td valign="top">2.&nbsp;</td>
      <td><a rel="nofollow" href="{_redirect('https://realpython.com/async-io-python/')}"
             class="result-link">Async IO in Python</a></td></tr>
  <tr><td>&nbsp;</td><td class="result-snippet">A complete walkthrough.</td></tr>
</table>
<a href="/settings">Settings</a>
</body></html>
"""

DDG_FULL_HTML = f"""
<html><body>
<div class="result results_links results_links_deep web-result">
  <h2 class="result__title">
    <a rel="nofollow" class="result__a" href="{_redirect('https://example.org/article')}">Заголовок статьи</a>
  </h2>
  <a class="result__snippet" href="{_redirect('https://example.org/article')}">Краткое описание статьи.</a>
</div>
<div class="result results_links">
  <h2 class="result__title">
    <a rel="nofollow" class="result__a" href="{_redirect('https://second.example/page')}">Вторая ссылка</a>
  </h2>
  <a class="result__snippet">Ещё одно описание.</a>
</div>
</body></html>
"""

MOJEEK_HTML = """
<html><body>
<ul class="results-standard">
  <li><h2><a href="https://mojeek.example/one">Первый результат</a></h2>
      <p class="s">Описание первого результата.</p></li>
  <li><h2><a href="https://mojeek.example/two">Второй результат</a></h2>
      <p class="s">Описание второго.</p></li>
</ul>
</body></html>
"""

ANOMALY_HTML = (
    "<html><body><p>Unfortunately, bots use DuckDuckGo too.</p>"
    "<script src=anomaly.js></script></body></html>"
)


def _context() -> ToolContext:
    return ToolContext(notification=Notification(type=NotificationType.NIGHTLY_TASK, message="тест"))


# -- разбор вёрстки ----------------------------------------------------------


def test_lite_results_survive_the_redirect_wrapper() -> None:
    """Регрессия: раньше здесь получался пустой список — все ссылки шли через duckduckgo.com."""
    results = parse_duckduckgo_lite(DDG_LITE_HTML)

    assert [result.url for result in results] == [
        "https://docs.python.org/3/library/asyncio.html",
        "https://realpython.com/async-io-python/",
    ]
    assert results[0].title == "asyncio — Asynchronous I/O"
    assert "concurrent code" in results[0].snippet


def test_lite_snippet_is_taken_from_the_snippet_cell_not_the_next_row() -> None:
    """Между ссылкой и описанием бывает строка с адресом — она не должна подменять описание."""
    results = parse_duckduckgo_lite(DDG_LITE_HTML)
    assert "docs.python.org/3/library" not in results[0].snippet


def test_html_layout_is_parsed() -> None:
    results = parse_duckduckgo_html(DDG_FULL_HTML)

    assert [result.url for result in results] == ["https://example.org/article", "https://second.example/page"]
    assert results[0].title == "Заголовок статьи"
    assert results[0].snippet == "Краткое описание статьи."


def test_mojeek_layout_is_parsed() -> None:
    results = parse_mojeek(MOJEEK_HTML)

    assert [result.url for result in results] == ["https://mojeek.example/one", "https://mojeek.example/two"]
    assert results[1].snippet == "Описание второго."


def test_internal_navigation_is_still_dropped() -> None:
    """Распаковка редиректов не должна тащить в выдачу настройки и «следующую страницу»."""
    html = '<html><body><a class="result-link" href="//duckduckgo.com/settings">Settings</a></body></html>'
    assert parse_duckduckgo_lite(html) == []


# -- перебор источников ------------------------------------------------------


async def test_falls_back_to_the_next_backend_when_the_first_is_blocked() -> None:
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        if request.url.host == "html.duckduckgo.com":
            return httpx.Response(403)
        if request.url.host == "lite.duckduckgo.com":
            return httpx.Response(200, text=DDG_LITE_HTML)
        return httpx.Response(200, text=MOJEEK_HTML)

    tool = WebSearchTool(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    outcome = await tool.search("python asyncio")

    assert seen[:2] == ["html.duckduckgo.com", "lite.duckduckgo.com"]
    assert outcome.backend == "ddg-lite"
    assert len(outcome.results) == 2
    assert outcome.failed is False


async def test_anti_bot_page_is_treated_as_failure_not_as_empty_results() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "www.mojeek.com":
            return httpx.Response(200, text=MOJEEK_HTML)
        return httpx.Response(200, text=ANOMALY_HTML)

    tool = WebSearchTool(client=httpx.AsyncClient(transport=httpx.MockTransport(handler)))
    outcome = await tool.search("что угодно")

    assert outcome.backend == "mojeek"
    assert len(outcome.results) == 2


async def test_all_backends_down_is_reported_as_an_error() -> None:
    tool = WebSearchTool(
        client=httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(503)))
    )
    outcome = await tool.search("python asyncio")

    assert outcome.failed is True
    assert "503" in outcome.error
    assert outcome.render().startswith("error:")


async def test_empty_results_everywhere_is_not_an_error() -> None:
    """Пустая выдача — честный ответ «не нашлось», а не поломка: их нельзя путать."""
    tool = WebSearchTool(
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, text="<html><body></body></html>"))
        )
    )
    outcome = await tool.search("заведомо несуществующий запрос")

    assert outcome.failed is False
    assert outcome.results == []
    assert "ничего не нашлось" in outcome.render()


# -- интеграция с инструментом ----------------------------------------------


async def test_execute_returns_rendered_results_and_journals_them() -> None:
    recorded: list[tuple[str, str]] = []

    class _Journal:
        async def record_web_lookup(
            self, *, query: str, digest: str, chat_id: int | None = None, thread_id: int | None = None
        ) -> int:
            recorded.append((query, digest))
            return 1

    tool = WebSearchTool(
        client=httpx.AsyncClient(
            transport=httpx.MockTransport(lambda request: httpx.Response(200, text=DDG_FULL_HTML))
        ),
        journal=_Journal(),
    )
    text = await tool.execute({"query": "статья"}, _context())

    assert text.startswith("Результаты поиска:")
    assert "https://example.org/article" in text
    assert len(recorded) == 1
    assert recorded[0][0] == "статья"


async def test_empty_query_is_refused() -> None:
    tool = WebSearchTool(client=httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(200))))
    assert await tool.execute({"query": "   "}, _context()) == "error: query must not be empty"


@pytest.mark.parametrize("status", [403, 429, 500])
async def test_journal_is_not_touched_when_nothing_was_found(status: int) -> None:
    """«Ничего не нашлось» — не опыт: в память такое не пишется."""

    class _Journal:
        def __init__(self) -> None:
            self.calls = 0

        async def record_web_lookup(self, **kwargs: object) -> int:
            self.calls += 1
            return 1

    journal = _Journal()
    tool = WebSearchTool(
        client=httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(status))),
        journal=journal,
    )
    await tool.execute({"query": "что-нибудь"}, _context())

    assert journal.calls == 0
