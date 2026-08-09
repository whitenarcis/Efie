"""
Тесты дашборда целиком: поднимается настоящий DashboardServer на свободном
порту, и запросы идут по HTTP — то есть проверяется вся цепочка (маршруты,
сборка снимков, SQL, отдача статики, доступ по токену), а не отдельные
функции в вакууме.

Отдельно проверяется главное свойство дашборда: он ТОЛЬКО читает. Любой
метод кроме GET/HEAD обязан получать отказ, а не выполняться.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from efi.config.schema import (
    DashboardSettings,
    EndpointConfig,
    LLMRolesSettings,
    PathsSettings,
    RoleRoute,
    Settings,
    TelegramSettings,
)
from efi.dashboard.logbus import LogBuffer
from efi.dashboard.metrics import LLMMetricsCollector
from efi.dashboard.server import DashboardServer
from efi.dashboard.snapshot import DashboardContext
from efi.db.core import Database
from efi.db.history_repository import SqliteHistoryRepository
from efi.db.models import MIGRATIONS
from efi.llm.schemas import DiaryEntry, DiaryEntryMetadata, Message, Role
from efi.memory.beliefs import BeliefStore
from efi.memory.diary import Diary
from efi.memory.working_memory import WorkingMemory
from efi.notifications.manager import NotificationManager
from efi.tools.registry import ToolRegistry

_OWNER_ID = 625207005


def _endpoint(model: str) -> EndpointConfig:
    return EndpointConfig(base_url="https://omni.example/v1", api_key="secret", model=model)


def _settings(tmp_path: Path, **dashboard: object) -> Settings:
    return Settings(
        telegram=TelegramSettings(
            api_id=1, api_hash="x", owner_id=_OWNER_ID, chat_labels={-100: "Флудилка"}
        ),
        llm_roles=LLMRolesSettings(
            main=RoleRoute(primary=_endpoint("main-model")),
            fast=RoleRoute(primary=_endpoint("fast-model")),
            vision=RoleRoute(primary=_endpoint("vision-model")),
        ),
        paths=PathsSettings(base_dir=tmp_path, session_name="test"),
        dashboard=DashboardSettings(port=0, **dashboard),  # type: ignore[arg-type]
    )


class _Harness:
    """Поднятый дашборд вместе со всем, на что он смотрит."""

    def __init__(self, server: DashboardServer, base_url: str, context: DashboardContext) -> None:
        self.server = server
        self.base_url = base_url
        self.context = context

    def client(self, **kwargs: object) -> httpx.AsyncClient:
        return httpx.AsyncClient(base_url=self.base_url, timeout=10.0, **kwargs)  # type: ignore[arg-type]


async def _harness(tmp_path: Path, **dashboard: object) -> _Harness:
    settings = _settings(tmp_path, **dashboard)
    database = Database(tmp_path / "efi.db", migrations=MIGRATIONS)
    context = DashboardContext(
        settings=settings,
        logs=LogBuffer(capacity=100, level=logging.INFO),
        metrics=LLMMetricsCollector(history=50),
        started_at=datetime.now(UTC),
        database=database,
        diary=Diary(tmp_path / "diary"),
        working_memory=WorkingMemory(tmp_path / "working_memory.json"),
        history=SqliteHistoryRepository(database),
        beliefs=BeliefStore(database),
        notifications=NotificationManager(worker_count=2),
        tools=ToolRegistry(),
    )
    server = DashboardServer(context, settings.dashboard)
    await server.start()
    return _Harness(server, server.url.rstrip("/"), context)


@pytest.fixture
async def harness(tmp_path: Path):  # type: ignore[no-untyped-def]
    started = await _harness(tmp_path)
    try:
        yield started
    finally:
        await started.server.stop()


# -- страница и статика ------------------------------------------------------


async def test_index_page_is_served(harness: _Harness) -> None:
    async with harness.client() as client:
        response = await client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "EFI" in response.text
    assert "/static/app.js" in response.text


async def test_static_assets_are_served_with_correct_types(harness: _Harness) -> None:
    async with harness.client() as client:
        css = await client.get("/static/styles.css")
        js = await client.get("/static/app.js")

    assert css.status_code == 200
    assert "text/css" in css.headers["content-type"]
    assert js.status_code == 200
    assert "javascript" in js.headers["content-type"]


async def test_static_path_traversal_is_refused(harness: _Harness) -> None:
    async with harness.client() as client:
        for path in ("/static/../server.py", "/static/..%2Fserver.py", "/static/sub/dir.css"):
            response = await client.get(path)
            assert response.status_code == 404, path


# -- только чтение -----------------------------------------------------------


async def test_write_methods_are_refused(harness: _Harness) -> None:
    async with harness.client() as client:
        for method in ("POST", "PUT", "DELETE", "PATCH"):
            response = await client.request(method, "/api/overview")
            assert response.status_code == 405, method


# -- разделы -----------------------------------------------------------------


async def test_overview_reports_state(harness: _Harness) -> None:
    async with harness.client() as client:
        payload = (await client.get("/api/overview")).json()

    assert payload["character_name"] == "Эфи"
    assert payload["telegram"]["owner_id"] == _OWNER_ID
    assert payload["queue"]["worker_count"] == 2
    assert payload["counts"]["messages"] == 0
    assert isinstance(payload["services"], list)
    assert {"enabled", "start_hour", "end_hour", "active_now"} <= payload["quiet_hours"].keys()


async def test_self_state_shows_working_memory_and_beliefs(harness: _Harness) -> None:
    assert harness.context.working_memory is not None
    assert harness.context.beliefs is not None
    await harness.context.working_memory.update_state(emotional_state="устала", energy=0.25)
    await harness.context.working_memory.add_item("вернуться к разговору про Годара")
    await harness.context.beliefs.upsert("нейросети", "полезны, но переоценены", confidence_score=0.8)

    async with harness.client() as client:
        payload = (await client.get("/api/self")).json()

    assert payload["working_memory"]["emotional_state"] == "устала"
    assert payload["working_memory"]["energy"] == 0.25
    assert [item["text"] for item in payload["working_memory"]["items"]] == ["вернуться к разговору про Годара"]
    assert payload["beliefs"][0]["topic"] == "нейросети"


async def test_diary_list_and_entry(harness: _Harness) -> None:
    assert harness.context.diary is not None
    await harness.context.diary.save(
        DiaryEntry(
            id="2026-08-09-vecher",
            metadata=DiaryEntryMetadata(confidence=1.0, usage_count=3, embedding=[0.1, 0.2, 0.3]),
            body="Сегодня он опять спорил со мной про монтаж, и мне это неожиданно понравилось.",
        )
    )

    async with harness.client() as client:
        listing = (await client.get("/api/diary")).json()
        entry = (await client.get("/api/diary/entry", params={"id": "2026-08-09-vecher"})).json()
        missing = await client.get("/api/diary/entry", params={"id": "нет-такой"})
        found = (await client.get("/api/diary", params={"q": "монтаж"})).json()
        empty = (await client.get("/api/diary", params={"q": "квантовая физика"})).json()

    assert listing["total"] == 1
    assert "монтаж" in listing["entries"][0]["preview"]
    assert "embedding" not in listing["entries"][0]  # вектор наружу не отдаётся
    assert listing["entries"][0]["embedding_dim"] == 3
    assert entry["body"].startswith("Сегодня он опять спорил")
    assert entry["is_ground_truth"] is True
    assert missing.status_code == 404
    assert found["total"] == 1
    assert empty["total"] == 0


async def test_chats_and_messages(harness: _Harness) -> None:
    assert harness.context.history is not None
    await harness.context.history.append(-100, Message(role=Role.USER, content="ты чего молчишь"))
    await harness.context.history.append(-100, Message(role=Role.ASSISTANT, content="думаю"))

    async with harness.client() as client:
        chats = (await client.get("/api/chats")).json()
        messages = (await client.get("/api/chats/messages", params={"chat_id": -100})).json()
        broken = await client.get("/api/chats/messages", params={"chat_id": "не-число"})

    assert chats["chats"][0]["chat_id"] == -100
    assert chats["chats"][0]["label"] == "Флудилка"  # подхватился chat_labels из конфига
    assert chats["chats"][0]["message_count"] == 2
    assert [message["content"] for message in messages["messages"]] == ["ты чего молчишь", "думаю"]
    assert broken.status_code == 400


async def test_logs_endpoint_returns_buffered_entries(harness: _Harness) -> None:
    harness.context.logs.emit(
        logging.LogRecord("efi.app", logging.WARNING, __file__, 1, "очередь не разгребается", None, None)
    )

    async with harness.client() as client:
        payload = (await client.get("/api/logs")).json()
        filtered = (await client.get("/api/logs", params={"level": "ERROR"})).json()

    assert payload["entries"][0]["message"] == "очередь не разгребается"
    assert payload["counts"]["WARNING"] == 1
    assert filtered["entries"] == []


async def test_log_stream_delivers_backlog_and_live_entries(harness: _Harness) -> None:
    harness.context.logs.install(logging.getLogger("efi.test.sse"))
    try:
        harness.context.logs.emit(logging.LogRecord("efi.app", logging.INFO, __file__, 1, "старое", None, None))

        async with harness.client() as client, client.stream("GET", "/api/logs/stream") as response:
            assert response.status_code == 200
            assert "text/event-stream" in response.headers["content-type"]

            seen: list[str] = []
            async for line in response.aiter_lines():
                if line.startswith("data: "):
                    seen.append(line)
                    if len(seen) == 1:
                        # Новая запись, появившаяся уже после подписки.
                        harness.context.logs.emit(
                            logging.LogRecord("efi.app", logging.INFO, __file__, 1, "новое", None, None)
                        )
                    if len(seen) == 2:
                        break
    finally:
        harness.context.logs.uninstall()

    assert "старое" in seen[0]
    assert "новое" in seen[1]


async def test_functions_lists_services_tools_and_roles(harness: _Harness) -> None:
    async with harness.client() as client:
        payload = (await client.get("/api/functions")).json()

    roles = {role["role"] for role in payload["llm_roles"]}
    assert roles == {"main", "fast", "background", "vision"}
    assert payload["behavior"]["memory"]["history_limit"] == 30
    assert any(service["name"] == "memory_pulse" for service in payload["services"])
    # Служба, которую никто не запускал, честно помечена как незапущенная.
    assert {service["state"] for service in payload["services"]} <= {"not_started", "disabled"}


async def test_unknown_api_route_is_404(harness: _Harness) -> None:
    async with harness.client() as client:
        assert (await client.get("/api/nonexistent")).status_code == 404


# -- доступ по токену --------------------------------------------------------


async def test_token_gate(tmp_path: Path) -> None:
    started = await _harness(tmp_path, token="s3cret-token")
    try:
        async with started.client() as client:
            anonymous = await client.get("/api/overview")
            by_header = await client.get("/api/overview", headers={"X-Efi-Token": "s3cret-token"})
            wrong = await client.get("/api/overview", headers={"X-Efi-Token": "wrong-token"})
            by_query = await client.get("/api/overview", params={"token": "s3cret-token"})

        assert anonymous.status_code == 401
        assert wrong.status_code == 401
        assert by_header.status_code == 200
        assert by_query.status_code == 200
        assert "efi_dashboard_token=" in by_query.headers.get("set-cookie", "")

        # Cookie, поставленная ссылкой, дальше пускает без параметра — именно
        # так работает EventSource, который своих заголовков не умеет.
        async with started.client(cookies={"efi_dashboard_token": "s3cret-token"}) as client:
            assert (await client.get("/api/overview")).status_code == 200
    finally:
        await started.server.stop()


async def test_token_page_is_shown_to_a_browser(tmp_path: Path) -> None:
    started = await _harness(tmp_path, token="s3cret-token")
    try:
        async with started.client() as client:
            response = await client.get("/", headers={"Accept": "text/html"})
        assert response.status_code == 401
        assert "токен" in response.text
    finally:
        await started.server.stop()
