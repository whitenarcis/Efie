"""
efi/dashboard/api.py

Маршруты `/api/*` — тонкий слой между HTTP и `snapshot.py`/`queries.py`:
разбор query-параметров, зажим лимитов и сериализация. Никакой логики
состояния здесь нет намеренно, поэтому раздел дашборда можно проверить
тестом, вызвав функцию сборки снимка напрямую, без HTTP.

Все маршруты — только чтение. Дашборд ничего не меняет в Эфи: страница,
которая умеет править её память или дёргать поведение, — это уже пульт
управления, и он требует отдельного разговора про то, кто и как получает к
нему доступ.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator

from efi.dashboard import queries, snapshot
from efi.dashboard.http import Handler, Request, Response, StreamResponse, sse_event
from efi.dashboard.snapshot import DashboardContext

logger = logging.getLogger(__name__)

#: Как часто слать комментарий-пульс в SSE-поток, если новых записей нет.
#: Без него незанятое соединение молча закрывают прокси и мобильные сети.
_SSE_HEARTBEAT_SECONDS = 20.0

_LEVEL_NAMES = logging.getLevelNamesMapping()


def _min_level(request: Request) -> int:
    """Числовой порог уровня из параметра `level` (имя уровня, регистр не важен)."""
    raw = request.param("level").strip().upper()
    if not raw or raw == "ALL":
        return 0
    return _LEVEL_NAMES.get(raw, 0)


def build_routes(context: DashboardContext) -> dict[str, Handler]:
    """Таблица маршрутов дашборда: точный путь -> обработчик."""

    async def overview(_: Request) -> Response:
        return Response.json(await snapshot.build_overview(context))

    async def self_state(_: Request) -> Response:
        return Response.json(await snapshot.build_self_state(context))

    async def functions(_: Request) -> Response:
        return Response.json(await snapshot.build_functions(context))

    async def prompts(_: Request) -> Response:
        return Response.json(await snapshot.build_prompts(context))

    async def logs(request: Request) -> Response:
        after_raw = request.param("after_id")
        after_id = int(after_raw) if after_raw.isdigit() else None
        entries = context.logs.snapshot(
            min_level=_min_level(request),
            query=request.param("q"),
            limit=request.int_param("limit", 200, minimum=1, maximum=2000),
            after_id=after_id,
        )
        return Response.json(
            {
                "entries": [entry.as_dict() for entry in entries],
                "counts": context.logs.counts(),
                "total": context.logs.total,
                "buffered": context.logs.buffered,
                "capacity": context.logs.capacity,
                "last_id": context.logs.last_id,
                "dropped": context.logs.dropped,
            }
        )

    async def logs_stream(request: Request) -> StreamResponse:
        min_level = _min_level(request)
        query = request.param("q")
        return StreamResponse(stream=_log_stream(context, min_level=min_level, query=query))

    async def metrics(request: Request) -> Response:
        limit = request.int_param("limit", 50, minimum=1, maximum=200)
        return Response.json(context.metrics.snapshot(recent_limit=limit))

    async def diary(request: Request) -> Response:
        return Response.json(
            await snapshot.build_diary_list(
                context,
                query=request.param("q"),
                limit=request.int_param("limit", 50, minimum=1, maximum=200),
                offset=request.int_param("offset", 0),
            )
        )

    async def diary_entry(request: Request) -> Response:
        entry_id = request.param("id")
        if not entry_id:
            return Response.error(400, "не указан id записи")
        entry = await snapshot.build_diary_entry(context, entry_id)
        if entry is None:
            return Response.error(404, "запись дневника не найдена")
        return Response.json(entry)

    async def memory(request: Request) -> Response:
        return Response.json(
            await snapshot.build_memory(
                context,
                limit=request.int_param("limit", 100, minimum=1, maximum=queries.MAX_ROWS),
                query=request.param("q"),
            )
        )

    async def people(request: Request) -> Response:
        return Response.json(
            await snapshot.build_people(
                context, limit=request.int_param("limit", 100, minimum=1, maximum=queries.MAX_ROWS)
            )
        )

    async def projects(request: Request) -> Response:
        return Response.json(
            await snapshot.build_projects(
                context, limit=request.int_param("limit", 50, minimum=1, maximum=queries.MAX_ROWS)
            )
        )

    async def chats(request: Request) -> Response:
        return Response.json(
            await snapshot.build_chats(
                context, limit=request.int_param("limit", 100, minimum=1, maximum=queries.MAX_ROWS)
            )
        )

    async def chat_messages(request: Request) -> Response:
        raw_chat_id = request.param("chat_id")
        try:
            chat_id = int(raw_chat_id)
        except ValueError:
            return Response.error(400, "не указан или некорректен chat_id")
        before_raw = request.param("before_id")
        before_id: int | None = None
        if before_raw:
            try:
                before_id = int(before_raw)
            except ValueError:
                before_id = None
        return Response.json(
            await snapshot.build_chat_messages(
                context,
                chat_id,
                limit=request.int_param("limit", 100, minimum=1, maximum=queries.MAX_ROWS),
                before_id=before_id,
            )
        )

    return {
        "/api/overview": overview,
        "/api/self": self_state,
        "/api/functions": functions,
        "/api/prompts": prompts,
        "/api/logs": logs,
        "/api/logs/stream": logs_stream,
        "/api/metrics": metrics,
        "/api/diary": diary,
        "/api/diary/entry": diary_entry,
        "/api/memory": memory,
        "/api/people": people,
        "/api/projects": projects,
        "/api/chats": chats,
        "/api/chats/messages": chat_messages,
    }


async def _log_stream(context: DashboardContext, *, min_level: int, query: str) -> AsyncIterator[bytes]:
    """
    Живая лента логов.

    Подписка оформляется ДО отправки накопленного хвоста: иначе между
    снимком и подпиской существует окно, в котором записи теряются
    безвозвратно — а лента логов, молча пропускающая именно ту строку, ради
    которой её открыли, хуже, чем отсутствие ленты. Возможный дубль на
    границе клиент отсекает сам по возрастающему `id`.
    """
    with context.logs.subscribe() as queue:
        backlog = context.logs.snapshot(min_level=min_level, query=query, limit=200)
        for entry in backlog:
            yield sse_event(entry.as_dict())
        last_id = backlog[-1].id if backlog else 0

        while True:
            try:
                entry = await asyncio.wait_for(queue.get(), timeout=_SSE_HEARTBEAT_SECONDS)
            except TimeoutError:
                yield b": ping\n\n"
                continue
            if entry.id <= last_id or entry.level_no < min_level or not entry.matches(query):
                continue
            last_id = entry.id
            yield sse_event(entry.as_dict())


__all__ = ["build_routes"]
