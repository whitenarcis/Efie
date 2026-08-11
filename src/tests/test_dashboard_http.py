"""
Тесты собственного HTTP-слоя дашборда (efi.dashboard.http).

Своего сервера в проекте раньше не было, поэтому проверяется именно то, что
обычно ломается в самописном HTTP: разбор строки запроса и заголовков,
keep-alive, HEAD без тела, лимиты, и SSE-поток, который обязан выживать
после того, как клиент отвалился на середине.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import suppress

from efi.dashboard.http import HttpServer, Request, Response, StreamResponse, sse_event


async def _start(handler) -> tuple[HttpServer, int]:  # type: ignore[no-untyped-def]
    server = HttpServer("127.0.0.1", 0, handler)
    await server.start()
    return server, server.port


async def _request(port: int, raw: bytes, *, read_all: bool = True) -> bytes:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(raw)
    await writer.drain()
    data = await reader.read(-1) if read_all else await reader.read(4096)
    writer.close()
    await writer.wait_closed()
    return data


async def test_get_returns_json_body() -> None:
    async def handler(request: Request) -> Response:
        return Response.json({"path": request.path, "who": request.param("who")})

    server, port = await _start(handler)
    try:
        # Кириллица в query приходит percent-encoded — она обязана дожить до
        # обработчика распакованной, а не превратиться в мусор.
        raw = await _request(
            port,
            b"GET /api/hello?who=%D0%AD%D1%84%D0%B8 HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n",
        )
    finally:
        await server.stop()

    assert b"200 OK" in raw
    assert b'"path": "/api/hello"' in raw
    assert "Эфи".encode() in raw


async def test_head_has_headers_but_no_body() -> None:
    async def handler(_: Request) -> Response:
        return Response.text("тело ответа")

    server, port = await _start(handler)
    try:
        raw = await _request(port, b"HEAD / HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
    finally:
        await server.stop()

    head, _, body = raw.partition(b"\r\n\r\n")
    assert b"Content-Length" in head
    assert body == b""


async def test_keep_alive_serves_two_requests_over_one_connection() -> None:
    seen: list[str] = []

    async def handler(request: Request) -> Response:
        seen.append(request.path)
        return Response.text("ok")

    server, port = await _start(handler)
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"GET /first HTTP/1.1\r\nHost: x\r\n\r\n")
        await writer.drain()
        first = await reader.readuntil(b"ok")
        writer.write(b"GET /second HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
        await writer.drain()
        second = await reader.read(-1)
        writer.close()
        await writer.wait_closed()
    finally:
        await server.stop()

    assert b"keep-alive" in first
    assert b"ok" in second
    assert seen == ["/first", "/second"]


async def test_malformed_request_line_gets_400() -> None:
    async def handler(_: Request) -> Response:  # pragma: no cover — не должен быть вызван
        return Response.text("ok")

    server, port = await _start(handler)
    try:
        raw = await _request(port, b"NONSENSE\r\nHost: x\r\n\r\n")
    finally:
        await server.stop()

    assert b"400" in raw


async def test_handler_exception_becomes_500_and_keeps_server_alive() -> None:
    async def handler(request: Request) -> Response:
        if request.path == "/boom":
            raise RuntimeError("сломалось")
        return Response.text("живой")

    server, port = await _start(handler)
    try:
        broken = await _request(port, b"GET /boom HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
        alive = await _request(port, b"GET /ok HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
    finally:
        await server.stop()

    assert b"500" in broken
    assert "живой".encode() in alive


async def test_stream_response_is_written_and_closed() -> None:
    async def numbers() -> AsyncIterator[bytes]:
        for value in range(3):
            yield sse_event({"value": value})

    async def handler(_: Request) -> StreamResponse:
        return StreamResponse(stream=numbers())

    server, port = await _start(handler)
    try:
        raw = await _request(port, b"GET /api/logs/stream HTTP/1.1\r\nHost: x\r\n\r\n")
    finally:
        await server.stop()

    assert b"text/event-stream" in raw
    assert raw.count(b"data: ") == 3


async def test_stream_survives_client_disconnect() -> None:
    """
    Обрыв на середине потока — норма (закрыли вкладку). Сервер обязан
    пережить его и продолжить обслуживать следующие запросы.
    """
    stopped = asyncio.Event()

    async def endless() -> AsyncIterator[bytes]:
        try:
            while True:
                yield sse_event({"tick": 1})
                await asyncio.sleep(0.01)
        finally:
            stopped.set()

    async def handler(request: Request) -> Response | StreamResponse:
        if request.path == "/stream":
            return StreamResponse(stream=endless())
        return Response.text("ещё жив")

    server, port = await _start(handler)
    try:
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(b"GET /stream HTTP/1.1\r\nHost: x\r\n\r\n")
        await writer.drain()
        await reader.read(200)
        writer.close()
        with suppress(ConnectionResetError, BrokenPipeError):
            await writer.wait_closed()

        # Главная проверка: генератор потока действительно доведён до конца
        # (сработал его finally), а не остался висеть навсегда.
        await asyncio.wait_for(stopped.wait(), timeout=5.0)
        alive = await _request(port, b"GET /ok HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")
    finally:
        await server.stop()

    assert "ещё жив".encode() in alive


async def test_stop_cancels_open_connections() -> None:
    """
    Регрессия: `Server.close()` закрывает только слушающий сокет, а уже
    принятые соединения продолжают жить. Брошенный обработчик уносит с собой
    открытое соединение с SQLite, а под каждое из них aiosqlite держит
    НЕ-daemon-поток — процесс после этого просто не завершается.
    """
    released = asyncio.Event()

    async def endless() -> AsyncIterator[bytes]:
        try:
            while True:
                yield sse_event({"tick": 1})
                await asyncio.sleep(0.05)
        finally:
            released.set()

    async def handler(_: Request) -> StreamResponse:
        return StreamResponse(stream=endless())

    server, port = await _start(handler)
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    writer.write(b"GET /stream HTTP/1.1\r\nHost: x\r\n\r\n")
    await writer.drain()
    await reader.read(120)

    await asyncio.wait_for(server.stop(), timeout=5.0)
    assert released.is_set()

    writer.close()
    with suppress(ConnectionResetError, BrokenPipeError):
        await writer.wait_closed()


def test_sse_event_splits_multiline_payload() -> None:
    """Перевод строки внутри данных обязан разъехаться на несколько `data:`, иначе браузер обрежет событие."""
    raw = sse_event("первая\nвторая").decode()
    assert raw == "data: первая\ndata: вторая\n\n"


def test_int_param_clamps_and_survives_garbage() -> None:
    request = Request(method="GET", path="/", query={"limit": ["9999"], "bad": ["abc"]}, headers={})
    assert request.int_param("limit", 50, minimum=1, maximum=200) == 200
    assert request.int_param("bad", 50) == 50
    assert request.int_param("missing", 7) == 7


def test_cookie_parsing() -> None:
    request = Request(
        method="GET", path="/", query={}, headers={"cookie": "a=1; efi_dashboard_token=secret; b=2"}
    )
    assert request.cookie("efi_dashboard_token") == "secret"
    assert request.cookie("nope") is None
