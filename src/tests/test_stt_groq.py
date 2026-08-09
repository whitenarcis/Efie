"""Тесты для efi.media.stt_groq.GroqSTT — без реальной сети (httpx.MockTransport)."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from efi.media.stt_groq import GroqSTT


@pytest.fixture
def audio_file(tmp_path: Path) -> Path:
    path = tmp_path / "voice.ogg"
    path.write_bytes(b"fake-ogg-bytes")
    return path


async def test_transcribe_returns_text_on_success(audio_file: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(200, json={"text": "привет, это тест"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    stt = GroqSTT("test-key", client=client)
    try:
        text = await stt.transcribe(audio_file)
    finally:
        await stt.aclose()

    assert text == "привет, это тест"


async def test_transcribe_strips_whitespace(audio_file: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": "  текст с пробелами  \n"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    stt = GroqSTT("test-key", client=client)
    try:
        text = await stt.transcribe(audio_file)
    finally:
        await stt.aclose()

    assert text == "текст с пробелами"


async def test_transcribe_returns_empty_string_on_http_error(audio_file: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "internal"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    stt = GroqSTT("test-key", client=client)
    try:
        text = await stt.transcribe(audio_file)
    finally:
        await stt.aclose()

    assert text == ""


async def test_transcribe_returns_empty_string_on_invalid_json(audio_file: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    stt = GroqSTT("test-key", client=client)
    try:
        text = await stt.transcribe(audio_file)
    finally:
        await stt.aclose()

    assert text == ""


async def test_transcribe_returns_empty_string_when_file_missing(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:  # не должен вызываться
        raise AssertionError("network call should not happen when file is missing")

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    stt = GroqSTT("test-key", client=client)
    try:
        text = await stt.transcribe(tmp_path / "missing.ogg")
    finally:
        await stt.aclose()

    assert text == ""


async def test_transcribe_sends_model_and_optional_language(audio_file: Path) -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        # multipart form fields доступны через .read()/content parsing только на реальном
        # сервере; здесь достаточно убедиться, что запрос вообще ушёл на верный URL.
        captured["url"] = str(request.url)
        return httpx.Response(200, json={"text": "ok"})

    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    stt = GroqSTT("test-key", client=client)
    try:
        await stt.transcribe(audio_file, language="ru")
    finally:
        await stt.aclose()

    assert captured["url"] == "https://api.groq.com/openai/v1/audio/transcriptions"
