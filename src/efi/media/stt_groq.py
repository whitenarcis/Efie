"""
efi/media/stt_groq.py

Прямой клиент Groq Speech-to-Text (Whisper-large-v3) поверх httpx.AsyncClient.

Отдельный модуль, а не очередной провайдер внутри efi/llm/providers/: Groq
STT — REST-эндпоинт с multipart/form-data загрузкой файла и собственным
JSON-ответом ({"text": "..."}), а не OpenAI Chat Completions контракт, под
который заточен efi.llm.providers.openai_compatible.OpenAICompatibleProvider
(и, соответственно, LLMRouter.transcribe_audio). LLMRouter остаётся
провайдер-агностичным основным путём распознавания речи (роль VISION,
см. efi/telegram/media/voice.py и video.py); GroqSTT — специализированный
прямой клиент, который efi/telegram/handlers.py использует ПЕРВЫМ, когда явно
настроен ключ Groq (efi.config.schema.SttSettings.groq_api_key), откатываясь
на LLMRouter при отсутствии ключа или пустом результате.

Тот же стиль, что и у efi.tools.web_tools.web_search.WebSearchTool:
инжектируемый httpx.AsyncClient, собственный aclose(), сетевые/HTTP ошибки
превращаются в пустой результат с предупреждением в лог, а не исключение —
обработка входящего голосового/видео-кружка не должна падать целиком из-за
сбоя одного конкретного STT-провайдера.
"""

from __future__ import annotations

import logging
from pathlib import Path

import aiofiles
import httpx

logger = logging.getLogger(__name__)

_TRANSCRIPTIONS_URL = "https://api.groq.com/openai/v1/audio/transcriptions"
_MODEL = "whisper-large-v3"
_REQUEST_TIMEOUT_SECONDS = 30.0


class GroqSTT:
    """Транскрибирует аудио/видео-файлы через Groq Whisper-large-v3."""

    def __init__(self, api_key: str, *, client: httpx.AsyncClient | None = None) -> None:
        self._api_key = api_key
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(_REQUEST_TIMEOUT_SECONDS))

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def transcribe(self, audio_path: Path, *, language: str | None = None) -> str:
        """
        Возвращает распознанный текст, либо пустую строку при любом сбое
        (файл не читается, сеть недоступна, Groq вернул ошибку/не-JSON) —
        пустая строка, а не исключение, чтобы вызывающая сторона могла сама
        решить, откатываться ли на другой STT-путь (см. докстринг модуля).
        """
        try:
            async with aiofiles.open(audio_path, mode="rb") as f:
                audio_bytes = await f.read()
        except OSError as exc:
            logger.warning("stt_groq: failed to read audio file %s: %s", audio_path, exc)
            return ""

        data: dict[str, str] = {"model": _MODEL}
        if language is not None:
            data["language"] = language
        files = {"file": (audio_path.name, audio_bytes, "application/octet-stream")}

        try:
            response = await self._client.post(
                _TRANSCRIPTIONS_URL,
                headers={"Authorization": f"Bearer {self._api_key}"},
                data=data,
                files=files,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            logger.warning("stt_groq: transcription request failed for %s: %s", audio_path, exc)
            return ""

        try:
            payload = response.json()
        except ValueError as exc:
            logger.warning("stt_groq: response for %s is not valid JSON: %s", audio_path, exc)
            return ""

        return str(payload.get("text", "")).strip()


__all__ = ["GroqSTT"]
