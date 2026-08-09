"""
efi/tools/media_tools/generate_voice.py

Генерация голосового ответа через ElevenLabs — текст в речь, сохраняется как
.ogg (Opus, opus_48000_64 — как и в текущей реализации) для отправки как
голосовое сообщение Telegram (TelegramClientWrapper.send_voice).

Примечание: конкретный формат запроса к ElevenLabs (output_format как
query-параметр, а не поле JSON-тела) сверен по памяти о публичном API на
момент написания — стоит перепроверить по актуальной документации
ElevenLabs перед первым реальным запуском, без сети в этой песочнице
runtime-проверка невозможна.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

import aiofiles
import aiofiles.os
import httpx
from pydantic import SecretStr

from efi.tools.base import Tool, ToolContext

logger = logging.getLogger(__name__)

_ELEVENLABS_BASE_URL = "https://api.elevenlabs.io/v1"
_REQUEST_TIMEOUT_SECONDS = 30.0
_DEFAULT_MODEL_ID = "eleven_multilingual_v2"
_OUTPUT_FORMAT = "opus_48000_64"


class GenerateVoiceTool(Tool):
    """Озвучивает переданный текст через ElevenLabs и сохраняет результат как .ogg-файл."""

    name = "generate_voice"
    description = "Озвучивает текст твоим голосом и готовит его для отправки как голосовое сообщение."
    parameters = {
        "type": "object",
        "properties": {
            "text": {"type": "string", "description": "Текст, который нужно озвучить"},
        },
        "required": ["text"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        api_key: SecretStr,
        voice_id: str,
        output_dir: Path,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_key = api_key
        self._voice_id = voice_id
        self._output_dir = output_dir
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(_REQUEST_TIMEOUT_SECONDS))

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        text = str(arguments.get("text", "")).strip()
        if not text:
            return "error: text must not be empty"

        url = f"{_ELEVENLABS_BASE_URL}/text-to-speech/{self._voice_id}"
        headers = {"xi-api-key": self._api_key.get_secret_value(), "Content-Type": "application/json"}

        try:
            response = await self._client.post(
                url,
                headers=headers,
                params={"output_format": _OUTPUT_FORMAT},
                json={"text": text, "model_id": _DEFAULT_MODEL_ID},
            )
            response.raise_for_status()
            audio_bytes = response.content
        except httpx.HTTPError as exc:
            logger.warning("generate_voice: request failed: %s", exc)
            return f"error: voice generation request failed: {exc}"

        await aiofiles.os.makedirs(self._output_dir, exist_ok=True)
        output_path = self._output_dir / f"{uuid.uuid4().hex}.ogg"
        async with aiofiles.open(output_path, mode="wb") as f:
            await f.write(audio_bytes)

        logger.info("generate_voice: saved %s (%d bytes)", output_path, len(audio_bytes))
        return f"Голосовое сообщение сгенерировано и сохранено: {output_path}"


__all__ = ["GenerateVoiceTool"]
