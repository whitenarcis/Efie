"""
efi/tools/media_tools/generate_image.py

Генерация изображений через Pollinations.ai — используется, в частности, для
селфи персонажа. Простой GET-запрос к image.pollinations.ai/prompt/<текст>,
ответ — сразу байты изображения (без JSON-обёртки).

КРИТИЧЕСКИ ВАЖНО (известный урок проекта, см. память): при enhance=true
Pollinations прогоняет промпт через собственную LLM-перезапись, которая на
длинных промптах теряет детали из ХВОСТА строки. Поэтому описание внешности
персонажа (appearance_prompt) должно идти ПЕРВЫМ элементом промпта, а не
дописываться в конец — иначе изображение перестаёт быть похоже на персонажа.
Порядок ниже (identity -> style -> composition) — это не стилистический
выбор, а обязательное требование.
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote

import aiofiles
import aiofiles.os
import httpx

from efi.tools.base import Tool, ToolContext

logger = logging.getLogger(__name__)

_POLLINATIONS_BASE_URL = "https://image.pollinations.ai/prompt/"
_REQUEST_TIMEOUT_SECONDS = 60.0  # генерация изображения ощутимо дольше обычного HTTP-запроса
_DEFAULT_STYLE_SUFFIX = "minimalist anime illustration, cel shading, flat colors, clean lines"


class GenerateImageTool(Tool):
    """
    Генерирует изображение (например, селфи персонажа) через Pollinations.ai
    и сохраняет его локально; возвращает путь к файлу — отправка результата
    собеседнику отдельным шагом (через send_telegram_message с описанием
    пути или отдельный "отправить фото" инструмент, который ещё предстоит
    добавить в TelegramClientWrapper.send_photo).
    """

    name = "generate_image"
    description = (
        "Генерирует изображение по текстовому описанию сцены (например, свою фотографию/селфи "
        "в конкретной ситуации). Опиши сцену и что на ней происходит — свою внешность "
        "описывать не нужно, она добавляется автоматически."
    )
    parameters = {
        "type": "object",
        "properties": {
            "scene_description": {
                "type": "string",
                "description": "Что происходит на изображении: поза, место, освещение, ситуация",
            },
        },
        "required": ["scene_description"],
        "additionalProperties": False,
    }

    def __init__(
        self,
        *,
        appearance_prompt: str,
        output_dir: Path,
        client: httpx.AsyncClient | None = None,
        style_suffix: str = _DEFAULT_STYLE_SUFFIX,
    ) -> None:
        self._appearance_prompt = appearance_prompt
        self._output_dir = output_dir
        self._style_suffix = style_suffix
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(_REQUEST_TIMEOUT_SECONDS))

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        scene_description = str(arguments.get("scene_description", "")).strip()
        if not scene_description:
            return "error: scene_description must not be empty"

        # Порядок частей промпта — identity (внешность) ПЕРВОЙ, затем стиль,
        # затем композиция/сцена. См. докстринг модуля про enhance=true.
        prompt = f"{self._appearance_prompt}, {self._style_suffix}, {scene_description}"
        url = _POLLINATIONS_BASE_URL + quote(prompt)

        try:
            response = await self._client.get(url, params={"nologo": "true", "enhance": "true"})
            response.raise_for_status()
            image_bytes = response.content
        except httpx.HTTPError as exc:
            logger.warning("generate_image: request failed: %s", exc)
            return f"error: image generation request failed: {exc}"

        await aiofiles.os.makedirs(self._output_dir, exist_ok=True)
        output_path = self._output_dir / f"{uuid.uuid4().hex}.png"
        async with aiofiles.open(output_path, mode="wb") as f:
            await f.write(image_bytes)

        logger.info("generate_image: saved %s (%d bytes)", output_path, len(image_bytes))
        return f"Изображение сгенерировано и сохранено: {output_path}"


__all__ = ["GenerateImageTool"]
