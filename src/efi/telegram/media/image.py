"""
efi/telegram/media/image.py

Описание входящих фотографий текстом через LLMRouter (роль VISION) — то же
превращение "медиа -> текст ДО Worker'а", что и для голосовых/кружков (см.
efi.telegram.media.voice/video).
"""

from __future__ import annotations

import logging
from pathlib import Path

from efi.config.schema import TaskRole
from efi.llm.errors import LLMError
from efi.llm.router import LLMRouter

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT = (
    "Опиши коротко и по существу, что изображено на этой фотографии — "
    "как будто объясняешь это словами человеку, который её не видит."
)


async def describe_photo(
    router: LLMRouter,
    image_path: Path,
    *,
    role: TaskRole = TaskRole.VISION,
    prompt: str = _DEFAULT_PROMPT,
) -> str:
    """Возвращает текстовое описание изображения. Никогда не бросает исключение — деградирует до плейсхолдера."""
    try:
        description = await router.describe_image(role, image_path, prompt=prompt)
    except LLMError as exc:
        logger.warning("image: description failed for %s: %s", image_path, exc)
        return "[фото — не удалось получить описание]"

    return description.strip() or "[фото — модель не дала описания]"


__all__ = ["describe_photo"]
