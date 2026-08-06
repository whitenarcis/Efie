"""
efi/telegram/media/video.py

Обработка видео-кружков (video note). В этом шаге — транскрипция звуковой
дорожки тем же путём, что и обычное голосовое (efi.telegram.media.voice):
Whisper-совместимый эндпоинт транскрибирует аудио независимо от того,
завёрнуто оно в .ogg или в видео-контейнер, если так поддерживает конкретный
провайдер.

Анализ самого видеоряда (сэмплирование кадров + vision-модель, как в
референсе) — отдельный, заметно более тяжёлый пайплайн, не реализован в этом
шаге; когда понадобится, встанет отдельной функцией в этом же модуле, не
меняя контракт transcribe_video_note.
"""

from __future__ import annotations

import logging
from pathlib import Path

from efi.config.schema import TaskRole
from efi.llm.errors import LLMError
from efi.llm.router import LLMRouter

logger = logging.getLogger(__name__)


async def transcribe_video_note(router: LLMRouter, video_path: Path, *, role: TaskRole = TaskRole.VISION) -> str:
    """Транскрибирует звуковую дорожку видео-кружка. См. предостережение в докстринге модуля про видеоряд."""
    try:
        transcription = await router.transcribe_audio(role, video_path)
    except LLMError as exc:
        logger.warning("video_note: transcription failed for %s: %s", video_path, exc)
        return "[видео-кружок — не удалось распознать речь]"

    text = transcription.text.strip()
    return text or "[видео-кружок — распознавание не дало текста]"


__all__ = ["transcribe_video_note"]
