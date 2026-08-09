"""
efi/telegram/media/voice.py

Транскрипция голосовых сообщений в текст — превращает файл, полученный от
Telegram, в текст ДО того, как событие попадёт в Worker (см.
efi.telegram.handlers.TelegramEventHandlers) — Worker и вся остальная
система работают только с текстом, ничего не зная о том, что исходно это
было аудио.
"""

from __future__ import annotations

import logging
from pathlib import Path

from efi.config.schema import TaskRole
from efi.llm.errors import LLMError
from efi.llm.router import LLMRouter

logger = logging.getLogger(__name__)


async def transcribe_voice_message(router: LLMRouter, audio_path: Path, *, role: TaskRole = TaskRole.VISION) -> str:
    """
    Транскрибирует голосовое сообщение.

    Роль по умолчанию — VISION: в этой архитектуре она покрывает не только
    изображения, но и прочие "специализированные"/служебные модели (см.
    TaskRole в efi/config/schema.py). Если транскрипция не удалась (сеть,
    недоступность всех кандидатов роли), возвращает текстовый плейсхолдер
    вместо исключения — обработка входящего сообщения не должна падать
    целиком из-за недоступности одной модели.
    """
    try:
        transcription = await router.transcribe_audio(role, audio_path)
    except LLMError as exc:
        logger.warning("voice: transcription failed for %s: %s", audio_path, exc)
        return "[голосовое сообщение — не удалось распознать речь]"

    text = transcription.text.strip()
    return text or "[голосовое сообщение — распознавание не дало текста]"


__all__ = ["transcribe_voice_message"]
