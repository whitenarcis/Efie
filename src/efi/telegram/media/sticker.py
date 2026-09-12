"""
efi/telegram/media/sticker.py

Описание входящего стикера текстом через LLMRouter (роль VISION) — то же
превращение «медиа -> текст ДО Worker'а», что и для фото/голосовых/кружков
(см. efi.telegram.media.image и voice/video).

От describe_photo отличается тремя вещами:

  1. Промптом: стикер это не фотография, и «что изображено» для него важнее,
     чем композиция/контекст.
  2. Контрактом: возвращает `str | None`, а не строку-плейсхолдер. None значит
     «описать не удалось», и вызывающий код НЕ должен кэшировать такой результат
     в sticker_descriptions — иначе однажды сорвавшийся запрос навсегда
     заморозил бы стикер без описания (см. efi/db/sticker_descriptions.py).
  3. Вызовом не из хендлера, а через кэш-обёртку: сперва efi.db.
     StickerDescriptionStore.find, vision только при промахе.
"""

from __future__ import annotations

import logging
from pathlib import Path

from efi.config.schema import TaskRole
from efi.llm.errors import LLMError
from efi.llm.router import LLMRouter

logger = logging.getLogger(__name__)

_DEFAULT_PROMPT = (
    "Опиши коротко и по существу, что изображено на этом стикере, — как будто "
    "объясняешь это словами человеку, который видит только подпись. Без оценки "
    "качества и без «это стикер с...»: сразу суть (кто/что на нём и какое "
    "настроение/действие)."
)


async def describe_sticker(
    router: LLMRouter,
    image_path: Path,
    *,
    role: TaskRole = TaskRole.VISION,
    prompt: str = _DEFAULT_PROMPT,
) -> str | None:
    """
    Возвращает текстовое описание стикера, или None, если распознать не удалось
    (сбой загрузки/запроса или пустой ответ модели). Никогда не бросает
    исключение: деградация стикера без описания — рабочая ситуация, в которой
    вызывающий код покажет эмодзи как последнюю доступную информацию.
    """
    try:
        description = await router.describe_image(role, image_path, prompt=prompt)
    except LLMError as exc:
        logger.warning("sticker: description failed for %s: %s", image_path, exc)
        return None

    stripped = description.strip()
    return stripped or None


__all__ = ["describe_sticker"]
