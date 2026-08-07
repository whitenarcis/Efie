"""
efi/memory/local_embeddings.py

Локальный embedding-движок поверх fastembed (ONNX Runtime, без PyTorch) —
считает эмбеддинги на устройстве, без сетевых вызовов и без зависимости от
того, поддерживает ли текущий LLM-провайдер embeddings вообще (известный
триггер: у Groq на момент написания нет embedding-моделей в принципе — см.
efi/memory/rag.py, где этот движок теперь основной путь, а LLMRouter.embedding()
остался вторичным фолбэком на случай, если облачный провайдер с эмбеддингами
всё же появится).

fastembed выбран сознательно вместо sentence-transformers: не тянет за собой
PyTorch — тяжёлую компиляцию/установку на ARM/Termux, ту же категорию боли,
что уже прошли с numpy/pydantic-core/tgcrypto в этом проекте. Инференс через
ONNX Runtime, для которого есть готовые ARM-wheels.

Модель по умолчанию — intfloat/multilingual-e5-large: fastembed не держит
`-small`-версию этого семейства (проверено через
`TextEmbedding.list_supported_models()` — в списке только `-large`), поэтому
дефолт тяжелее, чем изначально закладывалось (~1ГБ+ вместо ~470МБ), но с
заметно лучшим качеством на русском, чем более лёгкие EN-ориентированные
альтернативы (BAAI/bge-small-en-v1.5 и т.п.) или более компактная
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (~420МБ, тоже
многоязычная, без query/passage-асимметрии — жизнеспособный более лёгкий
вариант, если место на диске/скорость первого запуска станут проблемой).
Модель настраивается через config.schema.MemorySettings.local_embedding_model.

Важный нюанс моделей семейства e5 (в т.ч. дефолтная): они асимметричны —
для качественного поиска запрос и сохраняемый документ должны эмбеддиться с
разными текстовыми префиксами ("query: "/"passage: "), отсюда два разных
метода (embed_query/embed_document) вместо одного универсального embed().
Если модель заменить на семейство без этой конвенции (например, BGE — там
своя, отличная схема префиксов), эту логику нужно будет пересмотреть отдельно.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from efi.llm.schemas import EmbeddingVector

logger = logging.getLogger(__name__)

_DEFAULT_MODEL_NAME = "intfloat/multilingual-e5-large"


class LocalEmbeddingEngine:
    """
    Асинхронная обёртка над fastembed.TextEmbedding.

    Модель грузится лениво — при первом вызове embed_query()/embed_document(),
    не при конструировании — чтобы создание RAGMemory/EfiApp не зависело от
    того, готова ли модель прямо сейчас (на первый запуск веса ещё и
    скачиваются из сети), и не блокировало старт приложения.
    """

    def __init__(self, model_name: str = _DEFAULT_MODEL_NAME) -> None:
        self._model_name = model_name
        self._model: Any | None = None
        self._load_lock = asyncio.Lock()
        # threading.Lock, а не asyncio.Lock: сам инференс происходит в
        # отдельных потоках (asyncio.to_thread), и onnxruntime не гарантированно
        # потокобезопасен при параллельных вызовах на одной сессии модели.
        self._inference_lock = threading.Lock()

    async def embed_query(self, text: str) -> EmbeddingVector:
        """Эмбеддинг для ПОИСКОВОГО запроса (см. докстринг модуля про асимметрию e5-моделей)."""
        return await self._embed(f"query: {text}")

    async def embed_document(self, text: str) -> EmbeddingVector:
        """Эмбеддинг для СОХРАНЯЕМОГО текста (запись дневника и т.п.)."""
        return await self._embed(f"passage: {text}")

    async def _embed(self, prefixed_text: str) -> EmbeddingVector:
        model = await self._ensure_model()
        return await asyncio.to_thread(self._embed_sync, model, prefixed_text)

    async def _ensure_model(self) -> Any:
        if self._model is not None:
            return self._model
        async with self._load_lock:
            if self._model is not None:  # кто-то успел загрузить, пока мы ждали лок
                return self._model
            logger.info("local_embeddings: loading model %r (первый запуск может скачать веса из сети)", self._model_name)
            self._model = await asyncio.to_thread(self._load_model_sync)
            logger.info("local_embeddings: model %r loaded", self._model_name)
            return self._model

    def _load_model_sync(self) -> Any:
        try:
            from fastembed import TextEmbedding
        except ImportError as exc:
            raise RuntimeError(
                "fastembed не установлен — локальный embedding-движок недоступен. "
                "Установите: pip install fastembed --break-system-packages "
                "(или через extras проекта: pip install -e '.[local-embeddings]')"
            ) from exc
        return TextEmbedding(model_name=self._model_name)

    def _embed_sync(self, model: Any, prefixed_text: str) -> EmbeddingVector:
        with self._inference_lock:
            # TextEmbedding.embed() — генератор; на один текст берём единственный результат.
            (vector,) = model.embed([prefixed_text])
        return [float(component) for component in vector]


__all__ = ["LocalEmbeddingEngine"]
