"""
efi/llm/measurable.py

Обёртка метрик поверх LLMProvider: перехватывает каждый вызов (chat/
chat_streaming/embedding/transcribe_audio/describe_image), измеряет
длительность и (где применимо) токены/стоимость из Response, и публикует всё
через переданный `sink` — простой синхронный callback. Потребитель метрик
(efi/dashboard/metrics.py) подписывается на события через sink; ни провайдер,
ни роутер ничего не знают про дашборд или способ отображения.

Подключается через `LLMRouter(metrics_sink=...)`: роутер создаёт провайдеров
лениво (`_provider_for`), поэтому прозрачно обернуть их можно только там —
что он и делает, если sink передан. Без sink обёртки нет вовсе, и путь
LLM-вызова остаётся ровно таким же, как до появления метрик.
"""

from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass
from pathlib import Path

from efi.llm.base import LLMProvider
from efi.llm.schemas import AudioTranscription, EmbeddingVector, LLMParams, Response, Session

logger = logging.getLogger(__name__)


@dataclass(slots=True, frozen=True)
class CallMetric:
    """Одна запись метрики: какой провайдер/операция, сколько заняло, сколько токенов/во что обошлось."""

    provider_name: str
    operation: str
    duration_seconds: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float | None = None
    error: str | None = None


#: Синхронный callback. Не должен бросать исключения — MeasurableLLMProvider
#: их логирует и проглатывает, чтобы сбой в сборе метрик не мог сорвать
#: основной LLM-вызов.
MetricsSink = Callable[[CallMetric], None]


class MeasurableLLMProvider(LLMProvider):
    """
    Прозрачная обёртка вокруг любого LLMProvider: делегирует все вызовы
    исходному провайдеру, попутно измеряя длительность и публикуя каждый
    вызов через `sink` сразу после завершения операции (успешной или нет).
    """

    def __init__(self, inner: LLMProvider, sink: MetricsSink) -> None:
        self.name = inner.name
        self._inner = inner
        self._sink = sink

    async def chat(self, params: LLMParams, session: Session) -> Response:
        start = time.monotonic()
        try:
            response = await self._inner.chat(params, session)
        except Exception as exc:
            self._emit("chat", time.monotonic() - start, error=str(exc))
            raise
        self._emit(
            "chat",
            time.monotonic() - start,
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            cost=response.cost,
        )
        return response

    def chat_streaming(self, params: LLMParams, session: Session) -> AsyncIterator[Response]:
        # Обычная (не async) функция, возвращающая асинхронный генератор —
        # тот же контракт, что и у LLMProvider.chat_streaming (см. её докстринг).
        return self._chat_streaming_impl(params, session)

    async def _chat_streaming_impl(self, params: LLMParams, session: Session) -> AsyncIterator[Response]:
        start = time.monotonic()
        last_response: Response | None = None
        try:
            async for response in self._inner.chat_streaming(params, session):
                last_response = response
                yield response
        except Exception as exc:
            self._emit("chat_streaming", time.monotonic() - start, error=str(exc))
            raise

        usage = last_response.usage if last_response is not None else None
        self._emit(
            "chat_streaming",
            time.monotonic() - start,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            cost=last_response.cost if last_response is not None else None,
        )

    async def embedding(self, model: str, text: str) -> EmbeddingVector:
        start = time.monotonic()
        try:
            result = await self._inner.embedding(model, text)
        except Exception as exc:
            self._emit("embedding", time.monotonic() - start, error=str(exc))
            raise
        self._emit("embedding", time.monotonic() - start)
        return result

    async def transcribe_audio(self, model: str, audio_path: Path) -> AudioTranscription:
        start = time.monotonic()
        try:
            result = await self._inner.transcribe_audio(model, audio_path)
        except Exception as exc:
            self._emit("transcribe_audio", time.monotonic() - start, error=str(exc))
            raise
        self._emit("transcribe_audio", time.monotonic() - start)
        return result

    async def describe_image(self, model: str, image_path: Path, *, prompt: str) -> str:
        start = time.monotonic()
        try:
            result = await self._inner.describe_image(model, image_path, prompt=prompt)
        except Exception as exc:
            self._emit("describe_image", time.monotonic() - start, error=str(exc))
            raise
        self._emit("describe_image", time.monotonic() - start)
        return result

    async def aclose(self) -> None:
        await self._inner.aclose()

    def _emit(
        self,
        operation: str,
        duration_seconds: float,
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        cost: float | None = None,
        error: str | None = None,
    ) -> None:
        metric = CallMetric(
            provider_name=self.name,
            operation=operation,
            duration_seconds=duration_seconds,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
            error=error,
        )
        try:
            self._sink(metric)
        except Exception:
            logger.exception("measurable: metrics sink raised while handling %s", metric)


__all__ = ["CallMetric", "MetricsSink", "MeasurableLLMProvider"]
