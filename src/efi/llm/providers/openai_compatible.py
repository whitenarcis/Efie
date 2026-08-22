"""
efi/llm/providers/openai_compatible.py

Единый асинхронный клиент на httpx.AsyncClient для любого OpenAI-совместимого
chat/completions API. Используется как для OmniRoute (основной шлюз), так и для
Groq — разница между ними только в EndpointConfig (base_url/ключ/модель),
переданном при создании экземпляра; сам клиент провайдера не знает, с кем
именно разговаривает.

Обрабатывает:
    - нестриминговые запросы (chat);
    - SSE-стриминг (chat_streaming) с накоплением дельт и остановкой по `data: [DONE]`;
    - таймауты (httpx.TimeoutException -> LLMTimeoutError);
    - HTTP-ошибки провайдера, размаппленные в LLMAuthError/LLMRateLimitError/LLMServerError;
    - некорректные тела ответов (LLMInvalidResponseError).
"""

from __future__ import annotations

import base64
import json
import logging
import mimetypes
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any

import aiofiles
import httpx
from pydantic import ValidationError

from efi.config.schema import EndpointConfig
from efi.llm.base import LLMProvider
from efi.llm.errors import (
    LLMAuthError,
    LLMError,
    LLMInvalidResponseError,
    LLMRateLimitError,
    LLMServerError,
    LLMTimeoutError,
)
from efi.llm.schemas import (
    AudioTranscription,
    Choice,
    EmbeddingVector,
    LLMParams,
    Message,
    Response,
    Role,
    Session,
    Usage,
)

logger = logging.getLogger(__name__)

_CHAT_COMPLETIONS_PATH = "chat/completions"  # без ведущего "/" — см. _build_client()
_EMBEDDINGS_PATH = "embeddings"
_AUDIO_TRANSCRIPTIONS_PATH = "audio/transcriptions"
_ERROR_BODY_PREVIEW_LIMIT = 500


class OpenAICompatibleProvider(LLMProvider):
    """
    LLM-провайдер поверх любого OpenAI-совместимого HTTP API.

    Один экземпляр = одно (base_url, ключ) сочетание = один переиспользуемый
    httpx.AsyncClient с пулом соединений. Модель, которую использовать в
    конкретном запросе, приходит через `LLMParams.model` — сам провайдер её не
    фиксирует (это позволяет роутеру подменять модель на лету при fallback,
    не создавая новый провайдер).
    """

    def __init__(
        self,
        endpoint: EndpointConfig,
        *,
        name: str,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.name = name
        self._endpoint = endpoint
        # Если клиент передан снаружи (например, общий пул на несколько провайдеров),
        # закрывать его в aclose() — не наша ответственность.
        self._owns_client = client is None
        self._client = client or _build_client(endpoint)

    def _timeout_for(self, params: LLMParams) -> float:
        """
        Бюджет запроса: пер-запросный, если его посчитал роутер, иначе общий
        для эндпоинта. Провайдер ничего не решает сам — он только исполняет:
        знание о том, кто ждёт ответа (человек в чате или фоновая задача),
        живёт выше (efi/llm/router.py).
        """
        return params.timeout_seconds or self._endpoint.timeout_seconds

    async def chat(self, params: LLMParams, session: Session) -> Response:
        payload = self._build_payload(params, session, stream=False)
        timeout = self._timeout_for(params)
        try:
            http_response = await self._client.post(
                _CHAT_COMPLETIONS_PATH, json=payload, timeout=httpx.Timeout(timeout)
            )
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"{self.name}: request timed out after {timeout:.0f}s", provider=self.name
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMServerError(f"{self.name}: transport error: {exc}", provider=self.name) from exc

        await self._raise_for_status(http_response)

        try:
            data = http_response.json()
        except ValueError as exc:
            raise LLMInvalidResponseError(f"{self.name}: response body is not valid JSON", provider=self.name) from exc

        try:
            response = Response.model_validate(data)
        except ValidationError as exc:
            raise LLMInvalidResponseError(
                f"{self.name}: response does not match expected schema: {exc}", provider=self.name
            ) from exc

        # Пустой `choices` схему проходит (поле необязательное), но ответом не
        # является: обратиться к `Response.message` можно только исключением.
        # OpenAI-совместимые прокси реально отдают 200 с пустым choices — на
        # срабатывании модерации или когда апстрим отвалился на их стороне.
        # Без явной проверки такой ответ уходил из провайдера как успех, а
        # ValueError("no choices") вылетал уже у Worker'а, В ОБХОД LLMError —
        # то есть мимо и fallback'а роутера, и уведомления собеседника о сбое:
        # человек получал "прочитано" и тишину. Здесь это честный сбой
        # кандидата, и роутер идёт к следующему.
        if not response.choices:
            raise LLMInvalidResponseError(f"{self.name}: response contains no choices", provider=self.name)

        response.provider = response.provider or self.name
        return response

    async def chat_streaming(self, params: LLMParams, session: Session) -> AsyncIterator[Response]:
        payload = self._build_payload(params, session, stream=True)
        accumulated = Response(model=params.model, provider=self.name)
        try:
            async with self._client.stream("POST", _CHAT_COMPLETIONS_PATH, json=payload) as http_response:
                await self._raise_for_status(http_response, streaming=True)
                async for raw_line in http_response.aiter_lines():
                    line = raw_line.strip()
                    if not line or not line.startswith("data:"):
                        continue
                    data = line.removeprefix("data:").strip()
                    if data == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError as exc:
                        raise LLMInvalidResponseError(
                            f"{self.name}: malformed SSE chunk: {data!r}", provider=self.name
                        ) from exc
                    _merge_streaming_chunk(accumulated, chunk)
                    yield accumulated.model_copy(deep=True)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"{self.name}: stream timed out after {self._timeout_for(params):.0f}s", provider=self.name
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMServerError(f"{self.name}: transport error during streaming: {exc}", provider=self.name) from exc

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def embedding(self, model: str, text: str) -> EmbeddingVector:
        payload = {"model": model, "input": text}
        try:
            http_response = await self._client.post(_EMBEDDINGS_PATH, json=payload)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"{self.name}: embedding request timed out after {self._endpoint.timeout_seconds}s", provider=self.name
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMServerError(f"{self.name}: transport error: {exc}", provider=self.name) from exc

        await self._raise_for_status(http_response)

        try:
            data = http_response.json()
            vector = data["data"][0]["embedding"]
        except (ValueError, KeyError, IndexError, TypeError) as exc:
            raise LLMInvalidResponseError(
                f"{self.name}: unexpected /embeddings response shape", provider=self.name
            ) from exc

        return [float(component) for component in vector]

    async def transcribe_audio(self, model: str, audio_path: Path) -> AudioTranscription:
        try:
            async with aiofiles.open(audio_path, mode="rb") as f:
                audio_bytes = await f.read()
        except OSError as exc:
            raise LLMInvalidResponseError(
                f"{self.name}: could not read audio file {audio_path}: {exc}", provider=self.name
            ) from exc

        files = {"file": (audio_path.name, audio_bytes)}
        data = {"model": model, "response_format": "verbose_json"}
        try:
            http_response = await self._client.post(_AUDIO_TRANSCRIPTIONS_PATH, data=data, files=files)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"{self.name}: transcription request timed out after {self._endpoint.timeout_seconds}s",
                provider=self.name,
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMServerError(f"{self.name}: transport error: {exc}", provider=self.name) from exc

        await self._raise_for_status(http_response)

        try:
            payload = http_response.json()
            return AudioTranscription.model_validate(payload)
        except ValueError as exc:
            raise LLMInvalidResponseError(f"{self.name}: response body is not valid JSON", provider=self.name) from exc
        except ValidationError as exc:
            raise LLMInvalidResponseError(
                f"{self.name}: transcription response does not match expected schema: {exc}", provider=self.name
            ) from exc

    async def describe_image(self, model: str, image_path: Path, *, prompt: str) -> str:
        """
        Одноразовый vision-запрос, в обход общего _build_messages: формат
        OpenAI vision-контента (content как массив text+image_url частей)
        специфичен для этого единственного случая.
        """
        try:
            image_data_url = await _encode_image_as_data_url(image_path)
        except OSError as exc:
            raise LLMInvalidResponseError(
                f"{self.name}: could not read image file {image_path}: {exc}", provider=self.name
            ) from exc

        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": image_data_url}},
                    ],
                }
            ],
            "max_tokens": 512,
            "stream": False,
        }

        try:
            http_response = await self._client.post(_CHAT_COMPLETIONS_PATH, json=payload)
        except httpx.TimeoutException as exc:
            raise LLMTimeoutError(
                f"{self.name}: vision request timed out after {self._endpoint.timeout_seconds}s", provider=self.name
            ) from exc
        except httpx.HTTPError as exc:
            raise LLMServerError(f"{self.name}: transport error: {exc}", provider=self.name) from exc

        await self._raise_for_status(http_response)

        try:
            data = http_response.json()
            response = Response.model_validate(data)
        except ValueError as exc:
            raise LLMInvalidResponseError(f"{self.name}: response body is not valid JSON", provider=self.name) from exc
        except ValidationError as exc:
            raise LLMInvalidResponseError(
                f"{self.name}: vision response does not match expected schema: {exc}", provider=self.name
            ) from exc

        return response.text

    # -- построение запроса -------------------------------------------------

    def _build_payload(self, params: LLMParams, session: Session, *, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": params.model,
            "messages": self._build_messages(params, session),
            "max_tokens": params.max_output_tokens,
            "stream": stream,
        }
        optional_fields: dict[str, Any] = {
            "temperature": params.temperature,
            "top_p": params.top_p,
            "top_k": params.top_k,
            "min_p": params.min_p,
            "presence_penalty": params.presence_penalty,
            "repetition_penalty": params.repetition_penalty,
            "seed": params.seed,
        }
        for key, value in optional_fields.items():
            if value is not None:
                payload[key] = value
        if params.tools:
            payload["tools"] = params.tools
        return payload

    def _build_messages(self, params: LLMParams, session: Session) -> list[dict[str, Any]]:
        wire: list[dict[str, Any]] = []
        if params.system_prompt:
            wire.append({"role": Role.SYSTEM.value, "content": params.system_prompt})
        wire.extend(_message_to_wire(message) for message in session)
        return wire

    # -- обработка ошибок -----------------------------------------------

    async def _raise_for_status(self, response: httpx.Response, *, streaming: bool = False) -> None:
        if response.status_code < 400:
            return

        body = (await response.aread()).decode("utf-8", errors="replace") if streaming else response.text
        preview = body[:_ERROR_BODY_PREVIEW_LIMIT]
        status = response.status_code
        logger.debug("%s: HTTP %s from %s", self.name, status, response.request.url)

        if status in (401, 403):
            raise LLMAuthError(
                f"{self.name}: authentication failed ({status}): {preview}",
                provider=self.name, status_code=status,
            )
        if status == 429:
            retry_after = _parse_retry_after(response.headers.get("retry-after"))
            raise LLMRateLimitError(
                f"{self.name}: rate limited (429): {preview}", provider=self.name, retry_after=retry_after
            )
        if status >= 500:
            raise LLMServerError(
                f"{self.name}: server error ({status}): {preview}",
                provider=self.name, status_code=status,
            )
        raise LLMError(f"{self.name}: unexpected HTTP {status}: {preview}", provider=self.name, status_code=status)


def _build_client(endpoint: EndpointConfig) -> httpx.AsyncClient:
    # ВАЖНО: base_url должен заканчиваться на "/", а путь запроса — НЕ начинаться
    # с "/". httpx объединяет их через urljoin-семантику: "https://host/v1" + "/chat/completions"
    # даёт "https://host/chat/completions" (теряется "/v1"!), а "https://host/v1/" + "chat/completions"
    # даёт корректный "https://host/v1/chat/completions".
    base_url = endpoint.base_url.rstrip("/") + "/"
    return httpx.AsyncClient(
        base_url=base_url,
        headers={
            "Authorization": f"Bearer {endpoint.api_key.get_secret_value()}",
            "Content-Type": "application/json",
        },
        timeout=httpx.Timeout(endpoint.timeout_seconds),
    )


async def _encode_image_as_data_url(path: Path) -> str:
    """Кодирует файл изображения как data: URL (base64) для vision-запроса."""
    mime_type, _ = mimetypes.guess_type(str(path))
    mime_type = mime_type or "image/jpeg"
    async with aiofiles.open(path, mode="rb") as f:
        raw_bytes = await f.read()
    encoded = base64.b64encode(raw_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _parse_retry_after(value: str | None) -> float | None:
    """Разбирает заголовок Retry-After. Поддерживает только числовую форму (секунды)."""
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        # HTTP-date форму (RFC 7231) не разбираем — роутер в этом случае
        # использует свой дефолтный cooldown для 429.
        return None


def _message_to_wire(message: Message) -> dict[str, Any]:
    """Сериализует Message в OpenAI-совместимый JSON-объект сообщения."""
    wire: dict[str, Any] = {"role": message.role.value, "content": message.content}
    if message.role is Role.TOOL and message.tool_call_id:
        wire["tool_call_id"] = message.tool_call_id
    if message.tool_calls:
        wire["tool_calls"] = [
            {
                "id": call.id,
                "type": call.type or "function",
                "function": {"name": call.function.name, "arguments": call.function.arguments},
            }
            for call in message.tool_calls
        ]
    return wire


def _merge_streaming_chunk(accumulated: Response, chunk: dict[str, Any]) -> None:
    """Сливает один разобранный SSE-чанк (`data: {...}`) в накопленный Response."""
    if chunk.get("id"):
        accumulated.id = chunk["id"]
    if chunk.get("object"):
        accumulated.object = chunk["object"]
    if chunk.get("created"):
        accumulated.created = chunk["created"]
    if chunk.get("model"):
        accumulated.model = chunk["model"]
    if chunk.get("system_fingerprint"):
        accumulated.system_fingerprint = chunk["system_fingerprint"]
    if chunk.get("provider"):
        accumulated.provider = chunk["provider"]
    if chunk.get("usage"):
        accumulated.usage = Usage.model_validate(chunk["usage"])

    for choice_delta in chunk.get("choices") or []:
        index = choice_delta.get("index", 0)
        while len(accumulated.choices) <= index:
            accumulated.choices.append(Choice(index=len(accumulated.choices)))
        delta_message = Message.model_validate(choice_delta.get("delta") or {})
        accumulated.choices[index].message.accumulate(delta_message)
        if choice_delta.get("finish_reason"):
            accumulated.choices[index].finish_reason = choice_delta["finish_reason"]


__all__ = ["OpenAICompatibleProvider"]
