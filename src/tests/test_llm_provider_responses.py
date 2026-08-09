"""
Тесты для efi.llm.providers.openai_compatible.OpenAICompatibleProvider —
разбор ответа провайдера.

Регрессия, которая здесь закрыта: ответ 200 с пустым `choices` уходил из
провайдера как УСПЕХ. Схему он проходит (поле необязательное), но обратиться
к `Response.message` можно только исключением — и ValueError вылетал уже у
Worker'а, в обход LLMError. То есть мимо fallback'а роутера (следующий
кандидат даже не пробовался) и мимо уведомления собеседника о сбое: человек
получал "прочитано" и тишину. OpenAI-совместимые прокси отдают такой ответ
на практике — на срабатывании модерации или когда апстрим отвалился у них.
"""

from __future__ import annotations

import httpx
import pytest

from efi.config.schema import EndpointConfig
from efi.llm.errors import LLMInvalidResponseError
from efi.llm.providers.openai_compatible import OpenAICompatibleProvider
from efi.llm.schemas import LLMParams, Message, Role, Session

_ENDPOINT = EndpointConfig(base_url="https://example.test/v1", api_key="k", model="m", timeout_seconds=5.0)


def _provider(payload: dict[str, object]) -> OpenAICompatibleProvider:
    transport = httpx.MockTransport(lambda _request: httpx.Response(200, json=payload))
    client = httpx.AsyncClient(transport=transport, base_url=_ENDPOINT.base_url)
    return OpenAICompatibleProvider(_ENDPOINT, name="test", client=client)


def _session() -> Session:
    return Session(messages=[Message(role=Role.USER, content="привет")])


async def test_empty_choices_is_a_provider_failure_not_a_success() -> None:
    provider = _provider({"id": "1", "choices": []})
    try:
        with pytest.raises(LLMInvalidResponseError):
            await provider.chat(LLMParams(model="m"), _session())
    finally:
        await provider.aclose()


async def test_a_normal_response_still_goes_through() -> None:
    provider = _provider(
        {"id": "1", "choices": [{"index": 0, "message": {"role": "assistant", "content": "и тебе"}}]}
    )
    try:
        response = await provider.chat(LLMParams(model="m"), _session())
        assert response.message.content == "и тебе"
    finally:
        await provider.aclose()
