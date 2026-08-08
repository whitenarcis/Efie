"""
Тесты для efi.llm.router.LLMRouter: общий бюджет времени на весь перебор
кандидатов роли (регрессия на инцидент, когда primary и fallback роли MAIN
таймаутились подряд по 30с каждый, и Эфи не отвечала минутами).
"""

from __future__ import annotations

import asyncio

import pytest

from efi.config.schema import EndpointConfig, RoleRoute, TaskRole
from efi.llm.errors import LLMServerError, LLMTimeoutError
from efi.llm.router import LLMRouter


def _endpoint(model: str, timeout: float) -> EndpointConfig:
    return EndpointConfig(base_url="https://example.test/v1", api_key="k", model=model, timeout_seconds=timeout)


def _make_router(*, buffer: float = 0.05) -> LLMRouter:
    main_route = RoleRoute(primary=_endpoint("primary", 0.05), fallback=_endpoint("fallback", 0.05))
    fast_route = RoleRoute(primary=_endpoint("fast-primary", 0.05))
    vision_route = RoleRoute(primary=_endpoint("vision-primary", 0.05))
    routes = {
        TaskRole.MAIN: main_route,
        TaskRole.FAST: fast_route,
        TaskRole.BACKGROUND: fast_route,
        TaskRole.VISION: vision_route,
    }
    return LLMRouter(routes, role_timeout_buffer_seconds=buffer)


async def test_bounds_total_wait_to_sum_of_candidate_timeouts() -> None:
    router = _make_router(buffer=0.05)

    async def hangs_forever(provider: object, endpoint: EndpointConfig) -> str:
        await asyncio.sleep(10.0)
        raise AssertionError("should never complete")

    start = asyncio.get_running_loop().time()
    with pytest.raises(LLMTimeoutError):
        await router._attempt_with_fallback(TaskRole.MAIN, hangs_forever)
    elapsed = asyncio.get_running_loop().time() - start

    # main: primary(0.05) + fallback(0.05) + buffer(0.05) = 0.15s бюджет — с большим запасом меньше секунды.
    assert elapsed < 1.0


async def test_succeeds_within_budget_when_candidate_is_fast() -> None:
    router = _make_router()

    async def succeeds(provider: object, endpoint: EndpointConfig) -> str:
        return "ok"

    assert await router._attempt_with_fallback(TaskRole.MAIN, succeeds) == "ok"


async def test_falls_back_to_next_candidate_after_error() -> None:
    router = _make_router()
    calls: list[str] = []

    async def fails_then_succeeds(provider: object, endpoint: EndpointConfig) -> str:
        calls.append(endpoint.model)
        if endpoint.model == "primary":
            raise LLMServerError("boom", provider="primary")
        return "ok"

    result = await router._attempt_with_fallback(TaskRole.MAIN, fails_then_succeeds)
    assert result == "ok"
    assert calls == ["primary", "fallback"]


async def test_vision_role_budget_is_independent_of_main_role() -> None:
    """Роль с одним медленным (но настроенным именно так) кандидатом не должна урезаться чужим бюджетом."""
    router_settings = LLMRouter(
        {
            TaskRole.MAIN: RoleRoute(primary=_endpoint("main-primary", 0.05)),
            TaskRole.FAST: RoleRoute(primary=_endpoint("fast-primary", 0.05)),
            TaskRole.BACKGROUND: RoleRoute(primary=_endpoint("background-primary", 0.05)),
            TaskRole.VISION: RoleRoute(primary=_endpoint("vision-primary", 2.0)),
        },
        role_timeout_buffer_seconds=0.5,
    )

    async def sleeps_then_succeeds(provider: object, endpoint: EndpointConfig) -> str:
        await asyncio.sleep(0.05)
        return "ok"

    # Бюджет для VISION = 2.0 + 0.5 = 2.5s — с большим запасом хватает, несмотря на то что MAIN
    # настроен куда туже (широкие поля намеренно — тест не должен флакать под нагрузкой CI).
    result = await router_settings._attempt_with_fallback(TaskRole.VISION, sleeps_then_succeeds)
    assert result == "ok"
