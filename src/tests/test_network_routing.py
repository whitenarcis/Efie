"""
Тесты двух ярусов вычислений и устойчивости к отказам.

Проверяется то, ради чего вся конструкция и затевалась: работа не должна
теряться. Ноутбук уснул посреди генерации — задачу доделывает облако; провайдер
ответил 429 — запрос повторяется, а не пропадает; отвергнутый ключ — наоборот,
не повторяется, потому что четыре попытки дадут четыре одинаковых отказа.

Всё без сети и без настоящих пауз: транспорт подменяется MockTransport'ом,
сон — счётчиком. Иначе тест зависел бы от того, дома ли сейчас разработчик.
"""

from __future__ import annotations

import httpx
import pytest
from pydantic import SecretStr

from efi.config.schema import EndpointConfig, LaptopLinkSettings
from efi.llm.errors import LLMAuthError, LLMRateLimitError, LLMServerError
from efi.llm.network_router import LaptopLink, NetworkModelRouter
from efi.llm.resilience import (
    AttemptLog,
    Backend,
    ConcurrencyGate,
    RetryPolicy,
    is_transient,
    with_failover,
    with_retries,
)
from efi.llm.schemas import Choice, LLMParams, Message, Response, Role, Session


class _Clock:
    """Часы, которые двигаются только когда мы сами их двигаем."""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _Sleeps:
    """Сон, который ничего не ждёт, но помнит, сколько его просили ждать."""

    def __init__(self) -> None:
        self.waited: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.waited.append(delay)


def _endpoint(url: str = "http://192.168.0.109:8080/v1") -> EndpointConfig:
    return EndpointConfig(base_url=url, api_key=SecretStr("local"), model="claude-3-5-sonnet")


def _response(text: str) -> Response:
    return Response(choices=[Choice(message=Message(role=Role.ASSISTANT, content=text))])


# -- повторы ------------------------------------------------------------------


async def test_rate_limit_is_waited_out_not_dropped() -> None:
    sleeps = _Sleeps()
    calls = 0

    async def flaky() -> str:
        nonlocal calls
        calls += 1
        if calls < 3:
            raise LLMRateLimitError("too many requests", provider="laptop")
        return "готово"

    result = await with_retries(flaky, policy=RetryPolicy(jitter_seconds=0.0), sleeper=sleeps)

    assert result == "готово"
    assert calls == 3
    assert sleeps.waited == [2.0, 4.0], "экспонента, а не постоянная пауза"


async def test_provider_knows_better_than_our_formula() -> None:
    """Retry-After он присылает не из вежливости: он знает, когда лимит отпустит."""
    sleeps = _Sleeps()

    async def limited() -> str:
        raise LLMRateLimitError("wait", provider="laptop", retry_after=17.0)

    with pytest.raises(LLMRateLimitError):
        await with_retries(limited, policy=RetryPolicy(attempts=2), sleeper=sleeps)

    assert sleeps.waited == [17.0]


async def test_retry_after_is_capped() -> None:
    """На исчерпанном дневном лимите приходят часы — столько ждать бессмысленно."""
    policy = RetryPolicy(max_delay_seconds=90.0)

    assert policy.delay_for(1, LLMRateLimitError("wait", retry_after=7200.0)) == 90.0


def test_only_passing_failures_are_repeated() -> None:
    assert is_transient(LLMRateLimitError("429")) is True
    assert is_transient(LLMServerError("502")) is True
    assert is_transient(TimeoutError()) is True
    assert is_transient(LLMAuthError("invalid key")) is False
    assert is_transient(ValueError("что-то не то")) is False


async def test_a_rejected_key_is_not_retried() -> None:
    calls = 0

    async def denied() -> str:
        nonlocal calls
        calls += 1
        raise LLMAuthError("invalid api key", provider="laptop")

    with pytest.raises(LLMAuthError):
        await with_retries(denied, policy=RetryPolicy(), sleeper=_Sleeps())

    assert calls == 1


async def test_the_gate_lets_only_so_many_through_at_once() -> None:
    """Пачка файлов без потолка — лучший способ получить 429 на ровном месте."""
    gate = ConcurrencyGate(limit=2)
    peak = 0
    inside = 0

    async def work() -> None:
        nonlocal peak, inside
        inside += 1
        peak = max(peak, inside)
        inside -= 1

    import asyncio

    await asyncio.gather(*(gate.run(work) for _ in range(6)))
    assert peak <= 2


# -- переключение ярусов ------------------------------------------------------


async def test_a_dying_backend_hands_the_task_over() -> None:
    """Обрыв связи с ноутбуком посреди генерации не должен ронять задачу."""
    log = AttemptLog()

    async def laptop() -> str:
        raise LLMServerError("connection reset", provider="laptop")

    async def cloud() -> str:
        return "сделано облаком"

    result = await with_failover(
        [Backend("ноутбук", laptop), Backend("облако", cloud)],
        policy=RetryPolicy(attempts=1),
        sleeper=_Sleeps(),
        log=log,
    )

    assert result == "сделано облаком"
    assert log.failovers == 1
    assert log.backend == "облако"


# -- health-check -------------------------------------------------------------


def _link(handler: object, *, clock: _Clock | None = None) -> LaptopLink:
    return LaptopLink(
        _endpoint(),
        transport=httpx.MockTransport(handler),  # type: ignore[arg-type]
        clock=clock or _Clock(),
    )


async def test_a_laptop_that_answers_is_used() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/models")
        return httpx.Response(200, json={"data": []})

    assert await _link(handler).available() is True


async def test_a_sleeping_laptop_is_not_waited_for() -> None:
    """Телефон уехал из дома — проверка обязана ответить «нет», а не висеть."""

    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectTimeout("no route to host")

    assert await _link(handler).available() is False


async def test_the_check_is_cached_so_it_does_not_ping_before_every_file() -> None:
    clock = _Clock()
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json={"data": []})

    link = _link(handler, clock=clock)
    assert await link.available() is True
    assert await link.available() is True
    assert calls == 1, "второй вопрос за ту же минуту идёт из кэша"

    clock.now += 60.0
    assert await link.available() is True
    assert calls == 2, "через минуту проверяем заново"


async def test_a_laptop_that_came_back_is_picked_up_quickly() -> None:
    """Пришли домой, телефон поймал Wi-Fi — ждать пять минут никто не будет."""
    clock = _Clock()
    alive = False

    def handler(request: httpx.Request) -> httpx.Response:
        if not alive:
            raise httpx.ConnectError("down")
        return httpx.Response(200, json={"data": []})

    link = _link(handler, clock=clock)
    assert await link.available() is False

    alive = True
    clock.now += 11.0
    assert await link.available() is True


# -- роутер целиком -----------------------------------------------------------


class _CountingFallback:
    def __init__(self) -> None:
        self.calls = 0

    async def __call__(self, params: LLMParams, session: Session) -> Response:
        self.calls += 1
        return _response("ответ облака")


def _session() -> Session:
    return Session(messages=[Message(role=Role.USER, content="почини импорт")])


async def test_without_a_laptop_everything_works_exactly_as_before() -> None:
    """Главное свойство всей затеи: не настроен ноутбук — ничего не меняется."""
    fallback = _CountingFallback()
    router = NetworkModelRouter(None, fallback)

    response = await router.chat(LLMParams(model=""), _session())

    assert response.text == "ответ облака"
    assert router.has_laptop is False
    assert await router.tier() == "облачный кодер"


async def test_a_live_laptop_takes_the_heavy_thinking() -> None:
    class _LaptopProvider:
        name = "laptop"

        async def chat(self, params: LLMParams, session: Session) -> Response:
            assert params.model == "claude-3-5-sonnet", "модель подставляется своя"
            return _response("ответ ноутбука")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []})

    link = LaptopLink(
        _endpoint(),
        provider=_LaptopProvider(),  # type: ignore[arg-type]
        transport=httpx.MockTransport(handler),
    )
    fallback = _CountingFallback()
    router = NetworkModelRouter(link, fallback)

    response = await router.chat(LLMParams(model=""), _session())

    assert response.text == "ответ ноутбука"
    assert fallback.calls == 0
    assert "ноутбук" in await router.tier()


async def test_a_laptop_that_dies_mid_generation_does_not_lose_the_task() -> None:
    class _DyingProvider:
        name = "laptop"

        async def chat(self, params: LLMParams, session: Session) -> Response:
            raise LLMServerError("connection reset by peer", provider="laptop")

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"data": []})

    link = LaptopLink(
        _endpoint(),
        provider=_DyingProvider(),  # type: ignore[arg-type]
        transport=httpx.MockTransport(handler),
    )
    fallback = _CountingFallback()
    router = NetworkModelRouter(
        link, fallback, policy=RetryPolicy(attempts=1), sleeper=_Sleeps()
    )

    response = await router.chat(LLMParams(model=""), _session())

    assert response.text == "ответ облака", "задача доехала, хоть ноутбук и отвалился"
    assert fallback.calls == 1


# -- конфигурация -------------------------------------------------------------


def test_the_laptop_address_is_read_from_plain_env_vars() -> None:
    """
    Эти переменные правят чаще всего остального — руками, когда роутер выдал
    ноутбуку другой адрес. Поэтому они без префикса и без вложенности.
    """
    settings = LaptopLinkSettings.from_environment(
        {
            "OMNIROUTE_URL": "http://192.168.0.109:8080/v1",
            "OMNIROUTE_MODEL": "claude-3-5-sonnet",
            "OMNIROUTE_API_KEY": "секрет",
        }
    )

    endpoint = settings.as_endpoint()
    assert endpoint is not None
    assert endpoint.base_url == "http://192.168.0.109:8080/v1"
    assert endpoint.model == "claude-3-5-sonnet"
    assert endpoint.api_key.get_secret_value() == "секрет"


def test_half_configured_laptop_counts_as_absent() -> None:
    """Адрес без модели — это не «почти настроено», а «не настроено»: гадать имя модели нельзя."""
    assert LaptopLinkSettings.from_environment({"OMNIROUTE_URL": "http://x/v1"}).as_endpoint() is None
    assert LaptopLinkSettings.from_environment({}).is_configured is False
