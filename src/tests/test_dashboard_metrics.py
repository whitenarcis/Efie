"""
Тесты сборщика метрик LLM (efi.dashboard.metrics) и его подключения к
роутеру через `LLMRouter(metrics_sink=...)`.

Главное свойство подключения: без sink путь LLM-вызова остаётся ровно
прежним (никакой обёртки не появляется), а с sink обёртка ставится
прозрачно и не меняет результат вызова.
"""

from __future__ import annotations

from efi.config.schema import EndpointConfig, RoleRoute, TaskRole
from efi.dashboard.metrics import LLMMetricsCollector
from efi.llm.measurable import CallMetric, MeasurableLLMProvider
from efi.llm.router import LLMRouter


def _metric(**overrides: object) -> CallMetric:
    defaults: dict[str, object] = {
        "provider_name": "https://omni.example/v1::gemma",
        "operation": "chat",
        "duration_seconds": 1.5,
        "prompt_tokens": 100,
        "completion_tokens": 20,
        "cost": 0.001,
        "error": None,
    }
    defaults.update(overrides)
    return CallMetric(**defaults)  # type: ignore[arg-type]


def test_totals_aggregate_calls_tokens_and_errors() -> None:
    collector = LLMMetricsCollector()
    collector.sink(_metric())
    collector.sink(_metric(duration_seconds=2.5))
    collector.sink(_metric(error="429 Too Many Requests", duration_seconds=0.5, prompt_tokens=0, completion_tokens=0))

    totals = collector.totals()
    assert totals["calls"] == 3
    assert totals["errors"] == 1
    assert totals["prompt_tokens"] == 200
    assert totals["completion_tokens"] == 40
    assert totals["avg_seconds"] == round(4.5 / 3, 3)
    assert totals["error_rate"] == round(1 / 3, 4)


def test_aggregates_are_split_by_endpoint_and_operation() -> None:
    collector = LLMMetricsCollector()
    collector.sink(_metric())
    collector.sink(_metric(operation="embedding"))
    collector.sink(_metric(provider_name="https://api.groq.com/openai/v1::llama"))

    rows = {(row["model"], row["operation"]) for row in collector.aggregates()}
    assert rows == {("gemma", "chat"), ("gemma", "embedding"), ("llama", "chat")}


def test_endpoint_row_splits_base_url_and_model() -> None:
    collector = LLMMetricsCollector()
    collector.sink(_metric())

    row = collector.aggregates()[0]
    assert row["base_url"] == "https://omni.example/v1"
    assert row["model"] == "gemma"
    assert row["last_error"] is None


def test_recent_is_bounded_and_newest_first() -> None:
    collector = LLMMetricsCollector(history=5)
    for index in range(12):
        collector.sink(_metric(operation=f"call-{index}"))

    recent = collector.recent(limit=100)
    assert len(recent) == 5
    assert recent[0]["operation"] == "call-11"
    assert recent[-1]["operation"] == "call-7"


def test_error_is_remembered_on_the_aggregate() -> None:
    collector = LLMMetricsCollector()
    collector.sink(_metric(error="503 upstream"))

    row = collector.aggregates()[0]
    assert row["errors"] == 1
    assert row["last_error"] == "503 upstream"


# -- подключение к роутеру ---------------------------------------------------


def _routes() -> dict[TaskRole, RoleRoute]:
    endpoint = EndpointConfig(base_url="https://omni.example/v1", api_key="k", model="gemma")
    route = RoleRoute(primary=endpoint)
    return dict.fromkeys(TaskRole, route)


def test_router_without_sink_does_not_wrap_providers() -> None:
    router = LLMRouter(_routes())
    provider = router._provider_for(_routes()[TaskRole.MAIN].primary)
    assert not isinstance(provider, MeasurableLLMProvider)


def test_router_with_sink_wraps_providers_once() -> None:
    collector = LLMMetricsCollector()
    router = LLMRouter(_routes(), metrics_sink=collector.sink)

    endpoint = _routes()[TaskRole.MAIN].primary
    first = router._provider_for(endpoint)
    second = router._provider_for(endpoint)

    assert isinstance(first, MeasurableLLMProvider)
    assert first is second  # провайдеры кэшируются, обёртка не наслаивается
    assert first.name == "https://omni.example/v1::gemma"


def test_cooldown_snapshot_shows_only_active_cooldowns() -> None:
    from efi.llm.errors import LLMRateLimitError

    router = LLMRouter(_routes())
    endpoint = _routes()[TaskRole.MAIN].primary
    assert router.cooldown_snapshot() == {}

    router._mark_failure(endpoint, LLMRateLimitError("429", provider="omni", retry_after=30.0))
    snapshot = router.cooldown_snapshot()
    assert list(snapshot) == [("https://omni.example/v1", "gemma")]
    assert 0.0 < snapshot[("https://omni.example/v1", "gemma")] <= 30.0

    router._mark_success(endpoint)
    assert router.cooldown_snapshot() == {}
