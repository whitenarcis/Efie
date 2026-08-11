"""
efi/dashboard/metrics.py

Сборщик метрик LLM-вызовов — тот самый потребитель, ради которого писалась
`efi.llm.measurable.MeasurableLLMProvider`: она измеряет каждый вызов и
отдаёт `CallMetric` в синхронный `sink`, ничего не зная про дашборд.

Здесь `sink` наконец появляется: агрегаты по (эндпоинт, операция) для
таблицы состояния и кольцо последних вызовов для ленты. Всё держится в
памяти процесса и умирает вместе с ним — это индикатор "что происходит с
моделями прямо сейчас", а не система долговременного мониторинга: писать
ещё одну таблицу в SQLite ради графиков за прошлую неделю проект не просил.

Сборщик обязан быть быстрым и не бросать исключений: `sink` вызывается
внутри пути LLM-запроса, сразу после ответа модели.
"""

from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from efi.llm.measurable import CallMetric

_DEFAULT_HISTORY = 200


@dataclass(slots=True, frozen=True)
class MetricEvent:
    """Один зафиксированный вызов — строка ленты последних обращений к моделям."""

    timestamp: datetime
    provider: str
    operation: str
    duration_seconds: float
    prompt_tokens: int
    completion_tokens: int
    cost: float | None
    error: str | None

    @property
    def base_url(self) -> str:
        return self.provider.split("::", 1)[0]

    @property
    def model(self) -> str:
        parts = self.provider.split("::", 1)
        return parts[1] if len(parts) == 2 else ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "provider": self.provider,
            "base_url": self.base_url,
            "model": self.model,
            "operation": self.operation,
            "duration_seconds": round(self.duration_seconds, 3),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost": self.cost,
            "error": self.error,
        }


@dataclass(slots=True)
class _Aggregate:
    """Накопленные итоги по одной паре (эндпоинт, операция)."""

    provider: str
    operation: str
    calls: int = 0
    errors: int = 0
    total_seconds: float = 0.0
    max_seconds: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    last_at: datetime | None = None
    last_error: str | None = None

    def as_dict(self) -> dict[str, Any]:
        base_url, _, model = self.provider.partition("::")
        return {
            "provider": self.provider,
            "base_url": base_url,
            "model": model,
            "operation": self.operation,
            "calls": self.calls,
            "errors": self.errors,
            "avg_seconds": round(self.total_seconds / self.calls, 3) if self.calls else 0.0,
            "max_seconds": round(self.max_seconds, 3),
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost": round(self.cost, 6) if self.cost else 0.0,
            "last_at": self.last_at.isoformat() if self.last_at is not None else None,
            "last_error": self.last_error,
        }


class LLMMetricsCollector:
    """
    Потребитель `CallMetric`. Метод `sink` подходит под тип
    `efi.llm.measurable.MetricsSink` и передаётся в `LLMRouter(metrics_sink=...)`.
    """

    def __init__(self, *, history: int = _DEFAULT_HISTORY) -> None:
        self._aggregates: dict[tuple[str, str], _Aggregate] = {}
        self._events: deque[MetricEvent] = deque(maxlen=history)
        self._started_at = time.monotonic()

    def sink(self, metric: CallMetric) -> None:
        """Синхронный приём одной метрики (см. `MetricsSink`)."""
        event = MetricEvent(
            timestamp=datetime.now(UTC),
            provider=metric.provider_name,
            operation=metric.operation,
            duration_seconds=metric.duration_seconds,
            prompt_tokens=metric.prompt_tokens,
            completion_tokens=metric.completion_tokens,
            cost=metric.cost,
            error=metric.error,
        )
        self._events.append(event)

        key = (metric.provider_name, metric.operation)
        aggregate = self._aggregates.get(key)
        if aggregate is None:
            aggregate = _Aggregate(provider=metric.provider_name, operation=metric.operation)
            self._aggregates[key] = aggregate

        aggregate.calls += 1
        aggregate.total_seconds += metric.duration_seconds
        aggregate.max_seconds = max(aggregate.max_seconds, metric.duration_seconds)
        aggregate.prompt_tokens += metric.prompt_tokens
        aggregate.completion_tokens += metric.completion_tokens
        if metric.cost is not None:
            aggregate.cost += metric.cost
        aggregate.last_at = event.timestamp
        if metric.error is not None:
            aggregate.errors += 1
            aggregate.last_error = metric.error

    # -- чтение ------------------------------------------------------------

    def totals(self) -> dict[str, Any]:
        """Сводка по всем вызовам разом — для карточки на обзорной странице."""
        calls = sum(aggregate.calls for aggregate in self._aggregates.values())
        errors = sum(aggregate.errors for aggregate in self._aggregates.values())
        seconds = sum(aggregate.total_seconds for aggregate in self._aggregates.values())
        return {
            "calls": calls,
            "errors": errors,
            "error_rate": round(errors / calls, 4) if calls else 0.0,
            "avg_seconds": round(seconds / calls, 3) if calls else 0.0,
            "prompt_tokens": sum(aggregate.prompt_tokens for aggregate in self._aggregates.values()),
            "completion_tokens": sum(aggregate.completion_tokens for aggregate in self._aggregates.values()),
            "cost": round(sum(aggregate.cost for aggregate in self._aggregates.values()), 6),
        }

    def aggregates(self) -> list[dict[str, Any]]:
        """Итоги по каждой паре (эндпоинт, операция), самые нагруженные — первыми."""
        rows = [aggregate.as_dict() for aggregate in self._aggregates.values()]
        rows.sort(key=lambda row: (-int(row["calls"]), str(row["provider"])))
        return rows

    def recent(self, *, limit: int = 50) -> list[dict[str, Any]]:
        """Последние вызовы, свежие — первыми."""
        events = list(self._events)[-max(1, limit) :]
        events.reverse()
        return [event.as_dict() for event in events]

    def snapshot(self, *, recent_limit: int = 50) -> dict[str, Any]:
        return {
            "totals": self.totals(),
            "endpoints": self.aggregates(),
            "recent": self.recent(limit=recent_limit),
        }


__all__ = ["LLMMetricsCollector", "MetricEvent"]
