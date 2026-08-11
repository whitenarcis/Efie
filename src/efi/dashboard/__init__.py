"""
efi/dashboard/

Веб-дашборд Эфи: подробные логи, состояние всех подсистем и её самой,
просмотр дневника и всей накопленной памяти.

Устройство пакета:
    http.py      — минимальный асинхронный HTTP/1.1-сервер (без внешних зависимостей)
    logbus.py    — кольцевой буфер записей логирования + живая подписка (SSE)
    metrics.py   — сборщик метрик LLM-вызовов (потребитель efi.llm.measurable.CallMetric)
    queries.py   — read-only SQL к общей базе для табличных разделов
    snapshot.py  — DashboardContext и сборка снимков состояния подсистем
    api.py       — маршруты /api/*
    server.py    — DashboardServer: сборка маршрутов, статика, доступ по токену
    static/      — сам интерфейс (HTML/CSS/JS без сборки и без CDN)

Почему свой HTTP-сервер, а не FastAPI/aiohttp: Эфи рассчитана на запуск в том
числе в Termux на телефоне, и тянуть ради страницы состояния веб-фреймворк с
транзитивными зависимостями (а с ним и uvicorn) — несоразмерная цена. Нужен
GET, отдача статики и SSE; всё это укладывается в один небольшой модуль
поверх asyncio.start_server.
"""

from __future__ import annotations

from efi.dashboard.logbus import LogBuffer, LogEntry
from efi.dashboard.metrics import LLMMetricsCollector
from efi.dashboard.server import DashboardServer
from efi.dashboard.snapshot import DashboardContext

__all__ = [
    "DashboardContext",
    "DashboardServer",
    "LLMMetricsCollector",
    "LogBuffer",
    "LogEntry",
]
