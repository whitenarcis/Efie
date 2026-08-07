"""
efi/tools/system_tools/device.py

Termux-специфичные инструменты: статус батареи и вибрация устройства.
Перенос tools.get_battery_status/trigger_vibration из текущей реализации Эфи
на схему Kuni "один инструмент — один класс" — оба сгруппированы в одном
модуле (аналог tools/stickers.h у референса, где несколько родственных
инструментов уже живут в одном файле).

Termux не даёт Python-биндингов к своему API — общение идёт через shell-команды
`termux-battery-status`/`termux-vibrate` из пакета Termux:API. Оба инструмента
запускают внешний процесс асинхронно (asyncio.create_subprocess_exec), не
блокируя event loop, с таймаутом на случай зависшего/отсутствующего Termux:API.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from efi.tools.base import Tool, ToolContext

logger = logging.getLogger(__name__)

_SUBPROCESS_TIMEOUT_SECONDS = 10.0


async def _run_termux_api_command(*args: str) -> str:
    """
    Запускает команду termux-api и возвращает её stdout. Общая обвязка для
    battery/vibrate: таймаут, отсутствие Termux:API и ненулевой код возврата
    обрабатываются в одном месте.
    """
    try:
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(f"команда {args[0]} не найдена — установлено ли приложение Termux:API?") from exc

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=_SUBPROCESS_TIMEOUT_SECONDS)
    except TimeoutError as exc:
        process.kill()
        await process.wait()
        raise RuntimeError(f"{args[0]} не ответил за {_SUBPROCESS_TIMEOUT_SECONDS:.0f}с") from exc

    if process.returncode != 0:
        raise RuntimeError(f"{args[0]} завершился с кодом {process.returncode}: {stderr.decode(errors='replace').strip()}")

    return stdout.decode(errors="replace")


class GetBatteryStatusTool(Tool):
    """Возвращает текущий заряд и статус зарядки телефона, на котором работает Эфи."""

    name = "get_battery_status"
    description = "Проверяет уровень заряда и статус зарядки устройства, на котором ты сейчас работаешь (через Termux:API)."
    parameters = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        try:
            raw = await _run_termux_api_command("termux-battery-status")
        except RuntimeError as exc:
            logger.warning("get_battery_status: %s", exc)
            return f"error: {exc}"

        try:
            data = json.loads(raw)
            percentage = data["percentage"]
            status = data["status"]
        except (json.JSONDecodeError, KeyError) as exc:
            logger.warning("get_battery_status: unexpected termux-battery-status output (%s): %s", exc, raw[:200])
            return "error: could not parse battery status output"

        return f"Заряд батареи: {percentage}%, статус: {status}."


class TriggerVibrationTool(Tool):
    """Включает вибрацию устройства — физический "жест" в ответ на что-то важное/смешное."""

    name = "trigger_vibration"
    description = "Заставляет устройство завибрировать на заданное время. Используй нечасто, как выразительный физический жест, а не как спецэффект на каждое сообщение."
    parameters = {
        "type": "object",
        "properties": {
            "duration_ms": {
                "type": "integer",
                "description": "Длительность вибрации в миллисекундах (по умолчанию 500, от 50 до 5000)",
            },
        },
        "required": [],
        "additionalProperties": False,
    }

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        duration_ms = _coerce_duration_ms(arguments.get("duration_ms", 500))
        try:
            await _run_termux_api_command("termux-vibrate", "-d", str(duration_ms))
        except RuntimeError as exc:
            logger.warning("trigger_vibration: %s", exc)
            return f"error: {exc}"
        return f"Вибрация на {duration_ms}мс выполнена."


def _coerce_duration_ms(raw: Any) -> int:
    try:
        value = int(raw)
    except (TypeError, ValueError):
        value = 500
    return max(50, min(value, 5000))  # защита от абсурдных/потенциально вредных для устройства значений


__all__ = ["GetBatteryStatusTool", "TriggerVibrationTool"]
