"""
efi/utils/json_utils.py

Небольшие помощники для работы с JSON там, где отказоустойчивость (внешние
API, пользовательский ввод) важнее лаконичности стандартного
json.loads/json.dumps.
"""

from __future__ import annotations

import json
from typing import Any


def safe_json_loads(raw: str, *, default: Any = None) -> Any:
    """
    Парсит JSON, возвращая `default` вместо исключения при некорректном вводе — для мест, где сбой парсинга не должен
    ронять вызывающий код.
    """
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return default


def compact_json_dumps(value: Any) -> str:
    """
    JSON без лишних пробелов (и без экранирования не-ASCII) — компактнее для хранения в БД/логах, чем json.dumps по
    умолчанию.
    """
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


__all__ = ["safe_json_loads", "compact_json_dumps"]
