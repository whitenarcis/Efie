"""
efi/tools/web_tools/get_weather.py

Погода через Open-Meteo — бесплатный API без ключа и без регистрации
(geocoding + forecast, отдельные бесплатные эндпоинты одного проекта).

Отдельный инструмент, а не часть web_search: DuckDuckGo (см. web_search.py)
не даёт надёжных ЖИВЫХ числовых данных вроде текущей температуры — сниппеты
поиска могут быть закэшированы/устаревшими. Погода — самый частый на
практике пример такого запроса ("почекай погоду", "сколько там градусов") —
для него нужен прямой API, а не поиск.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

from efi.tools.base import Tool, ToolContext

logger = logging.getLogger(__name__)

_GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_REQUEST_TIMEOUT_SECONDS = 10.0

# Упрощённая расшифровка кодов погоды WMO (стандарт, который использует
# Open-Meteo) — только самые частые случаи, этого достаточно для
# разговорного описания, не для метеорологической точности.
_WEATHER_CODE_DESCRIPTIONS: dict[int, str] = {
    0: "ясно", 1: "малооблачно", 2: "переменная облачность", 3: "пасмурно",
    45: "туман", 48: "изморозь",
    51: "морось", 53: "морось", 55: "сильная морось",
    61: "небольшой дождь", 63: "дождь", 65: "сильный дождь",
    71: "небольшой снег", 73: "снег", 75: "сильный снегопад",
    80: "ливень", 81: "сильный ливень", 82: "очень сильный ливень",
    95: "гроза", 96: "гроза с градом", 99: "сильная гроза с градом",
}


class GetWeatherTool(Tool):
    """Смотрит текущую погоду в указанном городе/месте через Open-Meteo — без ключа, без лимитов оплаты."""

    name = "get_weather"
    description = (
        "Смотрит АКТУАЛЬНУЮ погоду (температуру, осадки, ветер, влажность) в указанном городе прямо сейчас. "
        "Обязательно используй этот инструмент, если тебя просят 'почекать погоду', 'сколько градусов', "
        "'какая погода' и т.п. — НЕ придумывай цифры от себя и не отшучивайся вместо ответа, вызови "
        "инструмент и назови реальное значение."
    )
    parameters = {
        "type": "object",
        "properties": {
            "location": {"type": "string", "description": "Название города или места, например 'Тамбов' или 'Moscow'"},
        },
        "required": ["location"],
        "additionalProperties": False,
    }

    def __init__(self, *, client: httpx.AsyncClient | None = None) -> None:
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(timeout=httpx.Timeout(_REQUEST_TIMEOUT_SECONDS))

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def execute(self, arguments: dict[str, Any], context: ToolContext) -> str:
        location = str(arguments.get("location", "")).strip()
        if not location:
            return "error: location must not be empty"

        coordinates = await self._geocode(location)
        if coordinates is None:
            return f"error: could not find location {location!r}"
        latitude, longitude, resolved_name = coordinates

        try:
            response = await self._client.get(
                _FORECAST_URL,
                params={
                    "latitude": latitude,
                    "longitude": longitude,
                    "current": "temperature_2m,weathercode,windspeed_10m,relative_humidity_2m",
                    "timezone": "auto",
                },
            )
            response.raise_for_status()
            data = response.json()
        except httpx.HTTPError as exc:
            logger.warning("get_weather: forecast request failed for %r: %s", location, exc)
            return f"error: weather request failed: {exc}"
        except ValueError as exc:
            logger.warning("get_weather: could not parse forecast response for %r: %s", location, exc)
            return "error: weather service returned an unparseable response"

        return _format_weather(resolved_name, data)

    async def _geocode(self, location: str) -> tuple[float, float, str] | None:
        try:
            response = await self._client.get(_GEOCODING_URL, params={"name": location, "count": 1, "language": "ru"})
            response.raise_for_status()
            data = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            logger.warning("get_weather: geocoding failed for %r: %s", location, exc)
            return None

        results = data.get("results") or []
        if not results:
            return None
        first = results[0]
        try:
            return float(first["latitude"]), float(first["longitude"]), str(first.get("name", location))
        except (KeyError, TypeError, ValueError):
            return None


def _format_weather(location_name: str, data: dict[str, Any]) -> str:
    current = data.get("current") or {}
    temperature = current.get("temperature_2m")
    wind_speed = current.get("windspeed_10m")
    humidity = current.get("relative_humidity_2m")
    weather_code = current.get("weathercode")

    if temperature is None:
        return f"error: no current weather data available for {location_name!r}"

    description = _WEATHER_CODE_DESCRIPTIONS.get(int(weather_code), "") if weather_code is not None else ""
    parts = [f"{location_name}: {temperature}°C"]
    if description:
        parts.append(description)
    if humidity is not None:
        parts.append(f"влажность {humidity}%")
    if wind_speed is not None:
        parts.append(f"ветер {wind_speed} км/ч")
    return ", ".join(parts)


__all__ = ["GetWeatherTool"]
