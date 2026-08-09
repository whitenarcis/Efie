"""
efi/llm/router.py

Роутер задач к LLM-провайдерам по ролям — MAIN (тяжёлая модель для диалога и
личности), FAST (лёгкая/быстрая модель для роутинга, суммаризации, работы с
дневником) и VISION (медиа/служебные задачи) — с per-endpoint cooldown и
graceful fallback: 429/5xx на одной модели не должны ронять задачу, если есть
на что переключиться.

`TaskRole` и `RoleRoute` определены в efi.config.schema (там же, где
`EndpointConfig` и `LLMRolesSettings`) и реэкспортируются отсюда для удобства —
это данные конфигурации, а не поведение роутера, и конфигу нужно строить их
самостоятельно (`LLMRolesSettings.build_router()`) без обратной зависимости от
этого модуля.

Cooldown ключуется по (base_url, model), а не по провайдеру целиком: известный
на практике вывод — per-model cooldown работает надёжнее, чем per-provider,
потому что одна перегруженная модель на OmniRoute не должна блокировать
остальные модели, доступные через тот же base_url.

Общий потолок на ВЕСЬ перебор кандидатов одной роли (`chat`/`embedding`/
`transcribe_audio`/`describe_image`, через `_attempt_with_fallback`) — сумма
`timeout_seconds` всех кандидатов цепочки роли (+небольшой буфер), а не
фиксированная константа: без него цепочка из нескольких кандидатов
(primary -> fallback -> degrade_to) могла копить таймауты один за другим
(например, 30с + 30с + 30с) и превращать один ответ Эфи в минуты ожидания
собеседником — именно так и произошло на практике: и primary, и fallback
роли MAIN словили LLMTimeoutError подряд. Бюджет считается ДИНАМИЧЕСКИ по
уже настроенным `EndpointConfig.timeout_seconds` (а не берётся с потолка),
поэтому не конфликтует с ролями, где отдельный кандидат намеренно ждёт
дольше (например, VISION с timeout_seconds=60 в behavior.toml).
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from types import TracebackType
from typing import Self, TypeVar

from efi.config.schema import EndpointConfig, RoleRoute, TaskRole
from efi.llm.base import LLMProvider
from efi.llm.errors import LLMAuthError, LLMError, LLMRateLimitError, LLMTimeoutError
from efi.llm.providers.openai_compatible import OpenAICompatibleProvider
from efi.llm.schemas import AudioTranscription, EmbeddingVector, LLMParams, Response, Session

logger = logging.getLogger(__name__)

_EndpointKey = tuple[str, str]  # (base_url, model)
_T = TypeVar("_T")


class LLMRouter:
    """
    Маршрутизирует chat/chat_streaming запросы к нужной модели по роли задачи,
    прозрачно уводя нагрузку с моделей, которые недавно вернули 429/5xx/ошибку
    авторизации, на резервные — не прерывая выполнение вызывающей задачи.

    `LLMParams.model`, переданный вызывающей стороной, роутер игнорирует и
    подставляет модель выбранного кандидата — вызывающему коду не нужно знать
    конкретные имена моделей, достаточно указать роль (`TaskRole`).
    """

    def __init__(
        self,
        routes: dict[TaskRole, RoleRoute],
        *,
        default_cooldown_seconds: float = 60.0,
        rate_limit_cooldown_seconds: float = 90.0,
        auth_cooldown_seconds: float = 600.0,
        role_timeout_buffer_seconds: float = 5.0,
    ) -> None:
        missing_roles = set(TaskRole) - set(routes)
        if missing_roles:
            raise ValueError(
                "LLMRouter требует маршрут для каждой роли; отсутствуют: "
                f"{sorted(role.value for role in missing_roles)}"
            )
        self._routes = routes
        self._default_cooldown_seconds = default_cooldown_seconds
        self._rate_limit_cooldown_seconds = rate_limit_cooldown_seconds
        self._auth_cooldown_seconds = auth_cooldown_seconds
        self._role_timeout_buffer_seconds = role_timeout_buffer_seconds

        self._cooldowns: dict[_EndpointKey, float] = {}  # endpoint_key -> time.monotonic() дедлайна
        self._providers: dict[_EndpointKey, LLMProvider] = {}

    async def chat(self, role: TaskRole, params: LLMParams, session: Session) -> Response:
        """Нестриминговый запрос с автоматическим fallback между кандидатами роли."""

        async def operation(provider: LLMProvider, endpoint: EndpointConfig) -> Response:
            attempt_params = params.model_copy(update={"model": endpoint.model})
            return await provider.chat(attempt_params, session)

        return await self._attempt_with_fallback(role, operation)

    async def chat_streaming(self, role: TaskRole, params: LLMParams, session: Session) -> AsyncIterator[Response]:
        """
        Стриминговый запрос с fallback.

        Fallback возможен, только если сбой произошёл ДО того, как в поток
        ушёл хотя бы один чанк: молча подменить провайдера в середине уже
        читаемого потребителем стрима нельзя — он получил бы рассинхронизированный
        ответ (часть от одной модели, часть от другой). Если ошибка случилась
        после первого yield, она пробрасывается вызывающей стороне как есть.

        Не переиспользует `_attempt_with_fallback` (в отличие от остальных
        методов) именно из-за этого условного, а не безусловного fallback —
        общий алгоритм рассчитан на операции, которые либо полностью успешны,
        либо полностью проваливаются, а стриминг может провалиться "наполовину".
        """
        last_error: LLMError | None = None
        for endpoint in self._candidates(role):
            provider = self._provider_for(endpoint)
            attempt_params = params.model_copy(update={"model": endpoint.model})
            yielded_anything = False
            try:
                async for response in provider.chat_streaming(attempt_params, session):
                    yielded_anything = True
                    yield response
            except LLMError as exc:
                if yielded_anything:
                    raise
                last_error = exc
                self._mark_failure(endpoint, exc)
                continue
            self._mark_success(endpoint)
            return
        raise last_error or LLMError(f"no endpoints configured for role {role.value}")

    async def embedding(self, role: TaskRole, text: str) -> EmbeddingVector:
        """
        Возвращает эмбеддинг текста, подбирая модель по роли — с тем же
        per-endpoint cooldown/fallback, что и `chat`. Используется
        memory/rag.py на критическом пути перед ответом, поэтому важно, чтобы
        перегруженная embedding-модель не блокировала задачу целиком.
        """

        async def operation(provider: LLMProvider, endpoint: EndpointConfig) -> EmbeddingVector:
            return await provider.embedding(endpoint.model, text)

        return await self._attempt_with_fallback(role, operation)

    async def transcribe_audio(self, role: TaskRole, audio_path: Path) -> AudioTranscription:
        """
        Транскрибирует аудиофайл в текст, подбирая модель по роли (обычно VISION). Используется efi/telegram/media/.
        """

        async def operation(provider: LLMProvider, endpoint: EndpointConfig) -> AudioTranscription:
            return await provider.transcribe_audio(endpoint.model, audio_path)

        return await self._attempt_with_fallback(role, operation)

    async def describe_image(self, role: TaskRole, image_path: Path, *, prompt: str) -> str:
        """
        Возвращает текстовое описание изображения, подбирая модель по роли (обычно VISION). Используется
        efi/telegram/media/image.py.
        """

        async def operation(provider: LLMProvider, endpoint: EndpointConfig) -> str:
            return await provider.describe_image(endpoint.model, image_path, prompt=prompt)

        return await self._attempt_with_fallback(role, operation)

    async def _attempt_with_fallback(
        self,
        role: TaskRole,
        operation: Callable[[LLMProvider, EndpointConfig], Awaitable[_T]],
    ) -> _T:
        """
        Общий алгоритм fallback для операций, которые либо полностью успешны,
        либо полностью проваливаются (chat/embedding/transcribe_audio/
        describe_image — но НЕ chat_streaming, см. её докстринг), зажатый в
        общий бюджет времени на всю цепочку кандидатов (см. докстринг модуля).
        """
        candidates = self._candidates(role)
        budget = sum(endpoint.timeout_seconds for endpoint in candidates) + self._role_timeout_buffer_seconds
        try:
            return await asyncio.wait_for(self._attempt_candidates(candidates, operation), timeout=budget)
        except TimeoutError as exc:
            raise LLMTimeoutError(
                f"role {role.value}: exceeded overall budget of {budget:.0f}s across {len(candidates)} candidate(s)",
                provider=role.value,
            ) from exc

    async def _attempt_candidates(
        self,
        candidates: list[EndpointConfig],
        operation: Callable[[LLMProvider, EndpointConfig], Awaitable[_T]],
    ) -> _T:
        """
        Перебирает уже готовый список кандидатов (см. _candidates): на
        LLMError переключается на следующего, помечая его в cooldown.
        """
        last_error: LLMError | None = None
        for endpoint in candidates:
            provider = self._provider_for(endpoint)
            try:
                result = await operation(provider, endpoint)
            except LLMError as exc:
                last_error = exc
                self._mark_failure(endpoint, exc)
                continue
            self._mark_success(endpoint)
            return result
        assert last_error is not None  # candidates гарантированно непусто — см. _candidates
        raise last_error

    async def aclose(self) -> None:
        """Закрывает все созданные провайдеры (и их httpx-клиенты)."""
        results = await asyncio.gather(
            *(provider.aclose() for provider in self._providers.values()), return_exceptions=True
        )
        for result in results:
            if isinstance(result, BaseException):
                logger.warning("llm_router: error while closing a provider during aclose()", exc_info=result)

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()

    # -- подбор кандидатов и cooldown ----------------------------------

    def _candidates(self, role: TaskRole) -> list[EndpointConfig]:
        """
        Упорядоченный список кандидатов для роли: сперва не находящиеся в
        cooldown (в порядке объявления в RoleRoute-цепочке), затем — те, что
        в cooldown, но отсортированные по времени истечения (скорее освободится —
        раньше пробуем). Так `chat`/`chat_streaming` почти никогда не остаются
        совсем без попытки — это и есть «не ломать выполнение задачи».
        """
        chain = _dedupe_by_key(self._resolve_chain(role, visited=set()), key=_endpoint_key)
        if not chain:
            raise LLMError(f"no endpoints configured for role {role.value}")

        now = time.monotonic()
        available = [endpoint for endpoint in chain if self._cooldown_until(endpoint) <= now]
        cooling_down = sorted(
            (endpoint for endpoint in chain if self._cooldown_until(endpoint) > now),
            key=self._cooldown_until,
        )
        return available + cooling_down

    def _resolve_chain(self, role: TaskRole, visited: set[TaskRole]) -> list[EndpointConfig]:
        if role in visited:
            return []  # защита от циклической деградации (MAIN -> FAST -> MAIN -> ...)
        visited.add(role)

        route = self._routes[role]
        chain = [route.primary]
        if route.fallback is not None:
            chain.append(route.fallback)
        if route.degrade_to is not None:
            chain.extend(self._resolve_chain(route.degrade_to, visited))
        return chain

    def _cooldown_until(self, endpoint: EndpointConfig) -> float:
        return self._cooldowns.get(_endpoint_key(endpoint), 0.0)

    def _mark_success(self, endpoint: EndpointConfig) -> None:
        self._cooldowns.pop(_endpoint_key(endpoint), None)

    def _mark_failure(self, endpoint: EndpointConfig, exc: LLMError) -> None:
        key = _endpoint_key(endpoint)
        if isinstance(exc, LLMAuthError):
            # Неверный ключ сам себя не починит — держим кандидата в стороне подольше.
            duration = self._auth_cooldown_seconds
        elif isinstance(exc, LLMRateLimitError):
            duration = exc.retry_after or self._rate_limit_cooldown_seconds
        else:
            duration = self._default_cooldown_seconds

        self._cooldowns[key] = time.monotonic() + duration
        logger.warning("llm_router: %s cooling down for %.0fs after %s: %s", key, duration, type(exc).__name__, exc)

    def _provider_for(self, endpoint: EndpointConfig) -> LLMProvider:
        key = _endpoint_key(endpoint)
        provider = self._providers.get(key)
        if provider is None:
            provider = OpenAICompatibleProvider(endpoint, name=f"{key[0]}::{key[1]}")
            self._providers[key] = provider
        return provider


def _endpoint_key(endpoint: EndpointConfig) -> _EndpointKey:
    return (endpoint.base_url, endpoint.model)


def _dedupe_by_key(
    items: list[EndpointConfig], *, key: Callable[[EndpointConfig], _EndpointKey]
) -> list[EndpointConfig]:
    """
    Убирает повторы, сохраняя порядок первого вхождения — тот же эндпоинт
    может встретиться дважды при деградации между ролями (например, если
    fallback роли MAIN совпадает с primary роли FAST).
    """
    seen: set[_EndpointKey] = set()
    result: list[EndpointConfig] = []
    for item in items:
        item_key = key(item)
        if item_key in seen:
            continue
        seen.add(item_key)
        result.append(item)
    return result


__all__ = ["TaskRole", "RoleRoute", "LLMRouter"]
