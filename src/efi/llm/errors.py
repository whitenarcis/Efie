"""
efi/llm/errors.py

Иерархия исключений уровня LLM-провайдера.

Провайдеры (efi/llm/providers/*) обязаны переводить в эти типы любую
HTTP/транспортную ошибку — весь вышестоящий код (efi/llm/router.py и выше)
полагается только на эту иерархию, не зная про httpx или конкретного провайдера.

Вынесено в отдельный модуль (а не в providers/openai_compatible.py), чтобы
router.py мог опираться на исключения, не импортируя конкретную реализацию
провайдера, а сами провайдеры не зависели от router.py.
"""

from __future__ import annotations


class LLMError(Exception):
    """Базовое исключение для всех ошибок уровня LLM-провайдера."""

    def __init__(self, message: str, *, provider: str | None = None, status_code: int | None = None) -> None:
        super().__init__(message)
        self.provider = provider
        self.status_code = status_code


class LLMAuthError(LLMError):
    """401/403 — неверный, отозванный или истёкший ключ. Повторной попыткой не устраняется."""


class LLMRateLimitError(LLMError):
    """429 — превышен лимит запросов у провайдера/модели."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        status_code: int = 429,
        retry_after: float | None = None,
    ) -> None:
        super().__init__(message, provider=provider, status_code=status_code)
        #: Рекомендованная провайдером пауза в секундах (заголовок Retry-After), если он её прислал.
        self.retry_after = retry_after


class LLMServerError(LLMError):
    """5xx или сбой транспортного уровня (обрыв соединения и т.п.) — как правило, временное явление."""


class LLMTimeoutError(LLMServerError):
    """Запрос или чтение стрима превысили таймаут. Наследует LLMServerError — тоже транзиентная ошибка."""


class LLMInvalidResponseError(LLMError):
    """Ответ провайдера не удалось разобрать как ожидаемый контракт (Response / SSE-чанк)."""


__all__ = [
    "LLMError",
    "LLMAuthError",
    "LLMRateLimitError",
    "LLMServerError",
    "LLMTimeoutError",
    "LLMInvalidResponseError",
]
