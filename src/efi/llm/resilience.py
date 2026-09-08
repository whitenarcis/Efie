"""
efi/llm/resilience.py

Устойчивость тяжёлых запросов: повторы с джиттером, потолок параллелизма и
подхват задачи резервным бэкендом прямо посреди генерации.

Зачем отдельный слой, если есть efi/llm/router.py. Роутер решает ДРУГУЮ
задачу: выбрать модель под роль и увести нагрузку с той, что недавно
отвечала 429. Он рассчитан на диалог — там ответа ждёт живой человек, и
правильная реакция на «занято» это сразу пойти к другой модели, а не ждать.
У работы с кодом всё наоборот: ждать некому (задача фоновая), а «другая
модель» часто хуже — переключение с Sonnet на маленькую модель посреди
правки означает не задержку, а другой результат. Поэтому здесь можно то,
чего нельзя в диалоге: подождать секунды и попросить ту же модель ещё раз.

Три механизма, и каждый закрывает свой класс отказа:

    RetryPolicy      — 429/5xx/таймаут: экспонента с джиттером и уважением
                       к Retry-After. Джиттер обязателен: без него пачка
                       файлов, стартовавшая одновременно, повторяется тоже
                       одновременно и снова выбивает лимит.
    ConcurrencyGate  — потолок одновременных запросов. Обработка пачки
                       файлов без него — это лучший способ получить 429 на
                       ровном месте, даже когда лимит щедрый.
    failover         — падение бэкенда целиком: задача уходит следующему по
                       списку, а не наверх исключением. Обрыв связи с
                       ноутбуком посреди генерации не должен ронять ни
                       задачу, ни процесс бота.

Модуль ничего не знает ни про Wi-Fi, ни про SWE-конвейер: на вход ему дают
корутины, на выход он отдаёт результат первой, которая справилась. Всё
поведение параметризовано и проверяется без сети и без реальных пауз
(`sleep` внедряется).
"""

from __future__ import annotations

import asyncio
import logging
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Generic, TypeVar

from efi.llm.errors import LLMError, LLMRateLimitError, LLMServerError

logger = logging.getLogger(__name__)

_T = TypeVar("_T")

#: Асинхронная пауза. Внедряется ради тестов: проверка лесенки повторов не
#: должна занимать минуту реального времени.
Sleeper = Callable[[float], Awaitable[None]]


@dataclass(slots=True, frozen=True)
class RetryPolicy:
    """
    Сколько раз и с какими паузами повторять то, что временно не получилось.

    Формула — `base ** n` с добавкой случайного джиттера, потолок на одну
    паузу. Значения по умолчанию рассчитаны на бесплатные тиры: четыре
    попытки покрывают типичное окно, за которое отпускает минутный лимит
    токенов, а полторы минуты — потолок, после которого ждать бессмысленно:
    фоновый цикл вернётся к задаче и сам.
    """

    attempts: int = 4
    base: float = 2.0
    jitter_seconds: float = 1.0
    max_delay_seconds: float = 90.0
    #: Уважать ли Retry-After от провайдера. Он знает, когда лимит отпустит,
    #: лучше любой нашей формулы, — но и его ответ режется потолком: на
    #: исчерпанном дневном лимите приходят часы.
    respect_retry_after: bool = True

    def delay_for(self, attempt: int, error: Exception | None = None) -> float:
        """Пауза перед попыткой номер `attempt` (нумерация с 1 — то есть перед ПЕРВЫМ повтором это 1)."""
        if self.respect_retry_after and isinstance(error, LLMRateLimitError) and error.retry_after:
            return min(float(error.retry_after), self.max_delay_seconds)
        backoff = self.base**attempt
        return min(backoff + random.random() * self.jitter_seconds, self.max_delay_seconds)


def is_transient(error: BaseException) -> bool:
    """
    Пройдёт ли это само. 429, 5xx и таймаут — да; отвергнутый ключ и
    несуществующая модель — нет, и повторять их значит потратить четыре
    запроса на четыре одинаковых отказа.
    """
    if isinstance(error, TimeoutError):
        return True
    if isinstance(error, LLMRateLimitError | LLMServerError):
        return True
    if isinstance(error, LLMError):
        status = error.status_code
        return status is not None and status >= 500
    return False


class ConcurrencyGate:
    """
    Потолок одновременных запросов — общий на весь SWE-конвейер.

    Именно общий, а не по бэкенду: лимит считает провайдер, и десять
    параллельных правок в десяти файлах выбьют его независимо от того, через
    какой роутер они пошли.
    """

    def __init__(self, limit: int = 2) -> None:
        if limit < 1:
            raise ValueError("предел параллелизма не может быть меньше одного")
        self._semaphore = asyncio.Semaphore(limit)
        self._limit = limit

    @property
    def limit(self) -> int:
        return self._limit

    async def run(self, operation: Callable[[], Awaitable[_T]]) -> _T:
        async with self._semaphore:
            return await operation()


@dataclass(slots=True)
class AttemptLog:
    """Что происходило по дороге к ответу — материал для лога и для живой реплики в чат."""

    backend: str = ""
    retries: int = 0
    failovers: int = 0
    errors: list[str] = field(default_factory=list)

    def note(self, error: BaseException) -> None:
        self.errors.append(str(error))


async def with_retries(
    operation: Callable[[], Awaitable[_T]],
    *,
    policy: RetryPolicy | None = None,
    name: str = "запрос",
    sleeper: Sleeper | None = None,
    log: AttemptLog | None = None,
) -> _T:
    """
    Повторяет операцию, пока отказ временный и не кончились попытки.

    Непроходящую ошибку (отвергнутый ключ, несуществующая модель) поднимает
    сразу: у неё другое лекарство, и оно не в ожидании.
    """
    policy = policy or RetryPolicy()
    sleep = sleeper or asyncio.sleep
    last: BaseException | None = None

    for attempt in range(1, policy.attempts + 1):
        try:
            return await operation()
        except Exception as exc:  # noqa: BLE001 — решение принимается по is_transient
            last = exc
            if log is not None:
                log.note(exc)
            if not is_transient(exc) or attempt == policy.attempts:
                raise
            delay = policy.delay_for(attempt, exc)
            if log is not None:
                log.retries += 1
            logger.info(
                "resilience: %s — %s, жду %.1fс и повторяю (%d/%d)",
                name, exc, delay, attempt, policy.attempts - 1,
            )
            await sleep(delay)

    # Недостижимо: последняя попытка либо возвращает, либо поднимает.
    raise last if last is not None else RuntimeError("with_retries: не осталось попыток")


@dataclass(slots=True, frozen=True)
class Backend(Generic[_T]):
    """
    Один способ выполнить задачу: имя для логов и корутина, которая её делает.

    Фабрика, а не готовая корутина: при повторе и при переключении задачу
    надо запускать заново, а корутину нельзя ждать дважды.
    """

    name: str
    run: Callable[[], Awaitable[_T]]


async def with_failover(
    backends: Sequence[Backend[_T]],
    *,
    policy: RetryPolicy | None = None,
    sleeper: Sleeper | None = None,
    log: AttemptLog | None = None,
) -> _T:
    """
    По очереди пробует бэкенды, каждый — со своими повторами.

    Смысл всей конструкции в одной фразе: обрыв связи с ноутбуком посреди
    генерации не должен ронять задачу. Первый бэкенд отвечает за качество
    (Sonnet на ноутбуке), последний — за то, что результат вообще будет.

    Непроходящая ошибка первого бэкенда — тоже повод переключиться, а не
    упасть: «нет такой модели» на ноутбуке лечится тем, что задачу доделает
    облачный кодер, и это лучше, чем ничего.
    """
    if not backends:
        raise ValueError("with_failover: не передано ни одного бэкенда")

    last: BaseException | None = None
    for index, backend in enumerate(backends):
        if log is not None:
            log.backend = backend.name
            if index:
                log.failovers += 1
        try:
            return await with_retries(
                backend.run, policy=policy, name=backend.name, sleeper=sleeper, log=log
            )
        except Exception as exc:  # noqa: BLE001 — падение бэкенда это повод для следующего, а не конец
            last = exc
            if index + 1 < len(backends):
                logger.warning(
                    "resilience: бэкенд %s не справился (%s) — передаю задачу дальше, в %s",
                    backend.name, exc, backends[index + 1].name,
                )
                continue
            logger.error("resilience: последний бэкенд %s тоже не справился: %s", backend.name, exc)
            raise

    raise last if last is not None else RuntimeError("with_failover: пустой перебор")


__all__ = [
    "AttemptLog",
    "Backend",
    "ConcurrencyGate",
    "RetryPolicy",
    "Sleeper",
    "is_transient",
    "with_failover",
    "with_retries",
]
