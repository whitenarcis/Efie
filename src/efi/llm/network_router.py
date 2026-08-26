"""
efi/llm/network_router.py

Связка «телефон в кармане ↔ ноутбук в домашней сети»: если ноутбук рядом и
не спит, тяжёлое думание уходит на него.

Зачем. Эфи живёт в Termux на телефоне, и всё, что она умеет, ограничено
бесплатными тирами облачных провайдеров: маленькая модель, минутные лимиты
токенов, 429 на четвёртом файле подряд. Но ноутбук в той же квартире держит
модель, которая на голову сильнее всего, что доступно телефону, — и стоит
это ноль, пока телефон в домашней Wi-Fi. Разница не в скорости: анализ
чужого репозитория и патч по трейсбэку либо получаются у сильной модели,
либо не получаются вовсе.

Отсюда два яруса:

    Tier 1 — ноутбук отвечает: архитектура, разбор чужого кода, патчи идут
             туда.
    Tier 2 — ноутбук не отвечает (спит, телефон в метро, сменилась сеть):
             прозрачно работает то, что работало раньше, — облачный кодер.
             Никаких «недоступно, попробуйте позже»: Эфи просто пишет чуть
             проще, а человек этого не замечает.

Ключевое слово — прозрачно. Модуль НЕ заменяет и не трогает существующий
конвейер (efi/dev/qwen_client.py, efi/llm/router.py): он надстраивается
сверху и в худшем случае вырождается в вызов того же самого кода, что и
раньше.

Про health-check. Проверка живости идёт с жёстким бюджетом (по умолчанию
600 мс) и кэшируется: пинговать ноутбук перед каждым из десятка файлов —
это десяток лишних задержек на ровном месте, а состояние сети за минуту
меняется редко. Отрицательный ответ кэшируется на меньшее время, чем
положительный: ноутбук, который только что проснулся, должен подхватываться
быстро, а не через пять минут.

Адрес и модель ноутбука настраиваются в .env (см. efi/config/schema.py,
LaptopLinkSettings):

    OMNIROUTE_URL=http://192.168.0.109:8080/v1
    OMNIROUTE_MODEL=claude-3-5-sonnet
"""

from __future__ import annotations

import logging
import time
from collections.abc import Awaitable, Callable

import httpx

from efi.config.schema import EndpointConfig
from efi.llm.base import LLMProvider
from efi.llm.errors import LLMError
from efi.llm.providers.openai_compatible import OpenAICompatibleProvider
from efi.llm.resilience import AttemptLog, Backend, RetryPolicy, Sleeper, with_failover
from efi.llm.schemas import LLMParams, Message, Response, Role, Session

logger = logging.getLogger(__name__)

#: Бюджет health-check'а. Больше — и проверка сама становится задержкой:
#: ноутбук в локальной сети отвечает за единицы миллисекунд, а если не
#: ответил за полсекунды, то его либо нет, либо он спит.
DEFAULT_HEALTH_TIMEOUT_SECONDS = 0.6

#: Сколько верить положительному ответу. Полминуты: ноутбук не засыпает
#: посреди минуты, а пинг перед каждым файлом — это десяток лишних задержек.
_ALIVE_TTL_SECONDS = 30.0

#: Сколько верить отрицательному. Меньше, чем положительному, и намеренно:
#: пришли домой, телефон поймал Wi-Fi — Эфи должна подхватить ноутбук через
#: несколько секунд, а не досиживать общий кэш.
_DEAD_TTL_SECONDS = 10.0

#: Что спрашивать у ноутбука. `/models` есть у любого OpenAI-совместимого
#: сервера, он дешёвый и не будит модель — в отличие от пробного запроса на
#: генерацию, который на локальной LLM означает загрузку весов в память.
_HEALTH_PATH = "/models"


class LaptopLink:
    """
    Один ноутбук: адрес, модель и знание о том, доступен ли он прямо сейчас.

    Состояние — только кэш живости; экземпляр переживает смену сети и не
    требует пересоздания при переходе телефона из домашней Wi-Fi в LTE.
    """

    def __init__(
        self,
        endpoint: EndpointConfig,
        *,
        provider: LLMProvider | None = None,
        health_timeout_seconds: float = DEFAULT_HEALTH_TIMEOUT_SECONDS,
        transport: httpx.AsyncBaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._endpoint = endpoint
        self._provider = provider or OpenAICompatibleProvider(endpoint, name="laptop")
        self._health_timeout_seconds = health_timeout_seconds
        #: Транспорт подменяется в тестах: проверка живости обязана
        #: проверяться без настоящей сети, иначе тест зависит от того, дома
        #: ли сейчас разработчик.
        self._transport = transport
        self._clock = clock
        self._alive: bool | None = None
        self._checked_at = 0.0

    @property
    def model(self) -> str:
        return self._endpoint.model

    @property
    def base_url(self) -> str:
        return self._endpoint.base_url

    @property
    def last_known_state(self) -> bool | None:
        """Что мы думали о ноутбуке в прошлый раз — без похода в сеть. None = ещё не проверяли."""
        return self._alive

    async def aclose(self) -> None:
        """Закрывает клиент ноутбука — он тоже свой собственный, а не роутерный."""
        await self._provider.aclose()

    def forget(self) -> None:
        """Сбросить кэш живости — например, после обрыва прямо посреди запроса."""
        self._alive = None
        self._checked_at = 0.0

    async def available(self) -> bool:
        """
        Отвечает ли ноутбук прямо сейчас. Результат кэшируется (см. TTL выше),
        так что вызывать можно перед каждой задачей.
        """
        now = self._clock()
        if self._alive is not None:
            ttl = _ALIVE_TTL_SECONDS if self._alive else _DEAD_TTL_SECONDS
            if now - self._checked_at < ttl:
                return self._alive

        alive = await self._probe()
        if alive != self._alive:
            logger.info(
                "network_router: ноутбук %s %s",
                self._endpoint.base_url, "на связи" if alive else "не отвечает — работаю сама",
            )
        self._alive = alive
        self._checked_at = now
        return alive

    async def _probe(self) -> bool:
        headers = {"Authorization": f"Bearer {self._endpoint.api_key.get_secret_value()}"}
        try:
            async with httpx.AsyncClient(
                base_url=self._endpoint.base_url.rstrip("/"),
                timeout=self._health_timeout_seconds,
                headers=headers,
                transport=self._transport,
            ) as client:
                response = await client.get(_HEALTH_PATH)
        except (httpx.HTTPError, OSError, TimeoutError) as exc:
            logger.debug("network_router: ноутбук не отозвался (%s)", exc)
            return False
        return response.status_code < httpx.codes.INTERNAL_SERVER_ERROR

    async def chat(self, params: LLMParams, session: Session) -> Response:
        """Запрос к ноутбуку. Модель подставляется своя — вызывающему знать её незачем."""
        return await self._provider.chat(params.model_copy(update={"model": self._endpoint.model}), session)


#: Резервный способ сделать ту же работу: то, что работало до появления
#: ноутбука (efi/dev/qwen_client.py поверх облачного кодера).
FallbackChat = Callable[[LLMParams, Session], Awaitable[Response]]


class NetworkModelRouter:
    """
    Выбор яруса и подхват задачи при обрыве.

    Единственная точка, где принимается решение «думаем на ноутбуке или
    сами». Всё остальное (SWE-движок, автопочинка) вызывает `chat` и не
    знает, где физически считалась генерация.
    """

    def __init__(
        self,
        laptop: LaptopLink | None,
        fallback: FallbackChat,
        *,
        fallback_name: str = "облачный кодер",
        policy: RetryPolicy | None = None,
        sleeper: Sleeper | None = None,
    ) -> None:
        self._laptop = laptop
        self._fallback = fallback
        self._fallback_name = fallback_name
        self._policy = policy or RetryPolicy()
        self._sleeper = sleeper

    @property
    def has_laptop(self) -> bool:
        """Настроен ли ноутбук вообще (а не доступен ли он сейчас)."""
        return self._laptop is not None

    async def aclose(self) -> None:
        """Закрывает то, чем владеет сам роутер. Резервный бэкенд закрывает тот, кто его создал."""
        if self._laptop is not None:
            await self._laptop.aclose()

    async def tier(self) -> str:
        """Кто будет думать в ближайшую минуту — для лога, дашборда и живой реплики."""
        if self._laptop is not None and await self._laptop.available():
            return f"ноутбук ({self._laptop.model})"
        return self._fallback_name

    async def chat(
        self, params: LLMParams, session: Session, *, log: AttemptLog | None = None
    ) -> Response:
        """
        Выполняет запрос на лучшем доступном ярусе.

        Порядок бэкендов определяется ДО запроса (по health-check), а
        переключение между ними — уже по факту отказа: ноутбук, уснувший
        посреди генерации, отдаёт задачу облаку, и снаружи это выглядит как
        просто чуть более долгий ответ.
        """
        backends: list[Backend[Response]] = []
        if self._laptop is not None and await self._laptop.available():
            laptop = self._laptop
            backends.append(
                Backend(name=f"ноутбук/{laptop.model}", run=lambda: laptop.chat(params, session))
            )
        backends.append(Backend(name=self._fallback_name, run=lambda: self._fallback(params, session)))

        try:
            return await with_failover(backends, policy=self._policy, sleeper=self._sleeper, log=log)
        except LLMError:
            # Ноутбук мог оборваться именно сейчас — пусть следующая задача
            # проверит его заново, а не досиживает положительный кэш.
            if self._laptop is not None:
                self._laptop.forget()
            raise


def as_fixer(
    router: NetworkModelRouter, *, max_output_tokens: int = 4096
) -> Callable[[str, str], Awaitable[str | None]]:
    """
    Роутер в виде «спроси модель текстом» — контракт, который ждут циклы
    починки (efi/dev/auto_fix.py, efi/dev/verify.py).

    Отказ модели превращается в None: для цикла починки это «правок не
    пришло», а не повод падать. Решение, что делать дальше, принимает он —
    ему виднее, остались ли круги.
    """

    async def call(system_prompt: str, request: str) -> str | None:
        params = LLMParams(model="", system_prompt=system_prompt, max_output_tokens=max_output_tokens)
        session = Session(messages=[Message(role=Role.USER, content=request)])
        try:
            response = await router.chat(params, session)
        except LLMError as exc:
            logger.warning("network_router: модель не ответила на запрос починки (%s)", exc)
            return None
        return response.text

    return call


__all__ = [
    "DEFAULT_HEALTH_TIMEOUT_SECONDS",
    "FallbackChat",
    "LaptopLink",
    "NetworkModelRouter",
    "as_fixer",
]
