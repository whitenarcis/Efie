"""
efi/llm/base.py

Абстрактный интерфейс LLM-провайдера.

Прямой аналог IOpenAIChat из C++-референса: роутер (efi/llm/router.py) и весь
домен (memory/, notifications/, tools/) работают только через этот контракт,
не зная, что конкретно за ним стоит — httpx-клиент поверх OmniRoute/Groq
(efi/llm/providers/openai_compatible.py) или, в тестах, мок.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from pathlib import Path
from types import TracebackType
from typing import Self

from efi.llm.schemas import AudioTranscription, EmbeddingVector, LLMParams, Response, Session


class LLMProvider(ABC):
    """
    Контракт LLM-провайдера: нестриминговый chat-запрос и его потоковая версия.

    Реализации отвечают за HTTP, маппинг транспортных/HTTP-ошибок в исключения
    из efi.llm.errors и за сборку ответа в форму Response/Message из
    efi.llm.schemas — вызывающая сторона (в первую очередь LLMRouter) полагается
    на то, что любая ошибка провайдера приходит именно в этой форме, а не как
    сырое исключение httpx.
    """

    #: Человекочитаемое имя провайдера — для логов, метрик и Response.provider.
    name: str

    @abstractmethod
    async def chat(self, params: LLMParams, session: Session) -> Response:
        """Выполняет один нестриминговый chat-запрос и возвращает полностью собранный ответ."""
        raise NotImplementedError

    @abstractmethod
    def chat_streaming(self, params: LLMParams, session: Session) -> AsyncIterator[Response]:
        """
        Выполняет стриминговый chat-запрос.

        Возвращает асинхронный итератор: каждый следующий `Response` — это
        снимок ответа, накопленного на данный момент (аналог `AProperty<Response>`
        у референса, но выраженный идиоматично для asyncio — через yield
        растущего состояния, а не через отдельный объект с подпиской на
        изменения). Последний элемент итератора — финальный, полностью
        собранный ответ.

        Метод намеренно объявлен обычной (не `async`) функцией: реализации —
        это `async def ...: yield ...` (асинхронные генераторы), а вызов такой
        функции сам по себе синхронный и сразу возвращает async-итератор,
        не требуя `await` на вызове.
        """
        raise NotImplementedError

    @abstractmethod
    async def embedding(self, model: str, text: str) -> EmbeddingVector:
        """
        Возвращает эмбеддинг текста как вектор float. Аналог IOpenAIChat::embedding.

        Намеренно не принимает LLMParams целиком (sampling-параметры для
        эмбеддингов бессмысленны) — только имя модели и сам текст, зеркалируя
        реальную форму OpenAI-совместимого /embeddings эндпоинта.
        """
        raise NotImplementedError

    @abstractmethod
    async def transcribe_audio(self, model: str, audio_path: Path) -> AudioTranscription:
        """
        Транскрибирует аудиофайл (голосовое сообщение, звуковая дорожка
        видео-кружка) в текст. Whisper-совместимый контракт — используется
        efi/telegram/media/voice.py и video.py, чтобы превратить медиа в
        текст ДО того, как событие попадёт в Worker.
        """
        raise NotImplementedError

    @abstractmethod
    async def describe_image(self, model: str, image_path: Path, *, prompt: str) -> str:
        """
        Возвращает текстовое описание изображения (vision-запрос). Не
        переиспользует общий Session/Message-конвейер: формат
        multimodal-контента (текст + image_url) специфичен для этого
        единственного случая, и не стоит тащить его в Message — она везде
        остальном работает с обычным текстом. Используется
        efi/telegram/media/image.py.
        """
        raise NotImplementedError

    async def aclose(self) -> None:
        """Освобождает ресурсы провайдера (HTTP-соединения и т.п.). По умолчанию — no-op."""
        return None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        await self.aclose()


__all__ = ["LLMProvider"]
