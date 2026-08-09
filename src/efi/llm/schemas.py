"""
efi/llm/schemas.py

Основные Pydantic-модели для взаимодействия с LLM-провайдерами (OpenAI-совместимый
контракт: Groq, OmniRoute) и для метаданных долгосрочной памяти (дневник).

Прямые аналоги из C++-референса (Kuni):
    IOpenAIChat::Message::Role       -> Role
    IOpenAIChat::Message::ToolCall   -> ToolCall
    IOpenAIChat::Message             -> Message
    IOpenAIChat::Response::Usage     -> Usage
    IOpenAIChat::Response::Choice    -> Choice
    IOpenAIChat::Response            -> Response
    IOpenAIChat::Session             -> Session
    IOpenAIChat::Params              -> LLMParams
    IOpenAIChat::AudioTranscription  -> AudioTranscription
    Diary::EntryEx::Metadata         -> DiaryEntryMetadata
    Diary::EntryEx                   -> DiaryEntry
    Diary::EntryExAndRelatedness     -> DiaryQueryResult
    Diary::QueryOpts                 -> DiaryQueryOptions

Все модели — данные, а не поведение: провайдер-специфичная логика (HTTP-запросы,
парсинг SSE) живёт в llm/providers/* и llm/streaming.py и оперирует этими типами.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, field_validator

EmbeddingVector: TypeAlias = list[float]


class Role(StrEnum):
    """Роль автора сообщения в диалоге с LLM."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class ToolCallFunction(BaseModel):
    """Имя и аргументы вызываемой функции. Аргументы — сырой JSON-текст, как их отдаёт провайдер."""

    name: str = ""
    arguments: str = ""

    def parsed_arguments(self) -> dict[str, Any]:
        """Парсит `arguments` как JSON-объект. Бросает ValueError на некорректный/неполный JSON."""
        if not self.arguments:
            return {}
        try:
            parsed = json.loads(self.arguments)
        except json.JSONDecodeError as exc:
            raise ValueError(f"tool call arguments are not valid JSON: {self.arguments!r}") from exc
        if not isinstance(parsed, dict):
            raise ValueError(f"tool call arguments must decode to an object, got {type(parsed).__name__}")
        return parsed

    def accumulate(self, delta: ToolCallFunction) -> None:
        """Накопление стримингового дельта-чанка (SSE) поверх текущего состояния."""
        self.name += delta.name
        self.arguments += delta.arguments


class ToolCall(BaseModel):
    """Единичный вызов инструмента, запрошенный моделью."""

    id: str = ""
    index: int = 0
    type: str = ""
    function: ToolCallFunction = Field(default_factory=ToolCallFunction)

    def accumulate(self, delta: ToolCall) -> None:
        """
        Накопление стримингового дельта-чанка.

        Аналог IOpenAIChat::Message::ToolCall::operator+=: index берётся из
        дельты как есть (провайдер присылает его целиком в каждом чанке),
        остальные строковые поля конкатенируются посимвольно/попостфиксно.

        Важно: default для `type` — пустая строка, а не "function". В SSE-потоке
        `id`/`type`/`function.name` присутствуют только в первом чанке для
        данного tool call, а во всех последующих — отсутствуют (там едет только
        `index` и очередной кусок `function.arguments`). Если бы `type`
        по умолчанию был "function", он конкатенировался бы на каждом чанке
        ("functionfunctionfunction..."). Пустая строка воспроизводит семантику
        OPTIONAL-полей C++-референса (там отсутствующее поле после
        AJSON_FIELDS-парсинга — это пустая AString, а не значение по умолчанию).
        """
        self.id += delta.id
        self.index = delta.index
        self.type += delta.type
        self.function.accumulate(delta.function)


class Message(BaseModel):
    """
    Сообщение в диалоге с LLM. Совместимо по форме с OpenAI Chat Completions API.

    Поддерживает накопление стриминговых дельт через `accumulate()` —
    аналог IOpenAIChat::Message::operator+=, используемый при сборке SSE-потока
    в llm/streaming.py.
    """

    role: Role = Role.USER
    content: str = ""
    tool_call_id: str | None = None
    reasoning: str = ""
    reasoning_content: str = ""  # DeepSeek-style «сырое» поле рассуждений
    tool_calls: list[ToolCall] = Field(default_factory=list)

    def accumulate(self, delta: Message) -> None:
        """
        Сливает стриминговый дельта-чанк в текущее сообщение.

        Роль стримингового ответа всегда ASSISTANT — аналог
        util/openai_streaming.h::collectTo, где role жёстко проставляется
        сборщиком, а не берётся из чанка (провайдеры присылают роль только
        в первом чанке потока, дальше поле часто пустое).
        """
        self.role = Role.ASSISTANT
        self.content += delta.content
        self.tool_call_id = delta.tool_call_id or self.tool_call_id
        self.reasoning += delta.reasoning
        self.reasoning_content += delta.reasoning_content

        for delta_call in delta.tool_calls:
            while len(self.tool_calls) <= delta_call.index:
                self.tool_calls.append(ToolCall(index=len(self.tool_calls)))
            self.tool_calls[delta_call.index].accumulate(delta_call)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


class Usage(BaseModel):
    """Статистика использования токенов одного запроса к LLM."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    prompt_cache_hit_tokens: int = 0
    prompt_cache_miss_tokens: int = 0


class Choice(BaseModel):
    """Один вариант ответа модели (на практике почти всегда единственный, n=1)."""

    index: int = 0
    message: Message = Field(default_factory=Message)
    finish_reason: str | None = None


class Response(BaseModel):
    """Полный ответ LLM-провайдера на chat-запрос (OpenAI-совместимая форма)."""

    id: str = ""
    object: str = ""
    created: int = Field(default_factory=lambda: int(datetime.now(UTC).timestamp()))
    model: str = ""
    provider: str | None = None
    system_fingerprint: str | None = None
    choices: list[Choice] = Field(default_factory=list)
    usage: Usage = Field(default_factory=Usage)
    cost: float | None = None
    cost_details: dict[str, Any] = Field(default_factory=dict)
    prompt_tokens_details: dict[str, Any] = Field(default_factory=dict)

    @property
    def message(self) -> Message:
        """
        Первое (обычно единственное) сообщение ответа.

        Бросает явный ValueError вместо неявного IndexError — в C++-референсе
        повсеместный доступ через `.choices.at(0)` был источником крашей на
        пустых ответах провайдера; здесь ошибка формулируется по существу.
        """
        if not self.choices:
            raise ValueError("LLM response contains no choices")
        return self.choices[0].message

    @property
    def text(self) -> str:
        """Текстовое содержимое первого выбора — самый частый способ обращения к ответу."""
        return self.message.content


class Session(BaseModel):
    """
    Диалоговая сессия — упорядоченная последовательность сообщений с идентификатором.

    В отличие от референса (`struct Session: AVector<Message>`), не наследует
    list напрямую: pydantic-модели плохо сочетаются с наследованием контейнеров.
    Вместо этого даёт list-подобный интерфейс через dunder-методы, оставаясь
    обычной pydantic-моделью (валидация, (де)сериализация, вложенность).

    Note: `__iter__` переопределён и итерирует по `messages`, а не по полям
    модели — это осознанный компромисс ради удобства (`for m in session`),
    из-за которого `dict(session)` вести себя как обычно не будет.
    """

    model_config = ConfigDict(frozen=False)

    session_id: str = Field(default_factory=lambda: f"session_{uuid.uuid4().hex[:12]}")
    messages: list[Message] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def append(self, message: Message) -> None:
        self.messages.append(message)

    def extend(self, messages: Iterable[Message]) -> None:
        self.messages.extend(messages)

    def __len__(self) -> int:
        return len(self.messages)

    def __iter__(self) -> Iterator[Message]:  # type: ignore[override]
        return iter(self.messages)

    def __getitem__(self, index: int) -> Message:
        return self.messages[index]

    def __bool__(self) -> bool:
        return bool(self.messages)


class LLMParams(BaseModel):
    """
    Параметры одного запроса к LLM. Аналог IOpenAIChat::Params — инкапсулирует
    всё, что варьируется от вызова к вызову (промпт, sampling, tools), оставляя
    неизменной часть конфигурации провайдера (EndpointConfig в config/schema.py).
    """

    system_prompt: str = ""
    model: str
    max_output_tokens: int = Field(default=8192, gt=0)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    top_p: float | None = Field(default=None, gt=0.0, le=1.0)
    top_k: float | None = None
    min_p: float | None = None
    presence_penalty: float | None = None
    repetition_penalty: float | None = None
    seed: int | None = None
    tools: list[dict[str, Any]] = Field(default_factory=list, description="JSON-схемы инструментов (см. tools/base.py)")


class AudioTranscriptionSegment(BaseModel):
    """Один сегмент транскрипции аудио (аналог IOpenAIChat::AudioTranscription::Segment)."""

    id: int = 0
    seek: int = 0
    start: float = 0.0
    end: float = 0.0
    text: str = ""
    tokens: list[int] = Field(default_factory=list)
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0


class AudioTranscription(BaseModel):
    """Результат распознавания речи (Whisper-совместимый формат)."""

    task: str = ""
    language: str = ""
    language_probability: float = 0.0
    duration: float = 0.0
    duration_after_vad: float = 0.0
    text: str = ""
    segments: list[AudioTranscriptionSegment] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Память: метаданные записей дневника
# ---------------------------------------------------------------------------


class DiaryEntryMetadata(BaseModel):
    """
    Метаданные записи дневника — аналог Diary::EntryEx::Metadata.

    Хранятся вместе с текстом записи (front-matter markdown-файла или отдельная
    колонка в SQLite — конкретное решение хранения принимается в memory/diary.py,
    здесь фиксируется только контракт данных).
    """

    score: float = Field(default=0.0, description="Similarity score, используется при ранжировании результатов запроса")
    confidence: float = Field(
        default=0.0,
        ge=-1.0,
        le=1.0,
        description=(
            "-1 = заведомая ложь/шутка, 0 = теория по умолчанию, 1 = подтверждённый факт (не меняется при ночной "
            "консолидации)"
        ),
    )
    last_used: datetime | None = Field(default=None, description="None означает «запись ещё ни разу не использовалась»")
    usage_count: int = Field(default=0, ge=0)
    embedding: EmbeddingVector = Field(default_factory=list)
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description=(
            "Когда запись реально появилась в дневнике — НЕ путать с last_used (когда её последний раз "
            "нашли поиском). Записи без last_used (ещё ни разу не использовались) не должны читаться как "
            "«старые»: свежая запись, которую пока никто не искал, — это норма, а не признак устаревания."
        ),
    )

    def touch(self) -> None:
        """Отмечает использование записи: увеличивает счётчик и обновляет last_used. Аналог incrementUsageCount()."""
        self.usage_count += 1
        self.last_used = datetime.now(UTC)

    @property
    def is_ground_truth(self) -> bool:
        return self.confidence >= 1.0

    @property
    def is_marked_false(self) -> bool:
        return self.confidence <= -1.0


class DiaryEntry(BaseModel):
    """Запись дневника: идентификатор, метаданные и свободный текст. Аналог Diary::EntryEx."""

    id: str
    metadata: DiaryEntryMetadata = Field(default_factory=DiaryEntryMetadata)
    body: str = ""

    @field_validator("id")
    @classmethod
    def _id_must_be_filesystem_safe(cls, value: str) -> str:
        if not value or any(char in value for char in "/\\:*?\"<>|"):
            raise ValueError(f"diary entry id is not filesystem-safe: {value!r}")
        return value

    @property
    def filename(self) -> str:
        """Имя markdown-файла, в котором хранится запись: `<id>.md`."""
        return f"{self.id}.md"


class DiaryQueryOptions(BaseModel):
    """
    Параметры запроса к дневнику. Аналог Diary::QueryOpts (без callback-фильтра — он передаётся отдельным аргументом
    функции).
    """

    confidence_factor: float = Field(
        default=0.01, ge=0.0, le=1.0,
        description="Вес confidence при ранжировании относительно чистого cosine similarity",
    )
    max_entry_count: int = Field(default=10, ge=1)
    min_relatedness: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Нижний порог relatedness результата (аналог diaryMinRelatedness); 0.0 = без отсечки",
    )


class DiaryQueryResult(BaseModel):
    """
    Результат запроса к дневнику: запись + нормализованная релевантность.
    Аналог Diary::EntryExAndRelatedness.
    """

    entry: DiaryEntry
    relatedness: float = Field(ge=0.0, le=1.0, description="0 = не связано, 1 = дословное совпадение")

    def __lt__(self, other: DiaryQueryResult) -> bool:
        return self.relatedness < other.relatedness


__all__ = [
    "EmbeddingVector",
    "Role",
    "ToolCallFunction",
    "ToolCall",
    "Message",
    "Usage",
    "Choice",
    "Response",
    "Session",
    "LLMParams",
    "AudioTranscriptionSegment",
    "AudioTranscription",
    "DiaryEntryMetadata",
    "DiaryEntry",
    "DiaryQueryOptions",
    "DiaryQueryResult",
]
