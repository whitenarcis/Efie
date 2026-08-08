"""
efi/config/schema.py

Типизированная конфигурация проекта на pydantic-settings.

Приоритет источников (от высшего к низшему), как и в предыдущей реализации
(config_loader.py): явные значения при создании объекта > переменные окружения
(.env / os.environ, с префиксом ``EFI_`` и разделителем вложенности ``__``) >
TOML-файл (``behavior.toml``) > значения по умолчанию, заданные в моделях ниже.

Пример переопределения через переменные окружения:
    EFI_TELEGRAM__API_HASH=xxx
    EFI_LLM_ROLES__MAIN__PRIMARY__API_KEY=xxx
    EFI_HUMANIZER__TYPING_WPM_MAX=160

Пример секции TOML (``behavior.toml``):
    [telegram]
    api_id = 123456
    owner_id = 625207005

    # Основной провайдер инфраструктуры — OmniRoute; три роли (MAIN/FAST/VISION)
    # покрывают задачи разной "тяжести", см. TaskRole/RoleRoute/LLMRolesSettings ниже.
    [llm_roles.main]
    degrade_to = "fast"

    [llm_roles.main.primary]
    base_url = "https://omni.thegoyhole.fun/v1"
    api_key = "..."
    model = "google/gemma-4-31b-it:free"

    [llm_roles.main.fallback]
    base_url = "https://api.groq.com/openai/v1"
    api_key = "..."
    model = "llama-3.1-8b-instant"

    [llm_roles.fast.primary]
    base_url = "https://api.groq.com/openai/v1"
    api_key = "..."
    model = "llama-3.1-8b-instant"

    [llm_roles.vision.primary]
    base_url = "https://omni.thegoyhole.fun/v1"
    api_key = "..."
    model = "qwen/qwen3.6-27b"

    [humanizer]
    typing_wpm_min = 120
    typing_wpm_max = 150

Примечание: для чтения TOML требуется пакет ``pydantic-settings[toml]``
(на Python 3.11+ он использует стандартный ``tomllib`` под капотом).

Загрузка конфигурации намеренно синхронна: чтение .env/TOML — одноразовая
операция на старте процесса, до создания event loop, поэтому async здесь не
даёт выигрыша и только усложнил бы инициализацию.
"""

from __future__ import annotations

import os
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    TomlConfigSettingsSource,
)

if TYPE_CHECKING:
    # Только для тайп-чекера — реальный импорт делается лениво внутри build_router(),
    # чтобы не создавать цикл: efi.llm.router импортирует типы из этого модуля.
    from efi.llm.router import LLMRouter

# Корень пакета: .../efi/efi/config/schema.py -> .../efi/efi -> .../efi (корень репозитория)
_PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = _PACKAGE_ROOT.parent
_DEFAULT_TOML_PATH = PROJECT_ROOT / "behavior.toml"

# Маркеры внешнего хранилища Android, которое Termux иногда монтирует в режиме,
# не поддерживающем sqlite journal/WAL-файлы (известная проблема: readonly database).
_TERMUX_READONLY_MARKERS = ("/sdcard", "/mnt/sdcard", "/storage/emulated")


def _termux_safe_path(path: Path) -> Path:
    """Переносит файл во внутреннее хранилище Termux, если путь указывает на /sdcard и т.п."""
    if any(marker in str(path) for marker in _TERMUX_READONLY_MARKERS):
        return Path.home() / ".efi_data" / path.name
    return path


class Environment(str, Enum):
    """Окружение исполнения — влияет на уровень логирования, отладочные тулы и т.п."""

    DEVELOPMENT = "development"
    PRODUCTION = "production"


class LockdownMode(str, Enum):
    """
    Режим доступа к личности Эфи, аналог Config::LockdownMode из референса.

    NONE           — публичный режим, отвечает в любом чате.
    CONTACTS_ONLY  — отвечает только контактам аккаунта.
    OWNER_ONLY     — отвечает только владельцу (папику); максимально закрытый режим.
    """

    NONE = "none"
    CONTACTS_ONLY = "contacts_only"
    OWNER_ONLY = "owner_only"


class PathsSettings(BaseModel):
    """
    Системные пути проекта.

    Каталоги (``data_dir``, ``diary_dir`` и т.п.) — вычисляемые свойства
    относительно ``base_dir``, а не хранимые поля: это исключает рассинхронизацию
    между «где лежит база» и «где лежит дневник», если кто-то поменяет только
    один путь через .env.
    """

    model_config = ConfigDict(frozen=True)

    base_dir: Path = Field(default=PROJECT_ROOT, description="Корень проекта")

    data_dir_name: str = "data"
    diary_dir_name: str = "diary"
    cache_dir_name: str = "cache"
    logs_dir_name: str = "logs"

    db_filename: str = "efi.db"
    session_name: str = "efi_session"
    thoughts_filename: str = "thoughts.txt"
    profile_filename: str = "user_profile.json"
    voice_reply_filename: str = "voice_reply.ogg"
    selfie_filename: str = "efi_selfie.png"

    @property
    def data_dir(self) -> Path:
        return self.base_dir / self.data_dir_name

    @property
    def diary_dir(self) -> Path:
        return self.data_dir / self.diary_dir_name

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / self.cache_dir_name

    @property
    def logs_dir(self) -> Path:
        return self.base_dir / self.logs_dir_name

    @property
    def db_path(self) -> Path:
        return _termux_safe_path(self.data_dir / self.db_filename)

    @property
    def session_path(self) -> Path:
        return _termux_safe_path(self.data_dir / self.session_name)

    @property
    def thoughts_path(self) -> Path:
        return self.data_dir / self.thoughts_filename

    @property
    def profile_path(self) -> Path:
        return self.data_dir / self.profile_filename

    @property
    def voice_reply_path(self) -> Path:
        return self.data_dir / self.voice_reply_filename

    @property
    def selfie_path(self) -> Path:
        return self.data_dir / self.selfie_filename

    def ensure_directories(self) -> None:
        """Создаёт все необходимые директории. Вызывается один раз при старте приложения."""
        for directory in (self.data_dir, self.diary_dir, self.cache_dir, self.logs_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_path.parent.mkdir(parents=True, exist_ok=True)


class TelegramSettings(BaseModel):
    """Параметры подключения Pyrogram (userbot-режим, MTProto)."""

    model_config = ConfigDict(frozen=True)

    api_id: int = Field(..., description="Telegram API ID, my.telegram.org")
    api_hash: SecretStr = Field(..., description="Telegram API hash, my.telegram.org")
    phone_number: SecretStr | None = Field(default=None, description="Нужен только при первой интерактивной авторизации")

    owner_id: int = Field(..., description="Telegram ID владельца — единственный безусловно доверенный собеседник")
    owner_display_name: str | None = Field(
        default=None,
        description="Как обращаться к владельцу в личности (подстановка {user_name} в personality.md). "
        "Если не задано — берётся Telegram-имя отправителя, когда он владелец, либо общее 'создатель'.",
    )
    allowed_chats: list[int] = Field(default_factory=list, description="Явный allowlist чатов помимо владельца")
    chat_labels: dict[int, str] = Field(default_factory=dict, description="Человекочитаемые метки чатов для контекста LLM")

    lockdown_mode: LockdownMode = LockdownMode.OWNER_ONLY
    check_chats_on_startup: bool = True
    can_join_chats: bool = False
    can_leave_chats: bool = True


class EndpointConfig(BaseModel):
    """
    Конфигурация одного LLM-эндпоинта: базовый URL + ключ + модель + таймаут.

    Прямой аналог связки Endpoint/EndpointAndModel из C++-референса — все
    параметры подключения к конкретному провайдеру инкапсулированы в одном
    объекте, который целиком передаётся в llm/providers/*.
    """

    model_config = ConfigDict(frozen=True)

    base_url: str
    api_key: SecretStr
    model: str
    timeout_seconds: float = Field(default=30.0, gt=0.0)


class GroqSettings(EndpointConfig):
    """
    Groq — быстрый провайдер для роутинга интентов и суммаризации.
    Не используется для генерации ответов от лица личности Эфи.
    """

    base_url: str = "https://api.groq.com/openai/v1"
    model: str = "llama-3.1-8b-instant"


class OmniRouteSettings(EndpointConfig):
    """
    OmniRoute — самохостируемый прокси на VPS, ротирующий аккаунты OpenRouter.
    Основной канал инференса личности Эфи.
    """

    base_url: str = "https://omni.thegoyhole.fun/v1"
    model: str = "google/gemma-4-31b-it:free"


class TaskRole(str, Enum):
    """
    Роль задачи, под которую подбирается модель.

    Основной провайдер инфраструктуры — OmniRoute; три роли ниже разделяют
    задачи разной "тяжести" и стоимости, чтобы не гонять всё через одну модель:
    """

    #: Тяжёлая модель для диалога и формирования личности Эфи.
    MAIN = "main"
    #: Лёгкая/быстрая модель для роутинга интентов, фоновой суммаризации, работы с памятью/дневником
    #: (например, Groq, либо облегчённая модель на том же OmniRoute).
    FAST = "fast"
    #: Модель для обработки медиа (vision) или прочих служебных задач.
    VISION = "vision"


class RoleRoute(BaseModel):
    """
    Маршрутизация одной роли: основной эндпоинт, необязательный резервный
    эндпоинт того же уровня и необязательная деградация на другую роль.

    Порядок перебора кандидатов при сбое (см. efi.llm.router.LLMRouter):
    `primary` -> `fallback` -> (рекурсивно) кандидаты роли `degrade_to`.
    Например, для MAIN можно задать `fallback` — Groq, на случай 429/5xx у
    основной модели, и `degrade_to=TaskRole.FAST` — как крайний случай, если
    недоступны и primary, и fallback.
    """

    model_config = ConfigDict(frozen=True)

    primary: EndpointConfig
    fallback: EndpointConfig | None = None
    degrade_to: TaskRole | None = None


class LLMRolesSettings(BaseModel):
    """
    Конфигурация всех трёх ролей LLM разом — единая точка входа для сборки LLMRouter.

    Типичная схема при основном провайдере OmniRoute:
        main.primary    -> OmniRoute, тяжёлая модель личности
        main.fallback   -> Groq (или облегчённая модель OmniRoute) на случай 429/5xx
        main.degrade_to -> TaskRole.FAST, как крайний случай
        fast.primary    -> Groq — быстрый роутинг/суммаризация/работа с дневником
        vision.primary  -> OmniRoute (или другой провайдер) с vision-моделью
    """

    model_config = ConfigDict(frozen=True)

    main: RoleRoute
    fast: RoleRoute
    vision: RoleRoute

    def as_routes(self) -> dict[TaskRole, RoleRoute]:
        """Приводит конфигурацию к виду, который принимает конструктор `LLMRouter`."""
        return {
            TaskRole.MAIN: self.main,
            TaskRole.FAST: self.fast,
            TaskRole.VISION: self.vision,
        }

    def build_router(self, **router_kwargs: Any) -> "LLMRouter":
        """
        Собирает `LLMRouter` из текущей конфигурации ролей.

        Импорт `LLMRouter` — намеренно локальный (внутри метода), а не на
        уровне модуля: `efi.llm.router` зависит от типов, определённых здесь
        (`EndpointConfig`, `TaskRole`, `RoleRoute`), поэтому импорт в обратную
        сторону на уровне модуля создал бы цикл. `router_kwargs` пробрасываются
        в конструктор `LLMRouter` как есть (например, `default_cooldown_seconds=`).
        """
        from efi.llm.router import LLMRouter

        return LLMRouter(self.as_routes(), **router_kwargs)


class HumanizerSettings(BaseModel):
    """
    Параметры «очеловечивания» вывода — прямой ответ на главную цель проекта:
    сделать диалог и поведение Эфи неотличимыми от реального человека.

    Объединяет симуляцию набора текста, генерацию опечаток и защиту от
    самоповторов (аналог util/typos.h и полей typingSimulation*/antiRepeat*
    из config.h референсного проекта).
    """

    model_config = ConfigDict(frozen=True)

    # --- Симуляция набора текста ---
    typing_wpm_min: int = Field(default=120, gt=0, description="Минимальная скорость набора, слов/мин")
    typing_wpm_max: int = Field(default=150, gt=0, description="Максимальная скорость набора, слов/мин")
    typing_thinking_pause_min_seconds: float = Field(default=1.0, ge=0.0, description="Пауза «осмысления» перед набором")
    typing_thinking_pause_max_seconds: float = Field(default=2.2, ge=0.0)
    typing_delay_min_seconds: float = Field(default=1.8, ge=0.0, description="Нижний предел суммарной задержки ответа")
    typing_delay_max_seconds: float = Field(default=7.0, gt=0.0, description="Верхний предел суммарной задержки ответа")

    # --- Опечатки ---
    typo_probability: float = Field(
        default=0.04, ge=0.0, le=1.0,
        description="Шанс алгоритмической опечатки на кусок сообщения (пропуск/сосед по клавише/перестановка "
        "соседних букв — см. efi/humanizer/typos.py); рекомендованный диапазон 3-5%",
    )
    typo_min_text_length: int = Field(default=10, ge=0, description="Не портим опечаткой слишком короткие сообщения")
    typo_self_correct_probability: float = Field(
        default=0.5, ge=0.0, le=1.0,
        description="Из тех сообщений, где случилась опечатка, доля тех, что она сама 'замечает' и исправляет "
        "через edit_message спустя короткую паузу — как реальный человек. Остальные остаются неисправленными "
        "(реальные люди тоже не всегда себя вычитывают).",
    )
    keyboard_neighbors: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Раскладка соседних клавиш (RU/EN) для правдоподобных опечаток; заполняется из behavior.toml",
    )

    # --- Защита от самоповторов ---
    anti_repeat_trigger_max: float = Field(default=0.95, ge=0.0, le=1.0, description="Порог схожести с любым из последних N сообщений")
    anti_repeat_trigger_avg: float = Field(default=0.85, ge=0.0, le=1.0, description="Порог средней схожести с последними N сообщениями")
    anti_repeat_max_history: int = Field(default=32, ge=1, description="Глубина истории для проверки на повторы")

    # --- Разбивка ответа на несколько сообщений ---
    max_messages_per_burst: int = Field(default=5, ge=1, description="Максимум сообщений в одной серии (///-разрывы)")

    # --- Anti-interrupt: группировка быстрых сообщений собеседника, ориентируясь
    # на живой статус "печатает" (efi/telegram/typing_tracker.py + debounce.py) ---
    debounce_post_typing_min_seconds: float = Field(
        default=0.1, ge=0.0,
        description="Минимальная пауза после того, как собеседник перестал печатать, перед реакцией",
    )
    debounce_post_typing_max_seconds: float = Field(
        default=1.0, gt=0.0,
        description="Максимальная пауза после того, как собеседник перестал печатать, перед реакцией",
    )
    debounce_typing_poll_interval_seconds: float = Field(
        default=0.3, gt=0.0,
        description="Как часто перепроверять статус 'печатает', пока он активен",
    )
    debounce_typing_ttl_seconds: float = Field(
        default=6.0, gt=0.0,
        description="Сколько секунд без нового сигнала считать статус 'печатает' ещё актуальным (Telegram обновляет его каждые ~5-6с)",
    )
    debounce_fallback_delay_seconds: float = Field(
        default=2.0, ge=0.0,
        description="Обычный таймер тишины, если статус 'печатает' вообще не отслеживается (TypingTracker не сработал ни разу для чата)",
    )
    debounce_max_wait_seconds: float = Field(
        default=15.0, gt=0.0,
        description="Жёсткий потолок ожидания от первого сообщения пачки — не даёт активному собеседнику бесконечно откладывать ответ",
    )

    @model_validator(mode="after")
    def _validate_ranges(self) -> "HumanizerSettings":
        if self.typing_wpm_min > self.typing_wpm_max:
            raise ValueError("typing_wpm_min не может быть больше typing_wpm_max")
        if self.typing_thinking_pause_min_seconds > self.typing_thinking_pause_max_seconds:
            raise ValueError("typing_thinking_pause_min_seconds не может быть больше *_max_seconds")
        if self.typing_delay_min_seconds > self.typing_delay_max_seconds:
            raise ValueError("typing_delay_min_seconds не может быть больше typing_delay_max_seconds")
        if self.debounce_post_typing_min_seconds > self.debounce_post_typing_max_seconds:
            raise ValueError("debounce_post_typing_min_seconds не может быть больше debounce_post_typing_max_seconds")
        return self

    def characters_per_second_range(self) -> tuple[float, float]:
        """Переводит WPM в диапазон символов/сек (1 «слово» ≈ 5 символов — стандартная метрика WPM)."""
        chars_per_word = 5.0
        return (
            self.typing_wpm_min * chars_per_word / 60.0,
            self.typing_wpm_max * chars_per_word / 60.0,
        )


class MemorySettings(BaseModel):
    """
    Параметры подсистемы памяти (efi/memory/) — пороги дневника и лимиты RAG-поиска.
    Аналоги diaryPlagiarismThreshold/diaryMinRelatedness из референса.
    """

    model_config = ConfigDict(frozen=True)

    diary_dir: Path | None = Field(
        default=None,
        description="Переопределяет paths.diary_dir, если задано; иначе используется вычисляемый путь из PathsSettings",
    )
    plagiarism_threshold: float = Field(
        default=0.97,
        ge=0.0,
        le=1.0,
        description="Порог relatedness, выше которого новая запись дневника считается дублем существующей (diaryPlagiarismThreshold)",
    )
    min_relatedness: float = Field(
        default=0.80,
        ge=0.0,
        le=1.0,
        description="Нижний порог relatedness для результатов RAG-поиска (diaryMinRelatedness); ниже — запись не считается релевантной",
    )
    max_rag_results: int = Field(default=10, ge=1, description="Максимум записей, возвращаемых RAG-поиском за один запрос")
    history_limit: int = Field(
        default=30, ge=1,
        description="Сколько последних сообщений диалога подмешивать в каждый запрос к LLM (Worker.history_limit). "
        "Больше — лучше короткая память в активном разговоре, но и больше риск упереться в TPM-лимиты "
        "узких бесплатных тиров (см. Worker._chat_with_size_retry, который подрезает историю при 413).",
    )
    novelization_lookback_days: int = Field(
        default=1, ge=1,
        description="На сколько дней назад заглядывать при первой ночной новеллизации чата, если для него ещё нет отметки 'докуда уже новеллизировано'",
    )
    novelization_min_messages: int = Field(
        default=6, ge=1,
        description="Минимум новых сообщений в чате с прошлой новеллизации, чтобы вообще запускать по нему извлечение памяти — не тратить LLM-вызов на пустяковую переписку",
    )
    novelization_char_limit: int = Field(
        default=10_000, ge=1,
        description=(
            "Сколько символов недавней переписки максимум передавать LLM за один запрос новеллизации "
            "(DiaryConsolidator._extract_memories). Раньше стояло 2000 — активный день переписки обрубался "
            "почти сразу, в дневник попадало только начало дня; см. novelization_max_output_tokens."
        ),
    )
    novelization_max_output_tokens: int = Field(
        default=2048, ge=1,
        description=(
            "Лимит токенов вывода при извлечении воспоминаний из переписки — дневник должен быть точным и "
            "подробным на этом шаге; сжатие уже сохранённых старых записей (summarize_stale_entries) — "
            "отдельная, намеренно более скупая операция, срабатывающая много позже (older_than)."
        ),
    )
    use_local_embeddings: bool = Field(
        default=True,
        description="Использовать локальный embedding-движок (fastembed/ONNX) как основной источник эмбеддингов вместо облачного LLMRouter",
    )
    local_embedding_model: str = Field(
        default="intfloat/multilingual-e5-large",
        description="Имя модели fastembed для локальных эмбеддингов (см. efi/memory/local_embeddings.py)",
    )

    def resolve_diary_dir(self, paths: PathsSettings) -> Path:
        """Возвращает diary_dir с учётом переопределения — используется при сборке Diary в app.py."""
        return self.diary_dir if self.diary_dir is not None else paths.diary_dir


class StateVectorSettings(BaseModel):
    """
    Параметры динамического блока текущего состояния личности в системном
    промпте (efi.prompts.builder._build_state_vector_block). mood и
    social_distance считаются на лету из efi.memory.beliefs.BeliefStore и
    efi.behavior.affinity.AffinityTracker — здесь настраивается только текст
    жёсткой инструкции подавления угодливости, который остаётся неизменным
    вне зависимости от конкретного mood/social_distance.
    """

    model_config = ConfigDict(frozen=True)

    sycophancy_protection_text: str = Field(
        default=(
            "Тебе строго запрещено соглашаться с собеседником просто чтобы не спорить, льстить или "
            "подстраивать мнение под его ожидания. Если ты не согласна — так и скажи, прямо, даже если "
            "это неприятно. Мгновенная капитуляция в споре без новых веских аргументов — провал "
            "характера, а не вежливость."
        ),
        description="Жёсткая инструкция подавления угодливости — попадает в блок текущего состояния промпта",
    )
    relevant_beliefs_limit: int = Field(
        default=3, ge=1, description="Сколько релевантных убеждений максимум подмешивать в блок состояния"
    )


class SttSettings(BaseModel):
    """
    Настройки распознавания речи. `groq_api_key` — необязательное поле:
    если не задано, efi.telegram.handlers.TelegramEventHandlers.GroqSTT
    просто не конструируется в app.py, и голосовые/видео-кружки идут по
    прежнему пути через LLMRouter (роль VISION) — та же логика "готово, но
    подключается только при наличии конфигурации", что и у
    GenerateImageTool/GenerateVoiceTool (см. efi/app.py).
    """

    model_config = ConfigDict(frozen=True)

    groq_api_key: SecretStr | None = Field(
        default=None,
        description="API-ключ Groq для прямой транскрипции (efi.media.stt_groq.GroqSTT, whisper-large-v3)",
    )


class LifeEngineSettings(BaseModel):
    """
    Параметры движка фоновой автономии (efi.behavior.life_engine.BackgroundLifeWorker):
    как часто проверять семена любопытства (efi.behavior.curiosity.CuriosityTracker)
    и с какого веса находка считается достаточно важной, чтобы Эфи сама
    написала о ней (efi.behavior.organic_ping.OrganicPingGenerator).
    """

    model_config = ConfigDict(frozen=True)

    check_interval_seconds: float = Field(
        default=1800.0, gt=0.0, description="Как часто проверять pending-семена любопытства (раз в N минут)"
    )
    ping_importance_threshold: float = Field(
        default=0.6, ge=0.0, le=1.0,
        description="Минимальный вес семени, при котором находка достаточно важна для органического пинга",
    )


class BusyEngineSettings(BaseModel):
    """
    Параметры симуляции занятости (efi.behavior.busy_engine.BusyEngine):
    диапазон базовой задержки перед тем, как Worker вообще "заметит"
    уведомление, плюс поправки на то, что Эфи занята фоновым исследованием
    (efi.behavior.life_engine.BackgroundLifeWorker.is_researching), устала
    (WorkingMemorySnapshot.energy) или отвечает близкому человеку
    (efi.behavior.affinity.AffinityTracker).

    Дефолты намеренно скромные: эта задержка встаёт ДО обращения к LLM (см.
    efi/notifications/worker.py), а сама LLM (особенно при деградации между
    несколькими кандидатами роли — efi/llm/router.py) уже может занять
    десятки секунд. Заметная "занятость" не должна складываться с и без того
    небыстрым ответом провайдера в минуты ожидания.
    """

    model_config = ConfigDict(frozen=True)

    base_delay_min_seconds: float = Field(default=1.0, ge=0.0, description="Нижняя граница базовой ignore_delay")
    base_delay_max_seconds: float = Field(default=8.0, gt=0.0, description="Верхняя граница базовой ignore_delay")
    research_busy_multiplier: float = Field(
        default=1.5, gt=1.0,
        description="Во сколько раз растягивается верхняя граница базовой задержки, пока идёт фоновое исследование",
    )
    low_energy_extra_seconds: float = Field(
        default=10.0, ge=0.0, description="Максимальная добавка к задержке при энергии, стремящейся к 0"
    )
    high_affinity_discount_seconds: float = Field(
        default=5.0, ge=0.0, description="Максимальная скидка с задержки при близости/уважении, стремящихся к 1"
    )
    min_delay_seconds: float = Field(default=0.5, ge=0.0, description="Нижний потолок итоговой ignore_delay")
    max_delay_seconds: float = Field(default=25.0, gt=0.0, description="Верхний потолок итоговой ignore_delay")

    @model_validator(mode="after")
    def _validate_ranges(self) -> "BusyEngineSettings":
        if self.base_delay_min_seconds > self.base_delay_max_seconds:
            raise ValueError("base_delay_min_seconds не может быть больше base_delay_max_seconds")
        if self.min_delay_seconds > self.max_delay_seconds:
            raise ValueError("min_delay_seconds не может быть больше max_delay_seconds")
        return self


class QuietHoursSettings(BaseModel):
    """
    Ночные "тихие часы" для проактивных путей (efi.behavior.spontaneous_ping,
    efi.behavior.organic_ping, efi.behavior.silence_monitor) — окно, в
    котором Эфи не пишет первой сама. НЕ блокирует ответ на входящее
    сообщение пользователя: если собеседник написал сам, Эфи всё равно
    отвечает, независимо от часа.

    Без этого раньше проактивные сервисы будили собеседника пингами в 5 и 7
    утра наравне с днём — ни один из них не смотрел на время суток вообще.
    """

    model_config = ConfigDict(frozen=True)

    enabled: bool = Field(default=True)
    start_hour: int = Field(default=23, ge=0, le=23, description="Час начала тихих часов (локальное время сервера)")
    end_hour: int = Field(default=8, ge=0, le=23, description="Час окончания тихих часов (локальное время сервера)")


class Settings(BaseSettings):
    """
    Корневой объект конфигурации приложения.

    Инстанцировать напрямую обычно не нужно — используйте `get_settings()`,
    которая кэширует единственный экземпляр на процесс.
    """

    model_config = SettingsConfigDict(
        env_prefix="EFI_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        toml_file=str(_DEFAULT_TOML_PATH),
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    environment: Environment = Environment.PRODUCTION
    debug: bool = False
    character_name: str = "Эфи"
    personality_prompt: str = Field(
        default="",
        description="Базовое описание личности персонажа для системного промпта (секция [character]/personality_prompt в behavior.toml)",
    )

    paths: PathsSettings = Field(default_factory=PathsSettings)
    telegram: TelegramSettings
    llm_roles: LLMRolesSettings
    memory: MemorySettings = Field(default_factory=MemorySettings)
    humanizer: HumanizerSettings = Field(default_factory=HumanizerSettings)
    state_vector: StateVectorSettings = Field(default_factory=StateVectorSettings)
    stt: SttSettings = Field(default_factory=SttSettings)
    life_engine: LifeEngineSettings = Field(default_factory=LifeEngineSettings)
    busy_engine: BusyEngineSettings = Field(default_factory=BusyEngineSettings)
    quiet_hours: QuietHoursSettings = Field(default_factory=QuietHoursSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        # Порядок — от высшего приоритета к низшему: init > env > .env > TOML > file secrets.
        toml_path = Path(os.environ.get("EFI_CONFIG_TOML", str(_DEFAULT_TOML_PATH)))
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            TomlConfigSettingsSource(settings_cls, toml_file=toml_path),
            file_secret_settings,
        )

    def ensure_directories(self) -> None:
        """Идемпотентно создаёт всю файловую структуру данных приложения."""
        self.paths.ensure_directories()

    def build_router(self, **router_kwargs: Any) -> "LLMRouter":
        """Шорткат: `settings.build_router()` эквивалентно `settings.llm_roles.build_router()`."""
        return self.llm_roles.build_router(**router_kwargs)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Синглтон-фабрика конфигурации.

    Кэшируется на процесс: конфигурация читается один раз при первом вызове
    и переиспользуется всеми модулями (llm/, memory/, telegram/, dashboard/).
    Для тестов с другой конфигурацией используйте ``get_settings.cache_clear()``.
    """
    settings = Settings()
    settings.ensure_directories()
    return settings


__all__ = [
    "Environment",
    "LockdownMode",
    "PathsSettings",
    "TelegramSettings",
    "EndpointConfig",
    "GroqSettings",
    "OmniRouteSettings",
    "TaskRole",
    "RoleRoute",
    "LLMRolesSettings",
    "MemorySettings",
    "HumanizerSettings",
    "StateVectorSettings",
    "SttSettings",
    "LifeEngineSettings",
    "BusyEngineSettings",
    "Settings",
    "get_settings",
]
