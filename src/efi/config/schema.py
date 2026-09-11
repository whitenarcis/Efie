from __future__ import annotations

import ipaddress
import logging
import os
from collections.abc import Mapping, Sequence
from enum import StrEnum
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
    from efi.llm.router import LLMRouter

_CONFIG_DIR = Path(__file__).resolve().parent
_PACKAGE_ROOT = _CONFIG_DIR.parent
_SRC_ROOT = _PACKAGE_ROOT.parent
PROJECT_ROOT = _SRC_ROOT.parent
DEFAULT_CONFIG_DIR = _SRC_ROOT / "config"

_GROQ_HOST_MARKER = "api.groq.com"
_TERMUX_READONLY_MARKERS = ("/sdcard", "/mnt/sdcard", "/storage/emulated")


class ConfigurationError(RuntimeError):
    pass


def _termux_safe_path(path: Path) -> Path:
    if any(marker in str(path) for marker in _TERMUX_READONLY_MARKERS):
        return Path.home() / ".efi_data" / path.name
    return path


def _positive_float(raw: str | None, *, default: float) -> float:
    try:
        value = float(str(raw).strip())
    except (TypeError, ValueError):
        return default
    return value if value > 0 else default


def _is_loopback_host(host: str) -> bool:
    normalized = host.strip().strip("[]").lower()
    if not normalized:
        return False
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _is_public_host(host: str) -> bool:
    normalized = host.strip().strip("[]").lower()
    if not normalized or normalized in {"0.0.0.0", "::", "localhost"}:
        return False
    try:
        address = ipaddress.ip_address(normalized)
    except ValueError:
        return True
    return bool(address.is_global)


class Environment(StrEnum):
    DEVELOPMENT = "development"
    PRODUCTION = "production"


class LockdownMode(StrEnum):
    NONE = "none"
    CONTACTS_ONLY = "contacts_only"
    OWNER_ONLY = "owner_only"


class PathsSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    base_dir: Path = Field(default=PROJECT_ROOT)

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
        for directory in (self.data_dir, self.diary_dir, self.cache_dir, self.logs_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.session_path.parent.mkdir(parents=True, exist_ok=True)


class TelegramSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    api_id: int
    api_hash: SecretStr
    phone_number: SecretStr | None = None

    owner_id: int
    owner_display_name: str | None = None
    allowed_chats: list[int] = Field(default_factory=list)
    community_chats: list[int] = Field(default_factory=list)
    chat_labels: dict[int, str] = Field(default_factory=dict)

    lockdown_mode: LockdownMode = LockdownMode.OWNER_ONLY
    check_chats_on_startup: bool = True
    can_join_chats: bool = False
    can_leave_chats: bool = True


class EndpointConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    base_url: str
    api_key: SecretStr
    model: str
    timeout_seconds: float = Field(default=30.0, gt=0.0)


class GroqSettings(EndpointConfig):
    base_url: str = "https://api.groq.com/openai/v1"
    model: str = "llama-3.1-8b-instant"


class OmniRouteSettings(EndpointConfig):
    base_url: str = "https://omni.thegoyhole.fun/v1"
    model: str = "google/gemma-4-31b-it:free"


class TaskRole(StrEnum):
    MAIN = "main"
    FAST = "fast"
    BACKGROUND = "background"
    VISION = "vision"
    CODER = "coder"


class RoleRoute(BaseModel):
    model_config = ConfigDict(frozen=True)

    primary: EndpointConfig
    fallback: EndpointConfig | None = None
    degrade_to: TaskRole | None = None


class LLMRolesSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    main: RoleRoute
    fast: RoleRoute | None = None
    vision: RoleRoute | None = None
    background: RoleRoute | None = None
    coder: RoleRoute | None = None

    def as_routes(self) -> dict[TaskRole, RoleRoute]:
        fast = self.fast if self.fast is not None else self.main
        routes = {
            TaskRole.MAIN: self.main,
            TaskRole.FAST: fast,
            TaskRole.BACKGROUND: self.background if self.background is not None else fast,
            TaskRole.VISION: self.vision if self.vision is not None else self.main,
        }
        if self.coder is not None:
            routes[TaskRole.CODER] = self.coder
        return routes

    def describe_fallbacks(self) -> list[str]:
        notes: list[str] = []
        if self.fast is None:
            notes.append("FAST не задана — служебные вызовы пойдут в модель MAIN")
        if self.background is None:
            notes.append(
                "BACKGROUND не задана — дневник и фоновая жизнь пойдут в модель "
                + ("FAST" if self.fast is not None else "MAIN")
            )
        if self.vision is None:
            notes.append(
                "VISION не задана — фотографии пойдут в модель MAIN; если она не мультимодальная, разбор изображений работать не будет"
            )
        if self.coder is None:
            notes.append("CODER не задана — кодогенерация использует резервный эндпоинт")
        return notes

    def build_router(self, **router_kwargs: Any) -> LLMRouter:
        from efi.llm.router import LLMRouter

        return LLMRouter(self.as_routes(), **router_kwargs)


class HumanizerSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    typing_wpm_min: int = Field(default=120, gt=0)
    typing_wpm_max: int = Field(default=150, gt=0)
    typing_thinking_pause_min_seconds: float = Field(default=1.0, ge=0.0)
    typing_thinking_pause_max_seconds: float = Field(default=2.2, ge=0.0)
    typing_delay_min_seconds: float = Field(default=1.8, ge=0.0)
    typing_delay_max_seconds: float = Field(default=7.0, gt=0.0)

    typo_probability: float = Field(default=0.04, ge=0.0, le=1.0)
    typo_min_text_length: int = Field(default=10, ge=0)
    typo_self_correct_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    keyboard_neighbors: dict[str, list[str]] = Field(default_factory=dict)

    anti_repeat_trigger_max: float = Field(default=0.95, ge=0.0, le=1.0)
    anti_repeat_trigger_avg: float = Field(default=0.85, ge=0.0, le=1.0)
    anti_repeat_max_history: int = Field(default=32, ge=1)

    max_messages_per_burst: int = Field(default=12, ge=1)
    max_reply_chars_per_turn: int = Field(default=700, ge=120)
    short_bubble_delay_min_seconds: float = Field(default=0.3, ge=0.0)
    short_bubble_delay_max_seconds: float = Field(default=0.8, gt=0.0)

    debounce_window_min_seconds: float = Field(default=1.5, ge=0.0)
    debounce_window_max_seconds: float = Field(default=2.5, gt=0.0)
    debounce_typing_poll_interval_seconds: float = Field(default=0.3, gt=0.0)
    debounce_typing_ttl_seconds: float = Field(default=6.0, gt=0.0)
    debounce_max_wait_seconds: float = Field(default=15.0, gt=0.0)

    @model_validator(mode="after")
    def _validate_ranges(self) -> HumanizerSettings:
        if self.typing_wpm_min > self.typing_wpm_max:
            raise ValueError("typing_wpm_min не может быть больше typing_wpm_max")
        if self.typing_thinking_pause_min_seconds > self.typing_thinking_pause_max_seconds:
            raise ValueError("typing_thinking_pause_min_seconds не может быть больше *_max_seconds")
        if self.typing_delay_min_seconds > self.typing_delay_max_seconds:
            raise ValueError("typing_delay_min_seconds не может быть больше typing_delay_max_seconds")
        if self.debounce_window_min_seconds > self.debounce_window_max_seconds:
            raise ValueError("debounce_window_min_seconds не может быть больше debounce_window_max_seconds")
        if self.short_bubble_delay_min_seconds > self.short_bubble_delay_max_seconds:
            raise ValueError("short_bubble_delay_min_seconds не может быть больше short_bubble_delay_max_seconds")
        return self

    def characters_per_second_range(self) -> tuple[float, float]:
        chars_per_word = 5.0
        return (
            self.typing_wpm_min * chars_per_word / 60.0,
            self.typing_wpm_max * chars_per_word / 60.0,
        )


class MemorySettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    diary_dir: Path | None = None
    plagiarism_threshold: float = Field(default=0.97, ge=0.0, le=1.0)
    min_relatedness: float = Field(default=0.80, ge=0.0, le=1.0)
    max_rag_results: int = Field(default=10, ge=1)
    history_limit: int = Field(default=30, ge=1)
    novelization_lookback_days: int = Field(default=1, ge=1)
    novelization_min_messages: int = Field(default=3, ge=1)
    novelization_char_limit: int = Field(default=10_000, ge=1)
    novelization_max_output_tokens: int = Field(default=4096, ge=1)
    use_local_embeddings: bool = True
    local_embedding_model: str = "intfloat/multilingual-e5-large"

    def resolve_diary_dir(self, paths: PathsSettings) -> Path:
        return self.diary_dir if self.diary_dir is not None else paths.diary_dir


class MemoryPulseSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    check_interval_seconds: float = Field(default=600.0, gt=0.0)
    episode_idle_seconds: float = Field(default=900.0, gt=0.0)
    max_messages_before_flush: int = Field(default=30, ge=2)
    min_messages: int = Field(default=3, ge=1)
    lookback_hours: int = Field(default=12, ge=1)


class StateVectorSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    sycophancy_protection_text: str = Field(
        default=(
            "Тебе строго запрещено соглашаться с собеседником просто чтобы не спорить, льстить или "
            "подстраивать мнение под его ожидания. Если ты не согласна — так и скажи, спокойно и "
            "по-доброму, но не отступая от сути. Мгновенная капитуляция без новых веских аргументов — "
            "провал характера, а не вежливость; спорить при этом надо ради истины, а не ради победы."
        )
    )
    relevant_beliefs_limit: int = Field(default=3, ge=1)


class SttSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    groq_api_key: SecretStr | None = None


class WebSearchSettings(BaseModel):
    """
    Веб-поиск (efi/tools/web_tools/web_search.py). Основной путь — Tavily:
    1000 бесплатных кредитов/мес, ключ выдаётся на app.tavily.com. Ключ
    удобнее держать в .env (EFI_WEB_SEARCH__TAVILY_API_KEY), а не здесь —
    этот конфиг отслеживается git. Пустой ключ — не поломка: поиск уходит
    на бесплатный ddgs-фолбэк.
    """

    model_config = ConfigDict(frozen=True)

    tavily_api_key: SecretStr | None = None


class LifeEngineSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    check_interval_seconds: float = Field(default=1800.0, gt=0.0)
    ping_importance_threshold: float = Field(default=0.6, ge=0.0, le=1.0)


class BusyEngineSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    base_delay_min_seconds: float = Field(default=1.0, ge=0.0)
    base_delay_max_seconds: float = Field(default=8.0, gt=0.0)
    research_busy_multiplier: float = Field(default=1.5, gt=1.0)
    low_energy_extra_seconds: float = Field(default=10.0, ge=0.0)
    high_affinity_discount_seconds: float = Field(default=5.0, ge=0.0)
    min_delay_seconds: float = Field(default=0.5, ge=0.0)
    max_delay_seconds: float = Field(default=25.0, gt=0.0)

    active_conversation_window_seconds: float = Field(default=300.0, ge=0.0)
    active_conversation_delay_min_seconds: float = Field(default=0.2, ge=0.0)
    active_conversation_delay_max_seconds: float = Field(default=1.5, ge=0.0)

    @model_validator(mode="after")
    def _validate_ranges(self) -> BusyEngineSettings:
        if self.base_delay_min_seconds > self.base_delay_max_seconds:
            raise ValueError("base_delay_min_seconds не может быть больше base_delay_max_seconds")
        if self.min_delay_seconds > self.max_delay_seconds:
            raise ValueError("min_delay_seconds не может быть больше max_delay_seconds")
        if self.active_conversation_delay_min_seconds > self.active_conversation_delay_max_seconds:
            raise ValueError(
                "active_conversation_delay_min_seconds не может быть больше active_conversation_delay_max_seconds"
            )
        return self


class CommunitySettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    comment_probability: float = Field(default=0.35, ge=0.0, le=1.0)
    min_delay_seconds: float = Field(default=300.0, ge=0.0)
    max_delay_seconds: float = Field(default=1800.0, gt=0.0)
    thread_scan_interval_seconds: float = Field(default=1800.0, gt=0.0)
    max_replies_per_thread: int = Field(default=1, ge=1)
    topic_match_min_score: float = Field(default=0.34, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def _validate_delay_range(self) -> CommunitySettings:
        if self.min_delay_seconds > self.max_delay_seconds:
            raise ValueError("min_delay_seconds не может быть больше max_delay_seconds")
        return self


class LaptopLinkSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    url: str = ""
    model: str = ""
    api_key: SecretStr = Field(default=SecretStr("local"))
    timeout_seconds: float = Field(default=180.0, gt=0.0)
    health_timeout_seconds: float = Field(default=0.6, gt=0.0, le=5.0)

    @property
    def is_configured(self) -> bool:
        return bool(self.url.strip() and self.model.strip())

    def as_endpoint(self) -> EndpointConfig | None:
        if not self.is_configured:
            return None
        return EndpointConfig(
            base_url=self.url.strip(),
            api_key=self.api_key,
            model=self.model.strip(),
            timeout_seconds=self.timeout_seconds,
        )

    @classmethod
    def from_environment(cls, environ: Mapping[str, str] | None = None) -> LaptopLinkSettings:
        source = environ if environ is not None else os.environ
        raw_key = source.get("OMNIROUTE_API_KEY", "").strip()
        return cls(
            url=source.get("OMNIROUTE_URL", "").strip(),
            model=source.get("OMNIROUTE_MODEL", "").strip(),
            api_key=SecretStr(raw_key or "local"),
            timeout_seconds=_positive_float(source.get("OMNIROUTE_TIMEOUT"), default=180.0),
        )


class DevSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = False
    coder: EndpointConfig | None = None
    coder_model: str = "qwen-2.5-coder-32b"
    coder_base_url: str = "https://api.groq.com/openai/v1"
    coder_timeout_seconds: float = Field(default=90.0, gt=0.0)

    check_interval_seconds: float = Field(default=3600.0, gt=0.0)
    self_initiated_probability: float = Field(default=0.25, ge=0.0, le=1.0)
    max_fix_iterations: int = Field(default=3, ge=0, le=10)
    progress_probability: float = Field(default=0.5, ge=0.0, le=1.0)
    lint_generated_code: bool = True

    review_probability: float = Field(default=0.3, ge=0.0, le=1.0)
    review_interval_days: float = Field(default=7.0, gt=0.0)
    patch_importance_threshold: float = Field(default=0.5, ge=0.0, le=1.0)
    discuss_importance_threshold: float = Field(default=0.8, ge=0.0, le=1.0)

    swe_enabled: bool = True
    laptop: LaptopLinkSettings = Field(default_factory=LaptopLinkSettings)
    workspaces_dir: str = "/tmp/workspaces"
    max_repair_rounds: int = Field(default=4, ge=1, le=8)
    max_parallel_model_calls: int = Field(default=2, ge=1, le=8)
    keep_workspaces: bool = False

    github_token: SecretStr | None = None
    github_owner: str = ""
    github_ssh_key_path: Path | None = None
    repo_private: bool = False
    push_enabled: bool = True
    workspace_dir_name: str = "projects"

    def workspace_dir(self, paths: PathsSettings) -> Path:
        return paths.data_dir / self.workspace_dir_name


class QuietHoursSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    start_hour: int = Field(default=23, ge=0, le=23)
    end_hour: int = Field(default=8, ge=0, le=23)


class DashboardSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    enabled: bool = True
    host: str = "0.0.0.0"
    port: int = Field(default=8765, ge=0, le=65535)
    token: SecretStr | None = None
    log_buffer_size: int = Field(default=2000, ge=100, le=100_000)
    log_level: str = "INFO"
    metrics_history: int = Field(default=200, ge=10, le=5000)

    @model_validator(mode="after")
    def _validate_exposure(self) -> DashboardSettings:
        if not self.enabled:
            return self
        if self.log_level.upper() not in logging.getLevelNamesMapping():
            raise ValueError(f"dashboard.log_level: неизвестный уровень логирования {self.log_level!r}")
        if self.token is None and _is_public_host(self.host):
            raise ValueError(
                f"dashboard.host = {self.host!r} — публичный адрес, а dashboard.token не задан."
            )
        return self

    @property
    def log_level_no(self) -> int:
        return logging.getLevelNamesMapping()[self.log_level.upper()]

    @property
    def is_local_only(self) -> bool:
        return _is_loopback_host(self.host)


KNOWN_CONFIG_FILES: tuple[str, ...] = (
    "behavour.toml",
    "behavior.toml",
    "telegram.toml",
    "llm.toml",
    "dashboard.toml",
    "experemental.toml",
    "experimental.toml",
    "web_search.toml",
)


def _collect_toml_files(config_dir: Path) -> list[Path]:
    if not config_dir.is_dir():
        return []
    files: list[Path] = []
    seen: set[Path] = set()
    for name in KNOWN_CONFIG_FILES:
        p = config_dir / name
        if p.is_file():
            resolved = p.resolve()
            if resolved not in seen:
                files.append(p)
                seen.add(resolved)
    for p in sorted(config_dir.glob("*.toml")):
        resolved = p.resolve()
        if resolved not in seen:
            files.append(p)
            seen.add(resolved)
    return files


class ModularTomlSettingsSource(TomlConfigSettingsSource):
    def __init__(
        self,
        settings_cls: type[BaseSettings],
        toml_files: Sequence[Path | str] | None = None,
    ) -> None:
        self.toml_files = [Path(p) for p in toml_files] if toml_files else []
        super().__init__(settings_cls, toml_file=self.toml_files[0] if len(self.toml_files) == 1 else None)

    def _read_file(self, file_path: Path) -> dict[str, Any]:
        if not file_path.is_file():
            return {}
        try:
            return super()._read_file(file_path)
        except Exception:
            try:
                import tomllib
            except ImportError:
                import tomli as tomllib  # type: ignore[no-redef]
            try:
                with file_path.open("rb") as f:
                    return tomllib.load(f)
            except Exception:
                return {}

    @staticmethod
    def _deep_update(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
        for k, v in update.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                base[k] = ModularTomlSettingsSource._deep_update(base[k], v)
            else:
                base[k] = v
        return base

    def __call__(self) -> dict[str, Any]:
        data: dict[str, Any] = {}
        for file_path in self.toml_files:
            file_data = self._read_file(file_path)
            if isinstance(file_data, dict):
                data = self._deep_update(data, file_data)
        return data


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="EFI_",
        env_nested_delimiter="__",
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        frozen=True,
    )

    environment: Environment = Environment.PRODUCTION
    debug: bool = False
    character_name: str = "Эфи"
    personality_prompt: str = ""
    timezone: str = ""

    paths: PathsSettings = Field(default_factory=PathsSettings)
    telegram: TelegramSettings
    llm_roles: LLMRolesSettings
    memory: MemorySettings = Field(default_factory=MemorySettings)
    memory_pulse: MemoryPulseSettings = Field(default_factory=MemoryPulseSettings)
    humanizer: HumanizerSettings = Field(default_factory=HumanizerSettings)
    state_vector: StateVectorSettings = Field(default_factory=StateVectorSettings)
    stt: SttSettings = Field(default_factory=SttSettings)
    web_search: WebSearchSettings = Field(default_factory=WebSearchSettings)
    life_engine: LifeEngineSettings = Field(default_factory=LifeEngineSettings)
    busy_engine: BusyEngineSettings = Field(default_factory=BusyEngineSettings)
    quiet_hours: QuietHoursSettings = Field(default_factory=QuietHoursSettings)
    community: CommunitySettings = Field(default_factory=CommunitySettings)
    dev: DevSettings = Field(default_factory=DevSettings)
    dashboard: DashboardSettings = Field(default_factory=DashboardSettings)

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        env_toml = os.environ.get("EFI_CONFIG_TOML")
        env_dir = os.environ.get("EFI_CONFIG_DIR")
        if env_toml:
            p = Path(env_toml)
            toml_files = _collect_toml_files(p) if p.is_dir() else ([p] if p.is_file() else [])
        elif env_dir:
            toml_files = _collect_toml_files(Path(env_dir))
        else:
            toml_files = _collect_toml_files(DEFAULT_CONFIG_DIR)
            if not toml_files:
                legacy = PROJECT_ROOT / "behavior.toml"
                if legacy.is_file():
                    toml_files = [legacy]

        return (
            init_settings,
            env_settings,
            dotenv_settings,
            ModularTomlSettingsSource(settings_cls, toml_files=toml_files),
            file_secret_settings,
        )

    def ensure_directories(self) -> None:
        self.paths.ensure_directories()

    def unfilled_placeholders(self) -> list[str]:
        problems: list[str] = []

        if self.telegram.api_id <= 0:
            problems.append("telegram.api_id")
        if not self.telegram.api_hash.get_secret_value().strip():
            problems.append("telegram.api_hash")
        if self.telegram.owner_id <= 0:
            problems.append("telegram.owner_id")

        declared_routes = {
            "main": self.llm_roles.main,
            "fast": self.llm_roles.fast,
            "vision": self.llm_roles.vision,
            "background": self.llm_roles.background,
            "coder": self.llm_roles.coder,
        }
        for role_name, route in declared_routes.items():
            if route is None:
                continue
            for slot, endpoint in (("primary", route.primary), ("fallback", route.fallback)):
                if endpoint is None:
                    continue
                prefix = f"llm_roles.{role_name}.{slot}"
                if not endpoint.base_url.strip():
                    problems.append(f"{prefix}.base_url")
                if not endpoint.model.strip():
                    problems.append(f"{prefix}.model")
                if not endpoint.api_key.get_secret_value().strip():
                    problems.append(f"{prefix}.api_key")

        return problems

    def validate_ready(self) -> None:
        problems = self.unfilled_placeholders()
        if not problems:
            return
        raise ConfigurationError(
            "конфигурация не заполнена — осталось "
            f"{len(problems)} незаполненное(ых) поле(й):\n  - "
            + "\n  - ".join(problems)
        )

    def build_router(self, **router_kwargs: Any) -> LLMRouter:
        return self.llm_roles.build_router(**router_kwargs)

    def resolve_groq_api_key(self) -> SecretStr | None:
        if self.stt.groq_api_key is not None:
            return self.stt.groq_api_key

        for route in self.llm_roles.as_routes().values():
            for endpoint in (route.primary, route.fallback):
                if endpoint is not None and _GROQ_HOST_MARKER in endpoint.base_url:
                    return endpoint.api_key
        return None

    def resolve_laptop_endpoint(self) -> EndpointConfig | None:
        if self.dev.laptop.is_configured:
            return self.dev.laptop.as_endpoint()
        return LaptopLinkSettings.from_environment().as_endpoint()

    def resolve_coder_endpoint(self) -> EndpointConfig | None:
        if self.llm_roles.coder is not None:
            return self.llm_roles.coder.primary

        if self.dev.coder is not None:
            return self.dev.coder

        api_key = self.resolve_groq_api_key()
        if api_key is None:
            return None
        return EndpointConfig(
            base_url=self.dev.coder_base_url,
            api_key=api_key,
            model=self.dev.coder_model,
            timeout_seconds=self.dev.coder_timeout_seconds,
        )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    settings = Settings()
    settings.validate_ready()
    settings.ensure_directories()
    return settings


__all__ = [
    "ConfigurationError",
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
    "MemoryPulseSettings",
    "HumanizerSettings",
    "StateVectorSettings",
    "SttSettings",
    "WebSearchSettings",
    "LifeEngineSettings",
    "BusyEngineSettings",
    "CommunitySettings",
    "DevSettings",
    "QuietHoursSettings",
    "DashboardSettings",
    "Settings",
    "get_settings",
]