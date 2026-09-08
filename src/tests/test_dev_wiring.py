"""
Тесты сборки приложения с подсистемой разработки.

Подсистема состоит из восьми модулей и подключается в семи местах: промпт,
инструменты, обработчики сообщений, занятость, фоновая задача, дашборд,
конфиг. Ошибка в проводке (не тот порядок конструирования, забытая
зависимость) проявилась бы не тестом, а падением на старте у владельца — и
только у того, кто включил `dev.enabled`. Поэтому сборка проверяется в обе
стороны: с выключенной подсистемой и с включённой.

Приложение здесь именно КОНСТРУИРУЕТСЯ, а не запускается: `start()` полез бы
в Telegram и в сеть. Проверяется ровно то, что можно проверить без них, —
что объект собирается, что кодер подхватывается из ключа Groq и что
выключенная подсистема не поднимает ни воркера, ни инструментов.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from efi.app import EfiApp
from efi.config.schema import Settings
from efi.notifications.schemas import Notification, NotificationType
from efi.tools.base import ToolContext

_GROQ_ENDPOINT = {
    "base_url": "https://api.groq.com/openai/v1",
    "api_key": "gsk-x",
    "model": "llama-3.1-8b-instant",
}


@pytest.fixture(autouse=True)
def _isolate_from_repo_toml(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Без изоляции сюда подмешался бы behavior.toml из корня репозитория."""
    empty_toml = tmp_path / "empty.toml"
    empty_toml.write_text("", encoding="utf-8")
    monkeypatch.setenv("EFI_CONFIG_TOML", str(empty_toml))


def _settings(tmp_path: Path, **dev: object) -> Settings:
    return Settings(
        telegram={"api_id": 12345, "api_hash": "h", "owner_id": 42},
        llm_roles={"main": {"primary": dict(_GROQ_ENDPOINT)}},
        paths={"base_dir": tmp_path},
        dashboard={"enabled": False},
        dev=dev,
        _env_file=None,
        _secrets_dir=None,
    )


def _tool_names(app: EfiApp) -> set[str]:
    return {tool.name for tool in app._build_tools()}


def test_app_assembles_with_the_craft_switched_off(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.ensure_directories()

    app = EfiApp(settings)

    assert app._dev_worker is None
    # Хранилище и стол переговоров есть всегда: они дёшевы, а промпту и
    # инструментам нужно знать, что проектов нет, — а не падать на None.
    assert app._dev_store is not None
    assert "check_my_projects" in _tool_names(app), "на вопрос про проекты она отвечает честно и без конвейера"
    # А вот инструмента запуска быть не должно: иначе она «возьмётся» за
    # проект, которого некому делать, и человек будет ждать результата.
    assert "start_dev_project" not in _tool_names(app)
    assert app._collab_desk.pipeline_available is False


def test_app_assembles_with_the_craft_switched_on(tmp_path: Path) -> None:
    settings = _settings(tmp_path, enabled=True)
    settings.ensure_directories()

    app = EfiApp(settings)

    assert app._dev_worker is not None
    assert app._dev_worker.is_coding is False
    assert {"start_dev_project", "check_my_projects"} <= _tool_names(app)
    assert app._collab_desk.pipeline_available is True
    # Рабочий каталог проектов создаётся при сборке, а не при первом пуше:
    # ошибка прав должна проявиться на старте, а не через час фоновой работы.
    assert (settings.paths.data_dir / settings.dev.workspace_dir_name).is_dir()


def test_working_with_code_comes_up_together_with_the_craft(tmp_path: Path) -> None:
    """
    Движок работы с чужим кодом надстраивается над тем же кодером: включён
    dev — есть и он. Ноутбук при этом не обязателен: без него всё работает
    ровно как раньше, просто модель слабее.
    """
    settings = _settings(tmp_path, enabled=True)
    settings.ensure_directories()

    app = EfiApp(settings)

    assert app._dev_desk.available is True
    assert "work_on_repo" in _tool_names(app)

    # Зарегистрирован — да, но показывается модели только когда в этом чате
    # есть о чём говорить: реестр спрашивает is_available и при показе, и при
    # исполнении (efi/tools/registry.py).
    tool = next(item for item in app._build_tools() if item.name == "work_on_repo")
    context = ToolContext(
        notification=Notification(type=NotificationType.USER_MESSAGE, chat_id=1, message="привет")
    )
    assert tool.is_available(context) is False
    app._dev_desk.consider_message(1, "глянь https://github.com/user/repo и почини импорт")
    assert tool.is_available(context) is True


def test_the_craft_can_write_projects_without_touching_foreign_repositories(tmp_path: Path) -> None:
    """Одно без другого — рабочий режим: можно писать своё и не лезть в чужие репозитории."""
    settings = _settings(tmp_path, enabled=True, swe_enabled=False)
    settings.ensure_directories()

    app = EfiApp(settings)

    assert app._dev_worker is not None, "свои проекты по-прежнему пишутся"
    assert app._dev_desk.available is False
    assert "work_on_repo" not in _tool_names(app)


def test_the_laptop_is_optional_and_read_from_plain_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path, enabled=True)

    monkeypatch.delenv("OMNIROUTE_URL", raising=False)
    assert settings.resolve_laptop_endpoint() is None, "без ноутбука — работа через облачный кодер"

    monkeypatch.setenv("OMNIROUTE_URL", "http://192.168.0.109:8080/v1")
    monkeypatch.setenv("OMNIROUTE_MODEL", "claude-3-5-sonnet")
    endpoint = settings.resolve_laptop_endpoint()
    assert endpoint is not None
    assert endpoint.model == "claude-3-5-sonnet"


def test_coder_is_picked_up_from_the_groq_key_without_duplicating_it(tmp_path: Path) -> None:
    """Один и тот же секрет не должен требоваться в конфиге дважды — как и у STT."""
    settings = _settings(tmp_path, enabled=True)

    endpoint = settings.resolve_coder_endpoint()

    assert endpoint is not None
    assert endpoint.model == settings.dev.coder_model
    assert endpoint.api_key.get_secret_value() == "gsk-x"


def test_without_any_groq_key_the_craft_stays_down(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """
    `enabled = true` без ключа кодера — это намерение, которое не сработало.
    Оно обязано быть слышно в логе, а не тихо ничего не делать.
    """
    settings = Settings(
        telegram={"api_id": 12345, "api_hash": "h", "owner_id": 42},
        llm_roles={"main": {"primary": {"base_url": "https://other.test/v1", "api_key": "k", "model": "m"}}},
        paths={"base_dir": tmp_path},
        dashboard={"enabled": False},
        dev={"enabled": True},
        _env_file=None,
        _secrets_dir=None,
    )
    settings.ensure_directories()

    assert settings.resolve_coder_endpoint() is None
    with caplog.at_level("WARNING"):
        app = EfiApp(settings)

    assert app._dev_worker is None
    assert any("кодер не настроен" in record.message for record in caplog.records)
