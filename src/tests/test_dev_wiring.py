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
    assert "check_my_projects" in _tool_names(app)


def test_app_assembles_with_the_craft_switched_on(tmp_path: Path) -> None:
    settings = _settings(tmp_path, enabled=True)
    settings.ensure_directories()

    app = EfiApp(settings)

    assert app._dev_worker is not None
    assert app._dev_worker.is_coding is False
    assert {"start_dev_project", "check_my_projects"} <= _tool_names(app)
    # Рабочий каталог проектов создаётся при сборке, а не при первом пуше:
    # ошибка прав должна проявиться на старте, а не через час фоновой работы.
    assert (settings.paths.data_dir / settings.dev.workspace_dir_name).is_dir()


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
