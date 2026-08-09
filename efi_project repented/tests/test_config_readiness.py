"""
Тесты для efi.config.schema.Settings.unfilled_placeholders/validate_ready —
проверки «конфиг заполнен до рабочего состояния».

Зачем отдельная проверка помимо схемы: `api_id = 0` и `api_key = ""` — с точки
зрения pydantic валидные значения нужных типов. Незаполненный шаблон проходил
валидацию и проявлялся уже в рантайме, глубоко внутри сторонних библиотек
(Pyrogram на авторизации с нулевым api_id, httpx на запросе к пустому
base_url), где связать сбой с конфигом по трейсбеку было нечем.
"""

from __future__ import annotations

import pytest

from efi.config.schema import ConfigurationError, Settings

_FILLED_ENDPOINT = {"base_url": "https://example.test/v1", "api_key": "sk-x", "model": "m"}


def _settings(**overrides: object) -> Settings:
    """
    Полностью заполненный Settings, поверх которого тест ломает одно поле.

    `_env_file=None` и `_secrets_dir=None` изолируют тест от окружения
    разработчика; TOML-источник изолируется фикстурой ниже.
    """
    base: dict[str, object] = {
        "telegram": {"api_id": 12345, "api_hash": "h", "owner_id": 42},
        "llm_roles": {
            "main": {"primary": dict(_FILLED_ENDPOINT)},
            "fast": {"primary": dict(_FILLED_ENDPOINT)},
            "vision": {"primary": dict(_FILLED_ENDPOINT)},
        },
    }
    base.update(overrides)
    return Settings(**base)  # type: ignore[arg-type]


@pytest.fixture(autouse=True)
def _isolate_from_repo_toml(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    """
    Без изоляции TOML-источник подмешал бы сюда behavior.toml из корня
    репозитория (pydantic-settings сливает источники вглубь), и тест видел бы
    пустые эндпоинты оттуда, а не то, что задал сам.
    """
    empty_toml = tmp_path / "empty.toml"
    empty_toml.write_text("", encoding="utf-8")
    monkeypatch.setenv("EFI_CONFIG_TOML", str(empty_toml))


def test_a_filled_config_has_nothing_to_complain_about() -> None:
    settings = _settings()
    assert settings.unfilled_placeholders() == []
    settings.validate_ready()  # не бросает


def test_zero_api_id_is_a_placeholder_not_a_value() -> None:
    settings = _settings(telegram={"api_id": 0, "api_hash": "h", "owner_id": 42})
    assert "telegram.api_id" in settings.unfilled_placeholders()


def test_blank_secrets_are_reported() -> None:
    settings = _settings(telegram={"api_id": 1, "api_hash": "   ", "owner_id": 0})
    problems = settings.unfilled_placeholders()
    assert "telegram.api_hash" in problems
    assert "telegram.owner_id" in problems


def test_blank_endpoint_fields_are_reported_per_role_and_slot() -> None:
    settings = _settings(
        llm_roles={
            "main": {"primary": {"base_url": "", "api_key": "", "model": ""}},
            "fast": {"primary": dict(_FILLED_ENDPOINT)},
            "vision": {"primary": dict(_FILLED_ENDPOINT)},
        }
    )
    problems = settings.unfilled_placeholders()
    assert "llm_roles.main.primary.base_url" in problems
    assert "llm_roles.main.primary.api_key" in problems
    assert "llm_roles.main.primary.model" in problems
    assert not any(problem.startswith("llm_roles.fast") for problem in problems)


def test_missing_background_section_is_not_a_problem() -> None:
    """BACKGROUND без своей секции переиспользует маршрут FAST — жаловаться не на что."""
    settings = _settings()
    assert settings.llm_roles.background is None
    assert not any(problem.startswith("llm_roles.background") for problem in settings.unfilled_placeholders())


def test_validate_ready_names_every_unfilled_field() -> None:
    settings = _settings(telegram={"api_id": 0, "api_hash": "", "owner_id": 0})
    with pytest.raises(ConfigurationError) as exc_info:
        settings.validate_ready()

    message = str(exc_info.value)
    assert "telegram.api_id" in message
    assert "telegram.api_hash" in message
    assert "telegram.owner_id" in message
