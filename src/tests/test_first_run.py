"""
Тесты первого запуска — того, с чем сталкивается человек, поставивший Эфи.

Аудит перед выпуском в open-source. Главное найденное препятствие: схема
требовала заполнить main, fast и vision, а шаблонный behavior.toml объявлял
ещё main.fallback и background — восемнадцать обязательных полей, из которых
пятнадцать про LLM. Человек с одним бесплатным ключом упирался в стену
раньше, чем видел хоть одно сообщение.

Ничто в архитектуре этого не требовало: регламент ролей — про то, КТО какой
канал занимает, а не про то, сколько у владельца ключей.

Здесь же — про утечку: адрес дашборда с токеном писался в лог. Логи люди
вставляют в issue, когда просят помощи, и вместе с логом уезжал ключ от
собственной переписки, дневника и профилей людей.
"""

from __future__ import annotations

import pathlib

import pytest

from efi.config.schema import (
    ConfigurationError,
    DashboardSettings,
    EndpointConfig,
    LLMRolesSettings,
    RoleRoute,
    Settings,
    TaskRole,
)

_TEMPLATE = pathlib.Path(__file__).resolve().parents[1] / "behavior.toml"


def _endpoint(model: str = "some/model") -> EndpointConfig:
    return EndpointConfig(base_url="https://example.invalid/v1", api_key="k", model=model)  # type: ignore[arg-type]


# -- порог входа ---------------------------------------------------------------


def test_one_model_is_enough_to_start() -> None:
    """
    Главная находка аудита: раньше без четырёх настроенных ролей Эфи просто
    не запускалась.
    """
    roles = LLMRolesSettings(main=RoleRoute(primary=_endpoint()))

    routes = roles.as_routes()

    assert set(routes) == set(TaskRole), "все четыре роли обязаны быть разрешимы"
    assert all(route.primary is not None for route in routes.values())


def test_unset_roles_fall_back_to_main() -> None:
    main = _endpoint("main/model")
    roles = LLMRolesSettings(main=RoleRoute(primary=main))

    routes = roles.as_routes()

    assert routes[TaskRole.FAST].primary is main
    assert routes[TaskRole.VISION].primary is main
    assert routes[TaskRole.BACKGROUND].primary is main


def test_background_still_prefers_fast_when_fast_is_configured() -> None:
    """Прежнее поведение: фоновая работа идёт в служебную модель, а не в диалоговую."""
    fast = _endpoint("fast/model")
    roles = LLMRolesSettings(main=RoleRoute(primary=_endpoint()), fast=RoleRoute(primary=fast))

    assert roles.as_routes()[TaskRole.BACKGROUND].primary is fast


def test_configured_roles_are_not_overridden() -> None:
    vision = _endpoint("vision/model")
    roles = LLMRolesSettings(main=RoleRoute(primary=_endpoint()), vision=RoleRoute(primary=vision))

    assert roles.as_routes()[TaskRole.VISION].primary is vision


def test_the_template_asks_for_six_fields_not_eighteen(tmp_path: pathlib.Path) -> None:
    """
    Проверка ровно того, что видит новый пользователь: сколько строк ему
    придётся заполнить, прежде чем Эфи вообще запустится.
    """
    copied = tmp_path / "behavior.toml"
    copied.write_text(_TEMPLATE.read_text(encoding="utf-8"), encoding="utf-8")

    import os

    original = os.environ.get("EFI_CONFIG_TOML")
    os.environ["EFI_CONFIG_TOML"] = str(copied)
    try:
        problems = Settings().unfilled_placeholders()
    finally:
        if original is None:
            os.environ.pop("EFI_CONFIG_TOML", None)
        else:
            os.environ["EFI_CONFIG_TOML"] = original

    assert problems == [
        "telegram.api_id",
        "telegram.api_hash",
        "telegram.owner_id",
        "llm_roles.main.primary.base_url",
        "llm_roles.main.primary.model",
        "llm_roles.main.primary.api_key",
    ]


def test_an_unfilled_config_says_exactly_what_to_fill() -> None:
    """
    Без этой проверки незаполненный конфиг проявлялся не на старте, а внутри
    сторонних библиотек: Pyrogram падал на нулевом api_id, httpx — на пустом
    base_url, и связать это с конфигом по трейсбеку было нечем.
    """
    settings = Settings(
        telegram={"api_id": 0, "api_hash": "", "owner_id": 0},  # type: ignore[arg-type]
        llm_roles=LLMRolesSettings(main=RoleRoute(primary=_endpoint())),
    )

    with pytest.raises(ConfigurationError) as excinfo:
        settings.validate_ready()

    message = str(excinfo.value)
    assert "telegram.api_id" in message
    assert "EFI_" in message, "человеку нужно показать и второй способ задать значение"


# -- предупреждения вместо тихих подстановок ------------------------------------


def test_falling_back_to_main_is_announced() -> None:
    """
    Молча подставить MAIN вместо VISION нельзя: текстовая модель на
    фотографию ответит ошибкой или выдумкой, и владелец должен узнать об этом
    при запуске, а не когда ему пришлют картинку.
    """
    notes = LLMRolesSettings(main=RoleRoute(primary=_endpoint())).describe_fallbacks()

    assert any("VISION" in note and "мультимодальн" in note for note in notes)
    assert any("FAST" in note for note in notes)
    assert any("BACKGROUND" in note for note in notes)


def test_a_fully_configured_setup_says_nothing() -> None:
    roles = LLMRolesSettings(
        main=RoleRoute(primary=_endpoint()),
        fast=RoleRoute(primary=_endpoint()),
        vision=RoleRoute(primary=_endpoint()),
        background=RoleRoute(primary=_endpoint()),
    )

    assert roles.describe_fallbacks() == []


# -- секреты не утекают в лог ----------------------------------------------------


def test_the_dashboard_url_carries_no_token() -> None:
    """
    Эта строка уходит в лог, а логи вставляют в issue, когда просят помощи.
    Вместе с логом уезжал бы ключ от переписки, дневника и профилей людей.
    """
    from efi.dashboard.server import DashboardServer

    settings = DashboardSettings(host="0.0.0.0", port=8765, token="s3cret-token")  # type: ignore[arg-type]
    server = DashboardServer.__new__(DashboardServer)
    server._settings = settings  # type: ignore[attr-defined]

    class _Http:
        port = 8765

    server._http = _Http()  # type: ignore[attr-defined]

    url = server.lan_url
    if url is not None:  # в песочнице LAN-адреса может не быть вовсе
        assert "s3cret-token" not in url
        assert "token=" not in url
