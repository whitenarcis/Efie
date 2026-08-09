"""
Тесты конфигурации дашборда (efi.config.schema.DashboardSettings).

Баланс, который проверяется: дашборд обязан открываться с других устройств
домашней сети без лишних обрядов (иначе он бесполезен — Эфи живёт на
телефоне, смотрят на неё с ноутбука), но публично маршрутизируемый адрес без
токена пропускать нельзя: наружу уходили бы дневник, переписка и профили.
"""

from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from efi.config.schema import DashboardSettings


def test_defaults_allow_other_devices_on_the_network() -> None:
    settings = DashboardSettings()
    assert settings.enabled is True
    assert settings.host == "0.0.0.0"  # noqa: S104 — сравнение с дефолтом, а не bind
    assert settings.token is None
    assert settings.is_local_only is False
    assert settings.log_level_no == logging.INFO


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "", "192.168.1.10", "10.0.0.5", "172.16.4.4", "169.254.1.1"])
def test_local_network_needs_no_token(host: str) -> None:
    """Все интерфейсы и частные адреса — домашний сценарий, токен не требуется."""
    assert DashboardSettings(host=host).host == host


@pytest.mark.parametrize("host", ["8.8.8.8", "1.1.1.1", "2001:4860:4860::8888", "efi.example.com"])
def test_public_host_without_token_is_refused(host: str) -> None:
    with pytest.raises(ValidationError) as error:
        DashboardSettings(host=host)
    assert "dashboard.token" in str(error.value)


def test_public_host_with_token_is_allowed() -> None:
    settings = DashboardSettings(host="8.8.8.8", token="s3cret")  # type: ignore[arg-type]
    assert settings.token is not None
    assert settings.token.get_secret_value() == "s3cret"


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "127.0.0.5"])
def test_loopback_is_recognised_as_local_only(host: str) -> None:
    settings = DashboardSettings(host=host)
    assert settings.is_local_only is True


def test_disabled_dashboard_skips_exposure_check() -> None:
    """Выключенный дашборд ничего не слушает, поэтому и требовать с него нечего."""
    assert DashboardSettings(enabled=False, host="8.8.8.8").enabled is False


def test_unknown_log_level_is_refused() -> None:
    with pytest.raises(ValidationError) as error:
        DashboardSettings(log_level="ОЧЕНЬ_ПОДРОБНО")
    assert "log_level" in str(error.value)


def test_log_level_is_case_insensitive() -> None:
    assert DashboardSettings(log_level="debug").log_level_no == logging.DEBUG
