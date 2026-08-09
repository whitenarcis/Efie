"""
Тесты конфигурации дашборда (efi.config.schema.DashboardSettings).

Смысл проверок один: дашборд отдаёт дневник, историю переписки и профили
людей, поэтому «выставить наружу» и «забыть про токен» не должны сходиться
в одной конфигурации даже по невнимательности.
"""

from __future__ import annotations

import logging

import pytest
from pydantic import ValidationError

from efi.config.schema import DashboardSettings


def test_defaults_are_local_and_open() -> None:
    settings = DashboardSettings()
    assert settings.enabled is True
    assert settings.host == "127.0.0.1"
    assert settings.token is None
    assert settings.log_level_no == logging.INFO


@pytest.mark.parametrize("host", ["0.0.0.0", "::", "192.168.1.10", "efi.local", ""])
def test_public_host_without_token_is_refused(host: str) -> None:
    with pytest.raises(ValidationError) as error:
        DashboardSettings(host=host)
    assert "dashboard.token" in str(error.value)


@pytest.mark.parametrize("host", ["127.0.0.1", "localhost", "::1", "127.0.0.5"])
def test_loopback_hosts_need_no_token(host: str) -> None:
    assert DashboardSettings(host=host).host == host


def test_public_host_with_token_is_allowed() -> None:
    settings = DashboardSettings(host="0.0.0.0", token="s3cret")  # type: ignore[arg-type]
    assert settings.token is not None
    assert settings.token.get_secret_value() == "s3cret"


def test_disabled_dashboard_skips_exposure_check() -> None:
    """Выключенный дашборд ничего не слушает, поэтому и требовать с него нечего."""
    assert DashboardSettings(enabled=False, host="0.0.0.0").enabled is False


def test_unknown_log_level_is_refused() -> None:
    with pytest.raises(ValidationError) as error:
        DashboardSettings(log_level="ОЧЕНЬ_ПОДРОБНО")
    assert "log_level" in str(error.value)


def test_log_level_is_case_insensitive() -> None:
    assert DashboardSettings(log_level="debug").log_level_no == logging.DEBUG
