#!/usr/bin/env python3
"""
scripts/run.py

Исполняемая точка входа: разбирает CLI-аргументы, настраивает логирование,
поднимает EfiApp и корректно останавливает её по SIGINT/SIGTERM.

Запуск:
    python scripts/run.py
    python scripts/run.py --log-level DEBUG --log-file logs/efi.log
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import signal
import sys
import tomllib
from pathlib import Path

from pydantic import ValidationError

# Позволяет запускать файл напрямую (python scripts/run.py), не только как
# модуль пакета — добавляем корень репозитория в sys.path до импорта efi.*.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from efi.app import EfiApp  # noqa: E402 — импорт после правки sys.path, иначе он и не нужен
from efi.config.schema import ConfigurationError, Settings, get_settings  # noqa: E402

logger = logging.getLogger("efi")

#: Код возврата при нерабочей конфигурации — отличается от 1 (падение в
#: рантайме), чтобы systemd/supervisor могли отличить «конфиг не заполнен»
#: (перезапуск не поможет) от «упало по ходу работы» (перезапуск осмыслен).
_CONFIG_EXIT_CODE = 2


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Efi — Telegram userbot-компаньон")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Уровень логирования (по умолчанию INFO)",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Путь к файлу лога в дополнение к stdout (по умолчанию логи только в stdout)",
    )
    return parser.parse_args(argv)


def _configure_logging(level: str, log_file: str | None) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))

    logging.basicConfig(
        level=getattr(logging, level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        handlers=handlers,
        force=True,  # перезаписывает конфигурацию логирования, если что-то уже успело её выставить (например, тесты)
    )

    # Библиотеки третьих сторон обычно многословнее на DEBUG, чем нужно для нашего собственного кода.
    third_party_level = max(logging.WARNING, getattr(logging, level))
    logging.getLogger("pyrogram").setLevel(third_party_level)
    logging.getLogger("httpx").setLevel(third_party_level)
    logging.getLogger("httpcore").setLevel(third_party_level)


def _load_settings() -> Settings:
    """
    Читает конфигурацию, превращая три типовых способа её сломать в короткое
    сообщение вместо трейсбека из недр pydantic/tomllib:

        - файл не разбирается как TOML (классика — значение-плейсхолдер без
          кавычек: `api_id = HERE`);
        - в файле нет обязательной секции (`[telegram]`, `[llm_roles.*]`);
        - секции есть, но поля остались пустыми (см. Settings.validate_ready).

    Ни один из трёх случаев не чинится перезапуском, поэтому выходим с
    отдельным кодом возврата, а не падаем как при рантайм-ошибке.
    """
    try:
        return get_settings()
    except tomllib.TOMLDecodeError as exc:
        logger.error(
            "behavior.toml не разбирается как TOML: %s\n"
            "Проверьте, что все значения — валидный TOML: строки в кавычках, числа без кавычек "
            "(частая ошибка — оставленный плейсхолдер вида `api_id = HERE`).",
            exc,
        )
    except ConfigurationError as exc:
        logger.error("%s", exc)
    except ValidationError as exc:
        logger.error("конфигурация не прошла валидацию схемы:\n%s", exc)
    raise SystemExit(_CONFIG_EXIT_CODE)


async def _main_async(args: argparse.Namespace) -> None:
    _configure_logging(args.log_level, args.log_file)

    settings = _load_settings()
    app = EfiApp(settings)

    loop = asyncio.get_running_loop()
    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            loop.add_signal_handler(sig, app.request_stop)
        except NotImplementedError:
            # add_signal_handler не реализован на некоторых платформах
            # (например, стандартный event loop на Windows) — тогда
            # полагаемся на KeyboardInterrupt, который asyncio.run() и так
            # пробрасывает штатно для SIGINT.
            logger.debug("add_signal_handler for %s is not supported on this platform", sig)

    await app.start()
    logger.info("efi is running, press Ctrl+C to stop")
    try:
        await app.wait_until_stopped()
    finally:
        await app.stop()


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    try:
        asyncio.run(_main_async(args))
    except KeyboardInterrupt:
        logger.info("interrupted, shutting down")


if __name__ == "__main__":
    main()
