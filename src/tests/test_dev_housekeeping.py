"""
Тесты уборки за конвейером разработки.

Ни один из этих тестов не про поведение Эфи — они про то, что она проживёт
на телефоне дольше недели. Конвейер клонирует чужие репозитории в /tmp и
держит собственные HTTP-клиенты к кодеру и к ноутбуку; и то и другое
накапливается тихо. Клоны съедают место, которого в Termux и так немного,
а незакрытые соединения переживают остановку и не дают процессу завершиться.

Ни одно из двух не проявилось бы тестом на поведение: приложение работает,
просто через неделю на телефоне кончается место, а `efi stop` не
возвращает управление.
"""

from __future__ import annotations

import os
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from efi.db.core import Database
from efi.db.models import MIGRATIONS
from efi.dev.reporter import DevReporter
from efi.dev.store import DevTaskStore
from efi.dev.worker import _WORKSPACE_TTL, DevWorker
from efi.dev.workspace import WorkspaceManager
from efi.notifications.manager import NotificationManager


def _aged(path: Path, age: timedelta) -> None:
    """Состаривает каталог: prune смотрит на mtime, а не на имя."""
    when = time.time() - age.total_seconds()
    os.utime(path, (when, when))


# -- рабочие копии не живут вечно -------------------------------------------------


def test_old_workspaces_are_swept_away(tmp_path: Path) -> None:
    root = tmp_path / "workspaces"
    stale = root / "session_old"
    stale.mkdir(parents=True)
    (stale / "big.bin").write_text("x" * 1024, encoding="utf-8")
    _aged(stale, _WORKSPACE_TTL + timedelta(days=1))

    removed = WorkspaceManager(root).prune_older_than(_WORKSPACE_TTL)

    assert removed == 1
    assert not stale.exists()


def test_a_fresh_workspace_is_left_alone(tmp_path: Path) -> None:
    """
    В свежей копии лежит ветка, которую человек ещё может захотеть забрать.
    Удалить её сразу после работы — значит отдать ссылку на то, чего нет.
    """
    root = tmp_path / "workspaces"
    recent = root / "session_new"
    recent.mkdir(parents=True)

    assert WorkspaceManager(root).prune_older_than(_WORKSPACE_TTL) == 0
    assert recent.exists()


def test_sweeping_an_empty_root_is_not_an_error(tmp_path: Path) -> None:
    """Первый запуск: каталога ещё нет, и это нормальный ход событий."""
    assert WorkspaceManager(tmp_path / "never-created").prune_older_than(_WORKSPACE_TTL) == 0


class _CountingWorkspaces:
    """Уборщик, который умеет только считать, сколько раз его позвали."""

    def __init__(self) -> None:
        self.sweeps = 0

    def prune_older_than(self, max_age: timedelta) -> int:
        self.sweeps += 1
        return 0


def _worker(tmp_path: Path, workspaces: Any) -> DevWorker:
    """Воркер без единой живой зависимости: здесь проверяется только уборка."""
    database = Database(tmp_path / "efi.db", migrations=MIGRATIONS)
    return DevWorker(
        DevTaskStore(database),
        object(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        DevReporter(NotificationManager(worker_count=1)),
        workspaces=workspaces,
        self_initiated_probability=0.0,
    )


async def test_the_sweep_repeats_while_she_runs(tmp_path: Path) -> None:
    """
    Главное здесь. Уборка была разовой — на первом тике, — и это ровно та
    ошибка, которую не видно: Termux на телефоне живёт неделями без
    перезапуска, а значит, уборка за всё это время случалась один раз.
    """
    workspaces = _CountingWorkspaces()
    worker = _worker(tmp_path, workspaces)

    worker._sweep_workspaces()
    worker._workspaces_swept_at = datetime.now(UTC) - timedelta(hours=2)
    worker._sweep_workspaces()

    assert workspaces.sweeps == 2


async def test_the_sweep_does_not_run_on_every_tick(tmp_path: Path) -> None:
    """Обход каталога синхронный — держать на нём цикл каждую минуту незачем."""
    workspaces = _CountingWorkspaces()
    worker = _worker(tmp_path, workspaces)

    for _ in range(5):
        worker._sweep_workspaces()

    assert workspaces.sweeps == 1


async def test_a_failed_sweep_does_not_stop_the_work(tmp_path: Path) -> None:
    """Не убралось — место кончится когда-нибудь, а работа встанет прямо сейчас."""

    class _BrokenWorkspaces:
        def prune_older_than(self, max_age: timedelta) -> int:
            raise OSError(13, "Permission denied")

    worker = _worker(tmp_path, _BrokenWorkspaces())

    worker._sweep_workspaces()  # не должно бросить


# -- соединения закрываются вместе с приложением ----------------------------------


async def test_dev_clients_are_closed_on_stop() -> None:
    """
    Клиенты кодера и ноутбука держат пул httpx с живыми сокетами. Незакрытый
    пул переживает остановку и не даёт процессу завершиться — на телефоне это
    выглядит как «Эфи не выключается».
    """
    from efi.config.schema import EndpointConfig
    from efi.dev.qwen_client import QwenCoderClient
    from efi.llm.network_router import LaptopLink

    coder = QwenCoderClient(
        EndpointConfig(base_url="https://example.invalid/v1", api_key="k", model="m")
    )
    link = LaptopLink(
        EndpointConfig(base_url="http://192.168.0.2:8080/v1", api_key="none", model="m")
    )

    await coder.aclose()
    await link.aclose()

    # Повторное закрытие — обычное дело при остановке по ошибке, и оно не
    # должно превращать остановку в исключение.
    await coder.aclose()
    await link.aclose()
