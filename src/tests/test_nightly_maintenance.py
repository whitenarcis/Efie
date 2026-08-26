"""
Тесты ночного обслуживания памяти.

Ночью Эфи делает пять вещей подряд: дописывает дневник за прожитый день,
чистит дубли, сворачивает старое в мемуары и прибирает две таблицы. Раньше
всё это связывал один общий `try`, и первый же сбой уносил ночь целиком.

Первой при этом идёт новеллизация — единственный этап, который ходит в LLM.
Таймаут на бесплатном тире там обычное дело (ровно он и рвал дневники), и
из-за него не проходили ни dedup, ни чистка таблиц, которым модель вообще не
нужна: одна недоступная сеть оставляла базу неубранной на сутки, а потом ещё
на сутки.
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from efi.app import EfiApp
from efi.config.schema import Settings
from efi.llm.errors import LLMTimeoutError

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


def _app(tmp_path: Path) -> EfiApp:
    settings = Settings(
        telegram={"api_id": 12345, "api_hash": "h", "owner_id": 42},
        llm_roles={"main": {"primary": dict(_GROQ_ENDPOINT)}},
        paths={"base_dir": tmp_path},
        dashboard={"enabled": False},
        _env_file=None,
        _secrets_dir=None,
    )
    settings.ensure_directories()
    return EfiApp(settings)


class _Consolidator:
    """Консолидатор, который умеет считать вызовы и падать по указанию."""

    def __init__(self, *, novelization_fails: bool = False, dedup_fails: bool = False) -> None:
        self.calls: list[str] = []
        self._novelization_fails = novelization_fails
        self._dedup_fails = dedup_fails

    async def novelize_recent_history(self, **_kwargs: Any) -> int:
        self.calls.append("novelize")
        if self._novelization_fails:
            raise LLMTimeoutError("превышен таймаут 15.0s")
        return 2

    async def deduplicate(self, **_kwargs: Any) -> int:
        self.calls.append("dedup")
        if self._dedup_fails:
            raise RuntimeError("векторное хранилище недоступно")
        return 1

    async def summarize_stale_entries(self) -> None:
        self.calls.append("memoir")
        return None


async def test_a_quiet_night_runs_every_stage(tmp_path: Path) -> None:
    app = _app(tmp_path)
    consolidator = _Consolidator()
    app._consolidator = consolidator  # type: ignore[assignment]

    await app._run_nightly_maintenance()

    assert consolidator.calls == ["novelize", "dedup", "memoir"]


async def test_a_timed_out_diary_does_not_cancel_the_cleanup(tmp_path: Path) -> None:
    """
    Главная регрессия. Новеллизация ходит в LLM и падает по таймауту чаще
    всего остального, а чистке таблиц модель не нужна вовсе — связывать их
    судьбу было нечем.
    """
    app = _app(tmp_path)
    consolidator = _Consolidator(novelization_fails=True)
    app._consolidator = consolidator  # type: ignore[assignment]

    await app._run_nightly_maintenance()

    assert consolidator.calls == ["novelize", "dedup", "memoir"]


async def test_a_failure_in_the_middle_does_not_stop_what_follows(tmp_path: Path) -> None:
    app = _app(tmp_path)
    consolidator = _Consolidator(dedup_fails=True)
    app._consolidator = consolidator  # type: ignore[assignment]

    await app._run_nightly_maintenance()

    assert "memoir" in consolidator.calls


async def test_the_tables_are_pruned_even_when_the_model_is_down(tmp_path: Path) -> None:
    """
    Чистка идёт по базе и не зависит ни от сети, ни от модели. Ради этого
    этапы и разделены: файл базы лежит на телефоне, где место кончается.
    """
    app = _app(tmp_path)
    app._consolidator = _Consolidator(novelization_fails=True)  # type: ignore[assignment]

    pruned: list[str] = []

    class _History:
        async def prune_old_messages(self, **_kwargs: Any) -> int:
            pruned.append("messages")
            return 7

    class _Social:
        async def prune_old(self, **_kwargs: Any) -> int:
            pruned.append("social")
            return 3

    app._history = _History()  # type: ignore[assignment]
    app._social_memory = _Social()  # type: ignore[assignment]

    await app._run_nightly_maintenance()

    assert pruned == ["messages", "social"]


async def test_shutdown_is_not_swallowed(tmp_path: Path) -> None:
    """
    Отмену глушить нельзя: иначе выключение будет ждать всю ночную работу до
    конца, а на телефоне остановки бывают резкими.
    """
    app = _app(tmp_path)
    reached_dedup = False

    class _CancelledMidway:
        async def novelize_recent_history(self, **_kwargs: Any) -> int:
            raise asyncio.CancelledError

        async def deduplicate(self, **_kwargs: Any) -> int:
            nonlocal reached_dedup
            reached_dedup = True
            return 0

    app._consolidator = _CancelledMidway()  # type: ignore[assignment]

    with pytest.raises(asyncio.CancelledError):
        await app._run_nightly_maintenance()

    assert not reached_dedup, "остановка должна прерывать ночь, а не переживать её"
