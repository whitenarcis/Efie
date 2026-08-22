"""
efi/dev/workspace.py

Одноразовая рабочая копия репозитория и запуск процессов в ней — без Docker,
потому что Docker'а в Termux нет и не будет.

Что здесь считается изоляцией. Не песочница в строгом смысле — её на телефоне
без рута не построить, — а три честных ограничения, каждое из которых
закрывает свой реальный способ испортить вечер:

    отдельный каталог  — работа идёт в /tmp/workspaces/{session_id}, а не в
                         репозитории человека. Клон, а не оригинал: правка
                         чужого рабочего дерева «на месте» — это потерянные
                         несохранённые изменения, и извиняться за это поздно.
    свой venv          — зависимости чужого проекта ставятся в одноразовое
                         окружение, а не поверх окружения Эфи. Иначе первый
                         же `pip install` чужого requirements.txt меняет
                         версии, на которых работает она сама.
    лимиты и таймауты  — resource.setrlimit на CPU, память и размер файла
                         плюс жёсткий таймаут на процесс. Бесконечный цикл в
                         чужом тесте не должен ни повесить фоновый цикл, ни
                         съесть батарею телефона.

Чего здесь НЕТ и не будет: запуска сгенерированного кода вне этого модуля.
Граница та же, что в efi/dev/sandbox.py: свои проекты Эфи проверяет статикой
и не исполняет вовсе, а тесты чужого репозитория запускаются только здесь —
в клоне, с лимитами, и только потому, что иначе «прогнать тесты» невозможно
в принципе.

Процессы запускаются через `create_subprocess_exec` списком аргументов —
никогда через оболочку: команды собираются в том числе из имён файлов,
пришедших от модели.
"""

from __future__ import annotations

import asyncio
import logging
import os
import resource
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

#: Где живут рабочие копии. /tmp, потому что это мусор по определению:
#: пережить перезагрузку телефона он не обязан.
DEFAULT_WORKSPACES_ROOT = Path("/tmp/workspaces")  # noqa: S108 — намеренно и задокументировано

#: Потолок на одну команду. Клонирование по мобильной сети бывает долгим,
#: тесты — тоже, но не бесконечными.
DEFAULT_COMMAND_TIMEOUT = 180.0
_CLONE_TIMEOUT = 300.0
_VENV_TIMEOUT = 300.0

#: Лимиты дочернего процесса. Значения подобраны так, чтобы обычный pytest
#: маленького проекта проходил, а бесконечный цикл — нет.
_CPU_SECONDS = 120
_MEMORY_BYTES = 2 * 1024**3
_FILE_SIZE_BYTES = 64 * 1024**2

#: Сколько вывода команды сохранять. Трейсбэк уходит модели, а модели нужен
#: конец вывода (где упало), а не начало (где всё было хорошо).
_MAX_OUTPUT_CHARS = 12_000


class WorkspaceError(RuntimeError):
    """Рабочую копию не удалось подготовить: не склонировалось, нет места, нет git."""


@dataclass(slots=True, frozen=True)
class CommandResult:
    """Результат одной команды в рабочей копии."""

    command: tuple[str, ...]
    exit_code: int
    stdout: str
    stderr: str
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out

    @property
    def output(self) -> str:
        """Вывод одним куском, хвостом вперёд: упало всегда в конце."""
        merged = "\n".join(part for part in (self.stdout, self.stderr) if part.strip())
        if len(merged) <= _MAX_OUTPUT_CHARS:
            return merged
        return "…(начало вывода срезано)…\n" + merged[-_MAX_OUTPUT_CHARS:]

    def render(self) -> str:
        head = " ".join(self.command)
        if self.timed_out:
            return f"$ {head}\n(не уложилось в таймаут)"
        return f"$ {head}\n(код возврата {self.exit_code})\n{self.output}".strip()


class Workspace:
    """
    Одна рабочая копия. Живёт от подготовки до `discard()`.

    Экземпляр знает свой каталог и своё окружение (venv, если создавали) и
    умеет ровно одно: запускать в нём команды с лимитами.
    """

    def __init__(self, root: Path, *, session_id: str, temporary: bool = True) -> None:
        self.root = root.resolve()
        self.session_id = session_id
        self._temporary = temporary
        self._venv: Path | None = None

    @property
    def python_executable(self) -> str:
        """Интерпретатор для команд проекта: свой venv, если он есть, иначе текущий."""
        if self._venv is not None:
            candidate = self._venv / "bin" / "python"
            if candidate.exists():
                return str(candidate)
        return sys.executable

    @property
    def has_venv(self) -> bool:
        return self._venv is not None

    def read(self, relative: str, *, max_chars: int = 40_000) -> str:
        """Файл рабочей копии как текст. Пусто, если файла нет: это не ошибка, а ответ."""
        target = (self.root / relative).resolve()
        if not target.is_relative_to(self.root) or not target.is_file():
            return ""
        try:
            return target.read_text(encoding="utf-8", errors="replace")[:max_chars]
        except OSError:
            return ""

    async def run(
        self,
        *command: str,
        # Таймаут здесь — на ПРОЦЕСС, а не на корутину: истёк — процесс убивается.
        timeout: float = DEFAULT_COMMAND_TIMEOUT,  # noqa: ASYNC109
        cwd: Path | None = None,
        env: dict[str, str] | None = None,
    ) -> CommandResult:
        """
        Одна команда в рабочей копии — с лимитами и таймаутом.

        Не поднимает исключений на ненулевой код возврата: «тесты упали» —
        это нормальный, ожидаемый и самый полезный исход, ради которого всё
        и затевалось (см. efi/dev/auto_fix.py).
        """
        try:
            process = await asyncio.create_subprocess_exec(
                *command,
                cwd=str(cwd or self.root),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=self._child_env(env),
                preexec_fn=_apply_limits,  # noqa: PLW1509 — POSIX-only и намеренно
            )
        except (FileNotFoundError, PermissionError, OSError) as exc:
            return CommandResult(command=command, exit_code=127, stdout="", stderr=str(exc))

        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
        except TimeoutError:
            process.kill()
            await process.wait()
            logger.warning("workspace: %s не уложилась в %.0fс", command[0], timeout)
            return CommandResult(command=command, exit_code=-1, stdout="", stderr="", timed_out=True)

        return CommandResult(
            command=command,
            exit_code=process.returncode or 0,
            stdout=stdout.decode("utf-8", "replace"),
            stderr=stderr.decode("utf-8", "replace"),
        )

    async def ensure_venv(self) -> bool:
        """
        Одноразовое окружение под зависимости проекта.

        Создаётся лениво и только когда действительно нужно ставить чужие
        пакеты: `python -m venv` на телефоне это десятки секунд, и платить их
        за проект на стандартной библиотеке незачем.
        """
        if self._venv is not None:
            return True
        target = self.root / ".efi-venv"
        result = await self.run(sys.executable, "-m", "venv", str(target), timeout=_VENV_TIMEOUT)
        if not result.ok:
            logger.warning("workspace: venv не создался: %s", result.output[:200])
            return False
        self._venv = target
        return True

    async def install_requirements(self, requirements: str = "requirements.txt") -> CommandResult | None:
        """Ставит зависимости проекта в свой venv. None — ставить нечего, и это нормальный исход."""
        if not (self.root / requirements).is_file():
            return None
        if not await self.ensure_venv():
            return None
        return await self.run(
            self.python_executable, "-m", "pip", "install", "-q", "-r", requirements, timeout=_VENV_TIMEOUT
        )

    def discard(self) -> None:
        """Удаляет рабочую копию. Для не-временных (свой проект) — ничего не делает."""
        if not self._temporary:
            return
        shutil.rmtree(self.root, ignore_errors=True)

    def _child_env(self, extra: dict[str, str] | None) -> dict[str, str]:
        env = dict(os.environ)
        # Чужой код не должен ни ходить в сеть за индексом пакетов на каждый
        # импорт, ни писать .pyc в рабочую копию, ни притворяться, что у него
        # есть терминал.
        env.update(
            {
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
                "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                "GIT_TERMINAL_PROMPT": "0",
                "TERM": "dumb",
                "NO_COLOR": "1",
            }
        )
        if self._venv is not None:
            env["VIRTUAL_ENV"] = str(self._venv)
            env["PATH"] = f"{self._venv / 'bin'}{os.pathsep}{env.get('PATH', '')}"
        if extra:
            env.update(extra)
        return env


def _apply_limits() -> None:
    """
    Лимиты дочернего процесса. Выполняется между fork и exec.

    Каждый лимит ставится отдельно и с проглатыванием ошибки: в PRoot часть
    из них недоступна, и это не повод не поставить остальные. Полный отказ от
    лимитов из-за одного неподдерживаемого — худший вариант.
    """
    for what, limit in (
        (resource.RLIMIT_CPU, _CPU_SECONDS),
        (resource.RLIMIT_FSIZE, _FILE_SIZE_BYTES),
        (resource.RLIMIT_AS, _MEMORY_BYTES),
        (resource.RLIMIT_CORE, 0),
    ):
        try:
            soft, hard = resource.getrlimit(what)
            ceiling = limit if hard == resource.RLIM_INFINITY else min(limit, hard)
            resource.setrlimit(what, (ceiling, hard))
        except (ValueError, OSError):  # pragma: no cover — зависит от ядра и PRoot
            continue


class WorkspaceManager:
    """
    Готовит рабочие копии: клонирует чужое, берёт в работу своё.

    Один менеджер на приложение; каталог задаётся конфигом, а имя копии —
    идентификатором задачи, чтобы две задачи никогда не работали в одном
    дереве.
    """

    def __init__(
        self,
        root: Path = DEFAULT_WORKSPACES_ROOT,
        *,
        git_executable: str = "git",
    ) -> None:
        self._root = root
        self._git = git_executable

    async def prepare(self, source: str, *, session_id: str) -> Workspace:
        """
        Рабочая копия по ссылке или по локальному пути.

        Локальный путь тоже клонируется, а не берётся как есть: Эфи не имеет
        права трогать рабочее дерево, в котором человек прямо сейчас что-то
        пишет. Не-git каталог копируется — с пропуском мусора, который в
        карту всё равно не попадёт.
        """
        destination = self._root / _safe_name(session_id)
        local = _prepare_destination(destination, source)
        if local is not None:
            await self._copy_local(local, destination)
        else:
            await self._clone(source, destination)

        workspace = Workspace(destination, session_id=session_id)
        logger.info("workspace: %s готов в %s", source, destination)
        return workspace

    async def _clone(self, url: str, destination: Path) -> None:
        result = await _run_plain(
            self._git, "clone", "--depth", "1", url, str(destination), timeout=_CLONE_TIMEOUT
        )
        if not result.ok:
            raise WorkspaceError(f"не удалось склонировать {url}: {result.output[:300] or 'нет вывода'}")

    async def _copy_local(self, source: Path, destination: Path) -> None:
        git_url = _local_git_url(source)
        if git_url:
            result = await _run_plain(
                self._git, "clone", "--depth", "1", git_url, str(destination), timeout=_CLONE_TIMEOUT
            )
            if result.ok:
                return
            logger.info("workspace: git-клон %s не вышел (%s), копирую файлами", source, result.output[:160])
        try:
            shutil.copytree(
                source, destination, ignore=shutil.ignore_patterns(*_COPY_IGNORE), dirs_exist_ok=True
            )
        except OSError as exc:
            raise WorkspaceError(f"не удалось скопировать {source}: {exc}") from exc


#: Что не тащить в рабочую копию при обычном копировании.
_COPY_IGNORE = (
    ".git", "__pycache__", ".venv", "venv", "node_modules", ".mypy_cache", ".ruff_cache",
    ".pytest_cache", "*.pyc", "build", "dist",
)


async def _run_plain(*command: str, timeout: float) -> CommandResult:  # noqa: ASYNC109 — см. Workspace.run
    """Команда вне рабочей копии (клонирование): лимиты те же, каталога ещё нет."""
    try:
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
        )
    except OSError as exc:
        return CommandResult(command=command, exit_code=127, stdout="", stderr=str(exc))

    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=timeout)
    except TimeoutError:
        process.kill()
        await process.wait()
        return CommandResult(command=command, exit_code=-1, stdout="", stderr="", timed_out=True)
    return CommandResult(
        command=command,
        exit_code=process.returncode or 0,
        stdout=stdout.decode("utf-8", "replace"),
        stderr=stderr.decode("utf-8", "replace"),
    )


def _prepare_destination(destination: Path, source: str) -> Path | None:
    """
    Готовит пустой каталог под рабочую копию и отвечает, локальный ли источник.

    Синхронно и одной функцией: работа с файловой системой из корутины —
    это блокирующие вызовы посреди цикла событий, и держать их в одном
    коротком месте честнее, чем размазывать по async-методам.
    """
    shutil.rmtree(destination, ignore_errors=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    local = Path(source).expanduser()
    return local if local.is_dir() else None


def _local_git_url(source: Path) -> str:
    """file://-адрес локального репозитория — или пусто, если это просто каталог с файлами."""
    return f"file://{source.resolve()}" if (source / ".git").is_dir() else ""


def _safe_name(session_id: str) -> str:
    """Имя каталога из идентификатора задачи: только то, что безопасно в пути."""
    cleaned = "".join(char if char.isalnum() or char in "-_" else "-" for char in session_id).strip("-")
    return cleaned[:64] or "session"


__all__ = [
    "DEFAULT_COMMAND_TIMEOUT",
    "DEFAULT_WORKSPACES_ROOT",
    "CommandResult",
    "Workspace",
    "WorkspaceError",
    "WorkspaceManager",
]
