"""
efi/dev/auto_fix.py

Закрытый цикл починки: прогнать проверки → взять трейсбэк → отдать модели →
применить правку → прогнать заново.

Что здесь принципиально нового по сравнению с efi/dev/engine.py. Там модель
пишет файл, а песочница смотрит на него статически: парсится ли, не ругается
ли линтер. Этого достаточно для «код выглядит правильным» и совершенно
недостаточно для «код работает». Разница видна на любом реальном баге:
`ImportError: cannot import name 'parse_line'` не находится ни компилятором,
ни ruff — он находится запуском.

Поэтому здесь проверки настоящие и в таком порядке (от дешёвых к дорогим):

    1. синтаксис    — компиляция всех изменённых файлов;
    2. импорты      — модуль реально импортируется (ловит ImportError,
                      циклы, опечатки в именах);
    3. линтер       — ruff, если он есть в окружении проекта;
    4. тесты        — pytest, если в проекте есть тесты.

Первая же упавшая проверка останавливает проход: чинить надо то, что
сломалось раньше всего, а не всё сразу — линтер по коду, который не
импортируется, скажет ерунду.

Вывод падения уходит модели дословно. Не пересказ, не «тесты не прошли», а
тот же текст, который увидел бы человек: трейсбэк — это и есть постановка
задачи, и любая его переработка теряет ровно ту строчку, по которой всё
чинится.

Цикл конечный (по умолчанию четыре круга). Модель, которая не починила
ошибку за четыре подхода, на пятом её не починит — она начнёт переписывать
соседние места, и это уже не починка, а порча.
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

from efi.dev.edits import EDIT_FORMAT_INSTRUCTIONS, apply_edits, parse_edits
from efi.dev.workspace import CommandResult, Workspace

logger = logging.getLogger(__name__)

#: Сколько кругов починки. Четыре — предел, за которым модель перестаёт
#: чинить и начинает переписывать вокруг.
DEFAULT_MAX_ROUNDS = 4

_PYTEST_TIMEOUT = 240.0
_QUICK_TIMEOUT = 90.0

#: Сколько вывода отдавать модели. Трейсбэк целиком обычно короче, а вот
#: вывод упавшего pytest на сто тестов вытеснит из контекста сам код.
_MAX_FAILURE_CHARS = 6000

_FIX_SYSTEM_PROMPT = (
    "Ты чинишь СВОЙ код по реальному выводу проверки: трейсбэку, ошибке импорта, замечанию линтера "
    "или упавшему тесту. Тебе дают вывод дословно и содержимое файлов, которых он касается.\n"
    "Правь минимально: причину падения, а не всё, что показалось некрасивым. Не переименовывай "
    "публичные имена, не меняй сигнатуры без нужды, не трогай файлы, к которым падение отношения не "
    "имеет. Если падает тест — чини КОД, а не тест, кроме случая, когда тест явно проверяет не то.\n"
    f"{EDIT_FORMAT_INSTRUCTIONS}"
)


class CheckKind(StrEnum):
    """Что именно проверяем. Порядок значений — порядок запуска."""

    SYNTAX = "syntax"
    IMPORTS = "imports"
    LINT = "lint"
    TESTS = "tests"

    @property
    def human(self) -> str:
        return _CHECK_WORDS[self]


_CHECK_WORDS: dict[CheckKind, str] = {
    CheckKind.SYNTAX: "синтаксис",
    CheckKind.IMPORTS: "импорты",
    CheckKind.LINT: "линтер",
    CheckKind.TESTS: "тесты",
}


@dataclass(slots=True, frozen=True)
class CheckOutcome:
    """Результат одной проверки."""

    kind: CheckKind
    ok: bool
    output: str = ""
    skipped: bool = False

    def render(self) -> str:
        if self.skipped:
            return f"{self.kind.human}: нечего проверять"
        return f"{self.kind.human}: {'ок' if self.ok else 'упало'}\n{self.output}".strip()


@dataclass(slots=True)
class RepairReport:
    """Чем кончился цикл починки — и что по дороге происходило."""

    rounds: int = 0
    green: bool = False
    checks: list[CheckOutcome] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    #: Короткие поводы для реплик в чат: «забыла добавить функцию в __all__»,
    #: «линтер задушил на типах». Не тексты сообщений — именно поводы, словами
    #: их делает Worker с личностью (см. efi/dev/reporter.py).
    notes: list[str] = field(default_factory=list)

    @property
    def last_failure(self) -> CheckOutcome | None:
        failures = [item for item in self.checks if not item.ok and not item.skipped]
        return failures[-1] if failures else None


#: Кто чинит: на вход системный промпт и запрос, на выход текст с правками.
#: Ровно та же форма, что у efi.dev.qwen_client.QwenCoderClient.write_document,
#: — чтобы сюда одинаково подходил и ноутбук через сетевой роутер, и облачный
#: кодер, который работал до него.
FixerCall = Callable[[str, str], Awaitable[str | None]]

#: Короткая реплика в чат по ходу починки. Необязательна: без неё цикл просто
#: молчит и чинит.
Narrator = Callable[[str], Awaitable[None]]


class RepairLoop:
    """
    Проверки и починка над одной рабочей копией.

    Не знает ни про git, ни про чаты: на вход рабочая копия и список
    изменённых файлов, на выход — зелёные проверки или честный отчёт о том,
    что не сошлось.
    """

    def __init__(
        self,
        fixer: FixerCall,
        *,
        max_rounds: int = DEFAULT_MAX_ROUNDS,
        narrator: Narrator | None = None,
        ruff_executable: str = "ruff",
    ) -> None:
        self._fixer = fixer
        self._max_rounds = max_rounds
        self._narrator = narrator
        self._ruff = ruff_executable

    async def run(self, workspace: Workspace, touched: list[str]) -> RepairReport:
        """
        Гоняет проверки и чинит найденное, пока не станет зелено или не
        кончатся круги.

        `touched` — файлы, которых касалась правка. По ним же идут дешёвые
        проверки: компилировать весь чужой репозиторий на каждом круге
        бессмысленно, а импортировать — и вредно.
        """
        report = RepairReport()
        for round_number in range(self._max_rounds + 1):
            outcome = await self._check(workspace, touched)
            report.checks.append(outcome)
            if outcome.ok or outcome.skipped:
                report.green = True
                logger.info("auto_fix: всё зелено за %d круг(ов)", report.rounds)
                return report

            if round_number == self._max_rounds:
                logger.warning(
                    "auto_fix: за %d кругов не починилось, последнее падение: %s",
                    self._max_rounds, outcome.output.splitlines()[:1],
                )
                return report

            await self._say(outcome, round_number, report)
            fixed = await self._repair(workspace, outcome, touched, report)
            if not fixed:
                logger.info("auto_fix: правка не пришла или не легла — дальше чинить нечем")
                return report
            report.rounds += 1
        return report

    async def _check(self, workspace: Workspace, touched: list[str]) -> CheckOutcome:
        """Первая упавшая проверка — она же и есть задача на этот круг."""
        for check in (self._syntax, self._imports, self._lint, self._tests):
            outcome = await check(workspace, touched)
            if not outcome.ok and not outcome.skipped:
                return outcome
        return CheckOutcome(kind=CheckKind.TESTS, ok=True)

    async def _syntax(self, workspace: Workspace, touched: list[str]) -> CheckOutcome:
        python_files = [path for path in touched if path.endswith(".py")]
        if not python_files:
            return CheckOutcome(kind=CheckKind.SYNTAX, ok=True, skipped=True)
        result = await workspace.run(
            workspace.python_executable, "-m", "py_compile", *python_files, timeout=_QUICK_TIMEOUT
        )
        return CheckOutcome(kind=CheckKind.SYNTAX, ok=result.ok, output=_trim(result))

    async def _imports(self, workspace: Workspace, touched: list[str]) -> CheckOutcome:
        """
        Импорт изменённых модулей — самая дешёвая проверка, которая ловит
        настоящие ошибки: ImportError, циклы, опечатки в именах.

        Импортируется именно изменённое, а не весь пакет: чужой репозиторий
        на импорте может поднять сервер или полезть в сеть, и делать это на
        каждом круге — плохая идея.
        """
        targets = [_import_target(workspace.root, path) for path in touched if path.endswith(".py")]
        found = [item for item in targets if item is not None]
        if not found:
            return CheckOutcome(kind=CheckKind.IMPORTS, ok=True, skipped=True)

        search_path = sorted({item[0] for item in found})
        modules = sorted({item[1] for item in found})
        script = (
            "import sys, importlib\n"
            f"sys.path[:0] = {search_path!r}\n"
            f"for name in {modules!r}:\n"
            "    importlib.import_module(name)\n"
        )
        result = await workspace.run(
            workspace.python_executable, "-c", script, timeout=_QUICK_TIMEOUT
        )
        return CheckOutcome(kind=CheckKind.IMPORTS, ok=result.ok, output=_trim(result))

    async def _lint(self, workspace: Workspace, touched: list[str]) -> CheckOutcome:
        python_files = [path for path in touched if path.endswith(".py")]
        if not python_files:
            return CheckOutcome(kind=CheckKind.LINT, ok=True, skipped=True)
        result = await workspace.run(
            self._ruff, "check", "--no-cache", "--select", "E9,F", *python_files, timeout=_QUICK_TIMEOUT
        )
        if result.exit_code == 127:
            # ruff в окружении нет — это рабочий режим, а не сбой (то же
            # решение, что в efi/dev/sandbox.py).
            return CheckOutcome(kind=CheckKind.LINT, ok=True, skipped=True)
        return CheckOutcome(kind=CheckKind.LINT, ok=result.ok, output=_trim(result))

    async def _tests(self, workspace: Workspace, touched: list[str]) -> CheckOutcome:
        if not _has_tests(workspace.root):
            return CheckOutcome(kind=CheckKind.TESTS, ok=True, skipped=True)
        result = await workspace.run(
            workspace.python_executable, "-m", "pytest", "-x", "-q", timeout=_PYTEST_TIMEOUT
        )
        if result.exit_code in (4, 5, 127):
            # 5 — тестов не нашлось, 4 — ошибка использования, 127 — нет
            # самого pytest. Ни одно из этого не является падением проекта.
            return CheckOutcome(kind=CheckKind.TESTS, ok=True, skipped=True)
        return CheckOutcome(kind=CheckKind.TESTS, ok=result.ok, output=_trim(result))

    async def _repair(
        self, workspace: Workspace, failure: CheckOutcome, touched: list[str], report: RepairReport
    ) -> bool:
        request = _render_request(workspace, failure, touched)
        answer = await self._fixer(_FIX_SYSTEM_PROMPT, request)
        if not answer:
            return False

        edits = parse_edits(answer)
        if not edits:
            logger.info("auto_fix: в ответе нет ни одного блока правки")
            return False

        changed, problems = apply_edits(workspace.root, edits)
        for path in changed:
            if path not in report.changed_files:
                report.changed_files.append(path)
            if path not in touched:
                touched.append(path)
        if problems:
            logger.info("auto_fix: часть правок отклонена: %s", "; ".join(problems)[:200])
        return bool(changed)

    async def _say(self, failure: CheckOutcome, round_number: int, report: RepairReport) -> None:
        """
        Короткая реплика по ходу — только на первом круге и только если есть
        что сказать. Комментировать каждый круг — это уже не «работаю рядом»,
        а лента уведомлений CI.
        """
        note = _failure_note(failure)
        report.notes.append(note)
        if self._narrator is None or round_number > 0:
            return
        await self._narrator(note)


def _render_request(workspace: Workspace, failure: CheckOutcome, touched: list[str]) -> str:
    """Запрос на починку: дословный вывод падения плюс файлы, которых он касается."""
    files = "\n\n".join(
        f"### {path}\n{workspace.read(path)}" for path in touched[:4] if workspace.read(path)
    )
    return (
        f"Проверка «{failure.kind.human}» упала. Вывод дословно:\n\n"
        f"{failure.output[:_MAX_FAILURE_CHARS]}\n\n"
        f"Файлы, которых это касается:\n\n{files}\n\n"
        "Почини причину падения и верни правки блоками SEARCH/REPLACE."
    )


def _failure_note(failure: CheckOutcome) -> str:
    """
    Повод для реплики — из того, что РЕАЛЬНО упало. Не «возникла ошибка», а
    строчка, по которой человек сразу понимает, о чём речь.
    """
    first = next((line.strip() for line in failure.output.splitlines() if line.strip()), "")
    last_error = next(
        (
            line.strip()
            for line in reversed(failure.output.splitlines())
            if "Error" in line or "error" in line or "assert" in line
        ),
        first,
    )
    return f"{failure.kind.human} не прошли: {last_error[:200]}"


def _trim(result: CommandResult) -> str:
    return result.output[-_MAX_FAILURE_CHARS:]


def _import_target(root: Path, path: str) -> tuple[str, str] | None:
    """
    Как импортировать этот файл: (что добавить в sys.path, имя модуля).

    Правило повторяет то, как Python на самом деле находит модули, а не то,
    как удобно нам. Файл внутри пакета (рядом лежит `__init__.py`)
    импортируется полным точечным именем от корня пакета. Одинокий скрипт —
    по имени файла, с его каталогом в sys.path: ровно то, что делает
    `python src/main.py`, и ровно поэтому `from parser import parse` в таком
    проекте работает.

    Без этого различия проверка «не импортируется» срабатывала бы на вполне
    рабочих проектах — то есть требовала бы от модели чинить то, что не
    сломано, и это худший вид ложной тревоги.
    """
    if not path.endswith(".py"):
        return None
    parts = [part for part in path[: -len(".py")].split("/") if part not in ("", ".")]
    if not parts or any(not part.isidentifier() for part in parts):
        return None

    directory = root.joinpath(*parts[:-1]) if len(parts) > 1 else root
    package_parts: list[str] = []
    probe = directory
    while probe != root and (probe / "__init__.py").is_file():
        package_parts.insert(0, probe.name)
        probe = probe.parent

    if parts[-1] == "__init__":
        if not package_parts:
            return None
        return _relative_entry(probe, root), ".".join(package_parts)

    module = ".".join([*package_parts, parts[-1]])
    return _relative_entry(probe if package_parts else directory, root), module


def _relative_entry(path: Path, root: Path) -> str:
    """Каталог для sys.path относительно корня рабочей копии (команды и так идут с cwd=корень)."""
    relative = str(path.relative_to(root))
    return "." if relative in ("", ".") else relative


def _has_tests(root: Path) -> bool:
    """Есть ли что запускать pytest'ом. Без этого он ищет тесты по всему дереву и находит чужие."""
    if (root / "tests").is_dir():
        return True
    return any(root.glob("test_*.py")) or any(root.glob("*_test.py"))


__all__ = [
    "DEFAULT_MAX_ROUNDS",
    "CheckKind",
    "CheckOutcome",
    "FixerCall",
    "Narrator",
    "RepairLoop",
    "RepairReport",
]
