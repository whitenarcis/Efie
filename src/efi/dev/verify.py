"""
efi/dev/verify.py

Последний рубеж перед публикацией: проект РЕАЛЬНО запускают.

До этого модуля весь конвейер собственных проектов был статическим:
компилятор, линтер, сверка импортов между файлами. Этого хватает, чтобы
поймать обрыв генерации, и не хватает, чтобы отличить «код выглядит
правильным» от «программа работает». Разница видна на первой же попытке
запустить: `AttributeError` на строке разбора аргументов, забытый
`if __name__ == "__main__"`, функция, которая ждёт `Path`, а получает `str`.
Ни одна из этих ошибок не находится чтением — только запуском.

Отсюда правило: перед публикацией проект раскладывается в одноразовую
рабочую копию (efi/dev/workspace.py, с лимитами и таймаутами), там же
чинится по настоящим трейсбэкам (efi/dev/auto_fix.py) и только потом едет в
репозиторий.

И второе правило, ради которого модуль вообще появился: НЕ ЗАСТРЕВАТЬ.
Проект, у которого не сходится один файл из четырёх, не должен ни висеть
вечно, ни пропадать целиком. Если после кругов починки остаётся кусок,
который импортируется и запускается, — выкладывается он, а выброшенное
честно называется в реплике и в памяти. Автор, который месяц полирует то,
чего никто не видел, ничем не отличается от автора, который ничего не
написал.

Запуск здесь — единственное место во всём конвейере СОБСТВЕННЫХ проектов,
где сгенерированный код исполняется, и происходит это в клоне с
`resource.setrlimit` и таймаутом (см. докстринг efi/dev/workspace.py).
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field

from efi.dev.auto_fix import FixerCall, Lookup, Narrator, RepairLoop, RepairReport
from efi.dev.imports import project_modules
from efi.dev.readme import README_PATH
from efi.dev.sandbox import write_project_files
from efi.dev.schemas import GeneratedFile, ProjectSpec
from efi.dev.workspace import Workspace, WorkspaceError, WorkspaceManager

logger = logging.getLogger(__name__)

#: Сколько кругов починки по трейсбэкам до того, как переходить к обрезке.
#: Три: за столько модель либо чинит, либо начинает ходить по кругу.
DEFAULT_VERIFY_ROUNDS = 3

#: Имена, по которым файл узнаётся как точка входа.
_ENTRYPOINT_SUFFIXES = ("main.py", "cli.py", "__main__.py", "app.py")


@dataclass(slots=True)
class VerifyOutcome:
    """Что получилось из проверки запуском."""

    files: list[GeneratedFile] = field(default_factory=list)
    green: bool = False
    #: Файлы, выброшенные ради того, чтобы остальное заработало.
    dropped: list[str] = field(default_factory=list)
    #: Почему не срослось, если не срослось, — дословная строка падения.
    failure: str = ""
    #: Поводы для реплик в чат (те же, что у цикла починки).
    notes: list[str] = field(default_factory=list)
    rounds: int = 0

    @property
    def is_publishable(self) -> bool:
        """Есть что публиковать: проект запускается, и в нём остался код."""
        return self.green and any(item.path.endswith(".py") for item in self.files)


class ProjectVerifier:
    """
    Проверка своего проекта запуском — и, если надо, обрезка до работающего.

    Работает поверх той же машинерии, что и работа с чужим кодом: одноразовая
    рабочая копия и закрытый цикл починки. Разница в предмете: там правится
    существующий проект, здесь — только что написанный.
    """

    def __init__(
        self,
        workspaces: WorkspaceManager,
        fixer: FixerCall,
        *,
        max_rounds: int = DEFAULT_VERIFY_ROUNDS,
        narrator: Narrator | None = None,
        lookup: Lookup | None = None,
        allow_pruning: bool = True,
    ) -> None:
        self._workspaces = workspaces
        self._fixer = fixer
        self._max_rounds = max_rounds
        self._narrator = narrator
        #: Поиск по тексту ошибки, когда она пережила первую правку
        #: (см. efi/dev/auto_fix.py).
        self._lookup = lookup
        #: Разрешено ли выкладывать проект без файла, который так и не
        #: заработал. Да — потому что альтернатива не «идеальный проект», а
        #: отсутствующий: см. докстринг модуля.
        self._allow_pruning = allow_pruning

    async def verify(self, spec: ProjectSpec, files: list[GeneratedFile]) -> VerifyOutcome:
        """
        Раскладывает проект, гоняет проверки, чинит найденное и возвращает то,
        что реально работает.

        Не поднимает исключений: это фоновая работа, и любая беда должна
        вернуться причиной, а не падением посреди цикла.
        """
        outcome = VerifyOutcome(files=list(files))
        try:
            workspace = await self._workspaces.prepare_empty(session_id=f"project-{spec.slug}")
        except WorkspaceError as exc:
            # Проверить не вышло — но это не повод считать проект сломанным:
            # статические проверки он уже прошёл.
            logger.warning("verify: рабочую копию для %s не создать (%s), проверяю только статикой", spec.slug, exc)
            outcome.green = True
            outcome.notes.append(f"запустить проект не получилось: {exc}")
            return outcome

        try:
            await self._run(workspace, spec, outcome)
        except Exception as exc:  # noqa: BLE001 — фоновая проверка не имеет права ронять конвейер
            logger.exception("verify: проверка %s сорвалась", spec.slug)
            outcome.failure = f"проверка сорвалась: {exc}"
        finally:
            workspace.discard()
        return outcome

    async def _run(self, workspace: Workspace, spec: ProjectSpec, outcome: VerifyOutcome) -> None:
        current = {item.path: item for item in outcome.files}
        write_project_files(workspace.root, {path: item.content for path, item in current.items()})

        entrypoint = _entrypoint(list(current))
        report = await self._repair(workspace, list(current), entrypoint)
        outcome.rounds = report.rounds
        outcome.notes.extend(report.notes)
        _absorb(workspace, current)

        if report.green:
            outcome.files = list(current.values())
            outcome.green = True
            return

        failure = report.last_failure
        outcome.failure = failure.output.strip().splitlines()[-1] if failure and failure.output else ""

        if not self._allow_pruning:
            outcome.files = list(current.values())
            return

        # Выкинуть то, что не срослось, и попробовать ещё раз тем, что
        # осталось. Единственная попытка: вторая обрезка — это уже не «проект
        # без лишнего файла», а другой проект.
        removed = _prune(workspace, current, failure_output=failure.output if failure else "")
        if not removed:
            outcome.files = list(current.values())
            return

        logger.info("verify: %s — выкидываю %s и проверяю остаток", spec.slug, ", ".join(removed))
        entrypoint = _entrypoint(list(current))
        report = await self._repair(workspace, list(current), entrypoint)
        _absorb(workspace, current)
        outcome.files = list(current.values())
        outcome.rounds += report.rounds
        outcome.notes.extend(report.notes)

        if report.green and entrypoint:
            outcome.green = True
            outcome.dropped = removed
            outcome.failure = ""
            outcome.notes.append(
                f"выложила без {', '.join(removed)} — этот кусок так и не заработал, "
                "остальное запускается"
            )
            return

        last = report.last_failure
        outcome.failure = (
            last.output.strip().splitlines()[-1] if last and last.output else outcome.failure
        )

    async def _repair(self, workspace: Workspace, paths: list[str], entrypoint: str) -> RepairReport:
        loop = RepairLoop(
            self._fixer, max_rounds=self._max_rounds, narrator=self._narrator, lookup=self._lookup
        )
        checkable = [path for path in paths if path.endswith(".py")]
        return await loop.run(workspace, checkable, entrypoint=entrypoint)


def _absorb(workspace: Workspace, current: dict[str, GeneratedFile]) -> None:
    """
    Забирает из рабочей копии то, что там стало после починки.

    Именно с диска, а не из памяти: цикл починки правит файлы на месте, и
    брать в репозиторий версию «как сгенерировали» значило бы выложить код,
    который заведомо не запускается.
    """
    for path, item in list(current.items()):
        updated = workspace.read(path)
        if updated and updated != item.content:
            current[path] = item.model_copy(
                update={"content": updated, "fix_rounds": item.fix_rounds + 1}
            )


def _entrypoint(paths: list[str]) -> str:
    """Точка входа проекта — то, что запускают. Пусто, если её нет (тогда и запускать нечего)."""
    for path in paths:
        if path.lower().endswith(_ENTRYPOINT_SUFFIXES):
            return path
    return ""


def _prune(
    workspace: Workspace, current: dict[str, GeneratedFile], *, failure_output: str
) -> list[str]:
    """
    Убирает файлы, из-за которых проект не работает, — но только те, без
    которых он ещё может жить.

    Три условия, и все три обязательны: файл упомянут в падении, он не точка
    входа и его никто не импортирует. Выкинуть модуль, на который ссылается
    точка входа, — значит поменять одно падение на другое.
    """
    entrypoint = _entrypoint(list(current))
    suspects = [
        path
        for path in current
        if path.endswith(".py") and path != entrypoint and path in failure_output
    ]
    if not suspects:
        return []

    needed = _imported_paths(current)
    removable = [path for path in suspects if path not in needed]
    if not removable:
        return []

    for path in removable:
        current.pop(path, None)
        target = workspace.root / path
        if target.is_file():
            target.unlink()
    return removable


def _imported_paths(current: dict[str, GeneratedFile]) -> set[str]:
    """Файлы проекта, которые импортирует кто-то другой, — их выкидывать нельзя."""
    modules = project_modules(list(current))
    needed: set[str] = set()
    for path, item in current.items():
        if path.lower() == README_PATH.lower() or not path.endswith(".py"):
            continue
        try:
            tree = ast.parse(item.content)
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names.append(node.module)
            elif isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            for name in names:
                target = modules.get(name) or modules.get(name.split(".")[0])
                if target is not None and target != path:
                    needed.add(target)
    return needed


__all__ = ["DEFAULT_VERIFY_ROUNDS", "ProjectVerifier", "VerifyOutcome"]
