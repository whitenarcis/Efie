"""
efi/dev/repo_map.py

Скелет репозитория: что где лежит и что оно объявляет — на пятнадцать
килобайт вместо мегабайтов исходников.

Задача. Чтобы поправить баг в чужом проекте, надо сначала понять, в каких
двух-трёх файлах он живёт. Отдать модели весь репозиторий нельзя (не влезет
и стоит денег), отдать список путей — мало: по `src/core/handlers.py` не
видно, что там внутри. Нужна середина: дерево файлов, а в каждом — имена
классов, функций и сигнатуры верхнего уровня, без тел.

Такая карта решает ровно один вопрос — «куда смотреть», — и этого достаточно:
дальше два-три выбранных файла читаются целиком и правятся точечно
(efi/dev/edits.py).

Разбор идёт через `ast`, а не регулярками: у регулярки нет шансов отличить
`def` внутри класса от `def` внутри строки, а ошибка здесь стоит не «чуть
хуже карта», а «модель правит файл, которого не понимает». Файл, который не
парсится, попадает в карту одной строкой «не разобрался» — это тоже
информация, и часто именно та, которую искали.

КОД НЕ ИСПОЛНЯЕТСЯ. Как и в efi/dev/sandbox.py: `ast.parse` разбирает текст
и не запускает его — единственный способ смотреть в чужой репозиторий, не
доверяя ему.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

#: Потолок карты. Пятнадцать килобайт — примерно четыре тысячи токенов:
#: столько можно отдать под «оглавление» проекта, не вытесняя из контекста
#: сам код, который потом придётся читать и править.
DEFAULT_MAP_BUDGET_BYTES = 15_000

#: Куда не ходим. Каталоги окружений и артефактов — это не код проекта, а
#: гарантированный способ забить карту чужими зависимостями.
_SKIP_DIRS = frozenset(
    {
        ".git", ".hg", ".svn", "__pycache__", ".venv", "venv", "env", "node_modules",
        ".mypy_cache", ".ruff_cache", ".pytest_cache", "build", "dist", ".tox", ".idea",
        ".eggs", "site-packages", ".next", "target",
    }
)

#: Файлы, которые важны сами по себе, даже если это не Python: по ним видно,
#: как проект ставится и запускается.
_NOTABLE_FILES = ("pyproject.toml", "setup.py", "requirements.txt", "Makefile", "README.md")

#: Максимум объявлений на файл. Файл на сотню функций в карте не нужен —
#: нужно понять, о чём он.
_MAX_SYMBOLS_PER_FILE = 25

#: Сколько файлов вообще рассматривать. Защита от репозитория на десять тысяч
#: файлов: карта всё равно обрежется по бюджету, а обход диска — нет.
_MAX_FILES_SCANNED = 2000


@dataclass(slots=True, frozen=True)
class FileOutline:
    """Один файл: путь, размер и его публичные объявления."""

    path: str
    size_bytes: int
    symbols: list[str] = field(default_factory=list)
    #: Почему файл не разобрался, если не разобрался.
    problem: str = ""

    def render(self) -> str:
        if self.problem:
            return f"{self.path}  ({self.problem})"
        if not self.symbols:
            return f"{self.path}"
        body = "\n".join(f"    {item}" for item in self.symbols)
        return f"{self.path}\n{body}"


@dataclass(slots=True, frozen=True)
class RepoMap:
    """Карта репозитория целиком."""

    root: Path
    files: list[FileOutline] = field(default_factory=list)
    #: Сколько файлов не поместилось в бюджет — честная строчка в конце карты
    #: важнее тишины: модель должна знать, что видит не всё.
    omitted: int = 0

    @property
    def paths(self) -> list[str]:
        return [item.path for item in self.files]

    def render(self) -> str:
        lines = [item.render() for item in self.files]
        if self.omitted:
            lines.append(f"... и ещё {self.omitted} файл(ов), не поместившихся в карту")
        return "\n".join(lines)


def build_repo_map(root: Path, *, budget_bytes: int = DEFAULT_MAP_BUDGET_BYTES) -> RepoMap:
    """
    Строит карту проекта под `root`.

    Порядок файлов — не алфавитный: сначала то, по чему проект узнаётся
    (README, pyproject, точки входа), потом остальной код по убыванию
    «содержательности» (число объявлений). При обрезке по бюджету теряется
    хвост из мелочи, а не половина ядра.
    """
    root = root.resolve()
    outlines = [_outline(path, root) for path in _walk(root)]
    outlines.sort(key=_importance, reverse=True)

    kept: list[FileOutline] = []
    used = 0
    for outline in outlines:
        rendered = len(outline.render().encode("utf-8")) + 1
        if used + rendered > budget_bytes and kept:
            break
        kept.append(outline)
        used += rendered

    kept.sort(key=lambda item: item.path)
    return RepoMap(root=root, files=kept, omitted=len(outlines) - len(kept))


def _walk(root: Path) -> list[Path]:
    found: list[Path] = []
    for path in sorted(root.rglob("*")):
        if len(found) >= _MAX_FILES_SCANNED:
            logger.info("repo_map: в %s слишком много файлов, смотрю первые %d", root, _MAX_FILES_SCANNED)
            break
        if not path.is_file():
            continue
        if any(part in _SKIP_DIRS for part in path.relative_to(root).parts):
            continue
        if path.suffix == ".py" or path.name in _NOTABLE_FILES:
            found.append(path)
    return found


def _outline(path: Path, root: Path) -> FileOutline:
    relative = str(path.relative_to(root))
    try:
        size = path.stat().st_size
    except OSError:
        size = 0

    if path.suffix != ".py":
        return FileOutline(path=relative, size_bytes=size)

    try:
        source = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return FileOutline(path=relative, size_bytes=size, problem=f"не читается: {exc}")

    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return FileOutline(path=relative, size_bytes=size, problem=f"не парсится: {exc.msg}")

    return FileOutline(path=relative, size_bytes=size, symbols=_symbols(tree))


def _symbols(tree: ast.Module) -> list[str]:
    """Публичные объявления верхнего уровня плюс методы классов — без тел."""
    found: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            found.append(f"class {node.name}{_bases(node)}:")
            found.extend(
                f"    {_signature(child)}"
                for child in node.body
                if isinstance(child, ast.FunctionDef | ast.AsyncFunctionDef)
                and not child.name.startswith("_")
            )
        elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
            if not node.name.startswith("_"):
                found.append(_signature(node))
        elif isinstance(node, ast.Assign):
            found.extend(
                f"{target.id} = ..."
                for target in node.targets
                if isinstance(target, ast.Name) and target.id.isupper()
            )
    return found[:_MAX_SYMBOLS_PER_FILE]


def _bases(node: ast.ClassDef) -> str:
    names = [ast.unparse(base) for base in node.bases]
    return f"({', '.join(names)})" if names else ""


def _signature(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    prefix = "async def" if isinstance(node, ast.AsyncFunctionDef) else "def"
    returns = f" -> {ast.unparse(node.returns)}" if node.returns is not None else ""
    return f"{prefix} {node.name}({ast.unparse(node.args)}){returns}"


def _importance(outline: FileOutline) -> tuple[int, int]:
    """
    Насколько файл важен для понимания проекта.

    Первым идёт то, по чему проект узнаётся снаружи (README, pyproject,
    точка входа), дальше — по числу объявлений: файл с двадцатью функциями
    рассказывает о проекте больше, чем файл с одной константой.
    """
    name = Path(outline.path).name
    head = 3 if name in ("README.md", "pyproject.toml") else 0
    if name in ("main.py", "__main__.py", "cli.py", "app.py"):
        head = max(head, 2)
    if Path(outline.path).parts[:1] in (("src",), ("app",)) and not head:
        head = 1
    return head, len(outline.symbols)


__all__ = ["DEFAULT_MAP_BUDGET_BYTES", "FileOutline", "RepoMap", "build_repo_map"]
