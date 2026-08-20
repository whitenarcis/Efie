"""
efi/dev/imports.py

Проверка проекта как ЦЕЛОГО, а не файла по отдельности.

Песочница (efi/dev/sandbox.py) смотрит на один файл: парсится ли, нет ли
неопределённых имён внутри него. Этого достаточно, чтобы поймать обрыв
генерации, и совершенно недостаточно, чтобы поймать главную болезнь
пофайловой генерации — расхождение между файлами. Каждый модуль по
отдельности безупречен, а вместе они не работают:

    src/main.py:    from parser import parse_line   # такой функции нет
    src/parser.py:  def parse(line: str) -> dict     # она называется иначе

Ни `compile()`, ни `ruff --select E9,F` этого не видят: для них соседний
модуль — внешний мир. Видит это только тот, кто держит в руках весь проект
сразу, — то есть код здесь.

Проверяется ровно то, что можно утверждать наверняка, без запуска:

    1. `from <свой модуль> import <имя>` — имя должно быть в том файле.
       Это самая частая и самая дорогая ошибка: проект выглядит готовым,
       ссылка отправлена, а `python src/main.py` падает на первой строке.
    2. `import <модуль>`, который не свой, не из стандартной библиотеки и не
       заявлен в стеке проекта, — выдуманный модуль (`import utils`, когда
       никакого utils нет).

Всё остальное — не наше дело: сторонняя библиотека, честно указанная в
стеке, это не ошибка, а зависимость (см. render_requirements).
"""

from __future__ import annotations

import ast
import logging
import re
import sys
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

logger = logging.getLogger(__name__)

#: Имена, которые есть у любого запуска Python, но которых нет в
#: sys.stdlib_module_names.
_ALWAYS_AVAILABLE = frozenset({"__future__", "__main__"})

#: Из строки стека («python 3.11», «rich>=13», «argparse (stdlib)») нужно
#: только имя пакета.
_STACK_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_.-]*")

#: Что в стеке означает не зависимость, а язык или его отсутствие.
_STACK_NOISE = frozenset({"python", "python3", "stdlib", "стандартная", "std", "cpython"})


@dataclass(slots=True, frozen=True)
class ImportProblem:
    """
    Одно расхождение между файлами проекта.

    `fatal` отделяет «проект точно не запустится» от «похоже на ошибку».
    Отсутствующее имя в своём же модуле — первое: это ImportError на первой
    строке, и публиковать такое нельзя. Неизвестный модуль — второе: он
    может оказаться настоящим пакетом, который кодер решил взять, не указав
    в стеке, и хоронить из-за этого готовый проект — хуже, чем выложить его
    с замечанием.
    """

    path: str
    message: str
    fatal: bool = False


def project_modules(paths: Iterable[str]) -> dict[str, str]:
    """
    Как модули проекта могут быть названы в import'е.

    Один файл отзывается на несколько имён, и это не небрежность, а правда
    про Python: `src/parser.py` при запуске `python src/main.py` импортируется
    как `parser` (каталог скрипта первым в sys.path), а при запуске из корня
    пакетом — как `src.parser`. Обе формы рабочие, поэтому обе считаются
    своими: задача проверки — ловить выдуманные модули, а не навязывать
    единственный способ запуска.
    """
    modules: dict[str, str] = {}
    for path in paths:
        if not path.endswith(".py"):
            continue
        parts = path[: -len(".py")].split("/")
        stem = parts[-1]
        if stem == "__init__" and len(parts) > 1:
            parts, stem = parts[:-1], parts[-2]
        modules.setdefault(".".join(parts), path)
        modules.setdefault(stem, path)
    return modules


def exported_names(source: str) -> set[str]:
    """
    Что модуль отдаёт наружу: функции, классы, константы и импортированные им
    же имена (реэкспорт — обычное дело для точки входа).

    Разбор по AST, а не по регулярке: `def` внутри класса или внутри `if` не
    должны попасть в список верхнего уровня, а `__all__` — не текст, а данные.
    """
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return set()

    names: set[str] = set()
    _collect_names(tree.body, names)
    return names


def _collect_names(body: list[ast.stmt], names: set[str]) -> None:
    """
    Имена верхнего уровня — включая те, что объявлены под `try:` или `if:`.

    Обёртка вокруг импорта или определения — обычный приём (`try: import
    tomllib except ImportError: import tomli as tomllib`), и имя из неё
    доступно снаружи ровно так же. Не заглядывать внутрь значило бы считать
    его несуществующим и требовать от кодера «починить» рабочий код.
    """
    for node in body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
        elif isinstance(node, ast.Assign):
            names.update(target.id for target in node.targets if isinstance(target, ast.Name))
        elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            names.add(node.target.id)
        elif isinstance(node, ast.Import):
            names.update((alias.asname or alias.name.split(".")[0]) for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update((alias.asname or alias.name) for alias in node.names)
        elif isinstance(node, ast.Try):
            _collect_names([*node.body, *node.orelse, *node.finalbody], names)
            for handler in node.handlers:
                _collect_names(handler.body, names)
        elif isinstance(node, ast.If | ast.For | ast.While | ast.With | ast.AsyncWith):
            _collect_names([*node.body, *getattr(node, "orelse", [])], names)


def stack_dependencies(stack: Iterable[str]) -> list[str]:
    """Имена сторонних пакетов из стека спеки: без версий, без языка, без дублей."""
    found: dict[str, None] = {}
    for item in stack:
        match = _STACK_NAME_RE.match(str(item).strip().lower())
        if match is None:
            continue
        name = match.group(0).rstrip(".-")
        if not name or name in _STACK_NOISE or name in sys.stdlib_module_names:
            continue
        found.setdefault(name, None)
    return list(found)


def cross_file_problems(files: Mapping[str, str], *, stack: Iterable[str] = ()) -> list[ImportProblem]:
    """
    Замечания к проекту целиком.

    Формулировки — как у линтера, потому что они уходят ровно туда же, куда
    его замечания: обратно кодеру в цикл исправлений (efi/dev/engine.py).
    """
    modules = project_modules(files)
    exports = {path: exported_names(source) for path, source in files.items() if path.endswith(".py")}
    known = set(stack_dependencies(stack))

    problems: list[ImportProblem] = []
    for path, source in files.items():
        if not path.endswith(".py"):
            continue
        problems.extend(_problems_in_file(path, source, modules=modules, exports=exports, known=known))
    return problems


def _problems_in_file(
    path: str,
    source: str,
    *,
    modules: Mapping[str, str],
    exports: Mapping[str, set[str]],
    known: set[str],
) -> list[ImportProblem]:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        # Битый синтаксис — забота песочницы, она скажет об этом точнее.
        return []

    problems: list[ImportProblem] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level:
                continue  # относительный импорт: разрешается пакетом, а не нами
            problems.extend(
                _check_from_import(path, node, modules=modules, exports=exports, known=known)
            )
        elif isinstance(node, ast.Import):
            for alias in node.names:
                problem = _check_module(path, node.lineno, alias.name, modules=modules, known=known)
                if problem:
                    problems.append(problem)
    return problems


def _check_from_import(
    path: str,
    node: ast.ImportFrom,
    *,
    modules: Mapping[str, str],
    exports: Mapping[str, set[str]],
    known: set[str],
) -> list[ImportProblem]:
    module = node.module or ""
    target = modules.get(module) or modules.get(module.split(".")[-1] if "." in module else module)
    if target is None:
        problem = _check_module(path, node.lineno, module, modules=modules, known=known)
        return [problem] if problem is not None else []
    if target == path:
        return []

    available = exports.get(target, set())
    if not available:
        return []  # файл не разобрался — судить не по чему
    missing = [alias.name for alias in node.names if alias.name != "*" and alias.name not in available]
    if not missing:
        return []
    return [
        ImportProblem(
            path=path,
            message=(
                f"{path}:{node.lineno}: ImportError: в модуле {module} ({target}) нет "
                f"{', '.join(missing)}; там определены: {', '.join(sorted(available)) or '(ничего)'}"
            ),
            fatal=True,
        )
    ]


def _check_module(
    path: str, lineno: int, module: str, *, modules: Mapping[str, str], known: set[str]
) -> ImportProblem | None:
    if not module:
        return None
    root = module.split(".")[0]
    if root in _ALWAYS_AVAILABLE or root in sys.stdlib_module_names:
        return None
    if module in modules or root in modules:
        return None
    if root in known:
        return None
    return ImportProblem(
        path=path,
        message=(
            f"{path}:{lineno}: ModuleNotFoundError: модуля {module} нет ни в проекте, ни в стандартной "
            f"библиотеке, ни в стеке проекта"
        ),
    )


def render_requirements(stack: Iterable[str]) -> str:
    """
    requirements.txt из стека — или пусто, если проект на стандартной
    библиотеке (тогда и файла быть не должно: пустой requirements.txt
    сообщает читателю ровно ничего).
    """
    dependencies = stack_dependencies(stack)
    if not dependencies:
        return ""
    return "\n".join(dependencies) + "\n"


__all__ = [
    "ImportProblem",
    "cross_file_problems",
    "exported_names",
    "project_modules",
    "render_requirements",
    "stack_dependencies",
]
