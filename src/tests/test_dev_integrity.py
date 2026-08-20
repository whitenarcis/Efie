"""
Тесты того, что проект работает КАК ЦЕЛОЕ, а не пофайлово.

Это самая дорогая болезнь пофайловой генерации и самая незаметная: каждый
файл по отдельности парсится, проходит линтер и выглядит образцово, а вместе
они не запускаются — точка входа зовёт функцию, которой в соседнем модуле
нет. Ссылка при этом уже отправлена.

Проверяется здесь три вещи, и все три без сети: сам разбор импортов
(efi/dev/imports.py), порядок написания файлов (модули раньше точки входа) и
поведение конвейера, когда расхождение всё-таки нашлось.
"""

from __future__ import annotations

import json
from typing import Any

from efi.config.schema import TaskRole
from efi.dev.engine import DevEngine
from efi.dev.imports import (
    cross_file_problems,
    exported_names,
    project_modules,
    render_requirements,
    stack_dependencies,
)
from efi.dev.sandbox import CodeSandbox
from efi.dev.schemas import FileSpec, ProjectSpec
from efi.llm.schemas import Choice, LLMParams, Message, Response, Role, Session

_SPEC_PAYLOAD = {
    "slug": "log-digest",
    "title": "Log Digest",
    "problem": "Разбирает логи nginx и показывает топ ошибок за период",
    "stack": ["python 3.11"],
    # Точка входа названа ПЕРВОЙ — так их и перечисляют модели, и так их
    # нельзя писать.
    "files": [
        {"path": "src/main.py", "purpose": "точка входа CLI"},
        {"path": "src/parser.py", "purpose": "разбор строк лога"},
    ],
}


def _spec(**overrides: Any) -> ProjectSpec:
    return ProjectSpec.model_validate(_SPEC_PAYLOAD | overrides)


# -- разбор импортов ----------------------------------------------------------


def test_missing_neighbour_function_is_caught() -> None:
    """Ровно тот случай, ради которого модуль существует: ImportError на первой строке запуска."""
    problems = cross_file_problems(
        {
            "src/main.py": "from parser import parse_line\n\nparse_line('x')\n",
            "src/parser.py": "def parse(line: str) -> dict:\n    return {}\n",
        }
    )

    assert len(problems) == 1
    assert problems[0].path == "src/main.py"
    assert problems[0].fatal is True
    assert "parse_line" in problems[0].message
    assert "parse" in problems[0].message, "кодеру нужно видеть, что там на самом деле есть"


def test_a_project_that_agrees_with_itself_has_no_problems() -> None:
    problems = cross_file_problems(
        {
            "src/main.py": "import argparse\n\nfrom src.parser import parse_line\n\nparse_line('x')\n",
            "src/parser.py": "def parse_line(line: str) -> dict:\n    return {}\n",
            "README.md": "# что-то\n",
        }
    )

    assert problems == []


def test_both_ways_of_naming_your_own_module_are_valid() -> None:
    """
    `python src/main.py` кладёт в sys.path каталог скрипта, запуск из корня —
    корень. Обе формы импорта рабочие, и требовать одну — значит браковать
    работающий проект.
    """
    modules = project_modules(["src/parser.py", "src/main.py"])

    assert modules["parser"] == "src/parser.py"
    assert modules["src.parser"] == "src/parser.py"


def test_invented_module_is_reported_but_does_not_bury_the_project() -> None:
    """
    `import utils`, которого нет, — ошибка. Но имя может оказаться и настоящим
    пакетом, который кодер взял, не указав в стеке, поэтому хоронить из-за
    этого готовый проект нельзя.
    """
    problems = cross_file_problems({"src/main.py": "import utils\n"})

    assert len(problems) == 1
    assert problems[0].fatal is False
    assert "utils" in problems[0].message


def test_declared_dependency_is_not_an_error() -> None:
    problems = cross_file_problems({"src/main.py": "import rich\n"}, stack=["python 3.11", "rich>=13"])

    assert problems == []


def test_names_defined_under_try_or_if_are_still_exported() -> None:
    """Иначе рабочий приём с запасным импортом выглядел бы как отсутствующее имя."""
    source = (
        "try:\n    import tomllib\nexcept ImportError:\n    tomllib = None\n\n"
        "if True:\n    LIMIT = 10\n\ndef read() -> None:\n    ...\n"
    )

    assert {"tomllib", "LIMIT", "read"} <= exported_names(source)


def test_stack_gives_requirements_only_when_there_is_something_to_install() -> None:
    assert stack_dependencies(["python 3.11", "argparse", "json"]) == []
    assert render_requirements(["python 3.11", "argparse"]) == ""
    assert render_requirements(["python 3.11", "rich>=13.0", "httpx"]) == "rich\nhttpx\n"


# -- конвейер -----------------------------------------------------------------


class _RecordingCoder:
    """Кодер, который отдаёт заготовленные файлы по путям и помнит, что ему показывали."""

    def __init__(self, sources: dict[str, str], *, fixes: dict[str, str] | None = None) -> None:
        self._sources = dict(sources)
        self._fixes = dict(fixes or {})
        self.unavailable_reason = ""
        self.last_answer_truncated = False
        self.order: list[str] = []
        self.contexts: dict[str, str] = {}
        self.fix_requests: list[tuple[str, str]] = []

    async def write_file(
        self, spec: ProjectSpec, file_spec: FileSpec, *, already_written: dict[str, str] | None = None
    ) -> str | None:
        self.order.append(file_spec.path)
        self.contexts[file_spec.path] = "\n".join((already_written or {}).values())
        return self._sources.get(file_spec.path)

    async def fix_file(self, path: str, source: str, diagnostics: str) -> str | None:
        self.fix_requests.append((path, diagnostics))
        return self._fixes.get(path)

    async def write_document(self, path: str, *, system_prompt: str, request: str) -> str | None:
        return _README


_README = (
    "# Log Digest\n\nУтилита.\n\n"
    "## Назначение\n\nРазбирает логи nginx и показывает топ ошибок за период — тем, кто держит "
    "сервер и не хочет читать гигабайты руками.\n\n"
    "## Установка\n\nPython 3.11+, зависимостей нет.\n\n```bash\ngit clone https://github.com/e/l.git\n```\n\n"
    "## Использование\n\n```bash\npython src/main.py access.log\n```\n\n"
    "## Структура\n\n- `src/parser.py` — разбор\n- `src/main.py` — вход\n"
)


class _StaticRouter:
    def __init__(self, text: str = "") -> None:
        self.text = text

    async def chat(self, role: TaskRole, params: LLMParams, session: Session) -> Response:
        return Response(choices=[Choice(message=Message(role=Role.ASSISTANT, content=self.text))])


def _engine(coder: Any, *, max_fix_iterations: int = 3) -> DevEngine:
    return DevEngine(
        _StaticRouter(json.dumps(_SPEC_PAYLOAD)),  # type: ignore[arg-type]
        coder,
        CodeSandbox(enable_linter=False),
        max_fix_iterations=max_fix_iterations,
    )


async def test_entrypoint_is_written_last_and_sees_the_real_interfaces() -> None:
    """
    Точка входа, написанная первой, выдумывает функции соседей — а соседи
    потом пишутся со своими именами. Порядок здесь дешевле любой проверки
    после.
    """
    coder = _RecordingCoder(
        {
            "src/parser.py": "def parse_line(line: str) -> dict:\n    return {}\n",
            "src/main.py": "from parser import parse_line\n\nparse_line('x')\n",
        }
    )

    build = await _engine(coder).build(_spec())

    assert coder.order == ["src/parser.py", "src/main.py"], "модули раньше точки входа"
    assert "def parse_line(line: str) -> dict:" in coder.contexts["src/main.py"]
    assert build.is_publishable is True


async def test_files_that_do_not_agree_are_sent_back_to_the_coder() -> None:
    coder = _RecordingCoder(
        {
            "src/parser.py": "def parse(line: str) -> dict:\n    return {}\n",
            "src/main.py": "from parser import parse_line\n\nparse_line('x')\n",
        },
        fixes={"src/main.py": "from parser import parse\n\nparse('x')\n"},
    )

    build = await _engine(coder).build(_spec())

    assert [path for path, _ in coder.fix_requests] == ["src/main.py"]
    assert "parse_line" in coder.fix_requests[0][1]
    assert build.is_publishable is True
    assert "from parser import parse\n" in build.as_file_map()["src/main.py"]


async def test_a_project_that_cannot_import_itself_is_not_published() -> None:
    """
    Проект, падающий ImportError'ом на первой строке, — это не «с
    замечаниями», это неработающий проект под её именем.
    """
    coder = _RecordingCoder(
        {
            "src/parser.py": "def parse(line: str) -> dict:\n    return {}\n",
            "src/main.py": "from parser import parse_line\n\nparse_line('x')\n",
        }
    )

    build = await _engine(coder).build(_spec())

    assert build.is_publishable is False
    assert build.broken_paths == ["src/main.py"]
    assert "parse_line" in build.failure_reason


async def test_repository_gets_the_files_every_repository_has() -> None:
    coder = _RecordingCoder(
        {
            "src/parser.py": "def parse_line(line: str) -> dict:\n    return {}\n",
            "src/main.py": "from parser import parse_line\n\nparse_line('x')\n",
        }
    )

    build = await _engine(coder).build(_spec(stack=["python 3.11", "rich"]))

    files = build.as_file_map()
    assert "__pycache__/" in files[".gitignore"]
    assert files["requirements.txt"] == "rich\n"
    assert "README.md" in files


async def test_no_requirements_file_when_the_project_is_pure_stdlib() -> None:
    """Пустой requirements.txt сообщает читателю ровно ничего — и его не должно быть."""
    coder = _RecordingCoder(
        {
            "src/parser.py": "def parse_line(line: str) -> dict:\n    return {}\n",
            "src/main.py": "from parser import parse_line\n\nparse_line('x')\n",
        }
    )

    build = await _engine(coder).build(_spec())

    assert "requirements.txt" not in build.as_file_map()
