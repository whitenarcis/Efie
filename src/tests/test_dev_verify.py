"""
Тесты последнего рубежа перед публикацией: проект РЕАЛЬНО запускают.

Здесь настоящие подпроцессы: только запуск отличает «код выглядит
правильным» от «программа работает», и подменять его нечем. Модель, как
обычно, подменена сценарием — проверяется наш код вокруг неё.

Два свойства, ради которых модуль и написан:
  * то, что чинится по трейсбэку, чинится — и уезжает в репозиторий рабочим;
  * то, что не чинится, не держит проект вечно: кусок выбрасывается, остаток
    выходит в свет, и об этом говорится вслух.
"""

from __future__ import annotations

from pathlib import Path

from efi.dev.schemas import GeneratedFile, ProjectSpec
from efi.dev.verify import ProjectVerifier
from efi.dev.workspace import WorkspaceManager

_SPEC = ProjectSpec.model_validate(
    {
        "slug": "log-digest",
        "title": "Log Digest",
        "problem": "Разбирает логи nginx и показывает топ ошибок за период",
        "stack": ["python 3.11", "argparse"],
        "files": [
            {"path": "src/parser.py", "purpose": "разбор строк"},
            {"path": "src/main.py", "purpose": "точка входа"},
        ],
    }
)

_WORKING_MAIN = (
    "import argparse\n\n"
    "from parser import parse_line\n\n\n"
    "def main() -> None:\n"
    "    argparse.ArgumentParser().parse_args()\n"
    "    parse_line('x')\n\n\n"
    "if __name__ == '__main__':\n"
    "    main()\n"
)
_WORKING_PARSER = "def parse_line(line: str) -> dict:\n    return {'raw': line}\n"


class _ScriptedFixer:
    """Кодер, чинящий по сценарию; помнит, что ему показывали."""

    def __init__(self, answers: list[str]) -> None:
        self._answers = list(answers)
        self.requests: list[str] = []

    async def __call__(self, system_prompt: str, request: str) -> str | None:
        self.requests.append(request)
        return self._answers.pop(0) if self._answers else None


def _verifier(fixer: _ScriptedFixer, tmp_path: Path, **kwargs: object) -> ProjectVerifier:
    return ProjectVerifier(
        WorkspaceManager(tmp_path / "workspaces"),
        fixer,
        **kwargs,  # type: ignore[arg-type]
    )


def _files(main: str, parser: str = _WORKING_PARSER) -> list[GeneratedFile]:
    return [
        GeneratedFile(path="src/parser.py", content=parser),
        GeneratedFile(path="src/main.py", content=main),
    ]


async def test_a_project_that_runs_goes_through_untouched(tmp_path: Path) -> None:
    fixer = _ScriptedFixer([])

    outcome = await _verifier(fixer, tmp_path).verify(_SPEC, _files(_WORKING_MAIN))

    assert outcome.green is True
    assert outcome.is_publishable is True
    assert outcome.dropped == []
    assert fixer.requests == [], "работающий проект чинить незачем"


async def test_a_crash_on_startup_is_fixed_by_its_own_traceback(tmp_path: Path) -> None:
    """
    Ровно тот класс ошибок, который не находится чтением: программа
    импортируется, но падает на первом же запуске.
    """
    broken = _WORKING_MAIN.replace("parse_line('x')", "parse_line()")
    fixer = _ScriptedFixer(
        [
            "src/main.py\n<<<<<<< SEARCH\n    parse_line()\n=======\n    parse_line('x')\n>>>>>>> REPLACE"
        ]
    )

    outcome = await _verifier(fixer, tmp_path).verify(_SPEC, _files(broken))

    assert outcome.green is True
    assert outcome.rounds == 1
    assert "TypeError" in fixer.requests[0], "модель видит настоящее падение, а не пересказ"
    fixed = {item.path: item.content for item in outcome.files}
    assert "parse_line('x')" in fixed["src/main.py"], "в репозиторий едет починенная версия"


async def test_a_module_that_never_works_is_dropped_and_the_rest_ships(tmp_path: Path) -> None:
    """
    Главное свойство: проект выходит в свет. Автор, который месяц полирует
    то, чего никто не видел, ничем не отличается от автора, который ничего не
    написал.
    """
    standalone_main = (
        "import argparse\n\n\n"
        "def main() -> None:\n"
        "    argparse.ArgumentParser().parse_args()\n\n\n"
        "if __name__ == '__main__':\n"
        "    main()\n"
    )
    files = [
        GeneratedFile(path="src/extra.py", content="import nonexistent_library_xyz\n"),
        GeneratedFile(path="src/main.py", content=standalone_main),
    ]
    fixer = _ScriptedFixer([])

    outcome = await _verifier(fixer, tmp_path, max_rounds=1).verify(_SPEC, files)

    assert outcome.green is True, outcome.failure
    assert outcome.dropped == ["src/extra.py"]
    assert [item.path for item in outcome.files] == ["src/main.py"]
    assert any("выложила без" in note for note in outcome.notes), "об урезанном проекте говорят вслух"


async def test_a_module_someone_imports_is_not_dropped(tmp_path: Path) -> None:
    """
    Выкинуть модуль, на который ссылается точка входа, — значит поменять одно
    падение на другое, и выложить то, что не запускается.
    """
    files = _files(_WORKING_MAIN, parser="import nonexistent_library_xyz\n\n\ndef parse_line(x):\n    return {}\n")
    fixer = _ScriptedFixer([])

    outcome = await _verifier(fixer, tmp_path, max_rounds=1).verify(_SPEC, files)

    assert outcome.green is False
    assert outcome.dropped == []
    assert outcome.is_publishable is False
    assert outcome.failure, "причина названа, а не проглочена"


async def test_a_project_without_an_entry_point_is_only_checked_statically(tmp_path: Path) -> None:
    """Запускать нечего — и это не повод считать проект сломанным."""
    files = [GeneratedFile(path="src/parser.py", content=_WORKING_PARSER)]
    fixer = _ScriptedFixer([])

    outcome = await _verifier(fixer, tmp_path).verify(_SPEC, files)

    assert outcome.green is True
    assert outcome.is_publishable is True


async def test_a_workspace_that_cannot_be_made_does_not_kill_the_project(tmp_path: Path) -> None:
    """
    Проверить не вышло — но статические проверки проект уже прошёл, и терять
    его из-за недоступного /tmp было бы худшим из решений.
    """
    blocked = tmp_path / "blocked"
    blocked.write_text("это файл, а не каталог", encoding="utf-8")
    verifier = ProjectVerifier(WorkspaceManager(blocked / "inside"), _ScriptedFixer([]))

    outcome = await verifier.verify(_SPEC, _files(_WORKING_MAIN))

    assert outcome.green is True
    assert any("запустить проект не получилось" in note for note in outcome.notes)
