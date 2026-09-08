"""
Тесты работы с чужим репозиторием целиком: от просьбы до ветки.

Здесь настоящие git и настоящие подпроцессы — как в тестах публикации
проектов (test_dev_publish.py), и по той же причине: изоляция рабочей копии,
права на файлы и поведение git проверяются только git'ом. Подменена ровно
одна вещь — модель: её ответы заданы сценарием, потому что проверяется наш код
вокруг них, а не то, что ответит Sonnet.

Главное, что здесь утверждается: ветка появляется ТОЛЬКО когда проверки
зелёные. Ветка с падающими тестами — не помощь, а работа для того, кто её
откроет.
"""

from __future__ import annotations

import asyncio
import shutil
from pathlib import Path

import pytest

from efi.dev.auto_fix import RepairLoop
from efi.dev.swe_engine import SweEngine, SweRequest
from efi.dev.workspace import Workspace, WorkspaceManager
from efi.llm.network_router import NetworkModelRouter
from efi.llm.resilience import ConcurrencyGate
from efi.llm.schemas import Choice, LLMParams, Message, Response, Role, Session

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="в системе нет git")


async def _git(cwd: Path, *args: str) -> None:
    process = await asyncio.create_subprocess_exec(
        "git", *args, cwd=str(cwd), stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await process.communicate()
    assert process.returncode == 0, stderr.decode("utf-8", "replace")


async def _repo(tmp_path: Path) -> Path:
    """Маленький, но настоящий репозиторий с настоящей поломкой: main зовёт то, чего нет."""
    root = tmp_path / "upstream"
    (root / "src").mkdir(parents=True)
    (root / "src" / "parser.py").write_text(
        '"""Разбор строк лога."""\n\n\ndef parse(line: str) -> dict:\n    return {"raw": line}\n',
        encoding="utf-8",
    )
    (root / "src" / "main.py").write_text(
        "from parser import parse_line\n\n\ndef main() -> None:\n    print(parse_line('x'))\n",
        encoding="utf-8",
    )
    (root / "README.md").write_text("# upstream\n", encoding="utf-8")

    await _git(root, "init", "-b", "main")
    await _git(root, "add", "-A")
    await _git(root, "-c", "user.name=T", "-c", "user.email=t@t", "commit", "-m", "initial")
    return root


class _ScriptedModel:
    """Модель, отвечающая по сценарию, и помнящая, о чём её спрашивали."""

    def __init__(self, answers: list[str]) -> None:
        self._answers = list(answers)
        self.prompts: list[str] = []

    async def __call__(self, params: LLMParams, session: Session) -> Response:
        self.prompts.append(session.messages[-1].content)
        text = self._answers.pop(0) if self._answers else ""
        return Response(choices=[Choice(message=Message(role=Role.ASSISTANT, content=text))])


def _engine(model: _ScriptedModel, tmp_path: Path, **kwargs: object) -> SweEngine:
    return SweEngine(
        NetworkModelRouter(None, model, fallback_name="кодер"),
        WorkspaceManager(tmp_path / "workspaces"),
        gate=ConcurrencyGate(limit=1),
        keep_workspace=True,
        **kwargs,  # type: ignore[arg-type]
    )


_GOOD_FIX = """Классика: функция называется иначе, чем её зовут.

src/main.py
<<<<<<< SEARCH
from parser import parse_line
=======
from parser import parse
>>>>>>> REPLACE

src/main.py
<<<<<<< SEARCH
    print(parse_line('x'))
=======
    print(parse('x'))
>>>>>>> REPLACE
"""


async def test_a_broken_import_is_found_fixed_and_landed_in_a_branch(tmp_path: Path) -> None:
    """
    Полный проход: карта → выбор файлов → точечная правка → проверки → ветка.
    Именно этого нельзя сделать статикой: ImportError находится запуском.
    """
    source = await _repo(tmp_path)
    model = _ScriptedModel(["ФАЙЛЫ: src/main.py\nТам зовут parse_line, а в парсере parse.", _GOOD_FIX])

    outcome = await _engine(model, tmp_path).work_on(
        SweRequest(source=str(source), instruction="почини импорт, main падает", session_id="t1")
    )

    assert outcome.ok is True, outcome.failure_reason
    assert outcome.branch.startswith("fix/"), "по имени ветки видно, что это починка"
    assert outcome.changed_files == ["src/main.py"]
    assert outcome.commit, "коммит есть, и у него есть хеш"

    # Правка легла в клон, а исходный репозиторий не тронут — это разные деревья.
    workspace = Path(outcome.workspace_path)
    assert "from parser import parse\n" in (workspace / "src" / "main.py").read_text(encoding="utf-8")
    assert "parse_line" in (source / "src" / "main.py").read_text(encoding="utf-8")


async def test_the_map_and_only_the_chosen_files_reach_the_model(tmp_path: Path) -> None:
    """Модель, которой не показали структуру, уверенно правит файл, которого нет."""
    source = await _repo(tmp_path)
    model = _ScriptedModel(["ФАЙЛЫ: src/main.py\nвот тут", _GOOD_FIX])

    await _engine(model, tmp_path).work_on(
        SweRequest(source=str(source), instruction="почини импорт", session_id="t2")
    )

    assert "src/parser.py" in model.prompts[0], "в карте видно всё"
    assert "def parse(line: str) -> dict" in model.prompts[0], "объявления, а не тела"
    assert "src/main.py" in model.prompts[1]
    assert "### src/parser.py" not in model.prompts[1], "открывается только выбранное"


async def test_a_project_that_stays_broken_leaves_no_branch(tmp_path: Path) -> None:
    """
    Ветка с падающими проверками — это не помощь, а работа для того, кто её
    откроет. Лучше честное «не смогла».
    """
    source = await _repo(tmp_path)
    still_broken = (
        "src/main.py\n<<<<<<< SEARCH\nfrom parser import parse_line\n=======\n"
        "from parser import still_missing\n>>>>>>> REPLACE"
    )
    model = _ScriptedModel(["ФАЙЛЫ: src/main.py\nага", still_broken, "", "", "", ""])

    outcome = await _engine(model, tmp_path, max_repair_rounds=1).work_on(
        SweRequest(source=str(source), instruction="почини импорт", session_id="t3")
    )

    assert outcome.ok is False
    assert outcome.branch == ""
    assert "не позеленели" in outcome.failure_reason


async def test_invented_file_paths_do_not_become_work(tmp_path: Path) -> None:
    """Путь, которого нет в карте, — это не выбор файла, а фантазия."""
    source = await _repo(tmp_path)
    model = _ScriptedModel(["ФАЙЛЫ: src/core/handlers.py\nправить там", _GOOD_FIX])

    outcome = await _engine(model, tmp_path).work_on(
        SweRequest(source=str(source), instruction="почини импорт", session_id="t4")
    )

    assert outcome.ok is False
    assert "какие файлы" in outcome.failure_reason


async def test_the_original_repository_is_never_touched(tmp_path: Path) -> None:
    """
    Правка чужого рабочего дерева «на месте» — это потерянные несохранённые
    изменения, и извиняться за это поздно.
    """
    source = await _repo(tmp_path)
    (source / "src" / "uncommitted.py").write_text("# работа человека\n", encoding="utf-8")
    model = _ScriptedModel(["ФАЙЛЫ: src/main.py\nтут", _GOOD_FIX])

    outcome = await _engine(model, tmp_path).work_on(
        SweRequest(source=str(source), instruction="почини импорт", session_id="t5")
    )

    assert outcome.ok is True
    assert (source / "src" / "uncommitted.py").read_text(encoding="utf-8") == "# работа человека\n"


# -- цикл починки отдельно ----------------------------------------------------


async def test_the_traceback_goes_back_to_the_model_verbatim(tmp_path: Path) -> None:
    """
    Трейсбэк — это и есть постановка задачи. Любой его пересказ теряет ту
    строчку, по которой всё чинится.
    """
    root = tmp_path / "w"
    root.mkdir()
    (root / "mod.py").write_text(
        "import nothing_like_this\n\nprint(nothing_like_this.value)\n", encoding="utf-8"
    )
    workspace = Workspace(root, session_id="w", temporary=False)
    seen: list[str] = []

    async def fixer(system_prompt: str, request: str) -> str | None:
        seen.append(request)
        return (
            "mod.py\n<<<<<<< SEARCH\nimport nothing_like_this\n\nprint(nothing_like_this.value)\n"
            "=======\nimport os\n\nprint(os.name)\n>>>>>>> REPLACE"
        )

    report = await RepairLoop(fixer, max_rounds=2).run(workspace, ["mod.py"])

    assert report.green is True
    assert report.rounds == 1
    assert "ModuleNotFoundError" in seen[0], "модель видит настоящую ошибку, а не «что-то упало»"
    assert "mod.py" in seen[0]


async def test_the_loop_gives_up_instead_of_grinding_forever(tmp_path: Path) -> None:
    """Модель, не починившая за отведённые круги, начинает переписывать вокруг — это уже порча."""
    root = tmp_path / "w"
    root.mkdir()
    (root / "mod.py").write_text("import nothing_like_this\n", encoding="utf-8")
    workspace = Workspace(root, session_id="w", temporary=False)
    calls = 0

    async def stubborn(system_prompt: str, request: str) -> str | None:
        nonlocal calls
        calls += 1
        return (
            "mod.py\n<<<<<<< SEARCH\nimport nothing_like_this\n=======\n"
            f"import also_missing_{calls}\n>>>>>>> REPLACE"
        )

    report = await RepairLoop(stubborn, max_rounds=2).run(workspace, ["mod.py"])

    assert report.green is False
    assert calls == 2
    assert report.last_failure is not None


async def test_silly_mistakes_become_a_remark_in_chat(tmp_path: Path) -> None:
    """
    «Забыла функцию в __all__ добавить, ща поправлю» — это то, что человек
    говорит между делом. Повод формулирует цикл, словами его делает Worker.
    """
    root = tmp_path / "w"
    root.mkdir()
    (root / "mod.py").write_text("import nothing_like_this\n", encoding="utf-8")
    workspace = Workspace(root, session_id="w", temporary=False)
    said: list[str] = []

    async def narrator(note: str) -> None:
        said.append(note)

    async def fixer(system_prompt: str, request: str) -> str | None:
        return (
            "mod.py\n<<<<<<< SEARCH\nimport nothing_like_this\n=======\n"
            "import os\n\nprint(os.name)\n>>>>>>> REPLACE"
        )

    await RepairLoop(fixer, max_rounds=2, narrator=narrator).run(workspace, ["mod.py"])

    assert said, "о глупой ошибке она говорит вслух"
    assert "импорт" in said[0].lower()
    assert "ModuleNotFoundError" in said[0], "в реплике настоящая ошибка, а не «что-то пошло не так»"


# -- рабочая копия ------------------------------------------------------------


async def test_a_runaway_process_is_killed_not_waited_for(tmp_path: Path) -> None:
    """Бесконечный цикл в чужом тесте не должен ни повесить фоновый цикл, ни съесть батарею."""
    root = tmp_path / "w"
    root.mkdir()
    workspace = Workspace(root, session_id="w", temporary=False)

    result = await workspace.run("python", "-c", "while True: pass", timeout=1.0)

    assert result.timed_out is True
    assert result.ok is False


async def test_a_missing_command_is_an_answer_not_a_crash(tmp_path: Path) -> None:
    root = tmp_path / "w"
    root.mkdir()
    workspace = Workspace(root, session_id="w", temporary=False)

    result = await workspace.run("this-command-does-not-exist")

    assert result.exit_code == 127
    assert result.ok is False
