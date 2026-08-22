"""
Тесты формата правок и карты репозитория — двух вещей, на которых держится
работа с чужим кодом.

Формат правок проверяется придирчиво, потому что это единственная защита от
самого дорогого способа испортить чужой проект: правка, применённая «примерно
туда». Здесь она обязана быть либо применена дословно, либо отвергнута.

Карта — про другое: она отвечает на вопрос «куда смотреть». Проверяется, что
в неё попадают объявления (а не тела), что она умещается в бюджет и что
непарсящийся файл честно помечается, а не выпадает молча.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from efi.dev.edits import EditError, apply_edit, apply_edits, parse_edits
from efi.dev.repo_map import build_repo_map
from efi.dev.sandbox import salvage_python

_ANSWER = """Тут всё просто: функция называется иначе.

src/parser.py
<<<<<<< SEARCH
def parse(line: str) -> dict:
=======
def parse_line(line: str) -> dict:
>>>>>>> REPLACE

И в точке входа тоже:

src/main.py
<<<<<<< SEARCH
from parser import parse
=======
from parser import parse_line
>>>>>>> REPLACE
"""


# -- разбор -------------------------------------------------------------------


def test_edits_are_taken_out_of_ordinary_prose() -> None:
    """Модель почти всегда объясняет замысел словами. Это не повод терять патч."""
    edits = parse_edits(_ANSWER)

    assert [item.path for item in edits] == ["src/parser.py", "src/main.py"]
    assert edits[0].search == "def parse(line: str) -> dict:\n"
    assert edits[0].replace == "def parse_line(line: str) -> dict:\n"


def test_path_survives_markdown_decoration() -> None:
    raw = "**Файл:** `src/core/handlers.py`\n<<<<<<< SEARCH\nx = 1\n=======\nx = 2\n>>>>>>> REPLACE"

    edits = parse_edits(raw)

    assert [item.path for item in edits] == ["src/core/handlers.py"]


def test_an_empty_search_block_means_a_new_file(tmp_path: Path) -> None:
    edits = parse_edits("src/new.py\n<<<<<<< SEARCH\n=======\nx = 1\n>>>>>>> REPLACE")

    assert edits[0].creates_file is True
    apply_edit(tmp_path, edits[0])
    assert (tmp_path / "src" / "new.py").read_text(encoding="utf-8") == "x = 1\n"


# -- применение ---------------------------------------------------------------


def _project(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "parser.py").write_text(
        "def parse(line: str) -> dict:\n    return {}\n", encoding="utf-8"
    )
    return tmp_path


def test_a_matching_edit_is_applied_verbatim(tmp_path: Path) -> None:
    root = _project(tmp_path)
    edits = parse_edits(_ANSWER)[:1]

    changed, problems = apply_edits(root, edits)

    assert changed == ["src/parser.py"]
    assert problems == []
    assert "def parse_line" in (root / "src" / "parser.py").read_text(encoding="utf-8")


def test_an_invented_fragment_is_refused(tmp_path: Path) -> None:
    """Придуманный контекст — это правка не туда. Лучше не применить вовсе."""
    root = _project(tmp_path)
    edits = parse_edits(
        "src/parser.py\n<<<<<<< SEARCH\ndef nothing_like_this() -> None:\n=======\nx = 1\n>>>>>>> REPLACE"
    )

    changed, problems = apply_edits(root, edits)

    assert changed == []
    assert "нет искомого фрагмента" in problems[0]


def test_an_ambiguous_fragment_is_refused(tmp_path: Path) -> None:
    """Две одинаковые строки — непонятно, какую править, а угадывать здесь нельзя."""
    root = tmp_path
    (root / "mod.py").write_text("x = 1\ny = 2\nx = 1\n", encoding="utf-8")
    edits = parse_edits("mod.py\n<<<<<<< SEARCH\nx = 1\n=======\nx = 3\n>>>>>>> REPLACE")

    changed, problems = apply_edits(root, edits)

    assert changed == []
    assert "2 раза" in problems[0]


def test_a_path_cannot_escape_the_workspace(tmp_path: Path) -> None:
    """Правка приезжает от модели: `../../.ssh/authorized_keys` не должен быть выразим."""
    edits = parse_edits("../../.ssh/authorized_keys\n<<<<<<< SEARCH\n=======\nключ\n>>>>>>> REPLACE")

    if edits:  # путь с пробелами и «..» отбрасывается ещё разбором — это тоже верный исход
        with pytest.raises(EditError, match="выходит за пределы"):
            apply_edit(tmp_path, edits[0])
    assert not (tmp_path.parent / ".ssh").exists()


def test_one_bad_edit_does_not_cancel_the_good_ones(tmp_path: Path) -> None:
    """Обычно из трёх блоков хорошие два — и терять их из-за третьего незачем."""
    root = _project(tmp_path)
    edits = parse_edits(
        "src/parser.py\n<<<<<<< SEARCH\ndef parse(line: str) -> dict:\n=======\n"
        "def parse_line(line: str) -> dict:\n>>>>>>> REPLACE\n\n"
        "src/missing.py\n<<<<<<< SEARCH\nчего-то нет\n=======\nбудет\n>>>>>>> REPLACE"
    )

    changed, problems = apply_edits(root, edits)

    assert changed == ["src/parser.py"]
    assert len(problems) == 1


def test_a_shifted_indent_is_forgiven(tmp_path: Path) -> None:
    """
    Модель регулярно сдвигает блок, копируя его из своего ответа. Это
    единственная поблажка: всё остальное («примерно похоже») не прощается.
    """
    root = tmp_path
    (root / "mod.py").write_text(
        "class A:\n    def run(self) -> int:\n        return 1\n", encoding="utf-8"
    )
    edits = parse_edits(
        "mod.py\n<<<<<<< SEARCH\ndef run(self) -> int:\n    return 1\n=======\n"
        "def run(self) -> int:\n    return 2\n>>>>>>> REPLACE"
    )

    changed, problems = apply_edits(root, edits)

    assert changed == ["mod.py"], problems
    assert (root / "mod.py").read_text(encoding="utf-8") == (
        "class A:\n    def run(self) -> int:\n        return 2\n"
    )


# -- карта репозитория --------------------------------------------------------


def _repo(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "core.py").write_text(
        "import os\n\n\n"
        "LIMIT = 10\n\n\n"
        "class Engine:\n"
        "    def run(self, path: str) -> int:\n        return 1\n\n"
        "    def _private(self) -> None:\n        ...\n\n\n"
        "def helper(value: int) -> str:\n    return str(value)\n",
        encoding="utf-8",
    )
    (tmp_path / "src" / "broken.py").write_text("def oops(:\n", encoding="utf-8")
    (tmp_path / "README.md").write_text("# Проект\n", encoding="utf-8")
    venv = tmp_path / ".venv" / "lib"
    venv.mkdir(parents=True)
    (venv / "junk.py").write_text("x = 1\n", encoding="utf-8")
    return tmp_path


def test_the_map_shows_declarations_not_bodies(tmp_path: Path) -> None:
    rendered = build_repo_map(_repo(tmp_path)).render()

    assert "class Engine:" in rendered
    assert "def run(self, path: str) -> int" in rendered
    assert "LIMIT = ..." in rendered
    assert "return 1" not in rendered, "тела в карту не попадают — иначе это не карта, а исходники"
    assert "_private" not in rendered, "приватное соседям не интересно"


def test_environments_and_caches_are_not_part_of_the_project(tmp_path: Path) -> None:
    assert "junk.py" not in build_repo_map(_repo(tmp_path)).render()


def test_a_file_that_does_not_parse_is_named_not_hidden(tmp_path: Path) -> None:
    """Часто именно этот файл и ищут — молчать про него хуже всего."""
    rendered = build_repo_map(_repo(tmp_path)).render()

    assert "broken.py" in rendered
    assert "не парсится" in rendered


def test_the_map_fits_the_budget_and_says_what_it_dropped(tmp_path: Path) -> None:
    root = tmp_path
    for index in range(60):
        (root / f"mod_{index:02d}.py").write_text(
            "\n".join(f"def function_{index}_{n}(argument: str) -> None:\n    ..." for n in range(20)),
            encoding="utf-8",
        )

    repo_map = build_repo_map(root, budget_bytes=4000)

    assert len(repo_map.render().encode("utf-8")) <= 4200
    assert repo_map.omitted > 0
    assert "не поместившихся" in repo_map.render()


# -- спасение оборванного файла ----------------------------------------------


def test_a_truncated_tail_is_cut_at_the_last_whole_definition() -> None:
    """
    Функция, у которой уцелели три строки тела, синтаксически безупречна и
    молча возвращает None — это хуже отсутствующего файла, потому что
    выглядит рабочей.
    """
    source = (
        "def first(value: int) -> int:\n    return value + 1\n\n\n"
        "def second(value: int) -> int:\n    return value + 2\n\n\n"
        "def third(rows: list[int]) -> int:\n    total = 0\n    for row in rows:\n        total += row(\n"
    )

    salvaged = salvage_python(source)

    assert "def first" in salvaged
    assert "def second" in salvaged
    assert "third" not in salvaged


def test_nothing_is_salvaged_from_a_stub() -> None:
    assert salvage_python("def only_one(\n") == ""
