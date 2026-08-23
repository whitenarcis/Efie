"""
Тесты того, ЧТО именно Эфи гуглит и зачем.

До этого фоновое исследование брало случайный интерес из worldview.json и
складывало в дневник факты — про эмуляцию, про новостные агрегаторы, про
алкогольные облака в космосе. Дашборд честно показывал по каждой такой
записи «использована 0 раз». Проверяется здесь ровно обратное: у запроса
должен быть адресат — падение, которое надо починить, или библиотека, с
которой она прямо сейчас работает.

И вторая половина той же мысли: найденное должно применяться. Ошибка,
пережившая первую правку, уходит в поиск, а найденное — обратно модели в
тот же цикл починки.
"""

from __future__ import annotations

from pathlib import Path

from efi.db.core import Database
from efi.db.models import MIGRATIONS
from efi.dev.auto_fix import RepairLoop
from efi.dev.research_topics import DevResearchTopics, normalize_error_query
from efi.dev.schemas import DevTaskStatus, ProjectSpec
from efi.dev.store import DevTaskStore
from efi.dev.workspace import Workspace

_SPEC = ProjectSpec.model_validate(
    {
        "slug": "lossless-grab",
        "title": "Lossless Grab",
        "problem": "Ищет и скачивает lossless-релизы по названию альбома",
        "stack": ["python 3.11", "mutagen"],
        "files": [{"path": "src/main.py", "purpose": "точка входа"}],
    }
)


def _store(tmp_path: Path) -> DevTaskStore:
    return DevTaskStore(Database(tmp_path / "efi.db", migrations=MIGRATIONS))


# -- откуда берутся вопросы ---------------------------------------------------


async def test_without_work_there_is_nothing_to_look_up(tmp_path: Path) -> None:
    """Нет задач — нет и технических вопросов; тогда работает обычное любопытство."""
    assert await DevResearchTopics(_store(tmp_path)).next_question() is None


async def test_a_real_failure_is_the_best_query_there_is(tmp_path: Path) -> None:
    """У чужого человека была ровно та же строчка, и ответ уже написан."""
    store = _store(tmp_path)
    task = await store.create("грабер", chat_id=1)
    await store.update(
        task,
        spec=_SPEC,
        status=DevTaskStatus.FAILED,
        error=(
            'File "/tmp/workspaces/project-x/src/main.py", line 42, in main\n'
            "    tags = mutagen.File(path).tags\n"
            "AttributeError: module 'mutagen' has no attribute 'File'"
        ),
    )

    question = await DevResearchTopics(store).next_question()

    assert question is not None
    assert question.task_id == task.id
    assert question.is_about_failure is True
    assert "AttributeError" in question.query
    assert "/tmp/workspaces" not in question.query, "путь на этой машине уникален и ничего не найдёт"
    assert "line 42" not in question.query


async def test_what_she_is_writing_now_is_worth_a_question_too(tmp_path: Path) -> None:
    """Так узнают, каким API пользоваться, ДО того как выдумать несуществующий."""
    store = _store(tmp_path)
    task = await store.create("грабер", chat_id=1)
    await store.update(task, spec=_SPEC, status=DevTaskStatus.CODING)

    question = await DevResearchTopics(store).next_question()

    assert question is not None
    assert "mutagen" in question.query
    assert question.task_id == task.id
    assert question.is_about_failure is False


def test_an_error_line_is_cleaned_up_not_pasted_whole() -> None:
    assert normalize_error_query("") == ""
    assert normalize_error_query("что-то пошло не так") == ""  # слишком коротко и без имени ошибки

    cleaned = normalize_error_query(
        'Traceback (most recent call last):\n  File "/home/u/x.py", line 7, in <module>\n'
        "ModuleNotFoundError: No module named 'requests'"
    )
    assert cleaned == "ModuleNotFoundError: No module named 'requests'"


# -- и главное: найденное применяется -----------------------------------------


async def test_an_error_that_survived_one_fix_is_looked_up(tmp_path: Path) -> None:
    """
    Первый круг модель почти всегда чинит сама, и лезть в сеть на каждую
    опечатку — трата времени. А вот когда ошибка пережила правку, знания
    модели по ней кончились: дальше помогает чужой ответ на ту же строчку.
    """
    root = tmp_path / "w"
    root.mkdir()
    (root / "mod.py").write_text("import nonexistent_library_xyz\n", encoding="utf-8")
    workspace = Workspace(root, session_id="w", temporary=False)

    searched: list[str] = []
    seen_requests: list[str] = []

    async def lookup(query: str) -> str:
        searched.append(query)
        return "Пакета nonexistent_library_xyz не существует; в стандартной библиотеке это os."

    async def fixer(system_prompt: str, request: str) -> str | None:
        seen_requests.append(request)
        if len(seen_requests) == 1:
            # Первая попытка — мимо: ошибка остаётся той же.
            return (
                "mod.py\n<<<<<<< SEARCH\nimport nonexistent_library_xyz\n=======\n"
                "import nonexistent_library_abc\n>>>>>>> REPLACE"
            )
        return (
            "mod.py\n<<<<<<< SEARCH\nimport nonexistent_library_abc\n=======\n"
            "import os\n\nprint(os.name)\n>>>>>>> REPLACE"
        )

    report = await RepairLoop(fixer, max_rounds=3, lookup=lookup).run(workspace, ["mod.py"])

    assert report.green is True
    assert searched, "на втором круге она идёт искать, а не ходит по кругу"
    assert "ModuleNotFoundError" in searched[0]
    assert "Что нашлось по этой ошибке" not in seen_requests[0], "на первом круге в сеть не ходим"
    assert "Что нашлось по этой ошибке" in seen_requests[1], "найденное уходит модели вместе с ошибкой"
    assert "в стандартной библиотеке это os" in seen_requests[1]


async def test_the_first_round_does_not_go_online(tmp_path: Path) -> None:
    root = tmp_path / "w"
    root.mkdir()
    (root / "mod.py").write_text("import nonexistent_library_xyz\n", encoding="utf-8")
    workspace = Workspace(root, session_id="w", temporary=False)
    searched: list[str] = []

    async def lookup(query: str) -> str:
        searched.append(query)
        return "неважно"

    async def fixer(system_prompt: str, request: str) -> str | None:
        return (
            "mod.py\n<<<<<<< SEARCH\nimport nonexistent_library_xyz\n=======\n"
            "import os\n\nprint(os.name)\n>>>>>>> REPLACE"
        )

    await RepairLoop(fixer, max_rounds=3, lookup=lookup).run(workspace, ["mod.py"])

    assert searched == [], "починилось с первого раза — искать было незачем"
