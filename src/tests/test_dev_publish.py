"""
Тесты публикации: нарезка коммитов, локальный репозиторий, GitHub API.

Локальная часть гоняется на НАСТОЯЩЕМ git — в этом и смысл: проверять
обёртку над git подставным git'ом значит проверять собственную выдумку о
том, как он себя ведёт. Сетевая часть, наоборот, подменяется транспортом
httpx: единственный поход в GitHub на проект выполняется молча и в бою, и
разбор его ответов (включая «репозиторий уже существует») обязан быть
проверен здесь.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import httpx
import pytest

from efi.dev.github_sync import GitHubSync, GitHubSyncError, plan_commits
from efi.dev.schemas import GeneratedFile, ProjectSpec

_SPEC = ProjectSpec.model_validate(
    {
        "slug": "log-digest",
        "title": "Log Digest",
        "problem": "Разбирает логи nginx и показывает топ ошибок за период",
        "stack": ["python 3.11"],
        "files": [
            {"path": "src/parser.py", "purpose": "разбор строк"},
            {"path": "src/main.py", "purpose": "точка входа"},
        ],
    }
)

_FILES = [
    GeneratedFile(path="src/parser.py", content="def parse() -> None:\n    pass\n"),
    GeneratedFile(path="src/main.py", content="def main() -> None:\n    pass\n"),
    GeneratedFile(path="tests/test_parser.py", content="def test_parse() -> None:\n    pass\n"),
    GeneratedFile(path="README.md", content="# Log Digest\n"),
]

_REPO_PAYLOAD = {
    "full_name": "efi/log-digest",
    "html_url": "https://github.com/efi/log-digest",
    "ssh_url": "git@github.com:efi/log-digest.git",
    "clone_url": "https://github.com/efi/log-digest.git",
}

requires_git = pytest.mark.skipif(shutil.which("git") is None, reason="в системе нет git")


# -- нарезка коммитов ---------------------------------------------------------


def test_commits_are_sliced_by_meaning() -> None:
    """
    Один коммит «initial commit» с двадцатью файлами — признак
    сгенерированного репозитория. Порядок здесь повторяет порядок настоящей
    работы: каркас, точка входа, модули, тесты.
    """
    slices = plan_commits(_FILES, _SPEC)

    assert [item.message for item in slices] == [
        "chore: scaffolding for log-digest",
        "feat: entrypoint for log-digest",
        "feat: implement core modules",
        "test: cover the core behaviour",
    ]
    assert slices[0].paths == ["README.md"]
    assert slices[1].paths == ["src/main.py"]
    assert slices[2].paths == ["src/parser.py"]
    assert slices[3].paths == ["tests/test_parser.py"]


def test_empty_slices_are_dropped() -> None:
    """Проект из двух файлов даёт два коммита, а не четыре, из которых два пустых."""
    slices = plan_commits([_FILES[0], _FILES[3]], _SPEC)

    assert [item.message for item in slices] == [
        "chore: scaffolding for log-digest",
        "feat: implement core modules",
    ]


# -- локальный репозиторий ----------------------------------------------------


def _git_log(repo: Path) -> list[str]:
    result = subprocess.run(  # noqa: S603 — фиксированная команда в тестовом каталоге
        ["git", "log", "--format=%s"],  # noqa: S607
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip().splitlines()


@requires_git
async def test_local_publish_creates_a_real_repository(tmp_path: Path) -> None:
    """
    Без токена работа не пропадает: репозиторий есть, коммиты нарезаны,
    ссылки нет — и это честный рабочий режим, а не заглушка.
    """
    sync = GitHubSync(tmp_path)

    result = await sync.publish(_SPEC, _FILES)

    assert result.pushed is False
    assert result.url == ""
    assert (result.local_path / "src" / "main.py").exists()
    assert (result.local_path / ".git").is_dir()
    # git log отдаёт новые сверху — сравниваем с обратным порядком нарезки.
    assert _git_log(result.local_path) == [
        "test: cover the core behaviour",
        "feat: implement core modules",
        "feat: entrypoint for log-digest",
        "chore: scaffolding for log-digest",
    ]


@requires_git
async def test_second_run_starts_from_a_clean_directory(tmp_path: Path) -> None:
    """Полуфабрикат прошлой попытки не должен смешиваться с новой генерацией."""
    sync = GitHubSync(tmp_path)
    first = await sync.publish(_SPEC, _FILES)
    (first.local_path / "src" / "leftover.py").write_text("x = 1", encoding="utf-8")

    second = await sync.publish(_SPEC, _FILES)

    assert not (second.local_path / "src" / "leftover.py").exists()


async def test_project_directory_cannot_escape_the_workspace(tmp_path: Path) -> None:
    """
    Slug приходит из ответа языковой модели. Схема его уже нормализует, но
    проверка повторяется там, где создаётся каталог, — цена ошибки тут не
    «некрасивое имя», а запись в чужую папку.
    """
    sync = GitHubSync(tmp_path / "workspace")
    spec = _SPEC.model_copy(update={"slug": "../escaped"})

    with pytest.raises(GitHubSyncError, match="за пределы"):
        await sync.publish(spec, _FILES)


# -- GitHub API ---------------------------------------------------------------


def _sync_with(tmp_path: Path, handler: object) -> GitHubSync:
    return GitHubSync(
        tmp_path,
        token="ghp_test",
        transport=httpx.MockTransport(handler),  # type: ignore[arg-type]
    )


async def test_repository_is_created_with_the_project_description(tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["auth"] = request.headers.get("authorization")
        seen["body"] = request.content.decode()
        return httpx.Response(201, json=_REPO_PAYLOAD)

    repo = await _sync_with(tmp_path, handler).create_repository(_SPEC)

    assert repo.ssh_url == "git@github.com:efi/log-digest.git"
    assert seen["url"] == "https://api.github.com/user/repos"
    assert seen["auth"] == "Bearer ghp_test"
    assert "log-digest" in str(seen["body"])


async def test_existing_repository_is_reused_not_failed(tmp_path: Path) -> None:
    """
    422 «имя занято» — обычное дело: проект могли начать в прошлый заход и не
    довести до пуша. Это повод забрать существующий репозиторий, а не бросать
    работу.
    """
    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "POST":
            return httpx.Response(422, json={"message": "name already exists on this account"})
        if request.url.path == "/repos/efi/log-digest":
            return httpx.Response(200, json=_REPO_PAYLOAD)
        return httpx.Response(200, json={"login": "efi"})

    sync = GitHubSync(tmp_path, token="ghp_test", owner="efi", transport=httpx.MockTransport(handler))

    assert (await sync.create_repository(_SPEC)).html_url == "https://github.com/efi/log-digest"


async def test_api_refusal_is_reported_as_a_sync_error(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(403, text="token has no repo scope")

    with pytest.raises(GitHubSyncError, match="403"):
        await _sync_with(tmp_path, handler).create_repository(_SPEC)


async def test_network_failure_is_reported_as_a_sync_error(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("нет сети")

    with pytest.raises(GitHubSyncError, match="недоступен"):
        await _sync_with(tmp_path, handler).create_repository(_SPEC)


def test_publishing_requires_both_a_token_and_permission(tmp_path: Path) -> None:
    assert GitHubSync(tmp_path).can_publish is False
    assert GitHubSync(tmp_path, token="ghp_test").can_publish is True
    assert GitHubSync(tmp_path, token="ghp_test", push_enabled=False).can_publish is False
