"""
efi/dev/github_sync.py

Публикация проекта: репозиторий на GitHub, локальный git и пуш.

Три решения, которые стоит объяснить сразу.

ПОЧЕМУ REST, А НЕ PyGithub. Единственное, что нужно от GitHub API, — создать
репозиторий и узнать его адреса; это один POST. Ради него тянуть зависимость
(PyGithub + её транзитивные) в проект, который ставится на телефон в Termux,
несоразмерно: httpx здесь уже есть и используется всеми провайдерами LLM.

ПОЧЕМУ git ЧЕРЕЗ ПОДПРОЦЕСС, А НЕ БИБЛИОТЕКОЙ. Пуш по SSH — это работа с
ключом, агентом и known_hosts, и системный git умеет это правильно, а
питоновские обёртки повторяют его настройки с переменным успехом. Команды
запускаются списком аргументов (`create_subprocess_exec`), без оболочки:
имя репозитория приходит в конечном счёте от языковой модели, и возможность
подставить `; rm -rf` в строку для shell здесь была бы не теоретической.

ПОЧЕМУ КОММИТОВ НЕСКОЛЬКО. Один коммит «initial commit» с двадцатью файлами
— признак сгенерированного репозитория. История, в которой сначала появился
каркас с README, потом модули по одному, читается как работа: именно это и
просили — «нарезка логичных коммитов». Нарезка идёт по смыслу файла
(каркас / точка входа / модули / тесты), а не по алфавиту.

Без токена подсистема работает в локальном режиме: репозиторий создаётся и
коммитится на диске, пуша нет. Это не заглушка ради тестов, а осмысленный
режим — проект остаётся, ссылки просто нет, и Эфи об этом честно скажет.
"""

from __future__ import annotations

import asyncio
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import httpx

from efi.dev.sandbox import write_project_files
from efi.dev.schemas import GeneratedFile, ProjectSpec

logger = logging.getLogger(__name__)

_GITHUB_API_URL = "https://api.github.com"
_GITHUB_API_VERSION = "2022-11-28"
_API_TIMEOUT_SECONDS = 30.0

#: Сколько ждать одну git-команду. Пуш по мобильной сети бывает медленным,
#: но не бесконечным; зависший `git push` не должен держать фоновый цикл.
_GIT_TIMEOUT_SECONDS = 120.0

#: Ветка по умолчанию. GitHub с 2020-го создаёт репозитории с `main`, и
#: локальная ветка обязана называться так же, иначе пуш создаст вторую.
_DEFAULT_BRANCH = "main"

#: Личность автора коммитов. Не настоящий e-mail: коммиты делает не человек,
#: и подписывать их чужим адресом было бы подлогом. `noreply`-домен GitHub
#: ровно для этого и существует.
_COMMIT_AUTHOR_NAME = "Efi"
_COMMIT_AUTHOR_EMAIL = "efi@users.noreply.github.com"


class GitHubSyncError(RuntimeError):
    """Публикация не удалась. Текст пойдёт в DevTask.error и в лог — не в чат."""


@dataclass(slots=True, frozen=True)
class RepoRef:
    """Созданный (или уже существовавший) репозиторий."""

    full_name: str
    html_url: str
    ssh_url: str
    clone_url: str


@dataclass(slots=True, frozen=True)
class PublishResult:
    """Что получилось из публикации: где лежит локально, куда запушено, сколько коммитов."""

    local_path: Path
    commits: list[str]
    repo: RepoRef | None = None
    pushed: bool = False

    @property
    def url(self) -> str:
        return self.repo.html_url if self.repo is not None else ""


@dataclass(slots=True, frozen=True)
class CommitSlice:
    """Один будущий коммит: сообщение и файлы, которые в него входят."""

    message: str
    paths: list[str]


class GitHubSync:
    """
    Всё, что происходит между «код готов» и «есть ссылка».

    Экземпляр не хранит состояния между проектами: путь к рабочему каталогу
    и настройки фиксируются в конструкторе, всё остальное — аргументы
    `publish`.
    """

    def __init__(
        self,
        workspace: Path,
        *,
        token: str = "",
        owner: str = "",
        ssh_key_path: Path | None = None,
        private: bool = False,
        push_enabled: bool = True,
        git_executable: str = "git",
        api_url: str = _GITHUB_API_URL,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._workspace = workspace
        self._token = token
        self._owner = owner
        self._ssh_key_path = ssh_key_path
        self._private = private
        self._push_enabled = push_enabled
        self._git_executable = git_executable
        self._api_url = api_url.rstrip("/")
        #: Транспорт httpx подменяется только в тестах: ходить в настоящий
        #: GitHub из теста нельзя, а проверять разбор ответов (включая 422
        #: «репозиторий уже есть») надо обязательно — это тот код, который в
        #: бою выполняется один раз на проект и молча.
        self._transport = transport

    @property
    def can_publish(self) -> bool:
        """Есть ли чем публиковать наружу. False — работаем локально, без ссылки."""
        return bool(self._token) and self._push_enabled

    async def publish(self, spec: ProjectSpec, files: list[GeneratedFile]) -> PublishResult:
        """
        Раскладывает проект на диск, нарезает коммиты и (если есть токен)
        создаёт удалённый репозиторий и пушит.

        Локальная часть выполняется всегда и первой: если GitHub недоступен,
        работа не должна пропадать — репозиторий остаётся на диске, и пуш
        можно повторить следующим заходом.
        """
        project_dir = self._prepare_directory(spec)
        _write_files(project_dir, files)

        await self._git(project_dir, "init", "-b", _DEFAULT_BRANCH)
        commits = await self._commit_in_slices(project_dir, files, spec)
        if not commits:
            raise GitHubSyncError("нечего коммитить: git не увидел ни одного файла")

        if not self.can_publish:
            logger.info(
                "github_sync: проект %s собран локально (%s), пуша не будет — нет токена или push выключен",
                spec.slug, project_dir,
            )
            return PublishResult(local_path=project_dir, commits=commits)

        repo = await self.create_repository(spec)
        await self._git(project_dir, "remote", "add", "origin", repo.ssh_url)
        await self._git(project_dir, "push", "-u", "origin", _DEFAULT_BRANCH)
        logger.info("github_sync: %s запушен в %s", spec.slug, repo.html_url)
        return PublishResult(local_path=project_dir, commits=commits, repo=repo, pushed=True)

    async def commit_revision(
        self, spec: ProjectSpec, files: list[GeneratedFile], *, message: str
    ) -> bool:
        """
        Правка в уже существующем проекте: перезаписать файлы, закоммитить
        одним коммитом и запушить в тот же репозиторий.

        Возвращает False, если коммитить было нечего (модель «исправила»
        файл в то же самое содержимое — обычное дело). Пустой коммит здесь
        хуже отсутствия правки: история проекта должна показывать работу, а
        не активность.

        Каталог проекта должен существовать — это клон, оставшийся от
        публикации. Если его нет (почистили диск, переехали), правка
        пропускается: перевыкладывать проект заново под видом «внёс правку»
        нельзя, это переписывание истории.
        """
        project_dir = (self._workspace / spec.slug).resolve()
        if not (project_dir / ".git").is_dir():
            raise GitHubSyncError(f"локального клона {spec.slug} нет — править нечего")

        _write_files(project_dir, files)
        paths = [item.path for item in files]
        await self._git(project_dir, "add", "--", *paths)
        if not (await self._git(project_dir, "status", "--porcelain", "--", *paths)).strip():
            logger.info("github_sync: правка в %s ничего не изменила, коммита не будет", spec.slug)
            return False

        await self._git(
            project_dir,
            "-c", f"user.name={_COMMIT_AUTHOR_NAME}",
            "-c", f"user.email={_COMMIT_AUTHOR_EMAIL}",
            "commit", "-m", message,
        )
        if self.can_publish:
            await self._git(project_dir, "push", "origin", _DEFAULT_BRANCH)
        logger.info("github_sync: %s — %s", spec.slug, message)
        return True

    async def create_repository(self, spec: ProjectSpec) -> RepoRef:
        """
        Создаёт репозиторий через REST API.

        Уже существующий репозиторий с тем же именем (422 от API) — не сбой:
        Эфи могла начать этот проект в прошлый раз и не довести до пуша.
        В этом случае просто забираем его адреса.
        """
        payload = {
            "name": spec.slug,
            "description": _repo_description(spec),
            "private": self._private,
            "auto_init": False,
            "has_issues": True,
            "has_wiki": False,
        }
        async with httpx.AsyncClient(
            timeout=_API_TIMEOUT_SECONDS, headers=self._api_headers(), transport=self._transport
        ) as client:
            try:
                response = await client.post(f"{self._api_url}/user/repos", json=payload)
            except httpx.HTTPError as exc:
                raise GitHubSyncError(f"GitHub API недоступен: {exc}") from exc

            if response.status_code == httpx.codes.UNPROCESSABLE_ENTITY:
                logger.info("github_sync: репозиторий %s уже существует, беру его", spec.slug)
                return await self._fetch_repository(client, spec.slug)
            if response.status_code >= httpx.codes.BAD_REQUEST:
                raise GitHubSyncError(
                    f"GitHub отказал в создании репозитория ({response.status_code}): "
                    f"{response.text.strip()[:200]}"
                )
            return _parse_repo(response.json())

    async def _fetch_repository(self, client: httpx.AsyncClient, slug: str) -> RepoRef:
        owner = self._owner or await self._resolve_owner(client)
        response = await client.get(f"{self._api_url}/repos/{owner}/{slug}")
        if response.status_code >= httpx.codes.BAD_REQUEST:
            raise GitHubSyncError(
                f"репозиторий {owner}/{slug} не создать и не найти ({response.status_code})"
            )
        return _parse_repo(response.json())

    async def _resolve_owner(self, client: httpx.AsyncClient) -> str:
        response = await client.get(f"{self._api_url}/user")
        if response.status_code >= httpx.codes.BAD_REQUEST:
            raise GitHubSyncError(f"не удалось узнать владельца токена ({response.status_code})")
        login = str(response.json().get("login", "")).strip()
        if not login:
            raise GitHubSyncError("GitHub не вернул login владельца токена")
        return login

    def _api_headers(self) -> dict[str, str]:
        return {
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": _GITHUB_API_VERSION,
            "Authorization": f"Bearer {self._token}",
        }

    def _prepare_directory(self, spec: ProjectSpec) -> Path:
        """
        Чистый каталог под проект. Если он остался с прошлой попытки — сносим:
        доводить чужой полуфабрикат до состояния «как будто так и было»
        дороже и опаснее, чем начать заново, а исходники всё равно
        перегенерированы.
        """
        project_dir = (self._workspace / spec.slug).resolve()
        if not project_dir.is_relative_to(self._workspace.resolve()):
            raise GitHubSyncError(f"имя проекта {spec.slug!r} уводит за пределы рабочего каталога")
        if project_dir.exists():
            shutil.rmtree(project_dir)
        project_dir.mkdir(parents=True)
        return project_dir

    async def _commit_in_slices(
        self, project_dir: Path, files: list[GeneratedFile], spec: ProjectSpec
    ) -> list[str]:
        messages: list[str] = []
        for commit in plan_commits(files, spec):
            await self._git(project_dir, "add", "--", *commit.paths)
            status = await self._git(project_dir, "status", "--porcelain", "--", *commit.paths)
            if not status.strip():
                continue  # нечего коммитить в этом срезе — файлы уже вошли в предыдущий
            await self._git(
                project_dir,
                "-c", f"user.name={_COMMIT_AUTHOR_NAME}",
                "-c", f"user.email={_COMMIT_AUTHOR_EMAIL}",
                "commit", "-m", commit.message,
            )
            messages.append(commit.message)
        return messages

    async def _git(self, cwd: Path, *args: str) -> str:
        """
        Одна git-команда. Только exec, никогда не через оболочку — см.
        докстринг модуля про то, откуда в аргументах берутся внешние данные.
        """
        process = await asyncio.create_subprocess_exec(
            self._git_executable,
            *args,
            cwd=str(cwd),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=self._git_env(),
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=_GIT_TIMEOUT_SECONDS)
        except TimeoutError as exc:
            process.kill()
            await process.wait()
            raise GitHubSyncError(f"git {args[0]} не уложился в {_GIT_TIMEOUT_SECONDS:.0f}с") from exc

        if process.returncode != 0:
            raise GitHubSyncError(
                f"git {' '.join(args[:2])} завершился с кодом {process.returncode}: "
                f"{stderr.decode('utf-8', 'replace').strip()[:300]}"
            )
        return stdout.decode("utf-8", "replace")

    def _git_env(self) -> dict[str, str]:
        """
        Окружение git-подпроцесса.

        `GIT_SSH_COMMAND` с явным ключом — чтобы пуш шёл ИМЕННО тем ключом,
        который выдали Эфи, а не первым попавшимся из ~/.ssh или из агента
        владельца. `GIT_TERMINAL_PROMPT=0` — чтобы git при отсутствии прав
        падал с ошибкой, а не вставал молча в ожидании пароля, которого в
        фоновом процессе никто не введёт.
        """
        env = dict(os.environ)
        env["GIT_TERMINAL_PROMPT"] = "0"
        if self._ssh_key_path is not None:
            env["GIT_SSH_COMMAND"] = (
                f"ssh -i {self._ssh_key_path} -o IdentitiesOnly=yes -o BatchMode=yes"
            )
        return env


def plan_commits(files: list[GeneratedFile], spec: ProjectSpec) -> list[CommitSlice]:
    """
    Нарезка коммитов по смыслу файлов. Чистая функция — проверяется тестами
    без git.

    Порядок повторяет порядок настоящей работы: сначала каркас (README и
    конфиги), потом точка входа, потом модули, потом тесты. Пустые срезы
    отбрасываются, поэтому проект из двух файлов даст два коммита, а не
    четыре пустых.
    """
    scaffolding: list[str] = []
    entrypoints: list[str] = []
    modules: list[str] = []
    tests: list[str] = []

    for item in files:
        name = item.path.lower()
        if name.startswith("test") or "/test" in name:
            tests.append(item.path)
        elif not name.endswith(".py"):
            scaffolding.append(item.path)
        elif _is_entrypoint(name):
            entrypoints.append(item.path)
        else:
            modules.append(item.path)

    slices = [
        CommitSlice(f"chore: scaffolding for {spec.slug}", scaffolding),
        CommitSlice(f"feat: {_core_subject(spec)}", entrypoints),
        CommitSlice("feat: implement core modules", modules),
        CommitSlice("test: cover the core behaviour", tests),
    ]
    return [item for item in slices if item.paths]


def _is_entrypoint(name: str) -> bool:
    return name.endswith(("main.py", "cli.py", "__main__.py", "app.py"))


def _core_subject(spec: ProjectSpec) -> str:
    """Тема головного коммита из названия проекта: `feat: cli entrypoint for log-digest`."""
    return f"entrypoint for {spec.slug}"


def _repo_description(spec: ProjectSpec) -> str:
    """GitHub обрезает описание на 350 символах — режем сами, чтобы не получить 422 на ровном месте."""
    return spec.problem.strip().replace("\n", " ")[:350]


def _parse_repo(payload: object) -> RepoRef:
    if not isinstance(payload, dict):
        raise GitHubSyncError(f"GitHub вернул не объект: {type(payload).__name__}")
    html_url = str(payload.get("html_url", "")).strip()
    ssh_url = str(payload.get("ssh_url", "")).strip()
    if not html_url or not ssh_url:
        raise GitHubSyncError("в ответе GitHub нет адресов репозитория")
    return RepoRef(
        full_name=str(payload.get("full_name", "")).strip(),
        html_url=html_url,
        ssh_url=ssh_url,
        clone_url=str(payload.get("clone_url", "")).strip(),
    )


def _write_files(project_dir: Path, files: list[GeneratedFile]) -> None:
    write_project_files(project_dir, {item.path: item.content for item in files})


__all__ = ["CommitSlice", "GitHubSync", "GitHubSyncError", "PublishResult", "RepoRef", "plan_commits"]
