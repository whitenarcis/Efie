"""
efi/dev/schemas.py

Модель предметной области «Эфи пишет проект»: спецификация, файлы, задача.

Спецификацию сочиняет главная модель (efi.dev.engine.DevEngine.design), и
приезжает она JSON'ом из свободного текста, поэтому это pydantic-модели с
нормализацией, а не dataclass'ы: имя проекта должно стать легальным именем
git-репозитория, пути файлов — не выходить за пределы рабочей папки, а сам
замысел — не быть «Hello World».

Про последнее отдельно. Модель, которую попросили придумать проект, по
умолчанию предлагает калькулятор, todo-лист и hello-world — это самые
частые учебные примеры в её обучающих данных. Такой проект не нужен никому,
включая саму Эфи: смысл ремесла в том, что вещь кому-то полезна. Поэтому
«учебность» — не вопрос вкуса, а валидация (`looks_like_junk`), и спека,
которая её не проходит, отправляется на переделку.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator

#: Максимум файлов в одном проекте. Не потолок амбиций, а граница одного
#: захода: каждый файл — это отдельный запрос к кодеру плюс до трёх
#: исправлений, и проект на тридцать файлов означал бы час работы фонового
#: цикла и сотню запросов на бесплатном тире.
MAX_PROJECT_FILES = 8

#: Учебный мусор. Проверяется по названию и по описанию проблемы: модель
#: любит предлагать то, что чаще всего встречается в туториалах.
_JUNK_MARKERS = (
    "hello world",
    "hello-world",
    "helloworld",
    "привет, мир",
    "hello, world",
    "todo list",
    "todo-list",
    "todo app",
    "тудушка",
    "список дел",
    "простой калькулятор",
    "simple calculator",
    "calculator app",
    "guess the number",
    "угадай число",
    "fizzbuzz",
    "fizz buzz",
    "учебный пример",
    "для практики",
    "demo project",
    "sample project",
    "тестовый проект",
)

#: Имя репозитория: то, что допускает GitHub и что не стыдно показать.
_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]{1,38}$")
_SLUG_CLEANUP_RE = re.compile(r"[^a-z0-9._-]+")

#: Пути внутри проекта. Никаких абсолютных путей, никаких «..» — спека
#: приходит от языковой модели, а по ней потом создаются реальные файлы на
#: диске; `../../.ssh/authorized_keys` не должен быть выразимым в принципе.
_ALLOWED_PATH_RE = re.compile(r"^[A-Za-z0-9_][A-Za-z0-9._/-]{0,79}$")


class DevTaskStatus(StrEnum):
    """
    Где сейчас задача. Порядок значений — порядок прохождения конвейера.

    Статус персистентный (см. efi/dev/store.py): проект пишется десятками
    минут, и перезапуск процесса посреди работы не должен превращать задачу
    в «висит непонятно где».
    """

    #: Идея принята, но конвейер к ней ещё не подходил.
    PENDING = "pending"
    #: Главная модель сочиняет спецификацию.
    SPECCING = "speccing"
    #: Кодер пишет файлы, песочница их проверяет.
    CODING = "coding"
    #: Код готов, идёт создание репозитория и пуш.
    PUBLISHING = "publishing"
    #: Запушено, ссылка есть.
    DONE = "done"
    #: Не получилось; `error` объясняет, на чём именно.
    FAILED = "failed"

    @property
    def is_terminal(self) -> bool:
        return self in (DevTaskStatus.DONE, DevTaskStatus.FAILED)


class FileSpec(BaseModel):
    """Один файл проекта по замыслу: что это и зачем."""

    model_config = ConfigDict(frozen=True)

    path: str = Field(description="Путь внутри репозитория, например src/parser.py")
    purpose: str = Field(default="", description="Что этот файл делает — задание для кодера")

    @field_validator("path")
    @classmethod
    def _validate_path(cls, value: str) -> str:
        """
        Путь приходит от языковой модели, а по нему создаётся реальный файл.
        Поэтому проверка не «на всякий случай», а по составляющим.

        Разбор идёт покомпонентно, а не одной чисткой строки: `lstrip("./")`
        выглядит как «убрать ведущий ./», но снимает ЛЮБЫЕ ведущие точки и
        слэши — и `../../.ssh/authorized_keys` превращается во вполне
        валидный `ssh/authorized_keys`, то есть проверка на «..» перестаёт
        что-либо ловить.
        """
        path = value.strip().replace("\\", "/").removeprefix("./")
        parts = path.split("/")
        if not path or path.startswith("/") or any(part in ("", ".", "..") for part in parts):
            raise ValueError(f"недопустимый путь файла: {value!r}")
        if not _ALLOWED_PATH_RE.match(path):
            raise ValueError(f"недопустимый путь файла: {value!r}")
        return path

    @property
    def is_python(self) -> bool:
        return self.path.endswith(".py")


class ProjectSpec(BaseModel):
    """
    Замысел проекта целиком — то, что главная модель придумала, а кодер
    будет исполнять.

    `slug` — он же имя репозитория на GitHub и имя папки в рабочем каталоге.
    """

    model_config = ConfigDict(frozen=True)

    slug: str = Field(description="Имя репозитория: латиница, цифры, дефисы")
    title: str = Field(description="Человеческое название проекта")
    problem: str = Field(description="Какую реальную проблему решает")
    stack: list[str] = Field(default_factory=list, description="Язык, библиотеки, инструменты")
    files: list[FileSpec] = Field(default_factory=list)
    readme: str = Field(default="", description="Содержимое README.md")

    @field_validator("slug")
    @classmethod
    def _normalize_slug(cls, value: str) -> str:
        slug = _SLUG_CLEANUP_RE.sub("-", value.strip().lower()).strip("-.")
        if not _SLUG_RE.match(slug):
            raise ValueError(f"невозможно получить имя репозитория из {value!r}")
        return slug

    @field_validator("files")
    @classmethod
    def _limit_files(cls, value: list[FileSpec]) -> list[FileSpec]:
        unique: dict[str, FileSpec] = {}
        for item in value:
            unique.setdefault(item.path, item)
        return list(unique.values())[:MAX_PROJECT_FILES]

    @property
    def python_files(self) -> list[FileSpec]:
        return [item for item in self.files if item.is_python]

    def looks_like_junk(self) -> bool:
        """
        Учебная пустышка ли это. См. докстринг модуля: проверка существует
        потому, что «придумай проект» без ограничений даёт калькулятор.
        """
        haystack = f"{self.slug} {self.title} {self.problem}".lower()
        return any(marker in haystack for marker in _JUNK_MARKERS)

    def is_substantial(self) -> bool:
        """Спека, по которой вообще есть что писать: есть код и сформулированная проблема."""
        return bool(self.python_files) and len(self.problem.strip()) >= 24

    def render_for_prompt(self) -> str:
        """Как проект выглядит в системном промпте — одной плотной строкой, без JSON."""
        stack = ", ".join(self.stack) if self.stack else "python"
        return f"«{self.title}» ({self.slug}, {stack}) — {self.problem.strip()}"


class GeneratedFile(BaseModel):
    """Готовый файл: путь из спеки плюс то, что реально написал кодер."""

    model_config = ConfigDict(frozen=True)

    path: str
    content: str
    #: Сколько раз файл пришлось переписывать после замечаний песочницы.
    fix_rounds: int = 0
    #: Осталась ли претензия песочницы к финальному варианту. Такой файл всё
    #: равно попадает в репозиторий (см. efi/dev/engine.py): проект с одним
    #: незакрытым замечанием линтера полезнее, чем несуществующий проект.
    unresolved_diagnostics: str = ""

    @property
    def is_clean(self) -> bool:
        return not self.unresolved_diagnostics


class DevTask(BaseModel):
    """
    Одна задача конвейера: идея, её текущий статус и всё, что о ней уже
    известно.

    `chat_id` — куда рассказывать о ходе работы. Для совместной задачи это
    чат, где договаривались; для собственной затеи — личка владельца.
    """

    id: int = 0
    chat_id: int | None = None
    idea: str
    #: Задача, которую предложил человек (в отличие от собственной затеи).
    #: Разница не косметическая: о совместной задаче Эфи отчитывается там,
    #: где о ней договорились, и не спрашивает разрешения писать первой —
    #: её попросили. См. efi/dev/reporter.py.
    is_collab: bool = False
    status: DevTaskStatus = DevTaskStatus.PENDING
    spec: ProjectSpec | None = None
    repo_url: str = ""
    error: str = ""
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    #: Когда Эфи последний раз перечитывала свой проект (efi/dev/maintenance.py).
    #: None — ни разу: значит, он ровно такой, каким его дописали.
    reviewed_at: datetime | None = None
    #: Сколько правок она внесла после релиза. Это и есть разница между
    #: «сгенерировала и забыла» и «у неё есть проект, к которому она
    #: возвращается».
    revisions: int = 0

    @property
    def status_label(self) -> str:
        """
        Статус её словами, а не значением перечисления. Одно определение на
        промпт, инструменты и дашборд: «coding» в интерфейсе для человека
        выглядит ровно так же неуместно, как в реплике Эфи.
        """
        return _STATUS_WORDS[self.status]

    def render_for_prompt(self) -> str:
        """Строчка о задаче для системного промпта — чтобы Эфи знала, чем сама сейчас занята."""
        if self.spec is not None:
            subject = self.spec.render_for_prompt()
        else:
            subject = f"замысел: {self.idea.strip()}"
        return f"{subject} — {self.status_label}"


#: Как статус звучит для самой Эфи. Не «status=coding», а то, что она могла
#: бы сказать вслух: этот текст уходит прямо в системный промпт.
_STATUS_WORDS: dict[DevTaskStatus, str] = {
    DevTaskStatus.PENDING: "ещё не начинала",
    DevTaskStatus.SPECCING: "продумываешь структуру",
    DevTaskStatus.CODING: "пишешь код",
    DevTaskStatus.PUBLISHING: "выкладываешь на GitHub",
    DevTaskStatus.DONE: "готово и запушено",
    DevTaskStatus.FAILED: "не вышло",
}


__all__ = [
    "MAX_PROJECT_FILES",
    "DevTask",
    "DevTaskStatus",
    "FileSpec",
    "GeneratedFile",
    "ProjectSpec",
]
