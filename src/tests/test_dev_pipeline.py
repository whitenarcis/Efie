"""
Тесты конвейера кодогенерации: спека -> код -> проверка -> исправление.

Проверяется не «модель что-то ответила», а то, что происходит с её ответом
дальше: как разбирается спека, приходящая свободным текстом; как из ответа
вырезается markdown-обёртка; как песочница ловит сломанный код и как цикл
исправлений доводит файл до состояния «парсится» — или честно признаёт, что
не довёл.

Все проверки без сети: и кодер, и главная модель подменяются, потому что
проверяется именно наш код вокруг них.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import SecretStr

from efi.config.schema import EndpointConfig, TaskRole
from efi.dev.engine import DevEngine, parse_spec
from efi.dev.qwen_client import QwenCoderClient, strip_code_fences
from efi.dev.readme import missing_sections, problems, render_fallback
from efi.dev.sandbox import CodeSandbox, write_project_files
from efi.dev.schemas import FileSpec, GeneratedFile, ProjectSpec
from efi.llm.errors import LLMServerError
from efi.llm.schemas import Choice, LLMParams, Message, Response, Role, Session

_GOOD_SPEC = {
    "slug": "log-digest",
    "title": "Log Digest",
    "problem": "Разбирает многогигабайтные логи nginx и показывает топ ошибок за период",
    "stack": ["python 3.11", "argparse"],
    "files": [
        {"path": "src/parser.py", "purpose": "разбор строк лога"},
        {"path": "src/main.py", "purpose": "точка входа CLI"},
    ],
    "readme": "# Log Digest\n\nразбор логов",
}


def _endpoint() -> EndpointConfig:
    return EndpointConfig(base_url="https://api.groq.com/openai/v1", api_key=SecretStr("k"), model="qwen")


def _spec() -> ProjectSpec:
    return ProjectSpec.model_validate(_GOOD_SPEC)


# -- разбор спеки -------------------------------------------------------------


def test_spec_survives_markdown_and_chatter() -> None:
    """Модель почти всегда оборачивает JSON в ```-блок и предваряет его фразой. Это не повод терять замысел."""
    raw = "Конечно! Вот спецификация:\n```json\n" + json.dumps(_GOOD_SPEC, ensure_ascii=False) + "\n```"

    spec, problem = parse_spec(raw)

    assert problem == ""
    assert spec is not None
    assert spec.slug == "log-digest"
    assert [item.path for item in spec.files] == ["src/parser.py", "src/main.py"]


def test_broken_file_entry_does_not_kill_the_whole_spec() -> None:
    """Один кривой путь не должен стоить всего замысла — остальные файлы доезжают."""
    payload = dict(_GOOD_SPEC)
    payload["files"] = [
        {"path": "../../.ssh/authorized_keys", "purpose": "нет"},
        {"path": "src/main.py", "purpose": "точка входа"},
    ]

    spec, _problem = parse_spec(json.dumps(payload))

    assert spec is not None
    assert [item.path for item in spec.files] == ["src/main.py"]


def test_spec_slug_is_normalized_into_a_repo_name() -> None:
    payload = dict(_GOOD_SPEC, slug="Log Digest!! 2.0")

    spec, _problem = parse_spec(json.dumps(payload))

    assert spec is not None
    assert spec.slug == "log-digest-2.0"


def test_junk_projects_are_recognized() -> None:
    """Учебный мусор — самый вероятный ответ модели на «придумай проект», и он должен отсеиваться."""
    junk = ProjectSpec.model_validate(
        dict(
            _GOOD_SPEC,
            slug="hello-world",
            title="Hello World",
            problem="Показывает приветствие пользователю в консоли, простой пример для практики",
        )
    )
    assert junk.looks_like_junk() is True
    assert _spec().looks_like_junk() is False


def test_spec_without_code_or_problem_is_not_substantial() -> None:
    thin = ProjectSpec.model_validate(dict(_GOOD_SPEC, problem="утилита", files=[]))
    assert thin.is_substantial() is False


# -- очистка ответа кодера ----------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("```python\nx = 1\n```", "x = 1"),
        ("```\nx = 1\n```", "x = 1"),
        ("Вот файл:\n```python\nx = 1\n```\nГотово!", "x = 1"),
        ("x = 1", "x = 1"),
        # Обрыв по лимиту токенов: закрывающего забора нет.
        ("```python\ndef f():\n    return 1", "def f():\n    return 1"),
    ],
)
def test_code_fences_are_stripped(raw: str, expected: str) -> None:
    assert strip_code_fences(raw) == expected


# -- песочница ----------------------------------------------------------------


async def test_sandbox_catches_broken_syntax() -> None:
    sandbox = CodeSandbox(enable_linter=False)

    report = await sandbox.check("src/main.py", "def f(:\n    pass\n")

    assert report.ok is False
    assert report.syntax_broken is True
    assert "SyntaxError" in report.render()


async def test_sandbox_passes_valid_code_and_skips_non_python() -> None:
    sandbox = CodeSandbox(enable_linter=False)

    assert (await sandbox.check("src/main.py", "def f() -> int:\n    return 1\n")).ok is True
    # README проверять нечем — и это не повод считать его плохим.
    assert (await sandbox.check("README.md", "# заголовок\n```")).ok is True


async def test_missing_linter_degrades_to_syntax_check() -> None:
    """
    ruff в Termux может не стоять вовсе. Это рабочий режим: проверка остаётся
    синтаксической, конвейер не встаёт.
    """
    sandbox = CodeSandbox(ruff_executable="ruff-that-does-not-exist")

    report = await sandbox.check("src/main.py", "import os\n")

    assert report.ok is True


def test_generated_files_cannot_escape_the_project(tmp_path: Any) -> None:
    """Путь провалидирован схемой, но перед записью на диск проверяется ещё раз — цена ошибки слишком велика."""
    with pytest.raises(ValueError, match="выходит за пределы"):
        write_project_files(tmp_path, {"../escaped.py": "x = 1"})

    written = write_project_files(tmp_path, {"src/main.py": "x = 1"})
    assert written[0].read_text(encoding="utf-8") == "x = 1"


# -- README как обязательный критерий -----------------------------------------


def test_readme_stub_is_not_accepted() -> None:
    """Два заголовка и строчка описания — это не документация, а её видимость."""
    assert missing_sections("# Проект\n\nкрутая штука") == [
        "Назначение",
        "Установка",
        "Использование",
        "Структура",
    ]


def test_readme_sections_are_recognized_by_synonyms() -> None:
    """
    Требовать дословных заголовков нельзя: модель пишет то «Установка», то
    «Как поставить», то «Installation» — и годный текст забраковывался бы
    из-за синонима.
    """
    text = (
        "# Утилита\n\nчто-то полезное\n\n"
        "## Зачем нужна\n\nрешает конкретную проблему конкретного человека, который устал делать это руками\n\n"
        "## Как поставить\n\nPython 3.11, зависимостей нет, склонировать и запустить\n\n"
        "## Быстрый старт\n\n`python main.py --help` покажет все доступные флаги и примеры вызова\n\n"
        "## Модули\n\n- `main.py` — точка входа и разбор аргументов командной строки\n"
    )

    assert missing_sections(text) == []


def test_short_readme_is_reported_as_short_not_as_missing_everything() -> None:
    """
    Заглушка, где формально упомянуты все четыре темы, всё равно не годится —
    но претензия к ней именно «слишком короткий». Соврать здесь значит
    отправить модели неправду про отсутствующие разделы.
    """
    text = "# Утилита\n\n## Назначение\nвсё\n## Установка\nvsё\n## Использование\nвсё\n## Структура\nвсё\n"

    issues = problems(text)

    assert missing_sections(text) == []
    assert any("короткий" in issue for issue in issues)


def test_fallback_readme_is_complete_and_built_from_real_paths() -> None:
    files = [
        GeneratedFile(path="src/parser.py", content="def parse() -> None:\n    ..."),
        GeneratedFile(path="src/main.py", content="def main() -> None:\n    ..."),
    ]

    text = render_fallback(_spec(), files)

    assert missing_sections(text) == []
    assert "python src/main.py" in text, "команда запуска — из точки входа проекта"
    assert "git clone" in text
    assert "`src/parser.py`" in text


# -- цикл «написать -> проверить -> починить» ---------------------------------


#: README, проходящий проверку обязательных разделов, — им отвечает кодер в
#: тестах, где предмет проверки не документация, а код.
_GOOD_README = (
    "# Log Digest\n\nУтилита для разбора логов.\n\n"
    "## Назначение\n\nРазбирает логи nginx и показывает топ ошибок за период — тем, кто держит "
    "сервер и не хочет читать гигабайты руками.\n\n"
    "## Установка\n\nТребуется Python 3.11+, внешних зависимостей нет.\n\n"
    "```bash\ngit clone https://github.com/efi/log-digest.git\ncd log-digest\n```\n\n"
    "## Использование\n\n```bash\npython src/main.py access.log --top 10\n```\n\n"
    "## Структура\n\n- `src/parser.py` — разбор строк\n- `src/main.py` — точка входа\n"
)


class _ScriptedCoder:
    """Кодер, отвечающий по заранее заданному сценарию: первый ответ битый, второй — рабочий."""

    def __init__(
        self, sources: list[str], *, fixes: list[str] | None = None, readme: str | None = _GOOD_README
    ) -> None:
        self._sources = list(sources)
        self._fixes = list(fixes or [])
        self._readme = readme
        self.fix_calls = 0
        self.readme_calls = 0

    async def write_file(self, spec: ProjectSpec, file_spec: FileSpec, **_: Any) -> str | None:
        return self._sources.pop(0) if self._sources else None

    async def fix_file(self, path: str, source: str, diagnostics: str) -> str | None:
        self.fix_calls += 1
        return self._fixes.pop(0) if self._fixes else None

    async def write_document(self, path: str, *, system_prompt: str, request: str) -> str | None:
        self.readme_calls += 1
        return self._readme


class _StaticRouter:
    """Главная модель, отвечающая одним и тем же текстом."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0

    async def chat(self, role: TaskRole, params: LLMParams, session: Session) -> Response:
        self.calls += 1
        return Response(choices=[Choice(message=Message(role=Role.ASSISTANT, content=self.text))])


class _FailingRouter:
    async def chat(self, role: TaskRole, params: LLMParams, session: Session) -> Response:
        raise LLMServerError("провайдер лёг", provider="test")


def _engine(coder: Any, router: Any = None, *, max_fix_iterations: int = 3) -> DevEngine:
    return DevEngine(
        router or _StaticRouter(json.dumps(_GOOD_SPEC)),  # type: ignore[arg-type]
        coder,
        CodeSandbox(enable_linter=False),
        max_fix_iterations=max_fix_iterations,
    )


async def test_broken_file_is_sent_back_to_the_coder_and_fixed() -> None:
    """Сердце автокоррекции: битый файл уходит обратно кодеру с трейсбеком и возвращается рабочим."""
    coder = _ScriptedCoder(
        ["def parse(:\n    pass\n", "def main() -> None:\n    pass\n"],
        fixes=["def parse() -> None:\n    pass\n"],
    )

    build = await _engine(coder).build(_spec())

    assert coder.fix_calls == 1
    assert build.is_publishable is True
    assert build.fix_rounds == 1
    assert "def parse() -> None" in build.as_file_map()["src/parser.py"]


async def test_file_that_never_parses_blocks_publication() -> None:
    """
    Файл, который не парсится и после трёх правок, — это не «неидеально», а
    отсутствующий файл: публиковать такой проект нельзя.
    """
    coder = _ScriptedCoder(["def f(:\n", "def main() -> None:\n    pass\n"], fixes=["still broken(:\n"] * 3)

    build = await _engine(coder).build(_spec())

    assert coder.fix_calls == 3, "должно быть ровно max_fix_iterations попыток"
    assert build.is_publishable is False
    assert build.broken_paths == ["src/parser.py"]


async def test_every_project_gets_a_readme_written_from_the_real_code() -> None:
    """
    README — обязательное условие публикации: репозиторий, по которому
    непонятно ни что это, ни как запустить, бесполезен для того, кто по
    ссылке пришёл.
    """
    coder = _ScriptedCoder(["x = 1\n", "y = 2\n"])

    build = await _engine(coder).build(_spec())

    readme = build.as_file_map()["README.md"]
    assert coder.readme_calls == 1
    assert missing_sections(readme) == []


async def test_unusable_readme_is_replaced_by_a_complete_one() -> None:
    """
    Модель регулярно отвечает заглушкой в две строки. «Обязательный раздел»
    должен быть свойством кода, а не пожеланием в промпте: годного README нет
    — собираем сами из спеки, но полный.
    """
    coder = _ScriptedCoder(["x = 1\n", "y = 2\n"], readme="# Log Digest\n\nкрутая штука\n")

    build = await _engine(coder).build(_spec())

    readme = build.as_file_map()["README.md"]
    assert coder.readme_calls == 2, "сначала просим дописать, и только потом собираем сами"
    assert missing_sections(readme) == []
    assert "src/main.py" in readme, "команда запуска — из реального файла, а не выдуманная"


async def test_readme_survives_a_silent_coder() -> None:
    coder = _ScriptedCoder(["x = 1\n", "y = 2\n"], readme=None)

    build = await _engine(coder).build(_spec())

    assert missing_sections(build.as_file_map()["README.md"]) == []


async def test_design_rejects_junk_and_gives_up_honestly() -> None:
    """
    Модель настаивает на hello world — значит, проекта не будет. Отказ лучше,
    чем репозиторий с учебным примером под её именем.
    """
    junk = dict(_GOOD_SPEC, slug="hello-world", title="Hello World", problem="Пример для практики: печатает привет")
    router = _StaticRouter(json.dumps(junk))

    spec = await _engine(_ScriptedCoder([]), router).design("")

    assert spec is None
    assert router.calls == 2, "вторая попытка с прямым указанием переделать"


async def test_design_survives_a_dead_provider() -> None:
    assert await _engine(_ScriptedCoder([]), _FailingRouter()).design("напиши парсер логов") is None


async def test_coder_errors_do_not_raise() -> None:
    """Сбой провайдера у кодера — это «файл не написался», а не исключение посреди фонового цикла."""
    client = QwenCoderClient(_endpoint(), provider=_FailingProvider())  # type: ignore[arg-type]

    assert await client.write_file(_spec(), _spec().files[0]) is None
    assert await client.fix_file("src/main.py", "x = 1", "E999") is None


class _FailingProvider:
    async def chat(self, params: LLMParams, session: Session) -> Response:
        raise LLMServerError("нет связи", provider="coder")


class _EchoProvider:
    """Провайдер, возвращающий заранее заданный текст и запоминающий, что у него спросили."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.last_user_content = ""

    async def chat(self, params: LLMParams, session: Session) -> Response:
        self.last_user_content = session.messages[-1].content
        return Response(choices=[Choice(message=Message(role=Role.ASSISTANT, content=self.text))])


async def test_coder_sees_the_interfaces_of_already_written_files() -> None:
    """
    Без этого кодер выдумывает интерфейс соседнего модуля заново, и проект
    разваливается на несовместимые куски, каждый из которых по отдельности
    проходит проверку.
    """
    provider = _EchoProvider("x = 1\n")
    client = QwenCoderClient(_endpoint(), provider=provider)  # type: ignore[arg-type]

    parser_source = "def parse_line(raw: str) -> dict:\n    ...\n\ndef _private() -> None:\n    ...\n"

    await client.write_file(_spec(), _spec().files[1], already_written={"src/parser.py": parser_source})

    assert "def parse_line(raw: str) -> dict:" in provider.last_user_content
    assert "_private" not in provider.last_user_content, "приватные имена соседу не нужны"
