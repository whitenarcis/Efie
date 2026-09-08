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
from efi.llm.errors import LLMAuthError, LLMRateLimitError, LLMServerError
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


def _client(provider: Any, **kwargs: Any) -> QwenCoderClient:
    """Кодер без пауз между повторами: проверяется поведение, а не терпение."""
    return QwenCoderClient(_endpoint(), provider=provider, **kwargs)


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


def test_files_come_in_three_shapes_and_all_three_count() -> None:
    """
    Модели отвечают структурой файлов тремя способами: списком объектов (как
    просили), списком путей и словарём «путь -> назначение». Понимать только
    первый — значит регулярно получать спеку «без единого файла» и отвергать
    вполне живой замысел как пустой.
    """
    as_strings = dict(_GOOD_SPEC, files=["src/parser.py", "src/main.py"])
    as_mapping = dict(_GOOD_SPEC, files={"src/parser.py": "разбор строк", "src/main.py": "точка входа"})

    for payload in (as_strings, as_mapping):
        spec, problem = parse_spec(json.dumps(payload, ensure_ascii=False))

        assert problem == ""
        assert spec is not None
        assert [item.path for item in spec.files] == ["src/parser.py", "src/main.py"]


def test_other_field_names_do_not_cost_the_whole_project() -> None:
    """
    «description» вместо «problem» и «name» вместо «slug» — самая частая
    вольность бесплатных моделей. Цена придирки к именам полей — потерянный
    проект, а выигрыша нет никакого.
    """
    raw = json.dumps(
        {
            "name": "Log Digest",
            "description": "Разбирает многогигабайтные логи nginx и показывает топ ошибок за период",
            "tech": ["python 3.11"],
            "structure": [{"file": "src/main.py", "role": "точка входа CLI"}],
        },
        ensure_ascii=False,
    )

    spec, problem = parse_spec(raw)

    assert problem == ""
    assert spec is not None
    assert spec.slug == "log-digest", "имя репозитория выводится из названия, если его не дали"
    assert spec.files[0].purpose == "точка входа CLI"


def test_spec_parse_failures_say_what_exactly_went_wrong() -> None:
    """Причина уходит наружу и доезжает до карточки проекта — общее «не вышло» не чинится ничем."""
    assert parse_spec("")[1] == "пустой ответ модели"
    assert "нет объекта JSON" in parse_spec("Конечно, давай сделаем парсер логов!")[1]
    assert "невалидный JSON" in parse_spec('{"slug": "log-digest", }')[1]


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
        self,
        sources: list[str],
        *,
        fixes: list[str] | None = None,
        readme: str | None = _GOOD_README,
        unavailable_reason: str = "",
        truncated: bool = False,
    ) -> None:
        self._sources = list(sources)
        self._fixes = list(fixes or [])
        self._readme = readme
        #: Непоправимый отказ провайдера (нет такой модели, отвергнут ключ) —
        #: см. QwenCoderClient.unavailable_reason.
        self.unavailable_reason = unavailable_reason
        #: Упёрся ли ответ в лимит вывода — см. QwenCoderClient.last_answer_truncated.
        self.last_answer_truncated = truncated
        self.fix_calls = 0
        self.readme_calls = 0
        self.diagnostics: list[str] = []

    async def write_file(self, spec: ProjectSpec, file_spec: FileSpec, **_: Any) -> str | None:
        return self._sources.pop(0) if self._sources else None

    async def fix_file(self, path: str, source: str, diagnostics: str) -> str | None:
        self.fix_calls += 1
        self.diagnostics.append(diagnostics)
        return self._fixes.pop(0) if self._fixes else None

    async def write_document(self, path: str, *, system_prompt: str, request: str) -> str | None:
        self.readme_calls += 1
        return self._readme


class _StaticRouter:
    """Главная модель, отвечающая одним и тем же текстом."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.calls = 0
        self.prompts: list[str] = []

    async def chat(self, role: TaskRole, params: LLMParams, session: Session) -> Response:
        self.calls += 1
        self.prompts.append(session.messages[-1].content)
        return Response(choices=[Choice(message=Message(role=Role.ASSISTANT, content=self.text))])


class _FailingRouter:
    async def chat(self, role: TaskRole, params: LLMParams, session: Session) -> Response:
        raise LLMServerError("провайдер лёг", provider="test")


class _TruncatingRouter:
    """Модель, чей ответ каждый раз упирается в лимит токенов (finish_reason='length')."""

    def __init__(self, text: str) -> None:
        self.text = text
        self.prompts: list[str] = []

    async def chat(self, role: TaskRole, params: LLMParams, session: Session) -> Response:
        self.prompts.append(session.messages[-1].content)
        return Response(
            choices=[
                Choice(message=Message(role=Role.ASSISTANT, content=self.text), finish_reason="length")
            ]
        )


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
    Файл, который не парсится ни после правок, ни после переписывания
    заново, — это не «неидеально», а отсутствующий файл: публиковать такой
    проект нельзя.
    """
    coder = _ScriptedCoder(
        ["def f(:\n", "def f( всё ещё сломано(:\n", "def main() -> None:\n    pass\n"],
        fixes=["still broken(:\n"] * 3,
    )

    build = await _engine(coder).build(_spec())

    assert coder.fix_calls == 3, "должно быть ровно max_fix_iterations попыток"
    assert build.is_publishable is False
    assert build.broken_paths == ["src/parser.py"]


async def test_a_file_that_never_parses_is_rewritten_from_scratch_before_giving_up() -> None:
    """
    Круг исправлений просит починить сломанный текст, и когда текст сломан
    обрывом посреди функции, кодер честно дописывает ту же функцию и
    упирается в тот же лимит. Просьба написать заново и компактнее рвёт этот
    круг — и часто спасает весь проект, а не только файл.
    """
    coder = _ScriptedCoder(
        ["def parse(:\n", "def parse_line(raw: str) -> dict:\n    return {}\n", "y = 2\n"],
        fixes=["ещё хуже(:\n"] * 3,
    )

    build = await _engine(coder).build(_spec())

    assert build.is_publishable is True
    assert "def parse_line" in build.as_file_map()["src/parser.py"]


async def test_a_truncated_tail_is_cut_off_instead_of_losing_the_whole_project() -> None:
    """
    Всё, что выше обрыва, — рабочий код. Терять из-за одной незавершённой
    функции в конце проект, где остальные файлы уже написаны, — худший из
    возможных обменов.
    """
    truncated = (
        "import re\n"
        "\n"
        "_LINE_RE = re.compile(r'^(?P<ip>\\S+) (?P<code>\\d+)$')\n"
        "\n"
        "\n"
        "def parse_line(raw: str) -> dict:\n"
        "    match = _LINE_RE.match(raw.strip())\n"
        "    if match is None:\n"
        "        return {}\n"
        "    return match.groupdict()\n"
        "\n"
        "\n"
        "def summarize(rows: list[dict]) -> dict:\n"
        "    totals: dict[str, int] = {}\n"
        "    for row in rows:\n"
        "        key = row.get(\n"
    )
    coder = _ScriptedCoder([truncated, truncated, "y = 2\n"], fixes=[None] * 3)  # type: ignore[list-item]

    build = await _engine(coder).build(_spec())

    parser = build.as_file_map()["src/parser.py"]
    assert build.is_publishable is True
    assert "def parse_line" in parser
    assert "summarize" not in parser, "оборванный хвост отрезан, а не выложен как есть"


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


async def test_a_dead_coder_stops_the_build_with_the_real_reason() -> None:
    """
    Снятая с обслуживания модель или отвергнутый ключ не чинятся к следующему
    файлу. Без остановки один неверный конфиг стоил бы десятка запросов на
    каждый проект и заканчивался бы невнятным «кодер не написал ни одного
    файла» — по такому сообщению причину не найти.
    """
    coder = _ScriptedCoder(
        [None, None],  # type: ignore[list-item]
        unavailable_reason="модель 'qwen-2.5-coder-32b' недоступна: coder: unexpected HTTP 404: model_decommissioned",
    )

    build = await _engine(coder).build(_spec())

    assert build.is_publishable is False
    assert "model_decommissioned" in build.failure_reason
    assert build.broken_paths == ["src/parser.py"], "второй файл даже не запрашивался"
    assert coder.readme_calls == 0, "README без кода писать не о чем"


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

    spec, reason = await _engine(_ScriptedCoder([]), router).design("")

    assert spec is None
    assert router.calls == 2, "вторая попытка с прямым указанием переделать"
    assert "учебный" in reason, "причина отказа — своя у каждого случая, а не общее «не придумалось»"


async def test_design_survives_a_dead_provider() -> None:
    spec, reason = await _engine(_ScriptedCoder([]), _FailingRouter()).design("напиши парсер логов")

    assert spec is None
    assert "провайдер лёг" in reason, "владельцу видно, что дело в провайдере, а не в фантазии модели"


async def test_she_does_not_write_the_same_project_twice() -> None:
    """
    Интересы меняются медленно, и «придумай себе проект» на одном и том же
    контексте даёт один и тот же ответ. Стоит это не только скуки: имя
    репозитория занято её же прошлым проектом, и пуш второго такого проекта
    отклоняется как непустая история.
    """
    router = _StaticRouter(json.dumps(_GOOD_SPEC, ensure_ascii=False))

    spec, reason = await _engine(_ScriptedCoder([]), router).design("", built=[_spec()])

    assert spec is None
    assert "повторяет" in reason
    assert "Это ты уже написала" in router.prompts[0], "список написанного уходит в промпт, а не только в проверку"


async def test_a_different_project_on_the_same_stack_is_not_a_repeat() -> None:
    """Иначе после второй утилиты на python она не смогла бы написать ничего."""
    other = dict(
        _GOOD_SPEC,
        slug="disk-watch",
        title="Disk Watch",
        problem="Следит за свободным местом на дисках и пишет в telegram, когда остаётся мало",
    )
    router = _StaticRouter(json.dumps(other, ensure_ascii=False))

    spec, reason = await _engine(_ScriptedCoder([]), router).design("", built=[_spec()])

    assert reason == ""
    assert spec is not None and spec.slug == "disk-watch"


async def test_truncated_answer_is_named_and_answered_with_write_shorter() -> None:
    """
    Оборванный по лимиту JSON не разбирается в принципе, и «невалидный JSON»
    как причина увело бы куда угодно, кроме настоящей: ответ не поместился.
    А повтор с той же просьбой дал бы ровно то же самое.
    """
    router = _TruncatingRouter(json.dumps(_GOOD_SPEC, ensure_ascii=False)[:120])

    spec, reason = await _engine(_ScriptedCoder([]), router).design("")

    assert spec is None
    assert "оборвал" in reason
    assert "КОРОТКО" in router.prompts[1], "во второй раз просим короче, а не «придумай другое»"


async def test_truncated_answer_does_not_hide_behind_invalid_json() -> None:
    """Причина обрыва не должна подменяться следом от разбора обрезанного текста."""
    router = _TruncatingRouter('{"slug": "log-dig')

    _spec_result, reason = await _engine(_ScriptedCoder([]), router).design("")

    assert "JSON" not in reason


async def test_coder_errors_do_not_raise() -> None:
    """Сбой провайдера у кодера — это «файл не написался», а не исключение посреди фонового цикла."""
    client = _client(_FailingProvider(), retry_delays=())

    assert await client.write_file(_spec(), _spec().files[0]) is None
    assert await client.fix_file("src/main.py", "x = 1", "E999") is None


class _FailingProvider:
    async def chat(self, params: LLMParams, session: Session) -> Response:
        raise LLMServerError("нет связи", provider="coder")


class _FlakyProvider:
    """Провайдер, который отказывает заданное число раз, а потом отвечает."""

    def __init__(self, error: Exception, *, failures: int) -> None:
        self._error = error
        self._left = failures
        self.calls = 0

    async def chat(self, params: LLMParams, session: Session) -> Response:
        self.calls += 1
        if self._left > 0:
            self._left -= 1
            raise self._error
        return Response(choices=[Choice(message=Message(role=Role.ASSISTANT, content="x = 1\n"))])


async def test_a_rate_limited_file_is_asked_for_again_instead_of_being_lost() -> None:
    """
    На бесплатном тире лимит токенов в минуту выбирается третьим-четвёртым
    файлом подряд. Без повтора файл просто не пишется — а за ним разваливается
    весь проект, хотя ждать было нужно полминуты и ждать было некому: конвейер
    фоновый.
    """
    provider = _FlakyProvider(LLMRateLimitError("too many requests", provider="coder"), failures=2)
    client = _client(provider, retry_delays=(0.0, 0.0, 0.0))

    source = await client.write_file(_spec(), _spec().files[0])

    assert source == "x = 1"
    assert provider.calls == 3


async def test_a_rejected_key_is_not_retried() -> None:
    """Отвергнутый ключ — это не «сейчас занято»: десять попыток дадут десять одинаковых ответов."""
    provider = _FlakyProvider(LLMAuthError("invalid api key", provider="coder"), failures=5)
    client = _client(provider, retry_delays=(0.0, 0.0))

    assert await client.write_file(_spec(), _spec().files[0]) is None
    assert provider.calls == 1
    assert "отверг ключ" in client.unavailable_reason


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
    client = _client(provider)

    parser_source = "def parse_line(raw: str) -> dict:\n    ...\n\ndef _private() -> None:\n    ...\n"

    await client.write_file(_spec(), _spec().files[1], already_written={"src/parser.py": parser_source})

    assert "def parse_line(raw: str) -> dict:" in provider.last_user_content
    assert "_private" not in provider.last_user_content, "приватные имена соседу не нужны"
