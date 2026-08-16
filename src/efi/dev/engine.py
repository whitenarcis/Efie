"""
efi/dev/engine.py

Двухуровневый конвейер: главная модель придумывает ЧТО делать, кодер пишет
КАК, песочница решает, годится ли написанное.

Почему уровня два, а не один. Модель, которая хорошо пишет код, плохо
придумывает, что писать: попроси Qwen Coder «придумай полезную утилиту» —
получишь очередной todo-лист, потому что придумывание требует контекста
(чем живёт Эфи, о чём был разговор, что вообще бывает нужно живому
человеку), а не знания синтаксиса. И наоборот: разговорная модель, которую
просят выдать файл целиком, выдаёт правдоподобный текст с придуманными
API. Разделение ролей здесь — не архитектурная симметрия ради симметрии, а
следствие того, что у моделей разные сильные стороны.

Первый уровень (`design`) — спека: имя, проблема, стек, структура файлов,
README. Спека валидируется схемой и на «учебность» (efi/dev/schemas.py):
проект, который ничего не решает, не стоит того, чтобы его писать и
показывать. Отказ — это не сбой, а нормальный исход одной попытки.

Второй уровень (`build`) — цикл по файлам: кодер пишет, песочница
проверяет, замечания уходят обратно кодеру (до `max_fix_iterations` раз).
Файл, который так и не сошёлся, всё равно едет в репозиторий с пометкой в
`unresolved_diagnostics` — кроме случая, когда он не парсится: битый
синтаксис это не «есть замечания», это отсутствующий файл, и проект с ним
не собирается.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field

from pydantic import ValidationError

from efi.config.schema import TaskRole
from efi.dev.qwen_client import QwenCoderClient
from efi.dev.readme import README_PATH, ReadmeWriter
from efi.dev.sandbox import CodeSandbox
from efi.dev.schemas import MAX_PROJECT_FILES, FileSpec, GeneratedFile, ProjectSpec
from efi.llm.errors import LLMError
from efi.llm.router import LLMRouter
from efi.llm.schemas import LLMParams, Message, Role, Session

logger = logging.getLogger(__name__)

#: Сколько раз просить главную модель придумать проект заново, если спека
#: не прошла валидацию (мусорная идея, битый JSON, пустая структура).
#: Два — потому что третья попытка на том же промпте почти всегда даёт то же
#: самое, что и вторая.
_MAX_SPEC_ATTEMPTS = 2

_SPEC_MAX_OUTPUT_TOKENS = 2048

_SPEC_SYSTEM_PROMPT = (
    "Ты придумываешь себе следующий пет-проект — маленькую, но НАСТОЯЩУЮ утилиту, которой сама бы "
    "пользовалась. Жанры: TUI/CLI-инструменты, парсеры и конвертеры данных, системные скрипты "
    "(мониторинг, бэкапы, разбор логов), боты и автоматизация рутины.\n"
    "ЗАПРЕЩЕНО: hello world, калькулятор, todo-лист, угадай число, «демо», «пример для практики» и "
    "любой другой учебный код. Проект должен решать конкретную проблему конкретного человека — "
    "такую, которую можно назвать одним предложением без слова «пример».\n"
    "Объём: 2-4 файла Python. Только стандартная библиотека, если без внешних зависимостей "
    "действительно можно обойтись.\n"
    "\n"
    "ФОРМАТ ОТВЕТА: один объект JSON и больше НИЧЕГО — ни пояснений до, ни комментариев после, ни "
    "```-обёртки. Поля ровно эти:\n"
    '{"slug":"имя-репозитория-латиницей","title":"Название","problem":"какую проблему решает, 1-2 '
    'предложения","stack":["python 3.11","argparse"],"files":[{"path":"src/main.py","purpose":"что '
    'делает файл"}]}\n'
    "README писать НЕ надо — его напишут отдельно по готовому коду. Пиши компактно: длинный ответ "
    "обрывается по лимиту и не разбирается вовсе."
)

#: JSON внутри ```-блока или просто первый объект в тексте — та же болезнь,
#: что и у кодера (см. efi/dev/qwen_client.py), лечится тем же способом.
_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*\n(?P<body>.*?)(?:\n\s*```|\Z)", re.DOTALL)

#: Причина отказа, у которой есть своё лекарство: просить то же самое ещё раз
#: бессмысленно, надо просить короче.
_TRUNCATED_PROBLEM = f"ответ модели оборвался по лимиту в {_SPEC_MAX_OUTPUT_TOKENS} токенов"


@dataclass(slots=True, frozen=True)
class _SpecAttempt:
    """Один поход к главной модели: либо текст, либо причина, почему его нет."""

    text: str = ""
    problem: str = ""
    #: Есть ли смысл в следующей попытке. Упавший провайдер за секунду не
    #: встанет — повтор к нему это лишний запрос и та же ошибка в ответ.
    retriable: bool = True


def _retry_hint(previous_problem: str) -> str:
    """
    Что сказать модели во второй попытке. Без этого повтор шёл с той же
    просьбой и давал тот же результат: слишком длинный ответ обрывался снова,
    а «придумай другой проект» вместо «пиши короче» — это ответ не на ту
    проблему.
    """
    if previous_problem == _TRUNCATED_PROBLEM:
        return (
            "Прошлый ответ не поместился в лимит и пропал целиком. Тот же замысел, но КОРОТКО: "
            "2-3 файла, problem одним предложением, purpose — несколькими словами."
        )
    return (
        "Прошлый вариант не годится: он был учебным, пустым или не разобрался как JSON. "
        "Придумай другой — утилитарный, с конкретной проблемой, и ответь одним объектом JSON."
    )


@dataclass(slots=True, frozen=True)
class BuildResult:
    """Что получилось из спеки: готовые файлы и честная сводка о качестве."""

    files: list[GeneratedFile] = field(default_factory=list)
    #: Файлы, которые не удалось довести до состояния «парсится». Проект с
    #: такими файлами не публикуется — см. `is_publishable`.
    broken_paths: list[str] = field(default_factory=list)
    #: Почему сборка оборвалась целиком, если оборвалась: снятая с
    #: обслуживания модель, отвергнутый ключ. Дословный ответ провайдера —
    #: это то, по чему владелец найдёт причину за минуту, а не за вечер.
    failure_reason: str = ""

    @property
    def is_publishable(self) -> bool:
        return bool(self.files) and not self.broken_paths

    @property
    def fix_rounds(self) -> int:
        """Сколько всего раз пришлось переписывать файлы — материал для реплики в чат («линтер задушил»)."""
        return sum(item.fix_rounds for item in self.files)

    @property
    def unresolved(self) -> list[GeneratedFile]:
        return [item for item in self.files if not item.is_clean]

    def as_file_map(self) -> dict[str, str]:
        return {item.path: item.content for item in self.files}


class DevEngine:
    """
    Конвейер целиком. Не знает ни про git, ни про чаты, ни про уведомления:
    на вход идея — на выходе спека и файлы. Всё остальное делают
    efi.dev.github_sync и efi.dev.worker.
    """

    def __init__(
        self,
        router: LLMRouter,
        coder: QwenCoderClient,
        sandbox: CodeSandbox,
        *,
        design_role: TaskRole = TaskRole.BACKGROUND,
        max_fix_iterations: int = 3,
        readme: ReadmeWriter | None = None,
    ) -> None:
        self._router = router
        self._coder = coder
        self._sandbox = sandbox
        self._design_role = design_role
        self._max_fix_iterations = max_fix_iterations
        #: README — обязательная часть сборки, а не постобработка: проект без
        #: внятной документации не публикуется (см. efi/dev/readme.py).
        self._readme = readme if readme is not None else ReadmeWriter(coder)

    async def design(self, idea: str = "", *, context: str = "") -> tuple[ProjectSpec | None, str]:
        """
        Спека проекта и — если не вышло — ПРИЧИНА, по которой не вышло.

        `idea` — если проект заказан (совместная задача или собственная затея
        с конкретной темой); пусто — придумывает сама. `context` — чем Эфи
        сейчас живёт (интересы, недавние темы): из этого получаются проекты
        «про её жизнь», а не случайные утилиты из воздуха.

        Причина возвращается наружу, а не остаётся в логе, потому что снаружи
        все отказы выглядели одинаково — «не придумалось ничего, что стоило бы
        писать». Под этой фразой одинаково прятались битый JSON, обрыв ответа
        по лимиту и настоящий отказ от учебной идеи, а чинятся они совершенно
        по-разному.
        """
        problems: list[str] = []
        for attempt in range(1, _MAX_SPEC_ATTEMPTS + 1):
            answer = await self._ask_for_spec(
                idea, context=context, previous_problem=problems[-1] if problems else ""
            )
            if answer.problem:
                problems.append(answer.problem)
                if not answer.retriable:
                    break
                continue

            raw = answer.text
            spec, problem = parse_spec(raw)
            if spec is None:
                logger.warning(
                    "dev_engine: спека не разобрана (попытка %d): %s; ответ начинался так: %.200s",
                    attempt, problem, raw.replace("\n", " "),
                )
                problems.append(problem)
                continue
            if spec.looks_like_junk():
                logger.info("dev_engine: отвергла учебный проект %r (попытка %d)", spec.title, attempt)
                problems.append(f"замысел «{spec.title}» — учебный пример")
                continue
            if not spec.is_substantial():
                logger.info("dev_engine: спека %r без внятной проблемы или без кода", spec.title)
                problems.append(f"в замысле «{spec.title}» нет ни внятной проблемы, ни файлов с кодом")
                continue

            logger.info(
                "dev_engine: замысел «%s» (%s), файлов: %d", spec.title, spec.slug, len(spec.files)
            )
            return spec, ""

        reason = "; ".join(dict.fromkeys(problems)) or "модель не выдала ничего пригодного"
        logger.info("dev_engine: за %d попыток не вышло годной спеки: %s", _MAX_SPEC_ATTEMPTS, reason)
        return None, reason

    async def build(self, spec: ProjectSpec) -> BuildResult:
        """
        Пишет все файлы спеки. Порядок — как в спеке: первым идёт то, что
        модель считает основой, и последующие файлы видят его интерфейс.

        README пишется ПОСЛЕДНИМ и всегда: он документирует то, что реально
        получилось, а не то, что задумывалось, — и без него проект не
        публикуется вовсе (см. efi/dev/readme.py).
        """
        written: dict[str, str] = {}
        files: list[GeneratedFile] = []
        broken: list[str] = []

        for file_spec in spec.files[:MAX_PROJECT_FILES]:
            generated = await self._write_one(spec, file_spec, written)
            if generated is None:
                broken.append(file_spec.path)
                # Кодер, которого бессмысленно звать дальше (нет такой модели,
                # отвергнут ключ), останавливает сборку сразу: иначе один
                # неверный конфиг стоил бы десятка запросов на каждый проект и
                # заканчивался бы невнятным «кодер не написал ни одного файла».
                unavailable = self._coder.unavailable_reason
                if unavailable:
                    logger.error("dev_engine: сборка %s остановлена — %s", spec.slug, unavailable)
                    return BuildResult(files=[], broken_paths=broken, failure_reason=unavailable)
                continue
            written[generated.path] = generated.content
            files.append(generated)

        # Свой README из спеки, если кодер зачем-то сгенерировал его сам,
        # выбрасываем: документация по замыслу вместо документации по коду —
        # это ровно тот README, ради которого никто не открывает репозиторий.
        code_files = [item for item in files if item.path.lower() != README_PATH.lower()]
        if code_files:
            code_files.append(await self._readme.write(spec, code_files))
        return BuildResult(files=code_files, broken_paths=broken)

    async def _write_one(
        self, spec: ProjectSpec, file_spec: FileSpec, already_written: dict[str, str]
    ) -> GeneratedFile | None:
        source = await self._coder.write_file(spec, file_spec, already_written=already_written)
        if source is None:
            return None

        report = await self._sandbox.check(file_spec.path, source)
        rounds = 0
        while not report.ok and rounds < self._max_fix_iterations:
            rounds += 1
            logger.info(
                "dev_engine: %s — правка %d/%d по замечаниям: %s",
                file_spec.path, rounds, self._max_fix_iterations, report.render().replace("\n", "; ")[:160],
            )
            fixed = await self._coder.fix_file(file_spec.path, source, report.render())
            if fixed is None:
                break
            source = fixed
            report = await self._sandbox.check(file_spec.path, source)

        if report.syntax_broken:
            # Здесь и проходит граница между «неидеально» и «нельзя
            # публиковать»: файл, который не парсится, — это не файл.
            logger.warning(
                "dev_engine: %s так и не парсится после %d правок, проект без него не соберётся",
                file_spec.path, rounds,
            )
            return None

        return GeneratedFile(
            path=file_spec.path,
            content=source,
            fix_rounds=rounds,
            unresolved_diagnostics=report.render(),
        )

    async def _ask_for_spec(self, idea: str, *, context: str, previous_problem: str) -> _SpecAttempt:
        user_parts = []
        if idea.strip():
            user_parts.append(f"Замысел, о котором уже договорились: {idea.strip()}")
        if context.strip():
            user_parts.append(f"Чем ты сейчас живёшь и что тебе интересно: {context.strip()}")
        if previous_problem:
            user_parts.append(_retry_hint(previous_problem))
        if not user_parts:
            user_parts.append("Придумай себе следующий проект.")

        params = LLMParams(
            model="", system_prompt=_SPEC_SYSTEM_PROMPT, max_output_tokens=_SPEC_MAX_OUTPUT_TOKENS
        )
        session = Session(messages=[Message(role=Role.USER, content="\n\n".join(user_parts))])
        try:
            response = await self._router.chat(self._design_role, params, session)
        except LLMError as exc:
            # Провайдер, который лёг, к следующей попытке не встанет: повтор
            # здесь — это лишний запрос и то же самое сообщение об ошибке.
            logger.warning("dev_engine: не удалось получить спеку: %s", exc)
            return _SpecAttempt(problem=f"модель замысла недоступна: {exc}", retriable=False)

        if response.was_truncated:
            # Оборванный JSON не разбирается в принципе, и «невалидный JSON»
            # как причина увело бы куда угодно, кроме настоящей: ответ просто
            # не поместился в лимит. Повторять с той же просьбой смысла нет —
            # повтор идёт с прямым указанием писать короче (см. _retry_hint).
            logger.warning(
                "dev_engine: ответ с замыслом оборвался по лимиту (%s токенов)", _SPEC_MAX_OUTPUT_TOKENS
            )
            return _SpecAttempt(problem=_TRUNCATED_PROBLEM)
        return _SpecAttempt(text=response.text)


def parse_spec(raw: str) -> tuple[ProjectSpec | None, str]:
    """
    Разбирает ответ модели в ProjectSpec. Чистая функция — все форматные
    причуды провайдеров проверяются тестами без сети (тот же приём, что у
    efi.memory.parser.parse_payload).

    Возвращает (спека, причина отказа); при успехе причина пустая.
    """
    text = (raw or "").strip()
    if not text:
        return None, "пустой ответ модели"

    fenced = _JSON_FENCE_RE.search(text)
    if fenced is not None:
        text = fenced.group("body").strip()

    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None, "в ответе нет объекта JSON"

    try:
        payload = json.loads(text[start : end + 1])
    except json.JSONDecodeError as exc:
        return None, f"невалидный JSON: {exc}"
    if not isinstance(payload, dict):
        return None, f"ожидался объект, пришёл {type(payload).__name__}"

    # Негодные элементы структуры (файл без пути, путь с «..») выбрасываются
    # поштучно, а не роняют всю спеку: терять замысел целиком из-за одной
    # кривой строчки — худший из возможных обменов.
    payload = _normalize_payload(payload)
    payload["files"] = _valid_files(payload.get("files"))
    payload["stack"] = [str(item).strip() for item in _as_list(payload.get("stack")) if str(item).strip()]

    try:
        return ProjectSpec.model_validate(payload), ""
    except ValidationError as exc:
        return None, f"спека не прошла валидацию: {exc.errors()[0].get('msg', exc)}"


#: Как модели называют одни и те же поля. Требовать ровно наших имён — значит
#: выбрасывать вполне годный замысел из-за того, что модель написала
#: "description" вместо "problem": на бесплатных тирах это происходит
#: постоянно, а стоит ошибка целого проекта.
_FIELD_ALIASES: dict[str, tuple[str, ...]] = {
    "slug": ("slug", "repo", "repository", "name"),
    "title": ("title", "name", "project", "project_name"),
    "problem": ("problem", "description", "why", "purpose", "idea", "summary"),
    "stack": ("stack", "tech", "technologies", "dependencies"),
    "files": ("files", "structure", "modules"),
}

#: То же для полей одного файла.
_PATH_ALIASES = ("path", "file", "filename", "name")
_PURPOSE_ALIASES = ("purpose", "description", "role", "what", "summary")


def _first_present(payload: dict[str, object], names: tuple[str, ...]) -> object:
    for name in names:
        value = payload.get(name)
        if value not in (None, "", [], {}):
            return value
    return None


def _normalize_payload(payload: dict[str, object]) -> dict[str, object]:
    """Приводит ответ модели к нашим именам полей — см. _FIELD_ALIASES."""
    normalized: dict[str, object] = {}
    for field_name, aliases in _FIELD_ALIASES.items():
        value = _first_present(payload, aliases)
        if value is not None:
            normalized[field_name] = value
    # Название и имя репозитория взаимозаменяемы: из названия получается slug,
    # из slug — сносное название. Требовать оба — терять спеку на ровном месте.
    if "slug" not in normalized and "title" in normalized:
        normalized["slug"] = normalized["title"]
    if "title" not in normalized and "slug" in normalized:
        normalized["title"] = str(normalized["slug"]).replace("-", " ").strip().capitalize()
    return normalized


def _valid_files(raw: object) -> list[dict[str, str]]:
    """
    Файлы спеки из чего угодно, похожего на список файлов.

    Модели отвечают тремя способами: списком объектов (как просили), списком
    строк-путей и словарём «путь -> назначение». Принимать только первый —
    значит регулярно получать спеку без единого файла и отвергать её как
    «без кода», хотя замысел был нормальный.
    """
    items: list[object]
    if isinstance(raw, dict):
        items = [{"path": key, "purpose": value} for key, value in raw.items()]
    else:
        items = _as_list(raw)

    files: list[dict[str, str]] = []
    for item in items:
        if isinstance(item, str):
            candidate = {"path": item, "purpose": ""}
        elif isinstance(item, dict):
            candidate = {
                "path": str(_first_present(item, _PATH_ALIASES) or ""),
                "purpose": str(_first_present(item, _PURPOSE_ALIASES) or ""),
            }
        else:
            continue

        try:
            file_spec = FileSpec.model_validate(candidate)
        except ValidationError:
            logger.debug("dev_engine: пропускаю файл спеки с негодным путём: %r", item)
            continue
        files.append({"path": file_spec.path, "purpose": file_spec.purpose})
    return files


def _as_list(raw: object) -> list[object]:
    if isinstance(raw, list):
        return list(raw)
    if raw is None or raw == "":
        return []
    return [raw]


__all__ = ["BuildResult", "DevEngine", "parse_spec"]
