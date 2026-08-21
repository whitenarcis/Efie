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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

from pydantic import ValidationError

from efi.config.schema import TaskRole
from efi.dev.imports import ImportProblem, cross_file_problems, render_requirements
from efi.dev.qwen_client import QwenCoderClient
from efi.dev.readme import README_PATH, ReadmeWriter
from efi.dev.sandbox import CodeSandbox, SandboxReport, salvage_python
from efi.dev.schemas import MAX_PROJECT_FILES, FileSpec, GeneratedFile, ProjectSpec
from efi.dev.showcase import significant_tokens
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

#: То же самое, но для файла с кодом. Уходит кодеру вместо голого
#: SyntaxError: по «почини синтаксис» он допишет тот же длинный файл и снова
#: не поместится, потратив все круги правок на один и тот же обрыв.
_REWRITE_HINT = (
    "ВАЖНО: прошлая попытка этого файла оказалась нерабочей — оборвалась по лимиту вывода или не "
    "разбиралась как Python. Напиши его заново и КОМПАКТНЕЕ: тот же публичный интерфейс, короткие "
    "docstring'и, без примеров использования в комментариях и без длинных таблиц данных в коде."
)

_TRUNCATED_FILE_HINT = (
    "Файл оборвался: ответ не поместился в лимит вывода. Напиши его ЗАНОВО и КОРОЧЕ — тот же "
    "публичный интерфейс, но компактнее: без длинных docstring'ов, без примеров использования "
    "в комментариях, без вынесенных в код таблиц данных."
)


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
    #: обслуживания модель, отвергнутый ключ, несходящиеся импорты. Дословная
    #: причина — это то, по чему владелец найдёт беду за минуту, а не за вечер.
    failure_reason: str = ""
    #: Повторять бессмысленно: причина не в удаче, а в конфигурации. Снятая с
    #: обслуживания модель к следующему часу не вернётся, а вот кодер, который
    #: разошёлся с собственным замыслом, со второй попытки часто сходится —
    #: разница ровно в этом флаге (см. efi/dev/worker.py).
    permanent: bool = False

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

    async def design(
        self, idea: str = "", *, context: str = "", built: Sequence[ProjectSpec] = ()
    ) -> tuple[ProjectSpec | None, str]:
        """
        Спека проекта и — если не вышло — ПРИЧИНА, по которой не вышло.

        `idea` — если проект заказан (совместная задача или собственная затея
        с конкретной темой); пусто — придумывает сама. `context` — чем Эфи
        сейчас живёт (интересы, недавние темы): из этого получаются проекты
        «про её жизнь», а не случайные утилиты из воздуха.

        `built` — что уже написано. Без этого списка модель раз за разом
        придумывает то же самое: интересы меняются медленно, а «придумай себе
        проект» на одном и том же контексте даёт один и тот же ответ. Стоит
        это не только скуки: имя репозитория занято, и пуш второго такого
        проекта отклоняется как непустая история.

        Причина возвращается наружу, а не остаётся в логе, потому что снаружи
        все отказы выглядели одинаково — «не придумалось ничего, что стоило бы
        писать». Под этой фразой одинаково прятались битый JSON, обрыв ответа
        по лимиту и настоящий отказ от учебной идеи, а чинятся они совершенно
        по-разному.
        """
        problems: list[str] = []
        for attempt in range(1, _MAX_SPEC_ATTEMPTS + 1):
            answer = await self._ask_for_spec(
                idea, context=context, built=built, previous_problem=problems[-1] if problems else ""
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
            repeat = _find_repeat(spec, built)
            if repeat is not None:
                logger.info(
                    "dev_engine: замысел «%s» повторяет уже написанный «%s»", spec.title, repeat.title
                )
                problems.append(f"замысел «{spec.title}» повторяет уже написанный «{repeat.title}»")
                continue

            logger.info(
                "dev_engine: замысел «%s» (%s), файлов: %d", spec.title, spec.slug, len(spec.files)
            )
            return spec, ""

        reason = "; ".join(dict.fromkeys(problems)) or "модель не выдала ничего пригодного"
        logger.info("dev_engine: за %d попыток не вышло годной спеки: %s", _MAX_SPEC_ATTEMPTS, reason)
        return None, reason

    async def build(self, spec: ProjectSpec, *, existing: Mapping[str, str] | None = None) -> BuildResult:
        """
        Пишет все файлы спеки, сводит их друг с другом и укомплектовывает
        проект тем, что есть у любого живого репозитория.

        `existing` — файлы, написанные в ПРОШЛЫЙ заход по этой же задаче
        (efi/dev/worker.py хранит их в задаче). Заново они не пишутся: если в
        прошлый раз проект развалился на четвёртом файле из-за лимита, то
        переписывать первые три — это и лишние запросы к тому же лимиту, и
        новый шанс разойтись с тем, что уже сходилось.

        Порядок написания — не порядок спеки: модули идут первыми, точка
        входа последней (см. `_writing_order`). Причина в том, что кодер
        видит интерфейсы только УЖЕ написанных файлов: main.py, написанный
        первым, выдумывает функции парсера, а парсер потом пишется со своими
        именами — и проект, у которого каждый файл по отдельности безупречен,
        не запускается вовсе.

        README пишется ПОСЛЕДНИМ и всегда: он документирует то, что реально
        получилось, а не то, что задумывалось, — и без него проект не
        публикуется вовсе (см. efi/dev/readme.py).
        """
        written: dict[str, str] = {}
        files: list[GeneratedFile] = []
        broken: list[str] = []
        reused = dict(existing or {})

        for file_spec in _writing_order(spec.files[:MAX_PROJECT_FILES]):
            carried = reused.get(file_spec.path)
            if carried is not None:
                logger.info("dev_engine: %s остался с прошлого захода, не переписываю", file_spec.path)
                written[file_spec.path] = carried
                files.append(GeneratedFile(path=file_spec.path, content=carried))
                continue
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
                    return BuildResult(
                        files=[], broken_paths=broken, failure_reason=unavailable, permanent=True
                    )
                continue
            written[generated.path] = generated.content
            files.append(generated)

        # Свой README из спеки, если кодер зачем-то сгенерировал его сам,
        # выбрасываем: документация по замыслу вместо документации по коду —
        # это ровно тот README, ради которого никто не открывает репозиторий.
        code_files = [item for item in files if item.path.lower() != README_PATH.lower()]
        if not code_files:
            return BuildResult(files=[], broken_paths=broken)

        code_files, unresolved_imports = await self._reconcile_imports(spec, code_files)
        if unresolved_imports:
            # Проект, который падает ImportError'ом на первой строке, — это не
            # «с замечаниями», это неработающий проект. Выкладывать такой под
            # своим именем незачем.
            logger.warning("dev_engine: %s не сходится по импортам: %s", spec.slug, unresolved_imports[0])
            return BuildResult(
                files=code_files,
                broken_paths=[*broken, *sorted({item.path for item in unresolved_imports})],
                failure_reason=unresolved_imports[0].message,
            )

        code_files.append(await self._readme.write(spec, code_files))
        code_files.extend(_scaffolding_files(spec))
        return BuildResult(files=code_files, broken_paths=broken)

    async def _reconcile_imports(
        self, spec: ProjectSpec, files: list[GeneratedFile]
    ) -> tuple[list[GeneratedFile], list[ImportProblem]]:
        """
        Сводит файлы друг с другом: то, чего не видит ни компилятор, ни линтер
        по одному файлу (см. efi/dev/imports.py).

        Найденное уходит кодеру теми же словами, что и замечания песочницы, —
        и повторяется, пока не сойдётся или пока не кончатся круги правок:
        одна правка часто рождает следующее расхождение.
        """
        by_path = {item.path: item for item in files}
        for _round in range(self._max_fix_iterations):
            problems = cross_file_problems(
                {path: item.content for path, item in by_path.items()}, stack=spec.stack
            )
            if not problems:
                return list(by_path.values()), []

            fixed_anything = False
            for path in sorted({item.path for item in problems}):
                diagnostics = "\n".join(item.message for item in problems if item.path == path)
                current = by_path[path]
                logger.info("dev_engine: %s не сходится с соседями: %s", path, diagnostics.split("\n")[0])
                repaired = await self._coder.fix_file(path, current.content, diagnostics)
                if repaired is None or repaired.strip() == current.content.strip():
                    continue
                report = await self._sandbox.check(path, repaired)
                if report.syntax_broken:
                    continue  # правка хуже болезни: до неё файл хотя бы парсился
                fixed_anything = True
                by_path[path] = current.model_copy(
                    update={
                        "content": repaired,
                        "fix_rounds": current.fix_rounds + 1,
                        "unresolved_diagnostics": report.render(),
                    }
                )
            if not fixed_anything:
                break

        remaining = cross_file_problems(
            {path: item.content for path, item in by_path.items()}, stack=spec.stack
        )
        for problem in remaining:
            if problem.fatal:
                continue
            # Несмертельное расхождение (неизвестный модуль, который может
            # оказаться настоящим пакетом) едет в репозиторий как замечание:
            # это материал и для реплики в чат, и для будущей ревизии.
            current = by_path[problem.path]
            by_path[problem.path] = current.model_copy(
                update={
                    "unresolved_diagnostics": "\n".join(
                        filter(None, [current.unresolved_diagnostics, problem.message])
                    )
                }
            )
        return list(by_path.values()), [item for item in remaining if item.fatal]

    async def _write_one(
        self, spec: ProjectSpec, file_spec: FileSpec, already_written: dict[str, str]
    ) -> GeneratedFile | None:
        source = await self._coder.write_file(spec, file_spec, already_written=already_written)
        if source is None:
            return None
        # Файл, оборванный по лимиту вывода, чинится не «исправь синтаксис»:
        # кодер честно допишет ту же функцию и упрётся в тот же лимит. Ему
        # нужно сказать, что случилось на самом деле.
        truncated_hint = _TRUNCATED_FILE_HINT if self._coder.last_answer_truncated else ""

        report = await self._sandbox.check(file_spec.path, source)
        rounds = 0
        while not report.ok and rounds < self._max_fix_iterations:
            rounds += 1
            logger.info(
                "dev_engine: %s — правка %d/%d по замечаниям: %s",
                file_spec.path, rounds, self._max_fix_iterations, report.render().replace("\n", "; ")[:160],
            )
            diagnostics = "\n".join(filter(None, [truncated_hint, report.render()]))
            truncated_hint = ""
            fixed = await self._coder.fix_file(file_spec.path, source, diagnostics)
            if fixed is None:
                break
            source = fixed
            report = await self._sandbox.check(file_spec.path, source)

        if report.syntax_broken:
            rescued = await self._rescue(spec, file_spec, source, already_written)
            if rescued is None:
                # Здесь и проходит граница между «неидеально» и «нельзя
                # публиковать»: файл, который не парсится, — это не файл.
                logger.warning(
                    "dev_engine: %s так и не парсится после %d правок и переписывания заново, "
                    "проект без него не соберётся",
                    file_spec.path, rounds,
                )
                return None
            source, report = rescued
            rounds += 1

        return GeneratedFile(
            path=file_spec.path,
            content=source,
            fix_rounds=rounds,
            unresolved_diagnostics=report.render(),
        )

    async def _rescue(
        self,
        spec: ProjectSpec,
        file_spec: FileSpec,
        broken_source: str,
        already_written: dict[str, str],
    ) -> tuple[str, SandboxReport] | None:
        """
        Последняя попытка спасти файл, который не парсится: сначала написать
        его ЗАНОВО, потом — отрезать оборванный хвост.

        Почему заново, а не ещё одна правка. Круг исправлений просит кодера
        починить сломанный текст, и когда текст сломан обрывом посреди
        функции, кодер честно дописывает ту же функцию — и упирается в тот же
        лимит. Просьба написать компактнее с нуля рвёт этот круг.

        Почему обрезка. Всё, что выше обрыва, — нормальный рабочий код, и
        терять из-за одной незавершённой функции в конце весь проект, где
        остальные файлы уже написаны, — худший из возможных обменов. Огрызок
        при этом не выдаётся за целый файл: если он не отдаёт того, что
        импортируют соседи, проект всё равно не соберётся (см.
        `_reconcile_imports`).
        """
        fresh = await self._coder.write_file(
            spec, file_spec, already_written=already_written, hint=_REWRITE_HINT
        )
        if fresh is not None:
            report = await self._sandbox.check(file_spec.path, fresh)
            if not report.syntax_broken:
                logger.info("dev_engine: %s переписан заново и на этот раз парсится", file_spec.path)
                return fresh, report

        salvaged = salvage_python(fresh or broken_source)
        if not salvaged:
            return None
        logger.info(
            "dev_engine: %s спасён обрезкой оборванного хвоста (%d строк из %d)",
            file_spec.path, len(salvaged.splitlines()), len((fresh or broken_source).splitlines()),
        )
        return salvaged, await self._sandbox.check(file_spec.path, salvaged)

    async def _ask_for_spec(
        self, idea: str, *, context: str, built: Sequence[ProjectSpec], previous_problem: str
    ) -> _SpecAttempt:
        user_parts = []
        if idea.strip():
            user_parts.append(f"Замысел, о котором уже договорились: {idea.strip()}")
        if context.strip():
            user_parts.append(f"Чем ты сейчас живёшь и что тебе интересно: {context.strip()}")
        if built:
            written = "\n".join(f"- {item.render_for_prompt()}" for item in built[:_MAX_BUILT_SHOWN])
            user_parts.append(
                f"Это ты уже написала — НЕ повторяйся ни темой, ни именем репозитория:\n{written}"
            )
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


#: Сколько уже написанных проектов показывать модели. Список нужен, чтобы не
#: повторяться, а не чтобы занять им весь промпт: десяток строк хватает,
#: дальше начинается пересказ портфолио вместо задания.
_MAX_BUILT_SHOWN = 10

#: С какой доли общих слов замысел считается повтором уже написанного. Порог
#: тот же, что у показа проекта в чужом разговоре (efi/dev/showcase.py): «оба
#: про логи» — совпадение, «оба на питоне» — нет, стоп-слова не в счёт.
_REPEAT_SCORE = 0.6


def _find_repeat(spec: ProjectSpec, built: Sequence[ProjectSpec]) -> ProjectSpec | None:
    """
    Не придумала ли она заново то, что уже написала.

    Проверяется и имя репозитория, и суть: одинаковый slug — это ещё и
    сорванный пуш (в непустой репозиторий история не заезжает), а одинаковая
    суть под новым именем — второй такой же проект в профиле, по которому
    видно, что автор себя не помнит.
    """
    subject = significant_tokens(f"{spec.title} {spec.problem}")
    for other in built:
        if other.slug == spec.slug:
            return other
        if not subject:
            continue
        overlap = subject & significant_tokens(f"{other.title} {other.problem}")
        if len(overlap) / len(subject) >= _REPEAT_SCORE:
            return other
    return None


#: Имена, по которым файл узнаётся как точка входа. Он пишется последним:
#: точке входа нужны чужие интерфейсы, а её собственный не нужен никому.
_ENTRYPOINT_SUFFIXES = ("main.py", "cli.py", "__main__.py", "app.py")


def _writing_order(files: list[FileSpec]) -> list[FileSpec]:
    """Модули вперёд, точка входа в конец — при устойчивом порядке внутри групп."""
    modules = [item for item in files if not item.path.lower().endswith(_ENTRYPOINT_SUFFIXES)]
    entrypoints = [item for item in files if item.path.lower().endswith(_ENTRYPOINT_SUFFIXES)]
    return [*modules, *entrypoints]


#: .gitignore проекта на Python. Не «на всякий случай»: без него первый же
#: запуск оставляет __pycache__, и репозиторий, в который никто не заглядывал
#: после релиза, выглядит именно так, как и есть.
_GITIGNORE = (
    "__pycache__/\n*.py[cod]\n*.egg-info/\n.venv/\nvenv/\n.env\n.ruff_cache/\n"
    ".pytest_cache/\n.mypy_cache/\n"
)


def _scaffolding_files(spec: ProjectSpec) -> list[GeneratedFile]:
    """
    Обвязка репозитория, которую не надо сочинять: .gitignore всегда,
    requirements.txt — только если в стеке действительно есть чужие пакеты.

    Пишется детерминированно, а не кодером: это не творческая задача, а
    разница между «сгенерированной папкой с файлами» и репозиторием, который
    не стыдно открыть. Пустой requirements.txt при этом хуже отсутствующего —
    он сообщает читателю ровно ничего.
    """
    files = [GeneratedFile(path=".gitignore", content=_GITIGNORE)]
    requirements = render_requirements(spec.stack)
    if requirements:
        files.append(GeneratedFile(path="requirements.txt", content=requirements))
    return files


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
