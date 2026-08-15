"""
efi/dev/qwen_client.py

Кодер: Qwen Coder через Groq. Отдельный клиент, а не ещё одна роль в
efi.llm.router.LLMRouter — и вот почему.

Роли роутера (MAIN/FAST/BACKGROUND/VISION) описывают, КТО говорит от лица
Эфи и с какой срочностью. Кодер не говорит от её лица вообще: он не знает ни
про личность, ни про собеседника, ни про историю чата, и его ответ — не
реплика, а содержимое файла. Подмешать его в общую ротацию значило бы, что
при недоступности Qwen на его место молча встанет разговорная модель и
начнёт писать код (или наоборот — кодер ответит собеседнику), а это ровно
тот класс отказа, который потом ищут неделю.

Поэтому здесь свой узкий контракт из двух методов — «напиши файл» и «почини
файл по замечаниям» — поверх того же OpenAI-совместимого провайдера, что
используется везде (efi/llm/providers/openai_compatible.py): Groq именно
такой API и предоставляет.

Отдельная забота — очистка ответа. Модели, обученные на markdown, почти
всегда заворачивают код в ```-блок и дописывают «Вот ваш файл:». Записать
это на диск дословно — получить файл, который не парсится с первой строки,
причём с идеально выглядящим кодом внутри. `strip_code_fences` разбирает
этот случай явно и покрыт тестами: формат ответа модели — не мелочь, а
самая частая причина мусора на выходе.
"""

from __future__ import annotations

import logging
import re

from efi.config.schema import EndpointConfig
from efi.dev.schemas import FileSpec, ProjectSpec
from efi.llm.base import LLMProvider
from efi.llm.errors import LLMAuthError, LLMError, LLMRateLimitError, LLMServerError
from efi.llm.providers.openai_compatible import OpenAICompatibleProvider
from efi.llm.schemas import LLMParams, Message, Role, Session

logger = logging.getLogger(__name__)

#: Бюджет вывода на один файл. Код дороже прозы в токенах, а обрыв на
#: середине функции — это не «неполный ответ», а неработающий файл.
_FILE_MAX_OUTPUT_TOKENS = 4096

#: Температура кодера. Ниже разговорной: от кода нужна предсказуемость, а не
#: разнообразие формулировок.
_CODER_TEMPERATURE = 0.2

_SYSTEM_PROMPT = (
    "Ты — опытный Python-разработчик. Пишешь ТОЛЬКО код одного файла, без объяснений, без markdown, "
    "без ```-блоков и без комментариев вида 'вот ваш файл'. Первая строка ответа — первая строка файла.\n"
    "Требования к коду: Python 3.11+, аннотации типов, короткие docstring'и по делу, стандартная "
    "библиотека в приоритете, никаких заглушек вида 'TODO: реализовать' и никакого мёртвого кода. "
    "Файл должен быть самодостаточным и работоспособным ровно в том виде, в каком ты его написал."
)

_FIX_SYSTEM_PROMPT = (
    "Ты — опытный Python-разработчик. Тебе дают файл и замечания статического анализатора к нему. "
    "Верни ИСПРАВЛЕННЫЙ файл целиком: только код, без markdown, без ```-блоков, без пояснений. "
    "Исправляй ровно то, на что указали, не переписывая остальное и не меняя публичный интерфейс."
)

#: ```python ... ``` в начале и в конце ответа. Ловится и без указания языка,
#: и с любым языком; закрывающий блок необязателен — при обрыве генерации по
#: лимиту токенов его просто не будет.
_FENCE_OPEN_RE = re.compile(r"^\s*```[a-zA-Z0-9_+-]*\s*\n")
_FENCE_CLOSE_RE = re.compile(r"\n\s*```\s*$")

#: Ответ, начинающийся с прозы («Конечно! Вот реализация:»), у которой ниже
#: идёт ```-блок: берём блок, а прозу выкидываем.
_FENCED_BLOCK_RE = re.compile(r"```[a-zA-Z0-9_+-]*\s*\n(?P<code>.*?)(?:\n\s*```|\Z)", re.DOTALL)


class QwenCoderClient:
    """
    Тонкая обёртка над OpenAI-совместимым эндпоинтом кодера.

    Ни один метод не бросает наружу: сбой провайдера — это «файл не
    написался», а не исключение посреди фонового цикла. Вызывающая сторона
    (efi.dev.engine.DevEngine) на None реагирует отказом от проекта, и это
    единственная разумная реакция — без файла проекта нет.
    """

    def __init__(self, endpoint: EndpointConfig, *, provider: LLMProvider | None = None) -> None:
        self._endpoint = endpoint
        #: Последняя причина отказа — дословный ответ провайдера. Нужна, чтобы
        #: «файл не написался» не оставалось единственным, что известно
        #: наружу: имя снятой с обслуживания модели или отвергнутый ключ видны
        #: только здесь (см. `unavailable_reason`).
        self._last_error: LLMError | None = None
        #: Провайдер внедряем ради тестов; в бою — обычный OpenAI-совместимый
        #: клиент, тот же, что ходит в Groq для остальных ролей. Имя видно в
        #: логах и метриках (efi/dashboard/): запрос кодера должен быть
        #: отличим от разговорного, иначе непонятно, кто съел лимит.
        self._provider = (
            provider if provider is not None else OpenAICompatibleProvider(endpoint, name="coder")
        )

    @property
    def model(self) -> str:
        return self._endpoint.model

    @property
    def unavailable_reason(self) -> str:
        """
        Почему кодера бессмысленно звать дальше — или пусто, если причин нет.

        Отличать НЕПОПРАВИМЫЙ отказ от временного здесь важнее, чем кажется.
        Модель, снятая с обслуживания (404 `model_decommissioned` у Groq —
        рядовое событие на бесплатных тирах), и отвергнутый ключ (401) не
        починятся ни к следующему файлу, ни к следующей правке: без этой
        проверки один такой конфиг стоил бы десятка запросов на каждый проект
        и заканчивался бы сообщением «кодер не написал ни одного файла», по
        которому причину не найти. Таймаут и 429, наоборот, поправимы сами
        собой — они сюда не попадают.
        """
        error = self._last_error
        if error is None:
            return ""
        if isinstance(error, LLMAuthError):
            return f"кодер отверг ключ: {error}"
        # LLMTimeoutError наследует LLMServerError — оба транзиентные, как и 429.
        if isinstance(error, LLMRateLimitError | LLMServerError):
            return ""
        status = error.status_code
        if status is not None and 400 <= status < 500:
            return f"модель {self._endpoint.model!r} недоступна: {error}"
        return ""

    async def write_file(
        self, spec: ProjectSpec, file_spec: FileSpec, *, already_written: dict[str, str] | None = None
    ) -> str | None:
        """
        Пишет один файл проекта.

        `already_written` — уже готовые файлы этого же проекта. Они уходят в
        промпт урезанными до сигнатур (см. `_render_context`): без них кодер
        каждый раз выдумывает интерфейс соседнего модуля заново, и проект
        разваливается на несовместимые куски, каждый из которых по
        отдельности проходит проверку.
        """
        user_content = (
            f"{_render_project_brief(spec)}\n\n"
            f"Сейчас напиши файл `{file_spec.path}`.\n"
            f"Назначение файла: {file_spec.purpose or 'см. структуру проекта выше'}\n"
            f"{_render_context(already_written or {})}"
        )
        return await self._ask(_SYSTEM_PROMPT, user_content, what=f"write {file_spec.path}")

    async def write_document(self, path: str, *, system_prompt: str, request: str) -> str | None:
        """
        Текстовый файл проекта, а не код: README и подобное.

        Отдельный метод со своим системным промптом, потому что документацию
        пишут иначе, чем модуль: «только код, без markdown» — ровно то, чего
        от README не надо. Пишет его всё равно кодер: он единственный видел
        реальные флаги и функции, а README про них и есть (см. efi/dev/readme.py).
        """
        return await self._ask(system_prompt, request, what=f"write {path}")

    async def fix_file(self, path: str, source: str, diagnostics: str) -> str | None:
        """Переписывает файл по замечаниям песочницы (efi/dev/sandbox.py)."""
        user_content = (
            f"Файл `{path}`:\n{source}\n\n"
            f"Замечания статического анализатора:\n{diagnostics}\n\n"
            "Верни исправленный файл целиком."
        )
        return await self._ask(_FIX_SYSTEM_PROMPT, user_content, what=f"fix {path}")

    async def _ask(self, system_prompt: str, user_content: str, *, what: str) -> str | None:
        params = LLMParams(
            model=self._endpoint.model,
            system_prompt=system_prompt,
            max_output_tokens=_FILE_MAX_OUTPUT_TOKENS,
            temperature=_CODER_TEMPERATURE,
        )
        session = Session(messages=[Message(role=Role.USER, content=user_content)])
        try:
            response = await self._provider.chat(params, session)
        except LLMError as exc:
            self._last_error = exc
            logger.warning("qwen: %s не удалось (%s)", what, exc)
            return None

        self._last_error = None

        if response.was_truncated:
            # У кода обрыв по лимиту неисправим в принципе: «последнее
            # законченное предложение» (salvage_truncated) для прозы работает,
            # а для функции — нет, останется висящий блок. Но обрубок всё
            # равно отдаём: песочница на нём поднимет SyntaxError, и кодер
            # получит шанс дописать файл в цикле исправлений.
            logger.warning("qwen: %s упёрлось в лимит вывода (%s токенов)", what, _FILE_MAX_OUTPUT_TOKENS)

        code = strip_code_fences(response.text)
        if not code.strip():
            logger.warning("qwen: %s вернуло пустой ответ", what)
            return None
        return code


def strip_code_fences(raw: str) -> str:
    """
    Убирает markdown-обёртку из ответа модели.

    Три случая, все реальные:
      1. Ответ целиком в ```-блоке — снимаем открывающий и закрывающий забор.
      2. Проза, а ниже ```-блок — берём содержимое первого блока.
      3. Чистый код — возвращаем как есть.

    Чистая функция: формат ответа провайдера — самая частая причина мусора
    на диске, и проверяться он должен без сети.
    """
    text = (raw or "").strip()
    if not text:
        return ""

    if _FENCE_OPEN_RE.match(text):
        text = _FENCE_OPEN_RE.sub("", text, count=1)
        text = _FENCE_CLOSE_RE.sub("", text, count=1)
        return text.strip("\n")

    fenced = _FENCED_BLOCK_RE.search(text)
    if fenced is not None:
        return fenced.group("code").strip("\n")

    return text


def _render_project_brief(spec: ProjectSpec) -> str:
    stack = ", ".join(spec.stack) if spec.stack else "python 3.11, стандартная библиотека"
    structure = "\n".join(f"  - {item.path}: {item.purpose}" for item in spec.files)
    return (
        f"Проект «{spec.title}» ({spec.slug}).\n"
        f"Решает: {spec.problem}\n"
        f"Стек: {stack}\n"
        f"Структура:\n{structure}"
    )


def _render_context(already_written: dict[str, str]) -> str:
    """
    Уже написанные файлы — сигнатурами, а не целиком: кодеру нужно знать, что
    он может импортировать, а не перечитывать весь проект. Полные тексты
    съедали бы контекст тем быстрее, чем дальше зашёл проект, — то есть
    ломались бы ровно на больших проектах, где связность важнее всего.
    """
    if not already_written:
        return ""
    lines: list[str] = ["Уже написанные файлы проекта (используй их, не выдумывай другой интерфейс):"]
    for path, source in already_written.items():
        signatures = _extract_signatures(source)
        rendered = "\n".join(f"    {line}" for line in signatures) if signatures else "    (без публичных имён)"
        lines.append(f"  {path}:\n{rendered}")
    return "\n".join(lines)


#: Определения ВЕРХНЕГО УРОВНЯ (без отступа): их и импортируют из соседнего
#: модуля. Методы внутри класса под это не попадают намеренно — они часть
#: класса, который уже перечислен.
_SIGNATURE_RE = re.compile(r"^(?:async\s+def|def|class)\s+(?P<name>[A-Za-z_]\w*)[^\n]*:", re.MULTILINE)
_MAX_SIGNATURES_PER_FILE = 20


def _extract_signatures(source: str) -> list[str]:
    """Публичные имена файла — регуляркой, а не ast.parse: файл мог не пройти проверку и не парситься."""
    return [
        match.group(0).strip()
        for match in _SIGNATURE_RE.finditer(source)
        if not match.group("name").startswith("_")
    ][:_MAX_SIGNATURES_PER_FILE]


__all__ = ["QwenCoderClient", "strip_code_fences"]
