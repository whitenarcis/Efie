"""
efi/dev/readme.py

README как обязательное условие публикации, а не как приятное дополнение.

Репозиторий без README — это не «проект, которому не хватает документации»,
а проект, которым невозможно воспользоваться: человек, открывший ссылку, не
понимает ни что это, ни как запустить, и закрывает вкладку. Поэтому здесь не
шаблон «# Название\\n\\nописание», а проверяемый минимум:

    Назначение   — что это и какую проблему решает;
    Установка    — как поставить, включая требования;
    Использование — как запустить, с примером команды;
    Структура    — из чего состоит, чтобы можно было залезть внутрь.

Порядок работы: сначала документацию пишет кодер (он единственный видел
реальный код и знает, какие там флаги и функции), затем текст ПРОВЕРЯЕТСЯ на
наличие всех разделов, и если чего-то нет — один запрос на доработку. Если и
это не помогло, README собирается детерминированно из спеки и списка файлов:
он будет суше, но все обязательные разделы в нём есть по построению.

Так «обязательный критерий» становится свойством кода, а не пожеланием в
промпте: у сгенерированного README есть только два исхода — годный или
заменённый на годный.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Protocol

from efi.dev.schemas import GeneratedFile, ProjectSpec

logger = logging.getLogger(__name__)

README_PATH = "README.md"

#: Минимальная длина README, ниже которой разговаривать не о чем: столько
#: занимает один заголовок с парой строк — то есть заглушка.
_MIN_README_LENGTH = 400


@dataclass(slots=True, frozen=True)
class ReadmeSection:
    """Обязательный раздел: как называется в требованиях и по каким словам узнаётся в тексте."""

    name: str
    #: Слова, любое из которых означает, что раздел есть. Несколько вариантов,
    #: потому что модель пишет то «Установка», то «Как поставить», то
    #: «Installation» — требовать дословный заголовок значило бы забраковывать
    #: годный текст из-за синонима.
    markers: tuple[str, ...]

    def is_present(self, lowered: str) -> bool:
        return any(marker in lowered for marker in self.markers)


REQUIRED_SECTIONS: tuple[ReadmeSection, ...] = (
    ReadmeSection(
        "Назначение",
        ("назначение", "что это", "зачем", "о проекте", "проблема", "описание", "about", "overview"),
    ),
    ReadmeSection(
        "Установка",
        ("установка", "как поставить", "требования", "зависимости", "install", "requirements"),
    ),
    ReadmeSection(
        "Использование",
        ("использование", "запуск", "как пользоваться", "примеры", "usage", "quick start", "быстрый старт"),
    ),
    ReadmeSection(
        "Структура",
        ("структура", "устройство", "как устроено", "модули", "файлы", "structure", "layout"),
    ),
)

_README_SYSTEM_PROMPT = (
    "Ты пишешь README.md для маленькой утилиты, которую сам(а) только что написал(а). Пиши по-русски, "
    "по делу, без маркетинга, без эмодзи и без фраз вида «этот замечательный инструмент».\n"
    "ОБЯЗАТЕЛЬНЫЕ разделы (именно такими заголовками, в этом порядке):\n"
    "  # Название\n"
    "  краткое описание одной-двумя строками\n"
    "  ## Назначение — какую конкретную проблему решает и кому нужно\n"
    "  ## Установка — требования (версия Python, зависимости) и точные команды\n"
    "  ## Использование — как запускать, с РЕАЛЬНЫМИ примерами команд и флагов из кода\n"
    "  ## Структура — какой файл за что отвечает\n"
    "Опирайся только на реальный код, который тебе показали: не выдумывай флаги, функции и "
    "возможности, которых в нём нет. Ответ — только markdown, без ```-обёртки вокруг всего текста."
)

#: Заголовок markdown любого уровня — по нему считаем, что текст размечен, а
#: не свален одним абзацем.
_HEADING_RE = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)


def missing_sections(text: str) -> list[str]:
    """
    Каких обязательных разделов не хватает. Пустой список = все разделы есть.

    Чистая функция: проверка обязательности не должна зависеть ни от сети, ни
    от того, кто именно писал текст.
    """
    lowered = (text or "").strip().lower()
    if not lowered:
        return [section.name for section in REQUIRED_SECTIONS]
    return [section.name for section in REQUIRED_SECTIONS if not section.is_present(lowered)]


def problems(text: str) -> list[str]:
    """
    Все претензии к README человеческими словами — то, что уйдёт обратно
    модели и в лог.

    Длина и разметка проверяются ОТДЕЛЬНО от разделов, а не сваливаются в
    «не хватает всего»: README на три строки, где формально упомянуты все
    четыре темы, — это заглушка, и сказать об этом надо именно так, иначе в
    логе и в запросе на доработку окажется неправда про отсутствующие
    разделы.
    """
    body = (text or "").strip()
    if not body:
        return ["README пустой"]

    issues: list[str] = []
    if len(body) < _MIN_README_LENGTH:
        issues.append(f"слишком короткий ({len(body)} символов, нужно хотя бы {_MIN_README_LENGTH})")
    if not _HEADING_RE.search(body):
        issues.append("нет ни одного заголовка markdown")
    gaps = missing_sections(body)
    if gaps:
        issues.append(f"нет разделов: {', '.join(gaps)}")
    return issues


def is_acceptable(text: str) -> bool:
    return not problems(text)


class DocumentAuthor(Protocol):
    """
    Кто пишет текстовые файлы проекта. Реализация —
    efi.dev.qwen_client.QwenCoderClient.write_document.
    """

    async def write_document(self, path: str, *, system_prompt: str, request: str) -> str | None: ...


class ReadmeWriter:
    """
    Пишет README руками кодера и следит, чтобы он прошёл проверку.

    Кодер, а не разговорная модель: README про то, как запускать конкретный
    код с конкретными флагами, и единственный, кто этот код видел, — тот, кто
    его написал.
    """

    def __init__(self, coder: DocumentAuthor, *, max_repair_rounds: int = 1) -> None:
        self._coder = coder
        self._max_repair_rounds = max_repair_rounds

    async def write(self, spec: ProjectSpec, files: list[GeneratedFile]) -> GeneratedFile:
        """
        Возвращает готовый README.md. Всегда — годный: если модель не
        справилась, отдаётся детерминированная сборка (см. render_fallback).
        """
        text = await self._ask(spec, files, issues=[])
        for _ in range(self._max_repair_rounds):
            if text is not None and is_acceptable(text):
                break
            issues = problems(text or "")
            logger.info("readme: %s — %s; прошу переписать", spec.slug, "; ".join(issues))
            text = await self._ask(spec, files, issues=issues, previous=text or "")

        if text is None or not is_acceptable(text):
            logger.warning(
                "readme: %s — модель не выдала годный README, собираю из спеки", spec.slug
            )
            text = render_fallback(spec, files)

        return GeneratedFile(path=README_PATH, content=text.strip() + "\n")

    async def _ask(
        self, spec: ProjectSpec, files: list[GeneratedFile], *, issues: list[str], previous: str = ""
    ) -> str | None:
        request = _render_request(spec, files)
        if issues:
            request += (
                f"\n\nПрошлый вариант не годится: {'; '.join(issues)}. "
                "Перепиши README целиком, со всеми обязательными разделами и по-человечески подробно.\n\n"
                f"Прошлый вариант:\n{previous[:2000]}"
            )
        return await self._coder.write_document(
            README_PATH, system_prompt=_README_SYSTEM_PROMPT, request=request
        )


def render_fallback(spec: ProjectSpec, files: list[GeneratedFile]) -> str:
    """
    README из того, что известно точно: спека, список файлов и найденная в
    коде точка входа.

    Суше, чем написанный моделью, но честный и полный: все обязательные
    разделы на месте, команды выведены из реальных путей, а не выдуманы.
    """
    stack = "\n".join(f"- {item}" for item in spec.stack) or "- Python 3.11+"
    structure = "\n".join(
        f"- `{item.path}` — {item.purpose}" for item in spec.files if item.purpose
    ) or "\n".join(f"- `{item.path}`" for item in spec.files)

    entrypoint = _find_entrypoint(files)
    run_command = f"python {entrypoint}" if entrypoint else "python -m project"
    requirements = "requirements.txt" if _has_requirements(files) else ""
    install_lines = [
        "```bash",
        f"git clone https://github.com/<owner>/{spec.slug}.git",
        f"cd {spec.slug}",
    ]
    if requirements:
        install_lines.append(f"pip install -r {requirements}")
    install_lines.append("```")

    return (
        f"# {spec.title}\n\n"
        f"{spec.problem.strip()}\n\n"
        "## Назначение\n\n"
        f"{spec.problem.strip()}\n\n"
        "## Установка\n\n"
        "Требования:\n\n"
        f"{stack}\n\n"
        + "\n".join(install_lines)
        + "\n\n## Использование\n\n"
        "```bash\n"
        f"{run_command}\n"
        "```\n\n"
        f"Скрипт запускается из корня репозитория; параметры смотрите в `{entrypoint or 'исходниках'}`.\n\n"
        "## Структура\n\n"
        f"{structure}\n"
    )


def _find_entrypoint(files: list[GeneratedFile]) -> str:
    """Файл, который запускают. Ищем по имени, а не по содержимому: имя здесь надёжнее эвристик по коду."""
    for suffix in ("__main__.py", "main.py", "cli.py", "app.py"):
        for item in files:
            if item.path.endswith(suffix):
                return item.path
    python_files = [item.path for item in files if item.path.endswith(".py")]
    return python_files[0] if python_files else ""


def _has_requirements(files: list[GeneratedFile]) -> bool:
    return any(item.path.endswith(("requirements.txt", "pyproject.toml")) for item in files)


#: Сколько символов кода одного файла показывать документатору. README пишется
#: по интерфейсу и флагам, а они всегда в начале файла; полные тексты вытеснили
#: бы из контекста саму задачу.
_MAX_FILE_PREVIEW = 1500


def _render_request(spec: ProjectSpec, files: list[GeneratedFile]) -> str:
    sources = "\n\n".join(
        f"### {item.path}\n{item.content[:_MAX_FILE_PREVIEW]}"
        for item in files
        if item.path.endswith(".py")
    )
    stack = ", ".join(spec.stack) if spec.stack else "python 3.11"
    return (
        f"Проект «{spec.title}» (репозиторий {spec.slug}).\n"
        f"Решает: {spec.problem}\n"
        f"Стек: {stack}\n\n"
        f"Код проекта:\n{sources}"
    )


__all__ = [
    "README_PATH",
    "DocumentAuthor",
    "REQUIRED_SECTIONS",
    "ReadmeSection",
    "ReadmeWriter",
    "is_acceptable",
    "missing_sections",
    "problems",
    "render_fallback",
]
