"""
efi/dev/sandbox.py

Проверка сгенерированного кода — и граница, которую подсистема не переходит.

КОД НИКОГДА НЕ ЗАПУСКАЕТСЯ. Ни здесь, ни где-либо ещё в efi/dev/. Проверка
статическая: разбор синтаксиса компилятором Python (`compile`, ровно то же,
что делает `py_compile`, но без записи .pyc) и, если он есть в системе,
`ruff check`. Это осознанная граница, а не недоделанная песочница.

Причина простая. Текст пишет языковая модель по замыслу другой языковой
модели, а процесс Эфи — это живой юзербот с сессией Telegram, ключами
провайдеров и SSH-ключом к GitHub в том же окружении. Между «проверить
код» и «выполнить код» здесь нет безопасного промежутка: любой `exec`
сгенерированного текста означал бы, что содержимое ответа LLM исполняется с
правами владельца аккаунта. Настоящее исполнение требует настоящей изоляции
(контейнер, отдельный пользователь, урезанный доступ к сети и ФС), которой у
телефона в Termux нет и не будет. Поэтому конвейер честно проверяет то, что
можно проверить не исполняя, и не делает вид, что тестирует.

Что это ловит на практике: обрыв генерации на середине функции,
несбалансированные скобки, смешанные отступы, `import` несуществующего
модуля стандартной библиотеки (F401/E999 у ruff), неиспользуемые имена,
битые f-строки. То есть ровно тот класс ошибок, который и порождает
пословная генерация кода.

`ruff` необязателен: если бинаря нет (обычное дело в Termux), проверка
деградирует до синтаксической и об этом сообщается один раз, а не на каждый
файл. Отсутствие линтера не должно останавливать конвейер.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

#: Сколько ждать ruff на один файл. Линтер локальный и быстрый; если он
#: завис — проблема не в коде, и держать из-за этого фоновый цикл незачем.
_RUFF_TIMEOUT_SECONDS = 20.0

#: Правила, по которым ruff гоняется на сгенерированном коде. Намеренно узкий
#: набор: E9 (синтаксис/ошибки выполнения парсера), F (pyflakes — настоящие
#: ошибки вроде неопределённых имён и неиспользуемых импортов). Стилистику
#: (длина строки, порядок импортов) здесь не проверяем: гонять кодера на
#: третий круг из-за пустой строки — трата запросов на бесплатном тире.
_RUFF_SELECT = "E9,F"

#: Сколько строк вывода линтера уходит обратно в модель. Кодеру нужен
#: конкретный список претензий, а не портянка: длинный вывод вытесняет из
#: контекста сам код, который надо чинить.
_MAX_DIAGNOSTIC_LINES = 25


@dataclass(slots=True, frozen=True)
class SandboxReport:
    """Результат проверки одного файла."""

    path: str
    #: Пусто = претензий нет.
    diagnostics: list[str] = field(default_factory=list)
    #: Нашлась ли синтаксическая ошибка (в отличие от замечаний линтера).
    #: Разница существенная: файл с битым синтаксисом бесполезен целиком, а
    #: файл с замечанием линтера работоспособен и в крайнем случае может
    #: уехать в репозиторий как есть.
    syntax_broken: bool = False

    @property
    def ok(self) -> bool:
        return not self.diagnostics

    def render(self) -> str:
        """Замечания одним текстом — ровно в том виде, в каком они уйдут кодеру."""
        return "\n".join(self.diagnostics[:_MAX_DIAGNOSTIC_LINES])


class CodeSandbox:
    """
    Статическая проверка файлов проекта.

    Экземпляр держит один флаг — сообщали ли уже об отсутствии ruff, — чтобы
    не повторять предупреждение на каждый файл каждого проекта.
    """

    def __init__(self, *, ruff_executable: str = "ruff", enable_linter: bool = True) -> None:
        self._ruff_executable = ruff_executable
        self._enable_linter = enable_linter
        self._linter_missing_reported = False

    async def check(self, path: str, source: str) -> SandboxReport:
        """
        Проверяет один файл. Не-Python файлы (README.md, .toml) проверять
        нечем — для них возвращается пустой отчёт: отсутствие проверки не
        повод считать файл плохим.
        """
        if not path.endswith(".py"):
            return SandboxReport(path=path)

        syntax_error = _check_syntax(path, source)
        if syntax_error is not None:
            # Линтер по коду, который не парсится, скажет ровно то же самое
            # и потратит на это процесс — синтаксис первичен.
            return SandboxReport(path=path, diagnostics=[syntax_error], syntax_broken=True)

        lints = await self._run_linter(path, source)
        return SandboxReport(path=path, diagnostics=lints)

    async def check_all(self, files: dict[str, str]) -> list[SandboxReport]:
        """Проверяет несколько файлов разом; отчёты возвращаются в порядке файлов."""
        return [await self.check(path, source) for path, source in files.items()]

    async def _run_linter(self, path: str, source: str) -> list[str]:
        if not self._enable_linter:
            return []

        try:
            process = await asyncio.create_subprocess_exec(
                self._ruff_executable,
                "check",
                "--no-cache",
                "--select",
                _RUFF_SELECT,
                "--output-format",
                "concise",
                "--stdin-filename",
                path,
                "-",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
        except (FileNotFoundError, NotADirectoryError, PermissionError):
            self._report_missing_linter()
            return []

        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(source.encode("utf-8")), timeout=_RUFF_TIMEOUT_SECONDS
            )
        except TimeoutError:
            process.kill()
            await process.wait()
            logger.warning("sandbox: ruff завис на %s, пропускаю линт этого файла", path)
            return []

        if process.returncode not in (0, 1):
            # 0 — чисто, 1 — есть замечания. Всё остальное это сам ruff
            # сломался (неизвестный флаг, битая установка), и претензии к
            # коду отсюда не следуют.
            logger.warning(
                "sandbox: ruff завершился с кодом %s на %s: %s",
                process.returncode, path, stderr.decode("utf-8", "replace").strip()[:200],
            )
            return []

        return [line.strip() for line in stdout.decode("utf-8", "replace").splitlines() if line.strip()]

    def _report_missing_linter(self) -> None:
        self._enable_linter = False
        if self._linter_missing_reported:
            return
        self._linter_missing_reported = True
        logger.info(
            "sandbox: ruff не найден (%r) — проверка кода остаётся синтаксической. "
            "Это рабочий режим, а не сбой: установите ruff, чтобы ловить ещё и "
            "неопределённые имена с неиспользуемыми импортами",
            self._ruff_executable,
        )


def _check_syntax(path: str, source: str) -> str | None:
    """
    Разбор файла компилятором Python. Ровно то же, что делает py_compile, но
    без записи .pyc и без импорта модуля: `compile()` парсит текст и НЕ
    исполняет его (см. докстринг модуля про то, почему это принципиально).
    """
    try:
        compile(source, path, "exec")
    except SyntaxError as exc:
        location = f"{path}:{exc.lineno or 0}:{exc.offset or 0}"
        return f"{location}: SyntaxError: {exc.msg}"
    except ValueError as exc:
        # Нулевые байты и прочий мусор в тексте: compile() отвечает на это
        # ValueError, а не SyntaxError.
        return f"{path}: невалидный исходник: {exc}"
    return None


def write_project_files(root: Path, files: dict[str, str]) -> list[Path]:
    """
    Раскладывает файлы проекта по диску под `root`.

    Пути уже провалидированы схемой (efi.dev.schemas.FileSpec), но проверка
    повторяется здесь же и по факту: между валидацией и записью на диск
    стоит целый конвейер, а цена ошибки — запись файла куда попало.
    """
    root = root.resolve()
    written: list[Path] = []
    for relative_path, content in files.items():
        target = (root / relative_path).resolve()
        if not target.is_relative_to(root):
            raise ValueError(f"путь {relative_path!r} выходит за пределы проекта")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        written.append(target)
    return written


__all__ = ["CodeSandbox", "SandboxReport", "write_project_files"]
