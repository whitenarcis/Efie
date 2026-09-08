"""
efi/dev/edits.py

Формат правки: блок «найди это — замени на это», а не файл целиком.

Почему не целиком. Переписывание файла — это способ, которым модель ломает
чужой код молча. Просят поправить одну функцию в файле на четыреста строк,
модель возвращает «весь файл», и в нём: сокращённые до `...` куски, которые
она сочла неважными, потерянные импорты, переписанные докстринги, съеденный
хвост из-за лимита вывода. Diff при этом выглядит как «изменён весь файл», и
понять, что именно она сделала, нельзя ни человеку, ни следующей проверке.

Точечный блок лишён всего этого. Он либо применяется дословно, либо не
применяется вовсе — третьего нет, и это главное свойство: правка, которая не
нашла своего места, обязана быть отвергнутой, а не применённой «примерно
туда». Формат тот же, что у Aider, потому что он уже есть в обучающих данных
моделей:

    path/to/file.py
    <<<<<<< SEARCH
    старый текст, дословно
    =======
    новый текст
    >>>>>>> REPLACE

Пустой блок SEARCH означает создание файла — это единственный случай, когда
допустима запись целиком, и он безобиден: нового файла ещё нет, ломать
нечего.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

#: Разбор блока. Заголовок с путём — строка перед `<<<<<<< SEARCH`; она может
#: быть обёрнута в ``` или в кавычки, потому что модели делают это постоянно.
_BLOCK_RE = re.compile(
    r"(?P<path>[^\n]*?)\n"
    r"[`\s]*<{5,9} SEARCH[^\n]*\n"
    r"(?P<search>.*?)"
    r"^={5,9}\s*\n"
    r"(?P<replace>.*?)"
    r"^>{5,9} REPLACE",
    re.DOTALL | re.MULTILINE,
)

#: Мусор вокруг пути: ```python, кавычки, markdown-выделение, «Файл:».
_PATH_NOISE_RE = re.compile(r"^[\s`'\"*#]*(?:файл|file)?[\s:`'\"*]*|[\s`'\"*]*$", re.IGNORECASE)


class EditError(RuntimeError):
    """Правку не удалось применить. Текст объясняет, почему именно, — он уходит обратно модели."""


@dataclass(slots=True, frozen=True)
class SearchReplaceEdit:
    """Одна правка одного файла."""

    path: str
    search: str
    replace: str

    @property
    def creates_file(self) -> bool:
        return not self.search.strip()

    def render(self) -> str:
        """Обратно в текстовый вид — для лога и для показа модели её же правки."""
        return f"{self.path}\n<<<<<<< SEARCH\n{self.search}=======\n{self.replace}>>>>>>> REPLACE"


def parse_edits(raw: str) -> list[SearchReplaceEdit]:
    """
    Достаёт все правки из ответа модели.

    Проза вокруг блоков игнорируется: модель почти всегда предваряет патч
    объяснением, и требовать «только блоки» — значит терять нормальный ответ
    из-за вежливости.
    """
    edits: list[SearchReplaceEdit] = []
    for match in _BLOCK_RE.finditer(raw or ""):
        path = _clean_path(match.group("path"))
        if not path:
            logger.debug("edits: блок без внятного пути пропущен")
            continue
        edits.append(
            SearchReplaceEdit(
                path=path, search=match.group("search"), replace=match.group("replace")
            )
        )
    return edits


def _clean_path(raw: str) -> str:
    """
    Путь из заголовка блока — или пусто, если это не путь.

    Ведущее `./` снимается через removeprefix, а НЕ через lstrip("./"):
    lstrip снял бы любые ведущие точки и слэши, и `../../.ssh/authorized_keys`
    превратился бы во вполне валидный `ssh/authorized_keys` — то есть проверка
    на выход за пределы рабочей копии перестала бы что-либо ловить (та же
    ошибка уже была в efi/dev/schemas.py::FileSpec и стоила бы дороже здесь:
    правки приезжают от модели, которая читала чужой репозиторий).
    """
    path = _PATH_NOISE_RE.sub("", raw.strip()).strip()
    # Последняя строка заголовка: модели пишут «Вот правка:\npath/to/file.py».
    path = path.splitlines()[-1].strip() if path else ""
    if not path or " " in path or path.startswith(("<", "=", ">")):
        return ""
    path = path.replace("\\", "/").removeprefix("./")
    parts = path.split("/")
    if path.startswith("/") or any(part in ("", ".", "..") for part in parts):
        return ""
    return path


def apply_edit(root: Path, edit: SearchReplaceEdit) -> Path:
    """
    Применяет одну правку к файлу под `root` и возвращает путь изменённого файла.

    Три отказа, и все три — по делу:
      * путь уводит за пределы рабочей копии (правка приехала от модели, а
        `../../.ssh/authorized_keys` не должен быть выразим в принципе);
      * искомого текста в файле нет — модель придумала контекст;
      * искомый текст встречается дважды — непонятно, какое место править,
        а угадывать здесь нельзя.
    """
    target = (root / edit.path).resolve()
    if not target.is_relative_to(root.resolve()):
        raise EditError(f"путь {edit.path!r} выходит за пределы рабочей копии")

    if edit.creates_file:
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(edit.replace, encoding="utf-8")
        return target

    if not target.is_file():
        raise EditError(f"файла {edit.path} нет — править нечего")

    source = target.read_text(encoding="utf-8")
    occurrences = source.count(edit.search)
    if occurrences == 0:
        relaxed = _match_ignoring_indent(source, edit.search)
        if relaxed is None:
            raise EditError(
                f"в {edit.path} нет искомого фрагмента — покажи текст, который там ДЕЙСТВИТЕЛЬНО есть"
            )
        source = source.replace(relaxed, _reindent(edit.replace, relaxed, edit.search), 1)
    elif occurrences > 1:
        raise EditError(
            f"фрагмент встречается в {edit.path} {occurrences} раза — возьми больше контекста, "
            "чтобы место правки было однозначным"
        )
    else:
        source = source.replace(edit.search, edit.replace, 1)

    target.write_text(source, encoding="utf-8")
    return target


def apply_edits(root: Path, edits: list[SearchReplaceEdit]) -> tuple[list[str], list[str]]:
    """
    Применяет пачку правок. Возвращает (изменённые пути, причины отказов).

    Отказ одной правки не отменяет остальные: обычно модель присылает три
    блока, и два из них хорошие. Причины уходят обратно ей — это и есть цикл
    автопочинки (efi/dev/auto_fix.py).
    """
    changed: list[str] = []
    problems: list[str] = []
    for edit in edits:
        try:
            path = apply_edit(root, edit)
        except EditError as exc:
            logger.info("edits: правка %s отклонена: %s", edit.path, exc)
            problems.append(str(exc))
            continue
        relative = str(path.relative_to(root.resolve()))
        # Один файл, две правки — это по-прежнему один изменённый файл. Без
        # этого он дважды уезжает в проверки и дважды называется в реплике.
        if relative not in changed:
            changed.append(relative)
    return changed, problems


def _match_ignoring_indent(source: str, search: str) -> str | None:
    """
    Тот же фрагмент, но с другим отступом.

    Единственная поблажка, которую здесь можно себе позволить: модель
    регулярно сдвигает блок на четыре пробела, копируя его из ответа. Всё
    остальное (пропущенная строка, «примерно похоже») не прощается — правка
    «примерно туда» хуже отсутствующей.
    """
    needle = [line.strip() for line in search.strip("\n").splitlines() if line.strip()]
    if not needle:
        return None
    lines = source.splitlines(keepends=True)
    stripped = [line.strip() for line in lines]

    for start in range(len(lines) - len(needle) + 1):
        window = [line for line in stripped[start : start + len(needle)]]
        if window != needle:
            continue
        return "".join(lines[start : start + len(needle)])
    return None


def _reindent(replacement: str, found: str, search: str) -> str:
    """Возвращает замене тот отступ, который был у найденного места, а не у ответа модели."""
    found_indent = _leading_indent(found)
    search_indent = _leading_indent(search)
    if found_indent == search_indent:
        return replacement
    shift = len(found_indent) - len(search_indent)
    lines = replacement.splitlines(keepends=True)
    if shift > 0:
        return "".join(" " * shift + line if line.strip() else line for line in lines)
    return "".join(line[-shift:] if line.startswith(" " * -shift) else line for line in lines)


def _leading_indent(text: str) -> str:
    for line in text.splitlines():
        if line.strip():
            return line[: len(line) - len(line.lstrip())]
    return ""


#: Как объяснить формат модели. Держится здесь, а не в промптах: формат и его
#: описание обязаны меняться вместе, иначе описание переживёт формат.
EDIT_FORMAT_INSTRUCTIONS = (
    "ФОРМАТ ПРАВОК. Отвечай только блоками вида:\n"
    "путь/к/файлу.py\n"
    "<<<<<<< SEARCH\n"
    "фрагмент, который есть в файле ДОСЛОВНО\n"
    "=======\n"
    "чем его заменить\n"
    ">>>>>>> REPLACE\n"
    "Правила: фрагмент SEARCH должен совпадать с файлом посимвольно и встречаться в нём РОВНО один "
    "раз — бери столько строк контекста, сколько нужно для однозначности. Не переписывай файл "
    "целиком и не сокращай код многоточиями. Новый файл — блок с пустым SEARCH. Блоков может быть "
    "несколько; коротко объяснить замысел словами перед ними можно."
)


__all__ = [
    "EDIT_FORMAT_INSTRUCTIONS",
    "EditError",
    "SearchReplaceEdit",
    "apply_edit",
    "apply_edits",
    "parse_edits",
]
