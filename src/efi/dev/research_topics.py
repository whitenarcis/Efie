"""
efi/dev/research_topics.py

О чём Эфи имеет смысл гуглить: вопросы из её собственной работы.

Фоновое исследование до этого модуля выглядело так: взять случайный интерес
из worldview.json, поискать его в вебе, записать в дневник «размышление».
Получалась лента фактов — про эмуляцию, про новостные агрегаторы, про
алкогольные облака в космосе, — которая ни разу никому не пригодилась
(дашборд честно показывает «использована 0 раз»). Такой поиск не бесполезен
теоретически: он должен подпитывать разговор. Но на практике он подпитывал
папку с фактами, годными для обоев телефона.

Здесь другой источник тем — то, что у неё СЕЙЧАС не получается или сейчас в
работе:

    падение     — дословный текст ошибки из задачи, которая не собралась.
                  Лучший поисковый запрос в природе: у чужого человека была
                  ровно та же строчка, и ответ уже написан.
    стек        — библиотека из спеки текущего проекта плюс то, что он
                  делает. Так узнают, каким API пользоваться, ДО того, как
                  выдумать несуществующий.
    замечание   — незакрытая претензия линтера или проверки, которую не
                  удалось починить с наскока.

Разница между этим и случайным фактом простая: у такого запроса есть
адресат. Найденное идёт не «в копилку», а в конкретную задачу, где его
можно применить сегодня.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from efi.dev.schemas import DevTask
from efi.dev.store import DevTaskStore

logger = logging.getLogger(__name__)

#: Сколько задач просматривать. Больше не нужно: интересно то, что горит
#: сейчас, а не всё, что когда-либо не собралось.
_RECENT_LIMIT = 5

#: Максимальная длина поискового запроса. Длинный трейсбэк целиком не ищется
#: нигде — ищется его последняя строка, та самая, что называет ошибку.
_MAX_QUERY_CHARS = 160

#: Шум, который в запросе только мешает: пути внутри рабочей копии, номера
#: строк, адреса объектов в памяти.
_NOISE_RE = re.compile(
    r'File "[^"]+", line \d+|/[\w./-]+\.py:\d+|0x[0-9a-f]+|\bline \d+\b', re.IGNORECASE
)

#: Строка, называющая ошибку. Без такого имени запрос бессмыслен: гуглится
#: не «что-то пошло не так», а `ModuleNotFoundError: No module named 'x'`.
_ERROR_NAME_RE = re.compile(r"\b\w*(?:Error|Exception|Warning)\b|\berror\b:", re.IGNORECASE)

#: Что из стека не является библиотекой и искать по нему нечего.
_STACK_NOISE = frozenset({"python", "python3", "python 3.11", "стандартная библиотека", "stdlib"})


@dataclass(slots=True, frozen=True)
class ResearchQuestion:
    """Один вопрос, который стоит задать поиску, и зачем он задан."""

    query: str
    #: Задача, ради которой ищем, — чтобы найденное вернулось именно к ней.
    task_id: int = 0
    #: Как это звучит в логе и в дневнике: «упало вот на этом», «выбираю
    #: библиотеку». Не для модели, а для человека, читающего дашборд.
    reason: str = ""
    #: Ищем по конкретному падению (в отличие от «что за библиотека такая»).
    #: Отдельным полем, а не разбором `reason`: смысл вопроса — это факт, а
    #: не то, какими словами его записали в лог.
    is_about_failure: bool = False


class DevResearchTopics:
    """
    Вопросы из работы. Читает задачи разработки и превращает их в поисковые
    запросы; ничего не ищет сам.
    """

    def __init__(self, store: DevTaskStore) -> None:
        self._store = store

    async def next_question(self) -> ResearchQuestion | None:
        """
        Самый насущный вопрос прямо сейчас — или None, если работы нет.

        Порядок не случайный: сначала то, что СЛОМАНО (там есть конкретная
        строчка ошибки и конкретная польза от ответа), потом то, что пишется
        (там польза в том, чтобы не выдумать несуществующий API).
        """
        try:
            failures = await self._store.recent_failures(limit=_RECENT_LIMIT)
            active = await self._store.active()
        except Exception:
            logger.warning("research_topics: не удалось прочитать задачи разработки", exc_info=True)
            return None

        for task in failures:
            question = _from_failure(task)
            if question is not None:
                return question
        for task in active:
            question = _from_stack(task)
            if question is not None:
                return question
        return None


def _from_failure(task: DevTask) -> ResearchQuestion | None:
    """Запрос из текста падения. Ошибка, по которой ничего не гуглится, — не запрос."""
    query = normalize_error_query(task.error)
    if not query:
        return None
    return ResearchQuestion(
        query=query, task_id=task.id, reason="упало вот на этом", is_about_failure=True
    )


def _from_stack(task: DevTask) -> ResearchQuestion | None:
    """Запрос из стека текущего проекта: библиотека плюс то, что ей делают."""
    spec = task.spec
    if spec is None:
        return None
    libraries = [
        item.strip()
        for item in spec.stack
        if item.strip() and item.strip().lower() not in _STACK_NOISE
    ]
    if not libraries:
        return None
    subject = " ".join(spec.problem.split()[:6])
    return ResearchQuestion(
        query=f"python {libraries[0]} {subject}"[:_MAX_QUERY_CHARS],
        task_id=task.id,
        reason=f"пишу проект на {libraries[0]}",
    )


def normalize_error_query(raw: str) -> str:
    """
    Превращает вывод падения в поисковый запрос.

    Берётся последняя содержательная строка (там называется ошибка) и из неё
    вычищается всё, что уникально для этой машины: пути, номера строк, адреса
    объектов. Иначе запрос будет уникальным и не найдёт ничего — а чужой
    ответ на ту же ошибку существует ровно потому, что строчка у всех
    одинаковая.
    """
    lines = [line.strip() for line in (raw or "").splitlines() if line.strip()]
    if not lines:
        return ""
    named = next((line for line in reversed(lines) if _ERROR_NAME_RE.search(line)), "")
    if not named:
        # Наша собственная формулировка провала («два файла так и не
        # собрались») — не поисковый запрос: у неё нет чужого ответа в
        # интернете, потому что она есть только у нас.
        return ""
    cleaned = _NOISE_RE.sub("", named).strip(" :,-")
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    if len(cleaned) < 12:
        return ""
    return cleaned[:_MAX_QUERY_CHARS]


__all__ = ["DevResearchTopics", "ResearchQuestion", "normalize_error_query"]
