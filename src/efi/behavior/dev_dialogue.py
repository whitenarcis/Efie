"""
efi/behavior/dev_dialogue.py

Разговор с напарником, а не с ботом: «глянь репу», «тест падает, в чём
дело», «давай допишем ту утилиту».

Соседний модуль (efi/behavior/collab_coding.py) отвечает за замысел НОВОГО
проекта и держит там одно правило — не соглашаться сразу. Здесь предмет
другой: код, который уже есть. Просьба «почини импорт» не нуждается в
обсуждении стека, её надо просто сделать; а вот «давай перепишем всё на
async» — очень даже нуждается, и согласиться на неё в ту же реплику значит
угробить чужой вечер. Поэтому разделение проходит не по «новое/старое», а по
цене ошибки:

    сделать сразу   — понятная точечная работа: посмотреть репозиторий,
                      починить конкретное падение, добавить маленькую вещь.
                      Спрашивать разрешения на это — то же самое, что
                      переспрашивать «точно почитать?» перед чтением.
    сперва спорить  — работа, которая меняет проект: переписать, выкинуть
                      слой, сменить библиотеку. Тут у неё должно быть своё
                      мнение, и высказать его надо ДО, а не после.

Никаких команд и префиксов: разбирается обычная речь. Ссылка на репозиторий
запоминается на чат — сказанное один раз «вот моя репа <url>» действует и
для следующих трёх просьб, потому что так говорят люди.

Модуль ничего не запускает и никуда не пишет: он распознаёт намерение и
хранит контекст разговора. Работу делает efi/dev/swe_engine.py, задачу
заводит инструмент, реплики формулирует efi/dev/reporter.py.
"""

from __future__ import annotations

import logging
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from efi.dev.schemas import DevTask, DevTaskKind
from efi.dev.store import DevTaskStore
from efi.utils.bounded import BoundedDict

logger = logging.getLogger(__name__)

#: Сколько чатов помним и сколько живёт контекст «мы сейчас про этот
#: репозиторий». Сутки: ссылку, названную утром, вечером ещё разумно помнить,
#: а через неделю — уже нет.
_MAX_TRACKED_CHATS = 64
_CONTEXT_TTL_SECONDS = 86_400.0

#: Сколько символов просьбы уносим в задачу.
_MAX_INSTRUCTION_LENGTH = 500

#: Ссылка на репозиторий в тексте. GitHub/GitLab/что угодно по https и
#: ssh-форма, которую копируют со страницы репозитория.
_REPO_URL_RE = re.compile(
    r"(?:https?://[\w.-]+/[\w.\-/]+?(?:\.git)?|git@[\w.-]+:[\w.\-/]+?\.git)(?=[\s,.;)]|$)"
)

#: Локальный путь: `~/projects/foo`, `/data/data/.../repo`, `./src`.
_LOCAL_PATH_RE = re.compile(r"(?:^|\s)(~?/[\w.\-/]{3,}|\./[\w.\-/]{3,})")

#: «Посмотри» — просьба разобраться, а не менять.
_LOOK_MARKERS = (
    "глянь", "гляди", "посмотри", "погляди", "зацени", "оцени", "разберись", "почитай код",
    "что там за", "что думаешь про репу", "изучи", "покопайся",
)

#: «Почини» — точечная работа по конкретной поломке.
_FIX_MARKERS = (
    "почини", "пофикси", "исправь", "падает", "не работает", "ошибка", "ошибку", "баг",
    "сломал", "сломан", "тест падает", "тесты падают", "traceback", "трейсбек", "трейсбэк",
    "exception", "error:", "в чём дело", "в чем дело", "разберись почему",
)

#: «Допиши» — добавить возможность в существующий код.
_CHANGE_MARKERS = (
    "допиши", "добавь", "допилим", "допишем", "доделай", "доработай", "прикрути", "внедри",
    "реализуй в", "сделай в репе", "поправь", "измени",
)

#: То, что менять нельзя молча: цена ошибки — чужой проект целиком.
_HEAVY_MARKERS = (
    "перепиши", "переписать", "перепишем", "мигрируй", "переведи на", "выкини", "выкинуть",
    "убери слой", "смени библиотеку", "поменяй архитектуру", "перевести на async",
    "перейти на", "рефактор", "рефакторинг", "с нуля",
)

_LOOK_RE = re.compile("|".join(re.escape(item) for item in _LOOK_MARKERS))
_FIX_RE = re.compile("|".join(re.escape(item) for item in _FIX_MARKERS))
_CHANGE_RE = re.compile("|".join(re.escape(item) for item in _CHANGE_MARKERS))
_HEAVY_RE = re.compile("|".join(re.escape(item) for item in _HEAVY_MARKERS))

#: Слова, по которым понятно, что речь о коде вообще. Нужны там, где сам
#: маркер двусмыслен: «посмотри» без этого срабатывало бы на «посмотри в окно».
_CODE_WORDS = (
    "реп", "repo", "git", "код", "модул", "функци", "класс", "импорт", "тест", "проект",
    "скрипт", "package", "ветк", "коммит", "pull request", "пулл", "pr ", "traceback",
    "утилит", "библиотек", "файл", ".py", "pytest", "линтер", "ruff",
)
_CODE_RE = re.compile("|".join(re.escape(item) for item in _CODE_WORDS))


class DevIntentKind(StrEnum):
    """Что от неё хотят."""

    #: Разобраться в репозитории и рассказать, что там.
    LOOK = "look"
    #: Починить конкретное падение.
    FIX = "fix"
    #: Добавить или поправить что-то по существу.
    CHANGE = "change"
    #: Крупное вмешательство — сначала спор, потом работа.
    HEAVY = "heavy"

    @property
    def needs_discussion(self) -> bool:
        """
        Нужен ли разговор до начала. Только для крупного: спрашивать
        разрешения починить импорт — это не вежливость, а трата чужого
        времени.
        """
        return self is DevIntentKind.HEAVY

    @property
    def human(self) -> str:
        return _KIND_WORDS[self]


_KIND_WORDS: dict[DevIntentKind, str] = {
    DevIntentKind.LOOK: "посмотреть код",
    DevIntentKind.FIX: "починить",
    DevIntentKind.CHANGE: "доработать",
    DevIntentKind.HEAVY: "переделать по-крупному",
}


@dataclass(slots=True, frozen=True)
class DevIntent:
    """Распознанная просьба по коду."""

    kind: DevIntentKind
    instruction: str
    #: Репозиторий, названный прямо в этой реплике. Пусто — берём из контекста чата.
    source: str = ""

    @property
    def is_actionable(self) -> bool:
        """Есть ли что делать прямо сейчас (в отличие от «поговорить об этом»)."""
        return self.kind is not DevIntentKind.HEAVY


@dataclass(slots=True)
class RepoContext:
    """О каком репозитории идёт речь в этом чате и что с ним уже делали."""

    chat_id: int
    source: str
    mentioned_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    #: Последняя просьба — чтобы «а теперь допиши тесты» понималось как
    #: продолжение, а не как задача из ниоткуда.
    last_instruction: str = ""
    #: Ветки, которые она уже сделала по этому репозиторию.
    branches: list[str] = field(default_factory=list)

    def render_for_prompt(self) -> str:
        tail = f"; уже сделала ветки: {', '.join(self.branches[-3:])}" if self.branches else ""
        return f"{self.source}{tail}"


def detect_dev_intent(text: str, *, in_code_context: bool = False) -> DevIntent | None:
    """
    Разбирает обычную реплику. Чистая функция — проверяется без чатов и БД.

    Условие двойное везде, где маркер сам по себе двусмыслен: «посмотри» плюс
    что-то про код. Цена ложного срабатывания — она полезет клонировать
    репозиторий вместо ответа на вопрос, и это заметнее, чем пропущенная
    просьба: пропущенную человек повторит.

    `in_code_context` снимает второе условие, и это не поблажка, а то, как
    устроен разговор: после «глянь репу X» реплика «допиши туда флаг --json»
    очевидно про код, хотя слова «код» в ней нет. Требовать его каждый раз —
    значит понимать только первую фразу из беседы.
    """
    normalized = (text or "").strip()
    if not normalized:
        return None
    lowered = normalized.lower()

    source = _extract_source(normalized)
    about_code = bool(_CODE_RE.search(lowered)) or bool(source) or in_code_context
    if not about_code:
        return None

    instruction = normalized[:_MAX_INSTRUCTION_LENGTH]
    if _HEAVY_RE.search(lowered):
        return DevIntent(kind=DevIntentKind.HEAVY, instruction=instruction, source=source)
    if _FIX_RE.search(lowered):
        return DevIntent(kind=DevIntentKind.FIX, instruction=instruction, source=source)
    if _CHANGE_RE.search(lowered):
        return DevIntent(kind=DevIntentKind.CHANGE, instruction=instruction, source=source)
    if _LOOK_RE.search(lowered) or (source and len(lowered) < 120):
        return DevIntent(kind=DevIntentKind.LOOK, instruction=instruction, source=source)
    return None


def _extract_source(text: str) -> str:
    """Ссылка или путь к репозиторию из реплики — то, с чем предстоит работать."""
    url = _REPO_URL_RE.search(text)
    if url is not None:
        return url.group(0).rstrip(".,;)")
    local = _LOCAL_PATH_RE.search(text)
    return local.group(1) if local is not None else ""


class DevPartnerDesk:
    """
    Память разговора о коде: какой репозиторий обсуждают и о чём попросили.

    Ровно как у стола переговоров про новые проекты (collab_coding), состояние
    держится в памяти процесса: «мы сейчас про эту репу» — свойство разговора,
    а не факт о мире. Сами задачи персистентны с первой секунды
    (efi/dev/store.py), и перезапуск их не теряет.
    """

    def __init__(
        self,
        store: DevTaskStore,
        *,
        available: bool = False,
        on_task_created: Callable[[], None] | None = None,
    ) -> None:
        self._store = store
        #: Поднят ли SWE-движок. Без него обсуждать код можно (это разговор),
        #: а браться нельзя: обещание, которое некому выполнить, хуже
        #: честного «не могу» — то же правило, что у collab_coding.
        self._available = available
        self._on_task_created = on_task_created
        self._contexts: BoundedDict[int, RepoContext] = BoundedDict(
            max_entries=_MAX_TRACKED_CHATS, ttl=_CONTEXT_TTL_SECONDS
        )
        self._intents: BoundedDict[int, DevIntent] = BoundedDict(
            max_entries=_MAX_TRACKED_CHATS, ttl=_CONTEXT_TTL_SECONDS
        )

    @property
    def available(self) -> bool:
        return self._available

    def attach_engine(self, *, available: bool, on_task_created: Callable[[], None] | None = None) -> None:
        """Поздняя привязка движка — как `CollabCodingDesk.attach_pipeline`, и по той же причине."""
        self._available = available
        self._on_task_created = on_task_created

    def consider_message(self, chat_id: int | None, text: str) -> DevIntent | None:
        """
        Вызывается на каждую входящую реплику. Возвращает распознанную
        просьбу, если она есть, — и заодно запоминает контекст.
        """
        if chat_id is None:
            return None
        known = self._contexts.get(chat_id)
        intent = detect_dev_intent(text, in_code_context=known is not None)
        if intent is None:
            return None

        context = known
        if intent.source:
            context = RepoContext(chat_id=chat_id, source=intent.source)
            self._contexts[chat_id] = context
        if context is not None:
            context.last_instruction = intent.instruction

        self._intents[chat_id] = intent
        logger.info(
            "dev_dialogue: в chat_id=%s просьба по коду (%s): %s",
            chat_id, intent.kind.value, intent.instruction[:80],
        )
        return intent

    def pending(self, chat_id: int | None) -> DevIntent | None:
        return self._intents.get(chat_id) if chat_id is not None else None

    def context(self, chat_id: int | None) -> RepoContext | None:
        return self._contexts.get(chat_id) if chat_id is not None else None

    def remember_repo(self, chat_id: int, source: str) -> None:
        """Запомнить репозиторий явно — например, когда его назвали в прошлом разговоре."""
        self._contexts[chat_id] = RepoContext(chat_id=chat_id, source=source)

    def remember_branch(self, chat_id: int | None, branch: str) -> None:
        """Отметить сделанную ветку: по ней потом можно сослаться в разговоре."""
        context = self.context(chat_id)
        if context is not None and branch and branch not in context.branches:
            context.branches.append(branch)

    def may_work(self, chat_id: int | None) -> bool:
        """
        Можно ли браться прямо сейчас: движок поднят, репозиторий известен, а
        просьба не из тех, что требуют предварительного спора.
        """
        if not self._available:
            return False
        intent = self.pending(chat_id)
        if intent is None or not intent.is_actionable:
            return False
        return bool(intent.source or self.context(chat_id) is not None)

    def resolve_source(self, chat_id: int | None, explicit: str = "") -> str:
        """Репозиторий для задачи: явно названный, из просьбы или из контекста чата."""
        if explicit.strip():
            return explicit.strip()
        intent = self.pending(chat_id)
        if intent is not None and intent.source:
            return intent.source
        context = self.context(chat_id)
        return context.source if context is not None else ""

    async def start(self, chat_id: int, *, instruction: str = "", source: str = "") -> DevTask | None:
        """
        Заводит задачу на работу с кодом. Возвращает None, если браться
        нельзя или непонятно, с чем работать.
        """
        if not self._available:
            return None
        resolved = self.resolve_source(chat_id, source)
        if not resolved:
            return None

        intent = self.pending(chat_id)
        text = instruction.strip() or (intent.instruction if intent is not None else "")
        if not text:
            return None

        if await self._store.has_open_task_for(chat_id):
            logger.info("dev_dialogue: в chat_id=%s уже есть задача в работе, вторую не беру", chat_id)
            return None

        task = await self._store.create(
            text, chat_id=chat_id, is_collab=True, kind=DevTaskKind.SWE, source=resolved
        )
        self._intents.pop(chat_id, None)
        logger.info("dev_dialogue: задача #%s по %s из chat_id=%s", task.id, resolved, chat_id)
        if self._on_task_created is not None:
            self._on_task_created()
        return task


__all__ = [
    "DevIntent",
    "DevIntentKind",
    "DevPartnerDesk",
    "RepoContext",
    "detect_dev_intent",
]
