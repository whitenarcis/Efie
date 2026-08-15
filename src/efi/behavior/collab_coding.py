"""
efi/behavior/collab_coding.py

Совместное проектирование: от «давай напишем X» до задачи в конвейере.

Модуль существует ради одного правила: НЕ СОГЛАШАТЬСЯ СРАЗУ. Согласиться на
предложение в ту же реплику — самое естественное поведение для языковой
модели («Отличная идея! Уже приступаю!») и самое бесполезное для дела: до
кода никто не обсудил ни стек, ни структуру, ни то, зачем эта штука нужна.
Живой человек, которому предложили вместе что-то писать, сперва задаёт
вопросы — и половина затей на этом честно заканчивается, что тоже результат.

Поэтому запрет здесь механический, а не воспитательный. Инструмент запуска
задачи (efi.tools.dev_tools.start_project.StartDevProjectTool) физически
недоступен модели, пока предложение не обсудили: `may_start()` требует, чтобы
после исходной реплики прошёл хотя бы один обмен. Промпт-блок при этом прямо
говорит, О ЧЁМ спорить (стек, структура, подводные камни) — одного запрета
без указания темы мало, модель начнёт спорить о смысле жизни.

Состояние — в памяти процесса, не в БД, и это осознанно: «мы сейчас обсуждаем
проект» живёт внутри одного разговора. Если процесс перезапустился, человек
повторит идею одной строчкой — а вот воскресшее через сутки «так что там с
нашим проектом?» было бы не памятью, а неловкостью. Сама задача, в отличие
от обсуждения, персистентна с первой секунды (efi/dev/store.py).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime

from efi.dev.schemas import DevTask
from efi.dev.store import DevTaskStore
from efi.utils.bounded import BoundedDict

logger = logging.getLogger(__name__)

#: Сколько чатов помним одновременно и как долго живёт необсуждённое
#: предложение. Час: разговор, вернувшийся к идее позже, начнётся заново — и
#: это правильно, за час контекст успевает смениться.
_MAX_TRACKED_CHATS = 64
_PROPOSAL_TTL_SECONDS = 3600.0

#: Сколько обменов репликами должно пройти ПОСЛЕ предложения, прежде чем
#: можно браться за работу. Один: этого хватает, чтобы Эфи задала свои
#: вопросы, а человек ответил, — и не превращает договорённость в допрос.
REQUIRED_DISCUSSION_TURNS = 1

#: «Давай напишем» и родня. Само по себе ничего не значит («давай сделаем
#: паузу»), поэтому требуется ещё и техническое существительное рядом.
_PROPOSAL_MARKERS = (
    "давай напишем", "давай сделаем", "давай запилим", "давай замутим", "давай соберём",
    "давай сделаю", "давай наваяем", "может напишем", "может сделаем", "может запилим",
    "а давай напишем", "не хочешь написать", "хочешь напишем", "напиши мне",
    "давай накидаем", "давай попробуем написать", "let's write", "let's build",
)

#: О чём именно предлагают. Без этого списка предложением считалось бы любое
#: «давай сделаем перерыв».
_ARTIFACT_MARKERS = (
    "бот", "скрипт", "парсер", "утилит", "тулз", "тулу", "тул ", "инструмент", "библиотек",
    "cli", "tui", "клиент", "приложени", "прогу", "программу", "сервис", "демон", "конвертер",
    "трекер", "мониторинг", "автоматиз", "плагин", "расширени", "api", "проект",
)

_MARKER_RE = re.compile("|".join(re.escape(marker) for marker in _PROPOSAL_MARKERS))
_ARTIFACT_RE = re.compile("|".join(re.escape(marker) for marker in _ARTIFACT_MARKERS))

#: Сколько символов реплики сохраняем как формулировку идеи.
_MAX_IDEA_LENGTH = 400


@dataclass(slots=True)
class Proposal:
    """Предложение, которое сейчас обсуждается в конкретном чате."""

    chat_id: int
    idea: str
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    #: Сколько реплик человек написал в этот чат ПОСЛЕ предложения. Именно
    #: обмены, а не время: договорённость измеряется разговором.
    turns_since: int = 0
    #: Уточнения, добавленные по ходу обсуждения (стек, ограничения) — они
    #: уезжают в задачу вместе с идеей.
    notes: list[str] = field(default_factory=list)

    @property
    def is_discussed(self) -> bool:
        return self.turns_since >= REQUIRED_DISCUSSION_TURNS

    def render_idea(self) -> str:
        """Идея с учётом всего, до чего договорились — то, что уйдёт в спеку."""
        if not self.notes:
            return self.idea
        return f"{self.idea}. Договорились: {'; '.join(self.notes)}"


class CollabCodingDesk:
    """
    Стол переговоров: помнит, что в этом чате обсуждают проект, и решает,
    можно ли уже браться за работу.

    Ничего не отправляет и ничего не генерирует — только состояние и
    решение. Говорит об этом промпт (efi/prompts/builder.py), запускает
    работу инструмент (efi/tools/dev_tools/), делает работу конвейер
    (efi/dev/).
    """

    def __init__(self, store: DevTaskStore) -> None:
        self._store = store
        self._proposals: BoundedDict[int, Proposal] = BoundedDict(
            max_entries=_MAX_TRACKED_CHATS, ttl=_PROPOSAL_TTL_SECONDS
        )

    async def consider_message(self, chat_id: int | None, text: str) -> None:
        """
        Вызывается на КАЖДУЮ входящую реплику (efi.telegram.handlers).

        Две работы разом: заметить новое предложение и досчитать обмены по
        уже идущему обсуждению. Второе не менее важно первого — именно
        счётчик обменов и отличает «договорились» от «поддакнула».
        """
        if chat_id is None:
            return

        existing = self._proposals.get(chat_id)
        if existing is not None:
            existing.turns_since += 1
            note = _extract_note(text)
            if note and note not in existing.notes:
                existing.notes.append(note)
            return

        idea = detect_proposal(text)
        if idea is None:
            return
        if await self._store.has_open_task_for(chat_id):
            # В этом чате уже что-то пишется. Взять вторую задачу параллельно
            # — верный способ не доделать обе.
            logger.info("collab: в chat_id=%s уже есть начатый проект, новое предложение не беру", chat_id)
            return

        self._proposals[chat_id] = Proposal(chat_id=chat_id, idea=idea)
        logger.info("collab: в chat_id=%s предложили проект: %s", chat_id, idea[:80])

    def pending(self, chat_id: int | None) -> Proposal | None:
        """Обсуждаемое сейчас предложение этого чата, если оно есть."""
        return self._proposals.get(chat_id) if chat_id is not None else None

    def may_start(self, chat_id: int | None) -> bool:
        """
        Можно ли уже браться за работу. Ровно это и запрещает соглашаться
        слепо: пока обсуждение не состоялось, ответ — нет.
        """
        proposal = self.pending(chat_id)
        return proposal is not None and proposal.is_discussed

    async def start(self, chat_id: int, *, idea: str = "") -> DevTask | None:
        """
        Переводит договорённость в задачу конвейера и закрывает обсуждение.

        `idea` — как её сформулировала сама Эфи по итогам разговора (это
        точнее исходной реплики человека: там уже учтён стек и всё, о чём
        договорились). Пусто — берём накопленное обсуждением.
        """
        proposal = self.pending(chat_id)
        if proposal is None or not proposal.is_discussed:
            return None

        final_idea = idea.strip() or proposal.render_idea()
        task = await self._store.create(final_idea, chat_id=chat_id, is_collab=True)
        self._proposals.pop(chat_id, None)
        logger.info("collab: задача #%s из обсуждения в chat_id=%s", task.id, chat_id)
        return task

    def drop(self, chat_id: int) -> None:
        """Забыть предложение — например, когда человек передумал."""
        self._proposals.pop(chat_id, None)


def detect_proposal(text: str) -> str | None:
    """
    Похоже ли на предложение вместе что-то написать. Чистая функция: цена
    ложного срабатывания — блок в промпте про несуществующий проект, поэтому
    условие двойное (маркер предложения И технический предмет).
    """
    normalized = (text or "").strip()
    if not normalized:
        return None
    lowered = normalized.lower()
    if not _MARKER_RE.search(lowered) or not _ARTIFACT_RE.search(lowered):
        return None
    return normalized[:_MAX_IDEA_LENGTH]


#: Уточнения по ходу обсуждения: на чём писать и чего не делать. Ищем ровно
#: те реплики, где человек называет технологию или ставит ограничение, —
#: остальное это разговор, а не спецификация.
_NOTE_MARKERS = (
    "на python", "на питоне", "через", "без ", "используй", "лучше ", "не надо", "не нужно",
    "должен уметь", "должна уметь", "чтобы он", "чтобы она", "главное",
)
_NOTE_RE = re.compile("|".join(re.escape(marker) for marker in _NOTE_MARKERS))
_MAX_NOTE_LENGTH = 200


def _extract_note(text: str) -> str:
    normalized = (text or "").strip()
    if not normalized or not _NOTE_RE.search(normalized.lower()):
        return ""
    return normalized[:_MAX_NOTE_LENGTH]


__all__ = [
    "REQUIRED_DISCUSSION_TURNS",
    "CollabCodingDesk",
    "Proposal",
    "detect_proposal",
]
