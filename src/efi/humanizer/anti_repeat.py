"""
efi/humanizer/anti_repeat.py

Отслеживание и предотвращение повторяющихся фраз/паттернов в ответах бота —
по чату (одна и та же фраза в РАЗНЫХ чатах не проблема, повтор ВНУТРИ одного
диалога — проблема). Использует difflib.SequenceMatcher (стандартная
библиотека, без внешних зависимостей) для оценки текстового сходства; сама
оценка — потенциально CPU-bound на длинной истории, поэтому вынесена в
asyncio.to_thread, как и остальные подобные вычисления в проекте (см.
memory/tfidf_fallback.py, memory/diary.py).
"""

from __future__ import annotations

import asyncio
import difflib
from collections import deque

from efi.config.schema import HumanizerSettings
from efi.utils.bounded import BoundedDict

#: Сколько чатов держим под наблюдением. Заметно больше, чем у кого-либо
#: бывает живых диалогов одновременно, — и при этом конечное число.
_MAX_TRACKED_CHATS = 256


class AntiRepeatTracker:
    """
    Хранит последние `anti_repeat_max_history` отправленных сообщений на чат
    и проверяет кандидата на схожесть с ними по двум порогам:
        - максимальному (`anti_repeat_trigger_max`) — ни с одним отдельным
          прошлым сообщением сходство не должно быть выше;
        - среднему (`anti_repeat_trigger_avg`) — защита от "не дословных, но
          всё равно однообразных" ответов, которые по отдельности не бьют в
          максимальный порог, но в совокупности создают ощущение зацикленности.
    """

    def __init__(self, settings: HumanizerSettings) -> None:
        self._settings = settings
        #: По чату — до `anti_repeat_max_history` последних реплик. Число
        #: ЧАТОВ тоже ограничено: у userbot'а их за месяцы набегает сколько
        #: угодно, а держим мы на каждый по три десятка строк. Вытесненный
        #: чат теряет защиту от повтора — приемлемо: это чат, в котором Эфи
        #: давно ничего не говорила, и повторяться там не с чем.
        self._history: BoundedDict[int, deque[str]] = BoundedDict(max_entries=_MAX_TRACKED_CHATS)

    def record(self, chat_id: int, text: str) -> None:
        """Регистрирует отправленное сообщение в истории чата. Вызывается ПОСЛЕ фактической отправки."""
        history = self._history.get(chat_id)
        if history is None:
            history = deque(maxlen=self._settings.anti_repeat_max_history)
            self._history[chat_id] = history
        history.append(text)

    async def is_repetitive(self, chat_id: int, candidate: str) -> bool:
        """Критический путь (перед отправкой): True, если кандидат слишком похож на недавнюю историю чата."""
        history = tuple(self._history.get(chat_id, ()))
        if not history:
            return False
        return await asyncio.to_thread(_is_repetitive_sync, candidate, history, self._settings)

    def clear(self, chat_id: int) -> None:
        """Сбрасывает историю чата (например, вместе с очисткой диалоговой истории)."""
        self._history.pop(chat_id, None)


def _is_repetitive_sync(candidate: str, history: tuple[str, ...], settings: HumanizerSettings) -> bool:
    similarities = [difflib.SequenceMatcher(a=candidate, b=previous).ratio() for previous in history]
    if not similarities:
        return False
    if max(similarities) > settings.anti_repeat_trigger_max:
        return True
    return (sum(similarities) / len(similarities)) > settings.anti_repeat_trigger_avg


__all__ = ["AntiRepeatTracker"]
