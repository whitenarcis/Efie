"""
efi/telegram/chat_orchestrator.py

ChatOrchestrator — владелец «кто прямо сейчас отвечает в этом чате» и
единственная точка, откуда устаревшая генерация снимается.

ЗАЧЕМ. Между приходом сообщения и последним отправленным бабблом проходит
много времени: busy-задержка, генерация LLM (секунды, иногда с несколькими
раундами tool-calling), потом паузы между бабблами по WPM. Всё это время
Worker считал, что отвечает на актуальную реплику. Если собеседник за эти
секунды дописывал мысль, Эфи договаривала ответ на устаревший вопрос, и
только потом отдельным ходом реагировала на новое — тот самый эффект
«запоздалого бота», когда разговор идёт с отставанием на реплику.

КАК. Worker выполняет обработку одного уведомления не прямо в своём цикле, а
внутри `asyncio.Task`, зарегистрированного здесь. Приход нового сообщения
(efi/telegram/buffer.py::InboundMessageBuffer.add, до всякого ожидания)
вызывает `interrupt()`, и таск снимается:

    - если Эфи ещё думает — запрос к LLM обрывается;
    - если уже печатает серию бабблов — отмена приходит в asyncio.sleep
      между бабблами, и неотправленный остаток просто не уходит
      (efi/telegram/queue.py::BubbleQueue знает, сколько было сброшено);
    - уже доставленные бабблы, разумеется, не отзываются — они сохраняются в
      историю как сказанное (см. Worker._handle), иначе следующая генерация
      не знала бы, что часть ответа собеседник уже прочитал, и повторилась бы.

Дальше пачка дособирается в буфере и уходит новым уведомлением, которое
NotificationManager отдаёт ТОМУ ЖЕ воркеру (маршрутизация по chat_id), так
что порядок обработки чата не нарушается.

ВАЖНО про различение отмен. `run()` возвращает False, когда таск сняли
отсюда, и пробрасывает CancelledError, когда останавливают само приложение.
Различаются они через `asyncio.Task.cancelling()` (3.11+): у внешней
остановки счётчик отмен вызывающего таска больше нуля, у нашей — ноль.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Coroutine
from typing import Any

logger = logging.getLogger(__name__)

#: Сколько ждать блок очистки прерванной генерации (запись уже отправленных
#: бабблов в историю). Обычно это миллисекунды; потолок нужен, чтобы залипшая
#: запись в БД не заткнула приём входящих — interrupt() вызывается прямо из
#: обработчика Telegram-события.
_CLEANUP_TIMEOUT_SECONDS = 5.0


class ChatOrchestrator:
    """
    Реестр активных генераций по chat_id.

    Один чат — не более одной активной генерации: маршрутизация
    NotificationManager закрепляет чат за одним воркером, поэтому
    конкурирующих обработок одного чата не бывает по построению, и здесь
    достаточно простого словаря без блокировок.
    """

    def __init__(self) -> None:
        self._active: dict[int, asyncio.Task[Any]] = {}

    def is_generating(self, chat_id: int) -> bool:
        """Идёт ли прямо сейчас генерация/отправка ответа в этом чате."""
        task = self._active.get(chat_id)
        return task is not None and not task.done()

    async def interrupt(self, chat_id: int) -> bool:
        """
        Снимает активную генерацию чата, если она есть. Возвращает True,
        если что-то реально было прервано.

        Дожидается фактического завершения таска: у него есть блок очистки
        (сохранение уже отправленных бабблов в историю), и следующая
        генерация должна стартовать уже после него — иначе она соберёт
        историю без последней реплики Эфи и повторила бы её.

        Собственную отмену НЕ проглатывает: если приложение останавливают
        прямо во время ожидания, CancelledError уходит наверх.
        """
        task = self._active.get(chat_id)
        if task is None or task.done():
            return False

        logger.info("orchestrator: interrupting a stale generation for chat_id=%s", chat_id)
        task.cancel()

        # asyncio.wait, а не await/wait_for: здесь мы НАБЛЮДАТЕЛЬ, а не
        # владелец таска — его результат (в т.ч. исключение) забирает run().
        # wait не отменяет таск по таймауту и не пробрасывает его ошибку, а
        # таймаут защищает входящий путь: interrupt() вызывается прямо из
        # обработчика Telegram-события, и залипший блок очистки не должен
        # затыкать приём новых сообщений.
        _done, pending = await asyncio.wait({task}, timeout=_CLEANUP_TIMEOUT_SECONDS)
        if pending:
            logger.warning(
                "orchestrator: cleanup of the interrupted generation for chat_id=%s is taking too long, moving on",
                chat_id,
            )
        return True

    async def run(self, chat_id: int | None, coro: Coroutine[Any, Any, None]) -> bool:
        """
        Выполняет обработку уведомления как отменяемый таск и возвращает,
        доведена ли она до конца (False — сняли как устаревшую).

        Уведомление без chat_id (например, глобальная ночная задача)
        прерывать не от чего и незачем — оно выполняется как обычный await.
        """
        if chat_id is None:
            await coro
            return True

        task = asyncio.create_task(coro, name=f"generation-{chat_id}")
        self._active[chat_id] = task
        try:
            await task
            return True
        except asyncio.CancelledError:
            if not _self_cancelled():
                logger.info("orchestrator: generation for chat_id=%s was superseded by a newer message", chat_id)
                return False
            # Останавливают приложение, а не эту генерацию. Отмена пришла в
            # точку await, но самого подопечного не тронула — снимаем его
            # явно и дожидаемся, иначе таск утёк бы («Task was destroyed but
            # it is pending») вместе с недоделанной записью в историю.
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            raise
        finally:
            if self._active.get(chat_id) is task:
                del self._active[chat_id]

    async def cancel_all(self) -> None:
        """
        Снимает все активные генерации — graceful shutdown (efi/app.py).
        Отменяем все разом, потом дожидаемся: последовательная отмена
        растянула бы остановку на сумму их блоков очистки.
        """
        tasks = [task for task in self._active.values() if not task.done()]
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self._active.clear()


def _self_cancelled() -> bool:
    """
    Отменяют ли ТЕКУЩИЙ таск (остановка приложения), а не тот, которым мы
    управляем. `Task.cancelling()` считает вызовы .cancel() именно на нём —
    ровно то различение, которого не даёт сам по себе CancelledError.
    """
    current = asyncio.current_task()
    return current is not None and current.cancelling() > 0


__all__ = ["ChatOrchestrator"]
