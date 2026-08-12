"""
efi/behavior/initiative.py

Право заговорить первой — общее для всех, кто это делает.

Проблема, ради которой модуль появился. Инициативных механизмов у Эфи
несколько (спонтанный пинг, монитор тишины, органический пинг с находкой), и
каждый решал сам за себя. Ни один не знал ни про остальных, ни — что гораздо
важнее — про то, ответил ли человек на прошлое сообщение. На практике это
выглядело так:

    09:00  ну чё там твой вайбкод, ещё не всё сломал?
    10:15  эй / ты там ещё не утонул в своём коде?
    10:28  эй / ты там не сдох от перетренированности?
    15:28  эй / ты там живой ещё или в коде утонул?
    16:28  эй / ты там ещё не окончательно в коде утонул?

Пять сообщений подряд, ни на одно не ответили. Живой человек так не пишет —
и дело не в частоте, которую можно подкрутить настройкой. Дело в правиле,
которого не было вовсе: **написал и не получил ответа — жди, а не пиши
снова**. Второе «эй» не увеличивает шанс ответа, оно только показывает, что
пишущий не заметил молчания.

Отсюда потолок: одно неотвеченное сообщение на чат. Не «не чаще раза в час»,
не «не больше трёх в день» — именно одно, до ответа. Такое правило не надо
подбирать, оно не ломается от смены интервалов и не зависит от того, сколько
инициативных механизмов появится дальше.

Что под правило НЕ подпадает: напоминания по прямой просьбе («напиши мне
через 10 минут»). Их человек заказал сам, и молчание в ответ на прошлое
сообщение не отменяет заказ — см. efi/behavior/reminders.py.

Состояние переживает перезапуск. Иначе правило обходилось бы само собой:
Эфи живёт на телефоне, процесс перезапускается регулярно, и после каждого
перезапуска она начинала бы писать снова как ни в чём не бывало.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta

from efi.memory.facts import FactStore

logger = logging.getLogger(__name__)

#: Ключ в FactStore под сущностью `chat:<id>`. Служебный: модель не должна
#: уметь его переписать (см. RESERVED_ATTRIBUTES в efi/memory/validator.py) —
#: иначе достаточно было бы уговорить её «забыть», что ответа не было.
UNANSWERED_KEY = "unanswered_initiative_at"

#: Через сколько неотвеченная инициатива перестаёт считаться висящей.
#:
#: Не «никогда»: молчание длиной в неделю — это уже не «он не ответил на то
#: сообщение», а просто пауза в общении, и заговорить снова нормально. Но и
#: не пара часов: смысл правила в том, чтобы не долбиться в тот же день.
DEFAULT_SILENCE_FORGIVENESS = timedelta(days=3)


class InitiativeGate:
    """
    Единственное место, которое решает, вправе ли Эфи писать первой в чат.

    Все инициативные службы спрашивают здесь, и ни одна не хранит своего
    мнения на этот счёт: правило «одно неотвеченное сообщение» имеет смысл
    только как общее. Три службы с тремя личными счётчиками дали бы ровно то,
    что было на скриншотах, — три сообщения вместо одного.
    """

    def __init__(
        self, facts: FactStore, *, forgiveness: timedelta = DEFAULT_SILENCE_FORGIVENESS
    ) -> None:
        self._facts = facts
        self._forgiveness = forgiveness

    async def may_initiate(self, chat_id: int, *, now: datetime | None = None) -> bool:
        """
        Можно ли сейчас написать в этот чат первой.

        Сбой чтения трактуется как «можно»: инициатива — не опасная операция,
        и глушить её из-за недоступной БД значило бы чинить одну проблему
        другой, менее заметной.
        """
        pending = await self._pending_since(chat_id)
        if pending is None:
            return True
        if (now or datetime.now(UTC)) - pending > self._forgiveness:
            # Молчание переросло в обычную паузу в общении — заговорить снова
            # нормально. Отметку снимаем здесь же, чтобы она не висела вечно.
            await self._clear(chat_id)
            return True
        logger.debug(
            "initiative: в chat_id=%s висит неотвеченное сообщение с %s — молчим",
            chat_id, pending.isoformat(timespec="minutes"),
        )
        return False

    async def record_initiative(self, chat_id: int, *, now: datetime | None = None) -> None:
        """
        Отмечает, что Эфи только что написала первой и ответа пока нет.

        Вызывается ПОСЛЕ фактической доставки (efi/notifications/worker.py):
        сообщение, которое не дошло, никого ни к чему не обязывает.

        Повторный вызов при уже висящей отметке её НЕ обновляет: иначе дата
        уползала бы вперёд с каждым сообщением, и `forgiveness` никогда бы не
        наступила.
        """
        if await self._pending_since(chat_id) is not None:
            return
        moment = now or datetime.now(UTC)
        try:
            await self._facts.upsert(f"chat:{chat_id}", UNANSWERED_KEY, moment.isoformat())
        except Exception:
            logger.warning("initiative: не удалось запомнить инициативу в chat_id=%s", chat_id, exc_info=True)

    async def record_reply(self, chat_id: int) -> None:
        """
        Человек написал — счёт обнуляется, и Эфи снова вправе заговорить
        первой, когда будет с чем.
        """
        if await self._pending_since(chat_id) is None:
            return
        await self._clear(chat_id)
        logger.debug("initiative: в chat_id=%s ответили — инициатива снова разрешена", chat_id)

    async def _pending_since(self, chat_id: int) -> datetime | None:
        try:
            raw = await self._facts.get(f"chat:{chat_id}", UNANSWERED_KEY)
        except Exception:
            logger.warning("initiative: не удалось прочитать состояние chat_id=%s", chat_id, exc_info=True)
            return None
        if not raw:
            return None
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            # Испорченное значение равносильно его отсутствию: молчать из-за
            # нечитаемой отметки хуже, чем один лишний раз написать.
            logger.warning("initiative: нечитаемая отметка %r в chat_id=%s, сбрасываю", raw, chat_id)
            await self._clear(chat_id)
            return None

    async def _clear(self, chat_id: int) -> None:
        try:
            await self._facts.upsert(f"chat:{chat_id}", UNANSWERED_KEY, "")
        except Exception:
            logger.warning("initiative: не удалось снять отметку в chat_id=%s", chat_id, exc_info=True)


__all__ = ["DEFAULT_SILENCE_FORGIVENESS", "UNANSWERED_KEY", "InitiativeGate"]
