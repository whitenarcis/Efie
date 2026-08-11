"""
efi/memory/working_memory.py

Рабочая память Эфи — аналог `important_things_to_remember` из C++-референса:
короткий горизонт (по умолчанию 3 дня), эмоциональное и физическое состояние
персонажа, список открытых задач/обещаний/напоминаний.

В отличие от Diary (десятки-сотни независимых записей с семантическим
поиском), рабочая память — один эволюционирующий снимок, поэтому хранится как
единый JSON-файл, а не набор markdown-файлов.

"Verbatim" в требовании означает: текст открытого пункта не переписывается
при плановой ротации/консолидации — обновляется только `last_updated` (через
`touch_item`) или `done` (через `mark_done`). Изменить сам текст пункта можно
только явно удалив старый и добавив новый через `add_item`.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

import aiofiles
import aiofiles.os
from pydantic import BaseModel, Field, ValidationError

from efi.behavior import energy as energy_model
from efi.utils.clock import local_now

logger = logging.getLogger(__name__)

_DEFAULT_HORIZON = timedelta(days=3)

#: Сколько живёт настроение, названное самой Эфи.
#:
#: Настроение — это состояние на сейчас, а не свойство характера. «Злая как
#: чёрт», сказанное во вторник, к пятнице неправда, и подставлять его в промпт
#: как текущее — значит заставлять её отыгрывать позавчерашний день. Восемь
#: часов — примерно «до конца этого куска суток»: утреннее настроение доживает
#: до вечера, но не до следующего утра.
STATE_TTL = timedelta(hours=8)


class WorkingMemoryItem(BaseModel):
    """Один открытый пункт: обещание, напоминание или незавершённая задача."""

    text: str
    created_at: datetime
    last_updated: datetime
    done: bool = False
    #: Срок, к которому обещание должно быть выполнено, и чат, в котором оно
    #: дано. Без них пункт был просто заметкой: «напиши мне через 10 минут»
    #: оседало текстом, и ничто в системе не могло узнать, что у него вообще
    #: есть срок и адресат (см. efi/behavior/reminders.py).
    due_at: datetime | None = None
    chat_id: int | None = None

    @property
    def is_overdue(self) -> bool:
        """Срок прошёл, а пункт всё ещё открыт — то самое «висит и ничего не происходит»."""
        return not self.done and self.due_at is not None and self.due_at < datetime.now(UTC)


class WorkingMemorySnapshot(BaseModel):
    """
    Снимок рабочей памяти целиком — то, что фактически подставляется в
    промпт как блок `<things_to_remember>` (формирование самого промпта —
    забота efi/prompts/, здесь только структурированные данные).
    """

    emotional_state: str = ""
    physical_state: str = ""
    energy: float = Field(
        default=0.7, ge=0.0, le=1.0,
        description="ЯКОРЬ энергии: последнее явно зафиксированное значение. Текущий уровень из него "
        "вычисляется на момент запроса (efi.behavior.energy.project), а не читается напрямую — "
        "энергия падает к ночи, тратится на разговор и восстанавливается со временем сама.",
    )
    energy_updated_at: datetime | None = Field(
        default=None,
        description="Когда якорь энергии был поставлен. None — снимок из версии до появления модели "
        "энергии: тогда якорь считается свежим, и старые файлы работают без миграции.",
    )
    #: Настроение — не постоянное свойство, а состояние на сейчас. Без отметки
    #: времени фраза «злая как чёрт», сказанная во вторник, ехала бы в промпт
    #: и в пятницу (см. STATE_TTL и WorkingMemory.describe).
    state_updated_at: datetime | None = None
    items: list[WorkingMemoryItem] = Field(default_factory=list)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


@dataclass(slots=True, frozen=True)
class SelfState:
    """
    Как Эфи себя ощущает прямо сейчас — то, что уходит в промпт и на дашборд.

    Пустых полей здесь нет по построению. «Состояние: не определено» —
    единственный вариант, которого у живого существа быть не может, а до
    появления этого типа именно он и стоял в промпте месяцами: строки
    заполнялись только добровольным вызовом инструмента, которого модель не
    делала.
    """

    emotional: str
    physical: str
    energy: energy_model.EnergyState
    #: True — состояние выведено из энергии и часа, а не названо самой Эфи.
    #: Дашборду это стоит показывать: «её слова» и «наша оценка» — разные
    #: вещи, и путать их не надо.
    is_derived: bool


class WorkingMemory:
    """
    Асинхронный доступ к рабочей памяти.

    `load()` — критический путь: читает JSON-файл один раз и дальше отдаёт
    закэшированный снимок, пока не будет вызвана мутирующая операция (`save`,
    `add_item`, `touch_item`, `mark_done`, `prune`, `update_state`) — каждая
    из них сама персистит изменения на диск. Если вызывающей стороне не важен
    момент, когда запись реально долетит до диска, она вольна обернуть вызов в
    `asyncio.create_task(...)`.
    """

    def __init__(self, path: Path, *, horizon: timedelta = _DEFAULT_HORIZON, timezone: str = "") -> None:
        self._path = path
        self._horizon = horizon
        #: Пояс нужен, потому что энергия зависит от ЧАСА СУТОК: по UTC на
        #: московском телефоне «глубокая ночь» пришлась бы на девять вечера.
        self._timezone = timezone
        self._lock = asyncio.Lock()
        self._cache: WorkingMemorySnapshot | None = None

    def describe(self, snapshot: WorkingMemorySnapshot, *, now: datetime | None = None) -> SelfState:
        """
        Текущее самоощущение: энергия на этот момент и состояние словами.

        Синхронная и без ввода-вывода — её зовут на критическом пути сборки
        промпта, где снимок уже загружен.

        Слова самой Эфи (`update_self_state`) имеют приоритет, пока не
        протухли; иначе состояние выводится из энергии и часа. Молчание
        модели больше не означает, что состояния нет.
        """
        moment = local_now(self._timezone, now=now)
        energy = energy_model.project(
            anchor=snapshot.energy, anchor_at=snapshot.energy_updated_at, now=moment
        )
        if self._explicit_state_is_fresh(snapshot, moment):
            return SelfState(
                emotional=snapshot.emotional_state or energy.label,
                physical=snapshot.physical_state or energy.label,
                energy=energy,
                is_derived=False,
            )
        return SelfState(emotional=energy.label, physical=energy.label, energy=energy, is_derived=True)

    @staticmethod
    def _explicit_state_is_fresh(snapshot: WorkingMemorySnapshot, moment: datetime) -> bool:
        if not snapshot.emotional_state and not snapshot.physical_state:
            return False
        if snapshot.state_updated_at is None:
            # Снимок из версии без отметки времени. Считаем состояние
            # протухшим, а не вечным: именно эти зависшие навсегда строки и
            # были проблемой.
            return False
        return moment - snapshot.state_updated_at < STATE_TTL

    async def load(self) -> WorkingMemorySnapshot:
        """Критический путь: возвращает текущий снимок, читая с диска только при первом обращении."""
        if self._cache is not None:
            return self._cache
        async with self._lock:
            if self._cache is not None:
                return self._cache
            self._cache = await self._read_from_disk()
            return self._cache

    async def _read_from_disk(self) -> WorkingMemorySnapshot:
        try:
            async with aiofiles.open(self._path, encoding="utf-8") as f:
                raw = await f.read()
        except FileNotFoundError:
            return WorkingMemorySnapshot()
        try:
            return WorkingMemorySnapshot.model_validate_json(raw)
        except ValidationError as exc:
            logger.warning("working_memory: corrupted snapshot at %s (%s), starting fresh", self._path, exc)
            return WorkingMemorySnapshot()

    async def save(self, snapshot: WorkingMemorySnapshot | None = None) -> WorkingMemorySnapshot:
        """
        Персистит снимок (текущий закэшированный, если явно не передан другой) на диск.

        Блокировка (`self._lock`) сериализует сами записи на диск — важно
        теперь, когда один экземпляр WorkingMemory может быть общим для
        нескольких параллельных Worker'ов (efi/notifications/worker.py):
        физическое/эмоциональное состояние у Эфи одно на всех чатов, а не
        по одному на чат. Если snapshot не передан явно, берём уже
        закэшированный объект НЕ через load() — load() сама может захватывать
        этот же лок при первом обращении, и вызов её изнутри уже занятого
        лока привёл бы к дедлоку (asyncio.Lock не реентерабельна).
        """
        if snapshot is None:
            snapshot = self._cache if self._cache is not None else await self.load()
        async with self._lock:
            snapshot.updated_at = datetime.now(UTC)
            await aiofiles.os.makedirs(self._path.parent, exist_ok=True)
            async with aiofiles.open(self._path, mode="w", encoding="utf-8") as f:
                await f.write(snapshot.model_dump_json(indent=2))
            self._cache = snapshot
        return snapshot

    async def update_state(
        self,
        *,
        emotional_state: str | None = None,
        physical_state: str | None = None,
        energy: float | None = None,
        now: datetime | None = None,
    ) -> WorkingMemorySnapshot:
        """
        Обновляет эмоциональное/физическое состояние и/или уровень энергии.

        Слова Эфи о себе сильнее любой модели: сказанное здесь становится
        новым якорем, от которого энергия дальше релаксирует как обычно.
        Поэтому вместе со значениями обязательно записывается момент — без
        него «я вымотана» осталось бы верным навсегда.
        """
        moment = now or datetime.now(UTC)
        snapshot = await self.load()
        if emotional_state is not None:
            snapshot.emotional_state = emotional_state
        if physical_state is not None:
            snapshot.physical_state = physical_state
        if emotional_state is not None or physical_state is not None:
            snapshot.state_updated_at = moment
        if energy is not None:
            snapshot.energy = max(0.0, min(energy, 1.0))
            snapshot.energy_updated_at = moment
        return await self.save(snapshot)

    async def spend_energy(self, *, turns: int = 1, now: datetime | None = None) -> WorkingMemorySnapshot:
        """
        Списывает энергию за проведённый разговор и переставляет якорь на
        «сейчас».

        Вызывается после КАЖДОГО отвеченного хода (efi/notifications/worker.py).
        Именно здесь энергия перестаёт быть константой: разговор её тратит,
        а время между разговорами возвращает к норме своего часа.

        Списывается от ТЕКУЩЕГО спроецированного значения, а не от старого
        якоря: иначе долгий перерыв, за который Эфи отдохнула, при первой же
        реплике откатился бы к позавчерашней усталости.
        """
        moment = now or datetime.now(UTC)
        snapshot = await self.load()
        projected = energy_model.project(
            anchor=snapshot.energy, anchor_at=snapshot.energy_updated_at, now=moment
        )
        snapshot.energy = energy_model.spend(projected.level, turns=turns)
        snapshot.energy_updated_at = moment
        return await self.save(snapshot)

    async def add_item(
        self, text: str, *, due_at: datetime | None = None, chat_id: int | None = None
    ) -> WorkingMemoryItem:
        """
        Добавляет новый открытый пункт (обещание/напоминание/задачу).

        `due_at`/`chat_id` необязательны: не у всякого обещания есть срок
        («скину ссылку, как найду»). Но если срок назван, он обязан дойти
        сюда — планировщик напоминаний берёт его именно отсюда, и пункт без
        срока для него не существует.
        """
        snapshot = await self.load()
        now = datetime.now(UTC)
        item = WorkingMemoryItem(
            text=text, created_at=now, last_updated=now, due_at=due_at, chat_id=chat_id
        )
        snapshot.items.append(item)
        await self.save(snapshot)
        return item

    async def touch_item(self, index: int) -> None:
        """
        Подтверждает актуальность пункта БЕЗ изменения текста (verbatim) —
        обновляет только `last_updated`, чтобы пункт не был вычищен `prune()`.
        """
        snapshot = await self.load()
        if 0 <= index < len(snapshot.items):
            snapshot.items[index].last_updated = datetime.now(UTC)
            await self.save(snapshot)

    async def mark_done(self, index: int) -> None:
        """Помечает пункт выполненным — он будет убран ближайшим `prune()`."""
        snapshot = await self.load()
        if 0 <= index < len(snapshot.items):
            snapshot.items[index].done = True
            await self.save(snapshot)

    async def find_and_mark_done(self, text_query: str) -> WorkingMemoryItem | None:
        """
        Находит первый ОТКРЫТЫЙ пункт, чей текст содержит `text_query`
        (регистронезависимая подстрока), и помечает его выполненным.
        Возвращает найденный пункт, либо None, если подходящего не нашлось.

        Текстовый поиск, а не индекс — предназначен для вызова инструментом
        модели (efi.tools.memory_tools.manage_promises.CompletePromiseTool),
        которой удобнее сослаться на обещание по смыслу, чем помнить его
        порядковый номер в списке; для короткого списка из нескольких
        открытых пунктов точного/подстрочного совпадения достаточно — тот же
        компромисс "дёшево и без ML", что и у memory/tfidf_fallback.py.
        """
        query = text_query.strip().lower()
        if not query:
            return None
        snapshot = await self.load()
        for item in snapshot.items:
            if not item.done and query in item.text.lower():
                item.done = True
                await self.save(snapshot)
                return item
        return None

    async def prune(self) -> int:
        """
        Убирает завершённые пункты и те, что не обновлялись дольше `horizon`
        (по умолчанию 3 дня — как в референсе). Возвращает число удалённых пунктов.
        """
        snapshot = await self.load()
        cutoff = datetime.now(UTC) - self._horizon
        kept = [item for item in snapshot.items if not item.done and item.last_updated >= cutoff]
        removed = len(snapshot.items) - len(kept)
        if removed:
            snapshot.items = kept
            await self.save(snapshot)
        return removed


__all__ = ["WorkingMemoryItem", "WorkingMemorySnapshot", "WorkingMemory"]
