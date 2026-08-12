"""
efi/utils/bounded.py

Словарь, который не растёт бесконечно.

Зачем. Эфи — не веб-сервис, который перезапускают на каждом деплое: она
месяцами работает одним процессом на телефоне. За это время через неё
проходят все чаты, куда её позвали, и все люди, которые там писали, — а
подсистемы поведения держат по записи на каждого:

    anti_repeat            последние 32 реплики НА КАЖДЫЙ чат
    conversation_lifecycle состояние диалога на КАЖДУЮ пару (человек, чат)
    affinity               снимок близости на каждый чат
    silence_monitor        время последней активности на каждый чат
    typing_tracker         когда собеседник последний раз печатал

Ни один из этих словарей ничего не удалял. В группе на пять тысяч человек,
где каждый однажды что-то написал, это пять тысяч записей, которые останутся
до перезапуска процесса, — и растут они только вверх. Ни одна отдельно взятая
запись не велика, поэтому проблема и не бросается в глаза: она просто тихо
съедает память месяц за месяцем на устройстве, где её и так немного.

Два ограничителя, потому что случаи разные:

  * `max_entries` — потолок по числу записей, вытесняется самая давно не
    использованная (LRU). Годится для кэшей: вытесненная запись просто
    перечитается из БД.
  * `ttl` — записи старше срока перестают существовать. Годится для
    состояния «когда это последний раз происходило»: такие записи бесполезны
    ровно с того момента, как устарели. Ноль — законное значение («протухает
    сразу»), им пользуются тесты, где нужен заведомо истёкший срок.

Реализация поверх `OrderedDict`, а не `functools.lru_cache`: тут нужен не
кэш функции, а именно словарь, в который пишут снаружи и который переживает
итерирование и удаление по ключу.
"""

from __future__ import annotations

import time
from collections import OrderedDict
from collections.abc import Iterator, MutableMapping
from typing import TypeVar

_K = TypeVar("_K")
_V = TypeVar("_V")


class BoundedDict(MutableMapping[_K, _V]):
    """
    Словарь с потолком по числу записей и (необязательно) по их возрасту.

    Ведёт себя как обычный dict; отличий два:

      * при переполнении вытесняется запись, к которой дольше всего не
        обращались (и чтение, и запись считаются обращением);
      * при `ttl` записи старше срока не видны и удаляются при первом же
        обращении к словарю.

    Возраст меряется `time.monotonic()`, а не системными часами: перевод
    времени на устройстве не должен ни воскрешать протухшие записи, ни
    убивать свежие.
    """

    __slots__ = ("_data", "_max_entries", "_ttl")

    def __init__(self, *, max_entries: int, ttl: float | None = None) -> None:
        if max_entries < 1:
            raise ValueError("max_entries должен быть >= 1")
        if ttl is not None and ttl < 0:
            raise ValueError("ttl не может быть отрицательным (None — значит «не протухает»)")
        self._max_entries = max_entries
        self._ttl = ttl
        #: ключ -> (значение, когда записали/трогали)
        self._data: OrderedDict[_K, tuple[_V, float]] = OrderedDict()

    def __getitem__(self, key: _K) -> _V:
        self._drop_expired()
        try:
            value, _stamp = self._data[key]
        except KeyError:
            raise KeyError(key) from None
        # Обращение обновляет и позицию в LRU, и отметку времени: запись, к
        # которой продолжают ходить, живая, сколько бы ей ни было от роду.
        self._data.move_to_end(key)
        self._data[key] = (value, time.monotonic())
        return value

    def __setitem__(self, key: _K, value: _V) -> None:
        self._drop_expired()
        self._data[key] = (value, time.monotonic())
        self._data.move_to_end(key)
        while len(self._data) > self._max_entries:
            self._data.popitem(last=False)

    def __delitem__(self, key: _K) -> None:
        del self._data[key]

    def __iter__(self) -> Iterator[_K]:
        self._drop_expired()
        # Копия ключей: вызывающая сторона часто удаляет по ходу обхода
        # («пройтись по чатам и убрать отработавшие»), а менять словарь во
        # время итерирования нельзя.
        return iter(list(self._data))

    def __len__(self) -> int:
        self._drop_expired()
        return len(self._data)

    def __repr__(self) -> str:
        return f"BoundedDict(max_entries={self._max_entries}, ttl={self._ttl}, len={len(self._data)})"

    def _drop_expired(self) -> None:
        if self._ttl is None or not self._data:
            return
        deadline = time.monotonic() - self._ttl
        # Порядок в OrderedDict — по последнему обращению, значит протухшие
        # записи всегда в начале: как только встретили свежую, дальше идти
        # незачем. Это делает уборку O(числа реально протухших), а не O(n) на
        # каждое обращение к словарю.
        while self._data:
            key = next(iter(self._data))
            _value, stamp = self._data[key]
            if stamp > deadline:
                return
            del self._data[key]


__all__ = ["BoundedDict"]
