"""
efi/db/sticker_descriptions.py

Кэш описаний стикеров: стабильный file_unique_id -> описание от vision.

Зачем отдельной таблицей, хотя описание можно генерировать на каждый приход
стикера. Стикер должен доходить до модели не как file_id+эмодзи, а как
описание («кот смеётся (id 7)») — иначе модель не понимает, ЧТО на стикере.
Но vision-описание это дорогой запрос к LLM, а один и тот же стикер люди
присылают десятки раз. Таблица позволяет описать стикер ровно один раз, а
дальше брать описание из кэша без единого обращения к vision.

Почему id у записи свой, а не Telegram file_unique_id. Модель видит в промпте
короткий id («(id 7)») и вызывает send_sticker по нему. Двадцатизначный
file_unique_id в промпте был бы тем же самым «стикер как id», от которого мы
уходим. Поэтому у каждой записи Telegram-ключ хранится под капотом, а наружу
для модели торчит короткий AUTOINCREMENT `sticker_id`.

`file_id` хранится отдельно от file_unique_id и обновляется при каждой новой
встрече стикера (touch): Telegram-овский file_id для одного и того же файла
может меняться между сессиями, и для send_sticker нужен именно актуальный.

Кэш поверх БД, как у efi/db/chat_directory.py: lookup читается на каждый
приход стикера, а меняется запись раз в жизни стикера (плюс счётчик встреч).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from efi.db.core import Database
from efi.utils.bounded import BoundedDict

logger = logging.getLogger(__name__)

#: Потолок кэша. Записи вытесняются по LRU и перечитываются из БД —
#: вытеснение здесь ничего не теряет.
_MAX_CACHED_STICKERS = 1024


@dataclass(slots=True, frozen=True)
class StickerDescription:
    """Что известно про один стикер. `sticker_id` — короткий id для модели."""

    sticker_id: int
    file_unique_id: str
    file_id: str
    description: str
    seen_count: int = 1
    last_seen_at: str = ""


class StickerDescriptionStore:
    """
    Персистентный кэш описаний стикеров поверх таблицы `sticker_descriptions`.

    Пишется из телеграм-обработчика входящих стикеров (efi/telegram/handlers.py:
    vision вызывается только при промахе кэша), читается инструментом send_sticker
    (по `sticker_id` -> рабочий file_id, efi/tools/telegram_actions/stickers.py)
    и сборщиком промпта (список известных стикеров, efi/prompts/builder.py).

    Методы не бросают наружу: кэш — вспомогательное знание, и ни один сбой
    SQLite не должен ни ронять доставку сообщения, ни отменять отправку стикера.
    Аналогично efi/db/chat_directory.py.
    """

    def __init__(self, database: Database) -> None:
        self._database = database
        # Ключ по file_unique_id — основной путь «пришёл стикер, есть ли описание».
        self._by_unique: BoundedDict[str, StickerDescription] = BoundedDict(max_entries=_MAX_CACHED_STICKERS)
        # Ключ по короткому sticker_id — путь send_sticker и промпт-блока.
        self._by_id: BoundedDict[int, StickerDescription] = BoundedDict(max_entries=_MAX_CACHED_STICKERS)

    def _cache(self, row: StickerDescription) -> StickerDescription:
        self._by_unique[row.file_unique_id] = row
        self._by_id[row.sticker_id] = row
        return row

    @staticmethod
    def _from_row(row: object) -> StickerDescription | None:
        record = dict(row)  # type: ignore[call-overload]  # aiosqlite.Row поддерживает dict()
        if "sticker_id" not in record or "file_unique_id" not in record:
            return None
        try:
            return StickerDescription(
                sticker_id=int(record["sticker_id"]),
                file_unique_id=str(record["file_unique_id"]),
                file_id=str(record.get("file_id") or ""),
                description=str(record.get("description") or ""),
                seen_count=int(record.get("seen_count") or 1),
                last_seen_at=str(record.get("last_seen_at") or ""),
            )
        except (TypeError, ValueError):
            return None

    async def find(self, file_unique_id: str) -> StickerDescription | None:
        """Описание стикера по его стабильному Telegram id, или None, если стикер ещё не описан."""
        if not file_unique_id:
            return None
        cached = self._by_unique.get(file_unique_id)
        if cached is not None:
            return cached

        try:
            row = await self._database.fetch_one(
                "SELECT * FROM sticker_descriptions WHERE file_unique_id = ?", (file_unique_id,)
            )
        except Exception:
            logger.warning(
                "sticker_descriptions: не удалось прочитать file_unique_id=%s", file_unique_id, exc_info=True
            )
            return None

        if row is None:
            return None
        stored = self._from_row(row)
        return self._cache(stored) if stored is not None else None

    async def remember(self, file_unique_id: str, file_id: str, description: str) -> StickerDescription | None:
        """
        Сохраняет первое описание стикера (upsert): вызывается ровно один раз,
        после успешного vision-зова. Повторный приход этого же стикера её не
        вызывает (см. touch). Возвращает запись с коротким sticker_id или None
        при сбое.
        """
        if not file_unique_id or not description.strip():
            return None
        now = datetime.now(UTC).isoformat()
        try:
            await self._database.execute(
                """
                INSERT INTO sticker_descriptions
                    (file_unique_id, file_id, description, seen_count, first_seen_at, last_seen_at)
                VALUES (?, ?, ?, 1, ?, ?)
                ON CONFLICT (file_unique_id) DO UPDATE SET
                    file_id      = excluded.file_id,
                    seen_count   = sticker_descriptions.seen_count + 1,
                    last_seen_at = excluded.last_seen_at
                """,
                (file_unique_id, file_id, description.strip(), now, now),
            )
        except Exception:
            logger.warning(
                "sticker_descriptions: не удалось запомнить file_unique_id=%s", file_unique_id, exc_info=True
            )
            return None

        # Читаем обратно: только так можно узнать AUTOINCREMENT sticker_id.
        # Сначала инвалидируем кэш: UPSERT мог обновить file_id/seen_count,
        # а лежащая в памяти запись уже устарела бы.
        self._by_unique.pop(file_unique_id, None)
        return await self.find(file_unique_id)

    async def touch(self, file_unique_id: str, file_id: str) -> StickerDescription | None:
        """
        Стикер с уже известным описанием встретился снова: обновляем актуальный
        file_id (он меняется между сессиями) и счётчик встреч. Описание НЕ
        трогается — ради этого кэш и существует.
        """
        if not file_unique_id:
            return None
        cached = self._by_unique.get(file_unique_id)
        if cached is None:
            return await self.find(file_unique_id)

        now = datetime.now(UTC).isoformat()
        try:
            await self._database.execute(
                """
                UPDATE sticker_descriptions
                SET file_id = ?, seen_count = seen_count + 1, last_seen_at = ?
                WHERE file_unique_id = ?
                """,
                (file_id or cached.file_id, now, file_unique_id),
            )
        except Exception:
            logger.debug("sticker_descriptions: не удалось обновить file_unique_id=%s", file_unique_id, exc_info=True)
            return cached

        updated = StickerDescription(
            sticker_id=cached.sticker_id,
            file_unique_id=cached.file_unique_id,
            file_id=file_id or cached.file_id,
            description=cached.description,
            seen_count=cached.seen_count + 1,
            last_seen_at=now,
        )
        return self._cache(updated)

    async def by_id(self, sticker_id: int) -> StickerDescription | None:
        """Запись по короткому id для модели — то, чем пользуется send_sticker."""
        cached = self._by_id.get(sticker_id)
        if cached is not None:
            return cached

        try:
            row = await self._database.fetch_one(
                "SELECT * FROM sticker_descriptions WHERE sticker_id = ?", (sticker_id,)
            )
        except Exception:
            logger.warning("sticker_descriptions: не удалось прочитать sticker_id=%s", sticker_id, exc_info=True)
            return None

        stored = self._from_row(row) if row is not None else None
        return self._cache(stored) if stored is not None else None

    async def recent(self, limit: int = 20) -> list[StickerDescription]:
        """Последние известные стикеры, самые свежие первыми — для блока «[Известные стикеры]» в промпте."""
        try:
            rows = await self._database.fetch_all(
                "SELECT * FROM sticker_descriptions ORDER BY last_seen_at DESC, sticker_id DESC LIMIT ?",
                (limit,),
            )
        except Exception:
            logger.warning("sticker_descriptions: не удалось прочитать список известных стикеров", exc_info=True)
            return []

        result: list[StickerDescription] = []
        for row in rows:
            stored = self._from_row(row)
            if stored is not None:
                result.append(self._cache(stored))
        return result


__all__ = ["StickerDescription", "StickerDescriptionStore"]
