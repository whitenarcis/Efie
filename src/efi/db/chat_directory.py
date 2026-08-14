"""
efi/db/chat_directory.py

Справочник чатов: chat_id -> тип чата и название.

Зачем он нужен отдельной таблицей, хотя тип чата приходит в каждом входящем
сообщении. Проактивные события (спонтанный пинг, пинг по затишью, follow-up,
органический пинг от фонового исследователя) рождаются НЕ из сообщения, а из
таймера: у них нет ни Pyrogram-объекта чата, ни отправителя — только
chat_id. До этой правки они и не несли ничего о чате, и системный промпт
собирался вообще без блока «[О чате]» (efi.prompts.builder.
_build_chat_context_block читает `payload["chat_type"]`). Наружу это выглядело
так: Эфи писала первой в группу, где у неё админка, ровно теми же словами,
какими пишет человеку в личку, — потому что из промпта было физически не
узнать, что это не личка.

Справочник закрывает разрыв: тип чата, увиденный однажды в реальном
сообщении, помнится и доступен в любой момент, включая проактивный путь.
Чего в нём ещё нет (чат, где Эфи ни разу не видела сообщения при живом
процессе), решает efi.telegram.chat_scope.classify_chat_id по самому id —
грубее, но без единого шанса ошибиться в главном вопросе «личка или нет».

Кэш поверх БД, как у efi.behavior.affinity.AffinityTracker: справочник
читается на каждом проактивном событии, а меняется он раз в жизни чата.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime

from efi.db.core import Database
from efi.telegram.chat_scope import ChatKind, resolve_chat_kind
from efi.utils.bounded import BoundedDict

logger = logging.getLogger(__name__)

#: Потолок кэша. Записи вытесняются по LRU и перечитываются из БД —
#: вытеснение здесь ничего не теряет.
_MAX_CACHED_CHATS = 512


@dataclass(slots=True, frozen=True)
class ChatDescriptor:
    """Что известно про чат. `chat_type` — имя pyrogram.enums.ChatType, пустая строка = не подтверждён."""

    chat_id: int
    chat_type: str = ""
    title: str = ""

    @property
    def kind(self) -> ChatKind:
        """Род чата: по подтверждённому типу, а если его нет — по самому id (см. efi/telegram/chat_scope.py)."""
        return resolve_chat_kind(self.chat_type or None, self.chat_id)


class ChatDirectory:
    """
    Персистентный справочник чатов поверх таблицы `chat_directory`.

    Пишется из телеграм-обработчиков (каждое входящее сообщение уже несёт
    тип и название чата), читается проактивным путём — воркером при сборке
    контекста уведомления и планировщиком спонтанных пингов при отборе
    чатов-кандидатов.

    Оба метода не бросают наружу: справочник — вспомогательное знание, и ни
    один сбой SQLite не должен ни ронять доставку сообщения, ни отменять уже
    принятое решение написать. При сбое чтения возвращается дескриптор без
    подтверждённого типа — то есть тот же ответ, что и для незнакомого чата.
    """

    def __init__(self, database: Database) -> None:
        self._database = database
        self._cache: BoundedDict[int, ChatDescriptor] = BoundedDict(max_entries=_MAX_CACHED_CHATS)

    async def remember(self, chat_id: int, *, chat_type: str | None = None, title: str | None = None) -> None:
        """
        Запоминает тип/название чата. Идемпотентно: если ровно это уже
        лежит в кэше, записи в БД не будет — метод вызывается на КАЖДОЕ
        входящее сообщение, а меняется тип чата примерно никогда.

        Пустые значения не затирают уже известные: у канала нет отправителя,
        у лички нет названия, и частичное знание не должно стирать полное.
        """
        descriptor = ChatDescriptor(
            chat_id=chat_id,
            chat_type=(chat_type or "").upper(),
            title=(title or "").strip(),
        )
        known = self._cache.get(chat_id)
        if known is not None:
            descriptor = ChatDescriptor(
                chat_id=chat_id,
                chat_type=descriptor.chat_type or known.chat_type,
                title=descriptor.title or known.title,
            )
            if descriptor == known:
                return

        self._cache[chat_id] = descriptor
        try:
            await self._database.execute(
                """
                INSERT INTO chat_directory (chat_id, chat_type, title, updated_at)
                VALUES (?, ?, ?, ?)
                ON CONFLICT (chat_id) DO UPDATE SET
                    chat_type = CASE WHEN excluded.chat_type = '' THEN chat_directory.chat_type
                                     ELSE excluded.chat_type END,
                    title     = CASE WHEN excluded.title = '' THEN chat_directory.title
                                     ELSE excluded.title END,
                    updated_at = excluded.updated_at
                """,
                (chat_id, descriptor.chat_type, descriptor.title, datetime.now(UTC).isoformat()),
            )
        except Exception:
            # Кэш уже обновлён — в пределах этого запуска справочник знает
            # про чат всё равно. Потерять запись на диске не страшно,
            # уронить из-за неё доставку сообщения — страшно.
            logger.warning("chat_directory: не удалось запомнить chat_id=%s", chat_id, exc_info=True)

    async def describe(self, chat_id: int) -> ChatDescriptor:
        """Что известно про чат. Для незнакомого — дескриптор без типа, род чата в нём выводится из id."""
        cached = self._cache.get(chat_id)
        if cached is not None:
            return cached

        try:
            row = await self._database.fetch_one(
                "SELECT chat_type, title FROM chat_directory WHERE chat_id = ?", (chat_id,)
            )
        except Exception:
            logger.warning("chat_directory: не удалось прочитать chat_id=%s", chat_id, exc_info=True)
            return ChatDescriptor(chat_id=chat_id)

        descriptor = (
            ChatDescriptor(chat_id=chat_id, chat_type=row["chat_type"] or "", title=row["title"] or "")
            if row is not None
            else ChatDescriptor(chat_id=chat_id)
        )
        self._cache[chat_id] = descriptor
        return descriptor

    async def kind_of(self, chat_id: int | None) -> ChatKind:
        """Род чата — короткий путь для тех, кому нужен только он (отбор кандидатов на пинг, гейт воркера)."""
        if chat_id is None:
            return ChatKind.UNKNOWN
        return (await self.describe(chat_id)).kind


__all__ = ["ChatDescriptor", "ChatDirectory"]
