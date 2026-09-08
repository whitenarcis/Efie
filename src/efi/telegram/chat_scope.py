"""
efi/telegram/chat_scope.py

Что за чат стоит за chat_id: личка один на один, группа или канал-вещание.

Модуль появился из конкретного случая. Спонтанный пинг ушёл в чат
`chat_id=-2041871692` — группу, где у Эфи админка, — и обработался ровно так
же, как личная переписка: системный промпт не содержал ни слова о том, что
это не диалог с человеком (блок «[О чате]» собирается по `payload.chat_type`,
а у проактивных уведомлений payload пустой), поэтому модель писала «первой»
в общий чат так, будто пишет одному собеседнику.

Отсюда два вывода, на которых стоит этот модуль:

  * тип чата обязан быть известен ВСЕГДА, а не только когда событие пришло
    из входящего сообщения Pyrogram (см. efi.db.chat_directory.ChatDirectory
    — она помнит тип между событиями);
  * даже когда про чат не известно ничего, сам по себе chat_id уже отвечает
    на главный вопрос — «личка это или нет». В Telegram id личного чата
    совпадает с user_id и всегда положителен, id группы всегда отрицателен.
    Ошибиться тут нельзя по построению, и это делает проверку доступной
    даже для чата, о котором в базе нет ни строчки, — например, для
    накопленной ДО этой правки истории.

Чистый модуль без I/O и без зависимости от Pyrogram: сюда передаются уже
готовые строки (имя `pyrogram.enums.ChatType`) и числа.
"""

from __future__ import annotations

from enum import StrEnum

#: Граница «Bot API»-представления id: супергруппы и каналы кодируются как
#: -100 + внутренний id, то есть всегда меньше этого числа. Обычные группы
#: лежат между ним и нулём. Отличить супергруппу от канала по одному лишь id
#: нельзя — формат у них общий, поэтому такой id даёт ChatKind.UNKNOWN, а не
#: догадку (см. classify_chat_id).
_CHANNEL_ID_BASE = -1_000_000_000_000


class ChatKind(StrEnum):
    """
    Род чата с точки зрения поведения Эфи.

    PRIVATE  — переписка один на один (включая ЛС с ботом): единственный род
               чата, где уместно писать первой.
    GROUP    — группа или супергруппа: несколько людей, общий контекст.
    CHANNEL  — канал-вещание: сообщение видят все подписчики.
    UNKNOWN  — точно НЕ личка (id отрицательный), но группа это или канал —
               по id неразличимо, а подтверждения типа ещё не было.
    """

    PRIVATE = "private"
    GROUP = "group"
    CHANNEL = "channel"
    UNKNOWN = "unknown"

    @property
    def is_one_on_one(self) -> bool:
        """
        Личка ли это. Именно этот вопрос решает право писать первой, поэтому
        UNKNOWN отвечает «нет»: неизвестный отрицательный id — это заведомо
        не диалог с человеком, и обращаться с ним как с личкой нельзя.
        """
        return self is ChatKind.PRIVATE


#: Имена pyrogram.enums.ChatType -> род чата. Строками, а не самим enum:
#: модуль не должен тянуть Pyrogram ради классификации, а в payload
#: уведомления тип и так уже лежит строкой (см. handlers._build_chat_context).
_CHAT_TYPE_NAMES: dict[str, ChatKind] = {
    "PRIVATE": ChatKind.PRIVATE,
    "BOT": ChatKind.PRIVATE,
    "GROUP": ChatKind.GROUP,
    "SUPERGROUP": ChatKind.GROUP,
    "CHANNEL": ChatKind.CHANNEL,
}


def classify_chat_id(chat_id: int | None) -> ChatKind:
    """
    Род чата по одному лишь id — то, что известно всегда и не требует ни
    базы, ни обращения к Telegram.

    Положительный id — личка (в Telegram id приватного чата равен user_id
    собеседника). Отрицательный — группа либо канал; какой именно, по id не
    определить, поэтому «-100…» даёт UNKNOWN. Для решения «можно ли писать
    сюда первой» этого достаточно: UNKNOWN — уже не личка.
    """
    if chat_id is None:
        return ChatKind.UNKNOWN
    if chat_id > 0:
        return ChatKind.PRIVATE
    if chat_id > _CHANNEL_ID_BASE:
        return ChatKind.GROUP
    return ChatKind.UNKNOWN


def kind_from_chat_type(chat_type: str | None) -> ChatKind | None:
    """
    Род чата по имени `pyrogram.enums.ChatType` ("PRIVATE", "SUPERGROUP", …).
    None — если типа нет или он незнаком; тогда вызывающая сторона
    откатывается на classify_chat_id.
    """
    if not chat_type:
        return None
    return _CHAT_TYPE_NAMES.get(chat_type.upper())


def resolve_chat_kind(chat_type: str | None, chat_id: int | None) -> ChatKind:
    """
    Лучшее, что известно про чат: подтверждённый тип, если он есть, иначе
    вывод из id.

    Порядок именно такой: тип из Telegram точен и различает группу и канал,
    id — грубее, но доступен всегда.
    """
    return kind_from_chat_type(chat_type) or classify_chat_id(chat_id)


__all__ = ["ChatKind", "classify_chat_id", "kind_from_chat_type", "resolve_chat_kind"]
