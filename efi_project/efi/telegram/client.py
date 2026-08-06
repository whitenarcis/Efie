"""
efi/telegram/client.py

Тонкая асинхронная обёртка над Pyrogram/Pyrofork Client: отправка сообщений
(с интеграцией Humanizer — typing-индикатор, задержка по WPM, редкие
опечатки), смена статусов, отправка медиа, реакции.

Структурно удовлетворяет efi.tools.telegram_actions.send_message.MessageSender
(есть async send_message(chat_id, text)), поэтому может быть передана
напрямую в SendMessageTool без дополнительной адаптации.

Примечание: конкретные имена методов Pyrogram — особенно `send_reaction`,
это относительно новая функциональность, различающаяся между ванильным
Pyrogram и форками вроде Pyrofork — стоит сверить с версией, реально
установленной в проекте, перед первым запуском.
"""

from __future__ import annotations

import logging
from pathlib import Path

from pyrogram import Client
from pyrogram.enums import ChatAction

from efi.config.schema import HumanizerSettings
from efi.humanizer.message_splitting import split_into_messages
from efi.humanizer.typing_simulation import simulate_typing_delay
from efi.humanizer.typos import inject_typo

logger = logging.getLogger(__name__)


class TelegramClientWrapper:
    """
    Владеет экземпляром Pyrogram Client и применяет Humanizer к каждому
    исходящему текстовому сообщению единообразно — независимо от того, что
    именно инициировало отправку (обычный ответ, спонтанный пинг, follow-up
    и т.п., всё идёт через SendMessageTool -> этот класс).

    Явно НЕ содержит проверку на самоповтор — это политика ("стоит ли вообще
    отправлять именно этот текст"), она решается в SendMessageTool ДО вызова
    send_message. Здесь — только механика доставки уже принятого решения.
    """

    def __init__(self, client: Client, humanizer_settings: HumanizerSettings) -> None:
        self._client = client
        self._humanizer_settings = humanizer_settings

    async def start(self) -> None:
        await self._client.start()
        logger.info("telegram: client started")

    async def stop(self) -> None:
        await self._client.stop()
        logger.info("telegram: client stopped")

    async def send_message(self, chat_id: int, text: str) -> None:
        """
        Отправляет текст, предварительно разбив его на цепочку сообщений
        (efi.humanizer.message_splitting) — каждое со своим typing-индикатором,
        задержкой по WPM и редкой опечаткой, отправляются последовательно.
        Так серия из нескольких коротких реплик выглядит как живой человек,
        печатающий одну за другой, а не как одно сообщение, порубленное
        символами переноса строки.
        """
        chunks = split_into_messages(text, self._humanizer_settings)
        if not chunks:
            logger.debug("telegram: send_message called with empty text for chat_id=%s, nothing to send", chat_id)
            return

        for chunk in chunks:
            humanized_chunk = inject_typo(chunk, self._humanizer_settings)

            try:
                await self._client.send_chat_action(chat_id, ChatAction.TYPING)
            except Exception:
                # Статус "печатает" — не критичная функциональность; если
                # Telegram его не принял (например, чат уже закрыт), само
                # сообщение всё равно должно уйти.
                logger.debug("telegram: failed to send typing action to chat_id=%s", chat_id, exc_info=True)

            await simulate_typing_delay(humanized_chunk, self._humanizer_settings)
            await self._client.send_message(chat_id, humanized_chunk)

    async def send_photo(self, chat_id: int, photo_path: str | Path, *, caption: str = "") -> None:
        await self._client.send_chat_action(chat_id, ChatAction.UPLOAD_PHOTO)
        await self._client.send_photo(chat_id, str(photo_path), caption=caption)

    async def send_voice(self, chat_id: int, voice_path: str | Path) -> None:
        await self._client.send_chat_action(chat_id, ChatAction.UPLOAD_VOICE)
        await self._client.send_voice(chat_id, str(voice_path))

    async def send_sticker(self, chat_id: int, sticker_file_id: str) -> None:
        await self._client.send_sticker(chat_id, sticker_file_id)

    async def edit_message(self, chat_id: int, message_id: int, text: str) -> None:
        await self._client.edit_message_text(chat_id, message_id, text)

    async def forward_message(self, from_chat_id: int, message_id: int, to_chat_id: int) -> None:
        await self._client.forward_messages(to_chat_id, from_chat_id, message_id)

    async def join_chat(self, chat_identifier: str) -> None:
        await self._client.join_chat(chat_identifier)

    async def leave_chat(self, chat_id: int) -> None:
        await self._client.leave_chat(chat_id)

    async def search_chats(self, query: str, *, limit: int = 10) -> list[tuple[int, str]]:
        """
        Ищет среди известных диалогов по названию/юзернейму. Pyrogram не даёт
        выделенного search-эндпоинта для локального списка диалогов —
        перебираем get_dialogs() и матчим подстроку регистронезависимо. На
        очень больших списках диалогов (тысячи) это не самое быстрое решение,
        но для личного аккаунта с разумным числом чатов — рабочее.
        """
        query_lower = query.lower()
        matches: list[tuple[int, str]] = []
        async for dialog in self._client.get_dialogs():
            chat = dialog.chat
            title = chat.title or chat.first_name or chat.username or str(chat.id)
            username = chat.username or ""
            if query_lower in title.lower() or query_lower in username.lower():
                matches.append((chat.id, title))
            if len(matches) >= limit:
                break
        return matches

    async def react(self, chat_id: int, message_id: int, emoji: str) -> None:
        """
        Ставит эмодзи-реакцию на сообщение. Требует Pyrofork либо достаточно
        свежий Pyrogram с поддержкой реакций — если метод недоступен в
        установленной версии, вызов бросит AttributeError, и это лучше
        обнаружить сразу при первом использовании, чем проглатывать молча.
        """
        await self._client.send_reaction(chat_id, message_id, emoji)


__all__ = ["TelegramClientWrapper"]
