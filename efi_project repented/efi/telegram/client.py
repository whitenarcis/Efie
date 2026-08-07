"""
efi/telegram/client.py

Тонкая асинхронная обёртка над Pyrogram/Pyrofork Client: отправка сообщений
(с интеграцией Humanizer — typing-индикатор, задержка по WPM, редкие
опечатки с шансом самокоррекции), смена статусов, отправка медиа, реакции,
отметка сообщений прочитанными.

Структурно удовлетворяет efi.tools.telegram_actions.send_message.MessageSender
(есть async send_message(chat_id, text, *, reply_to_message_id=None)),
поэтому может быть передана напрямую в SendMessageTool без дополнительной
адаптации.

Примечание: конкретные имена методов Pyrogram — особенно `send_reaction` —
это относительно новая функциональность, различающаяся между ванильным
Pyrogram и форками вроде Pyrofork — стоит сверить с версией, реально
установленной в проекте, перед первым запуском.
"""

from __future__ import annotations

import asyncio
import logging
import random
from pathlib import Path

from pyrogram import Client
from pyrogram.enums import ChatAction
from pyrogram.types import Message as PyrogramMessage

from efi.config.schema import HumanizerSettings
from efi.humanizer.message_splitting import first_chunk_typing_delay, split_into_messages
from efi.humanizer.typing_simulation import simulate_typing_delay
from efi.humanizer.typos import inject_typo

logger = logging.getLogger(__name__)

_SELF_CORRECT_DELAY_RANGE = (0.8, 2.5)


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
        # Держим ссылки на фоновые задачи самокоррекции, чтобы их не собрал
        # GC до завершения (стандартная идиома для "оторванных" asyncio.Task).
        self._background_tasks: set[asyncio.Task] = set()

    async def start(self) -> None:
        await self._client.start()
        logger.info("telegram: client started")

    async def stop(self) -> None:
        await self._client.stop()
        logger.info("telegram: client stopped")

    async def send_message(
        self,
        chat_id: int,
        text: str,
        *,
        reply_to_message_id: int | None = None,
        llm_generation_time: float | None = None,
    ) -> None:
        """
        Отправляет текст, предварительно разбив его на цепочку сообщений
        (efi.humanizer.message_splitting) — каждое со своим typing-индикатором,
        задержкой по WPM и редкой опечаткой, отправляются последовательно.

        `llm_generation_time` (если передан — см. efi.tools.telegram_actions.
        send_message.SendMessageTool) засчитывается как время печати ПЕРВОГО
        куска цепочки (efi.humanizer.message_splitting.first_chunk_typing_delay):
        Worker уже транслировал TYPING, пока ждал ответ LLM, так что не имеет
        смысла ждать ЕЩЁ раз с нуля — только оставшуюся разницу, если она
        вообще есть. Для всех последующих кусков ("///"-разбивка) действует
        обычный calculate_typing_delay.

        Если в конкретном куске случилась опечатка (inject_typo реально
        изменил текст), с вероятностью `typo_self_correct_probability`
        планирует её самокоррекцию — короткая пауза, затем edit_message_text
        с изначальным (правильным) текстом. Это то самое "иногда сама себя
        поправляет", которого не хватало — раньше опечатки никогда не
        исправлялись, что не похоже на реального человека.

        `reply_to_message_id` (если задан) применяется ТОЛЬКО к первому куску
        серии — явный Reply-статус имеет смысл один раз, на весь блок реплик,
        а не на каждый отдельный кусок ("///"-разбивку) по отдельности.

        Примечание: параметр Pyrogram называется `reply_to_message_id` в
        классическом MTProto API — в некоторых свежих версиях Pyrogram/Pyrofork
        (после перехода на Bot API 7.x-совместимые reply_parameters) имя
        параметра могло измениться; стоит свериться с установленной версией,
        если Reply не сработает.
        """
        chunks = split_into_messages(text, self._humanizer_settings)
        if not chunks:
            logger.debug("telegram: send_message called with empty text for chat_id=%s, nothing to send", chat_id)
            return

        for index, chunk in enumerate(chunks):
            humanized_chunk = inject_typo(chunk, self._humanizer_settings)

            try:
                await self._client.send_chat_action(chat_id, ChatAction.TYPING)
            except Exception:
                # Статус "печатает" — не критичная функциональность; если
                # Telegram его не принял (например, чат уже закрыт), само
                # сообщение всё равно должно уйти.
                logger.debug("telegram: failed to send typing action to chat_id=%s", chat_id, exc_info=True)

            if index == 0 and llm_generation_time is not None:
                delay = first_chunk_typing_delay(
                    humanized_chunk, self._humanizer_settings, llm_generation_time=llm_generation_time
                )
                if delay > 0:
                    await asyncio.sleep(delay)
            else:
                await simulate_typing_delay(humanized_chunk, self._humanizer_settings)

            reply_id = reply_to_message_id if index == 0 else None
            sent_message = await self._client.send_message(chat_id, humanized_chunk, reply_to_message_id=reply_id)

            if humanized_chunk != chunk:
                self._maybe_schedule_self_correction(chat_id, sent_message, chunk)

    def _maybe_schedule_self_correction(self, chat_id: int, sent_message: PyrogramMessage, correct_text: str) -> None:
        if random.random() >= self._humanizer_settings.typo_self_correct_probability:
            return
        message_id = getattr(sent_message, "id", None)
        if message_id is None:
            logger.debug("telegram: send_message did not return a usable message id, skipping self-correction")
            return
        task = asyncio.create_task(self._self_correct(chat_id, message_id, correct_text))
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _self_correct(self, chat_id: int, message_id: int, correct_text: str) -> None:
        """Короткая пауза (будто заметила опечатку и тут же поправилась), затем правка на изначально правильный текст."""
        await asyncio.sleep(random.uniform(*_SELF_CORRECT_DELAY_RANGE))
        try:
            await self._client.edit_message_text(chat_id, message_id, correct_text)
            logger.debug("telegram: self-corrected typo in message_id=%s (chat_id=%s)", message_id, chat_id)
        except Exception:
            logger.debug("telegram: self-correct edit failed for message_id=%s", message_id, exc_info=True)

    async def send_typing_action(self, chat_id: int) -> None:
        """
        Разовый пинг статуса "печатает", отдельно от полного send_message
        цикла — используется efi.notifications.worker.Worker, пока ждёт
        ответа LLM (может занимать несколько секунд, особенно с несколькими
        раундами tool-calling), чтобы собеседник видел живой TYPING, а не
        тишину между "прочитано" и первым сообщением. Не критичная
        функциональность — сбой не должен ничего ронять.
        """
        try:
            await self._client.send_chat_action(chat_id, ChatAction.TYPING)
        except Exception:
            logger.debug("telegram: failed to send typing pulse to chat_id=%s", chat_id, exc_info=True)

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

    async def mark_as_read(self, chat_id: int) -> None:
        """
        Отмечает историю чата прочитанной. Без этого сообщения собеседника
        навсегда остаются со статусом "отправлено/доставлено" и никогда не
        "прочитано" — для живого человека, реально читающего сообщения, это
        неестественно. Не критичная функциональность — сбой не должен ронять
        обработку самого сообщения.
        """
        try:
            await self._client.read_chat_history(chat_id)
        except Exception:
            logger.debug("telegram: failed to mark chat_id=%s as read", chat_id, exc_info=True)

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
