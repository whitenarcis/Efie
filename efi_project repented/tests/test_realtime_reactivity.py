"""
Тесты на «Эфи не отстаёт на реплику»: сборка быстрой пачки входящих,
прерывание устаревшей генерации, выборочный reply на строчку из пачки и
свободный ритм бабблов.

Общая регрессия, ради которой это всё: между приходом сообщения и последним
отправленным бабблом проходят десятки секунд (busy-задержка, генерация LLM,
паузы по WPM). Всё это время Эфи считала, что отвечает на актуальную реплику,
и договаривала ответ на устаревший вопрос, даже если разговор уже ушёл вперёд.
"""

from __future__ import annotations

import asyncio

import pytest

from efi.config.schema import HumanizerSettings
from efi.humanizer.message_splitting import is_short_bubble, short_bubble_delay, split_into_messages
from efi.humanizer.reply_selector import parse_reply_tags
from efi.telegram.buffer import InboundMessageBuffer
from efi.telegram.chat_orchestrator import ChatOrchestrator
from efi.telegram.queue import BubbleQueue, OutboundBubble

# -- буфер входящих: плавающее окно сборки ------------------------------------------


class _FlushRecorder:
    def __init__(self) -> None:
        self.batches: list[tuple[int, list[str]]] = []

    async def __call__(self, chat_id: int, items: list[str]) -> None:
        self.batches.append((chat_id, items))


def _buffer(recorder: _FlushRecorder, **overrides: object) -> InboundMessageBuffer[str]:
    defaults: dict[str, object] = dict(window_range=(0.05, 0.05), max_wait_seconds=5.0)
    defaults.update(overrides)
    return InboundMessageBuffer(recorder, **defaults)  # type: ignore[arg-type]


async def test_fast_messages_arrive_as_one_batch() -> None:
    """«найду романтику» / «и пох» / «пошел есть» — одна порция разговора, а не три запуска LLM."""
    recorder = _FlushRecorder()
    buffer = _buffer(recorder)

    for text in ("найду романтику", "и пох", "пошел есть"):
        await buffer.add(42, text)
        await asyncio.sleep(0.01)  # быстрее окна сборки

    await asyncio.sleep(0.2)
    assert recorder.batches == [(42, ["найду романтику", "и пох", "пошел есть"])]


async def test_each_message_slides_the_window() -> None:
    """
    Ключевое отличие от прежнего дебаунсера: окно сдвигается на каждом
    сообщении. Раньше буфер держался ровно столько, сколько горел статус
    «печатает», а между двумя короткими репликами он успевает погаснуть — и
    генерация запускалась на первую строчку.
    """
    recorder = _FlushRecorder()
    buffer = _buffer(recorder, window_range=(0.12, 0.12))

    await buffer.add(42, "первое")
    await asyncio.sleep(0.08)
    assert recorder.batches == [], "окно ещё не истекло"
    await buffer.add(42, "второе")
    await asyncio.sleep(0.08)
    assert recorder.batches == [], "второе сообщение должно было сдвинуть окно"

    await asyncio.sleep(0.12)
    assert recorder.batches == [(42, ["первое", "второе"])]


async def test_separate_chats_do_not_mix() -> None:
    recorder = _FlushRecorder()
    buffer = _buffer(recorder)

    await buffer.add(1, "чат один")
    await buffer.add(2, "чат два")
    await asyncio.sleep(0.2)

    assert sorted(recorder.batches) == [(1, ["чат один"]), (2, ["чат два"])]


async def test_nonstop_typing_still_hits_the_ceiling() -> None:
    """Собеседник, печатающий без остановки, не должен откладывать ответ бесконечно."""

    class _AlwaysTyping:
        def is_typing(self, chat_id: int) -> bool:
            return True

    recorder = _FlushRecorder()
    buffer = _buffer(recorder, typing_tracker=_AlwaysTyping(), max_wait_seconds=0.2)

    await buffer.add(42, "бесконечная мысль")
    await asyncio.sleep(0.5)
    assert recorder.batches == [(42, ["бесконечная мысль"])]


async def test_incoming_message_interrupts_immediately_on_arrival() -> None:
    """
    Прерывание идёт в момент ПРИЁМА, до всякого ожидания: смысл в том, что
    Эфи замолкает в ту же секунду, а не через полторы, когда закроется окно.
    """
    interrupted: list[int] = []

    async def _on_interrupt(chat_id: int) -> None:
        interrupted.append(chat_id)

    buffer = _buffer(_FlushRecorder(), on_interrupt=_on_interrupt)
    await buffer.add(42, "новая реплика")

    assert interrupted == [42], "сигнал должен уйти до окна сборки, а не после флаша"


async def test_shutdown_flushes_unfinished_batches() -> None:
    recorder = _FlushRecorder()
    buffer = _buffer(recorder, window_range=(10.0, 10.0))

    await buffer.add(42, "не потеряйся")
    await buffer.flush_all()

    assert recorder.batches == [(42, ["не потеряйся"])]


# -- оркестратор: отмена устаревшей генерации ------------------------------------


async def test_interrupt_cancels_a_running_generation() -> None:
    orchestrator = ChatOrchestrator()
    finished = False

    async def _slow_generation() -> None:
        nonlocal finished
        await asyncio.sleep(5.0)
        finished = True

    runner = asyncio.create_task(orchestrator.run(42, _slow_generation()))
    await asyncio.sleep(0.05)
    assert orchestrator.is_generating(42)

    assert await orchestrator.interrupt(42) is True
    assert await runner is False, "прерванная генерация не считается доведённой до конца"
    assert finished is False
    assert orchestrator.is_generating(42) is False


async def test_interrupt_waits_for_the_cleanup_block() -> None:
    """
    Прерванный ход дописывает в историю уже отправленные бабблы. Новая
    генерация должна стартовать ПОСЛЕ этого, иначе соберёт контекст без
    последней реплики Эфи и повторит её.
    """
    cleaned_up = False

    async def _generation_with_cleanup() -> None:
        nonlocal cleaned_up
        try:
            await asyncio.sleep(5.0)
        except asyncio.CancelledError:
            await asyncio.sleep(0.05)  # запись в историю
            cleaned_up = True
            raise

    orchestrator = ChatOrchestrator()
    runner = asyncio.create_task(orchestrator.run(42, _generation_with_cleanup()))
    await asyncio.sleep(0.05)

    await orchestrator.interrupt(42)
    assert cleaned_up is True
    await runner


async def test_interrupting_an_idle_chat_is_a_no_op() -> None:
    assert await ChatOrchestrator().interrupt(42) is False


async def test_finished_generation_reports_success() -> None:
    orchestrator = ChatOrchestrator()

    async def _quick() -> None:
        return None

    assert await orchestrator.run(42, _quick()) is True
    assert orchestrator.is_generating(42) is False


async def test_notifications_without_a_chat_are_not_interruptible() -> None:
    """Глобальная ночная задача не привязана к чату — прерывать её нечем и незачем."""
    orchestrator = ChatOrchestrator()
    done = False

    async def _global_task() -> None:
        nonlocal done
        done = True

    assert await orchestrator.run(None, _global_task()) is True
    assert done is True


async def test_generation_errors_are_not_swallowed_as_interruption() -> None:
    orchestrator = ChatOrchestrator()

    async def _failing() -> None:
        raise RuntimeError("LLM упала")

    with pytest.raises(RuntimeError):
        await orchestrator.run(42, _failing())
    assert orchestrator.is_generating(42) is False


async def test_shutdown_cancels_everything_in_flight() -> None:
    orchestrator = ChatOrchestrator()
    runner = asyncio.create_task(orchestrator.run(42, asyncio.sleep(5.0)))
    await asyncio.sleep(0.05)

    await orchestrator.cancel_all()
    assert await runner is False
    assert orchestrator.is_generating(42) is False


# -- очередь бабблов: что ушло и что сброшено --------------------------------------


def _queue(*texts: str) -> BubbleQueue:
    return BubbleQueue([OutboundBubble(text) for text in texts])


def test_queue_tracks_what_was_actually_delivered() -> None:
    queue = _queue("раз", "два", "три")

    queue.next_bubble()
    queue.mark_delivered()
    queue.next_bubble()
    queue.mark_delivered()

    assert queue.delivered_texts == ["раз", "два"]
    assert [bubble.text for bubble in queue.pending] == ["три"]


def test_interrupted_bubble_counts_as_undelivered() -> None:
    """Взятый в работу баббл подтверждает только mark_delivered — отмена посреди паузы его не доставила."""
    queue = _queue("раз", "два")
    queue.next_bubble()  # взяли в работу и прервались на паузе

    assert queue.delivered_texts == []
    assert [bubble.text for bubble in queue.pending] == ["раз", "два"]


def test_delivered_text_can_differ_from_the_intended_one() -> None:
    """В историю должна попасть отправленная версия — гуманизатор мог подмешать опечатку."""
    queue = _queue("привет")
    queue.next_bubble()
    queue.mark_delivered(text="привте")
    assert queue.delivered_texts == ["привте"]


def test_first_bubble_flag_flips_after_the_first_delivery() -> None:
    queue = _queue("раз", "два")
    queue.next_bubble()
    assert queue.is_first is True
    queue.mark_delivered()
    assert queue.is_first is False


def test_queue_refuses_to_skip_confirmation() -> None:
    queue = _queue("раз", "два")
    queue.next_bubble()
    with pytest.raises(RuntimeError):
        queue.next_bubble()


# -- выборочный reply на строчку из пачки ------------------------------------------


def test_reply_tag_binds_a_bubble_to_a_specific_message() -> None:
    bubbles = parse_reply_tags(
        ["[reply:102] это про вторую строчку", "а это просто продолжение"],
        incoming_message_ids=[101, 102],
    )
    assert bubbles[0].reply_to_message_id == 102
    assert bubbles[0].text == "это про вторую строчку"
    assert bubbles[1].reply_to_message_id is None


def test_invented_message_id_is_dropped() -> None:
    """Модель не может сослаться на старое сообщение: их id ей нигде не показываются."""
    bubbles = parse_reply_tags(["[reply:999] мимо", "и ещё"], incoming_message_ids=[101, 102])
    assert bubbles[0].reply_to_message_id is None
    assert bubbles[0].text == "мимо"


def test_single_incoming_message_never_gets_a_reply() -> None:
    """Свайп на единственную реплику, на которую ты и так отвечаешь, — чистый шум."""
    bubbles = parse_reply_tags(["[reply:101] ага"], incoming_message_ids=[101])
    assert bubbles[0].reply_to_message_id is None


def test_tagging_every_bubble_with_the_same_target_is_stripped() -> None:
    """Это не выборочный ответ, а привычка размечать всё подряд."""
    bubbles = parse_reply_tags(
        ["[reply:101] раз", "[reply:101] два"], incoming_message_ids=[101, 102]
    )
    assert all(bubble.reply_to_message_id is None for bubble in bubbles)


def test_partial_tagging_survives() -> None:
    bubbles = parse_reply_tags(
        ["[reply:101] отвечаю на первую", "общий текст"], incoming_message_ids=[101, 102]
    )
    assert bubbles[0].reply_to_message_id == 101


def test_tag_in_the_middle_is_left_as_text() -> None:
    """Середина текста — уже содержание реплики, а не разметка."""
    bubbles = parse_reply_tags(["смотри [reply:101] вот так"], incoming_message_ids=[101, 102])
    assert bubbles[0].reply_to_message_id is None
    assert "[reply:101]" in bubbles[0].text


def test_sloppy_tag_spacing_is_tolerated() -> None:
    bubbles = parse_reply_tags(
        ["[ reply : 102 ] норм", "второй"], incoming_message_ids=[101, 102]
    )
    assert bubbles[0].reply_to_message_id == 102


def test_empty_bubbles_are_dropped() -> None:
    assert parse_reply_tags(["[reply:101]", "  ", "текст"], incoming_message_ids=[101, 102]) == [
        OutboundBubble("текст")
    ]


# -- ритм бабблов -------------------------------------------------------------------


def _humanizer(**overrides: object) -> HumanizerSettings:
    return HumanizerSettings(**overrides)  # type: ignore[arg-type]


def test_long_stream_of_consciousness_is_not_capped_at_three() -> None:
    """
    «Поток мыслей» из 8 коротких реплик раньше схлопывался в 5, где последнее
    было слипшимся комом из остатка.
    """
    text = " /// ".join(f"мысль {index}" for index in range(8))
    assert split_into_messages(text, _humanizer()) == [f"мысль {index}" for index in range(8)]


def test_absurd_burst_still_hits_the_safety_ceiling() -> None:
    text = " /// ".join(f"кусок {index}" for index in range(40))
    chunks = split_into_messages(text, _humanizer(max_messages_per_burst=12))
    assert len(chunks) == 12
    assert "кусок 39" in chunks[-1], "остаток склеивается в последнее сообщение, а не отбрасывается"


def test_short_reply_stays_one_message() -> None:
    assert split_into_messages("ага, поняла", _humanizer()) == ["ага, поняла"]


def test_short_bubbles_are_recognized() -> None:
    assert is_short_bubble("прикинь") is True
    assert is_short_bubble("а ты?") is True
    assert is_short_bubble("я ток щас узнала") is False
    assert is_short_bubble("") is False


def test_short_bubble_delay_is_sub_second() -> None:
    """
    Обычный расчёт зажат снизу typing_delay_min_seconds (1.8с), из-за чего
    серия коротышей растягивалась на полминуты и читалась как медленный бот.
    """
    settings = _humanizer()
    delays = [short_bubble_delay(settings) for _ in range(50)]
    assert all(0.3 <= delay <= 0.8 for delay in delays)
    assert max(delays) < settings.typing_delay_min_seconds
