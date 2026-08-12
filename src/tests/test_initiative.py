"""
Тесты права заговорить первой (efi.behavior.initiative + ping_reason).

Регрессия из живой переписки. Инициативные сообщения Эфи выглядели так:

    09:00  ну чё там твой вайбкод, ещё не всё сломал?
    10:15  эй / ты там ещё не утонул в своём коде?
    10:28  эй / ты там не сдох от перетренированности?
    15:28  эй / ты там живой ещё или в коде утонул?
    16:28  эй / ты там ещё не окончательно в коде утонул?

Пять сообщений подряд, ни на одно не ответили. Причин ровно три, и каждая
чинится отдельно.

1. Правила «написал и не получил ответа — жди» не было вовсе. Инициативных
   служб три (спонтанный пинг, монитор тишины, органический пинг), каждая
   решала за себя, и ни одна не знала, ответил ли человек.

2. У инициативы не было повода. В уведомление подставлялось «просто чтобы
   напомнить о себе» — из такого текста «эй, ты там живой?» и есть
   единственный возможный вывод.

3. Одно сообщение приходило двумя бабблами («эй» + вопрос). Промпт этого
   не разрешал, но промпт — пожелание, а не гарантия.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from efi.behavior.initiative import DEFAULT_SILENCE_FORGIVENESS, InitiativeGate
from efi.behavior.ping_reason import PingReasonBuilder
from efi.db.core import Database
from efi.db.models import MIGRATIONS
from efi.memory.facts import FactStore
from efi.memory.validator import RESERVED_ATTRIBUTES

_CHAT_ID = 777123


def _gate(tmp_path: Path, *, db_name: str = "efi.db") -> tuple[InitiativeGate, FactStore]:
    facts = FactStore(Database(tmp_path / db_name, migrations=MIGRATIONS))
    return InitiativeGate(facts), facts


# -- правило «одно неотвеченное сообщение» -------------------------------------


async def test_the_first_initiative_is_allowed(tmp_path: Path) -> None:
    gate, _facts = _gate(tmp_path)

    assert await gate.may_initiate(_CHAT_ID) is True


async def test_a_second_message_without_an_answer_is_refused(tmp_path: Path) -> None:
    """
    Главная регрессия. Второе «эй» не увеличивает шанс ответа — оно только
    показывает, что пишущий не заметил молчания.
    """
    gate, _facts = _gate(tmp_path)
    await gate.record_initiative(_CHAT_ID)

    assert await gate.may_initiate(_CHAT_ID) is False


async def test_a_reply_reopens_the_right_to_write_first(tmp_path: Path) -> None:
    gate, _facts = _gate(tmp_path)
    await gate.record_initiative(_CHAT_ID)

    await gate.record_reply(_CHAT_ID)

    assert await gate.may_initiate(_CHAT_ID) is True


async def test_the_ban_is_per_chat(tmp_path: Path) -> None:
    """Молчание одного человека не запрещает писать другому."""
    gate, _facts = _gate(tmp_path)
    await gate.record_initiative(_CHAT_ID)

    assert await gate.may_initiate(-999) is True


async def test_the_ban_survives_a_restart(tmp_path: Path) -> None:
    """
    Иначе правило обходилось бы само собой: Эфи живёт на телефоне, процесс
    перезапускается регулярно, и после каждого перезапуска она начинала бы
    писать снова как ни в чём не бывало.
    """
    first, _facts = _gate(tmp_path, db_name="shared.db")
    await first.record_initiative(_CHAT_ID)

    second, _again = _gate(tmp_path, db_name="shared.db")

    assert await second.may_initiate(_CHAT_ID) is False


async def test_long_silence_eventually_stops_counting(tmp_path: Path) -> None:
    """
    Молчание длиной в несколько дней — это уже не «он не ответил на то
    сообщение», а обычная пауза в общении, и заговорить снова нормально.
    """
    gate, _facts = _gate(tmp_path)
    await gate.record_initiative(_CHAT_ID, now=datetime.now(UTC) - DEFAULT_SILENCE_FORGIVENESS - timedelta(hours=1))

    assert await gate.may_initiate(_CHAT_ID) is True


async def test_repeated_recording_does_not_push_the_deadline_forward(tmp_path: Path) -> None:
    """Иначе `forgiveness` не наступила бы никогда: каждая попытка сдвигала бы дату."""
    gate, facts = _gate(tmp_path)
    long_ago = datetime.now(UTC) - timedelta(days=1)
    await gate.record_initiative(_CHAT_ID, now=long_ago)

    await gate.record_initiative(_CHAT_ID, now=datetime.now(UTC))

    stored = await facts.get(f"chat:{_CHAT_ID}", "unanswered_initiative_at")
    assert stored == long_ago.isoformat()


async def test_a_corrupted_mark_does_not_silence_her_forever(tmp_path: Path) -> None:
    """Молчать из-за нечитаемой отметки хуже, чем один лишний раз написать."""
    gate, facts = _gate(tmp_path)
    await facts.upsert(f"chat:{_CHAT_ID}", "unanswered_initiative_at", "позавчера")

    assert await gate.may_initiate(_CHAT_ID) is True


def test_the_mark_is_out_of_the_models_reach() -> None:
    """
    Уговорив модель «забыть» отметку, правило можно было бы обойти прямо из
    разговора — поэтому ключ служебный.
    """
    assert "unanswered_initiative_at" in RESERVED_ATTRIBUTES


# -- повод для инициативы --------------------------------------------------------


async def test_without_a_reason_there_is_nothing_to_say(tmp_path: Path) -> None:
    """
    Главное правило: молчание лучше, чем «эй». Молчание читается как «занята
    своими делами», пустой пинг — как навязчивость.
    """
    assert await PingReasonBuilder().reason_for(_CHAT_ID) is None


async def test_an_incubated_thought_is_a_reason() -> None:
    async def thought() -> str | None:
        return "плёночные сканеры до сих пор быстрее половины современных"

    builder = PingReasonBuilder(incubated_thought_provider=thought)

    assert await builder.consume_incubated_thought() == "плёночные сканеры до сих пор быстрее половины современных"


async def test_a_remembered_fact_becomes_a_question(tmp_path: Path) -> None:
    """
    Ради этого и подключался конвейер строгой памяти: из «он работает
    монтажёром» получается нормальный человеческий вопрос про конкретную
    вещь, а не «ты там живой?».
    """
    from efi.memory.dedup import KnowledgeStore
    from efi.memory.people import PeopleStore
    from efi.memory.router import MemoryDomain
    from efi.memory.validator import ValidatedFact, compute_hash

    database = Database(tmp_path / "efi.db", migrations=MIGRATIONS)
    people = PeopleStore(database)
    await people.record_message(555, "привет", display_name="Рома", chat_id=_CHAT_ID)

    knowledge = KnowledgeStore(database)
    await knowledge.remember(
        ValidatedFact(
            domain=MemoryDomain.PERSONAL,
            entity_id="user:555",
            attribute="работа",
            value="монтажёр на телевидении",
            confidence=0.8,
            observed_at=datetime.now(UTC),
            source="test",
            normalized_hash=compute_hash(MemoryDomain.PERSONAL, "user:555", "работа", "монтажёр на телевидении"),
        )
    )

    reason = await PingReasonBuilder(knowledge=knowledge, people=people).reason_for(_CHAT_ID)

    assert reason is not None
    assert "монтажёр на телевидении" in reason
    assert "Спроси про это" in reason


async def test_facts_about_people_from_other_chats_are_not_a_reason(tmp_path: Path) -> None:
    """Иначе Эфи спрашивала бы одного человека про дела другого."""
    from efi.memory.dedup import KnowledgeStore
    from efi.memory.people import PeopleStore
    from efi.memory.router import MemoryDomain
    from efi.memory.validator import ValidatedFact, compute_hash

    database = Database(tmp_path / "efi.db", migrations=MIGRATIONS)
    people = PeopleStore(database)
    await people.record_message(555, "привет", display_name="Рома", chat_id=-1)

    knowledge = KnowledgeStore(database)
    await knowledge.remember(
        ValidatedFact(
            domain=MemoryDomain.PERSONAL,
            entity_id="user:555",
            attribute="работа",
            value="монтажёр",
            confidence=0.8,
            observed_at=datetime.now(UTC),
            source="test",
            normalized_hash=compute_hash(MemoryDomain.PERSONAL, "user:555", "работа", "монтажёр"),
        )
    )

    assert await PingReasonBuilder(knowledge=knowledge, people=people).reason_for(_CHAT_ID) is None


# -- инициативные службы соблюдают правило ----------------------------------------


async def test_the_spontaneous_ping_stays_quiet_without_a_reason() -> None:
    from efi.behavior.spontaneous_ping import SpontaneousPingScheduler
    from efi.notifications.manager import NotificationManager

    manager = NotificationManager(worker_count=1)

    async def candidates() -> list[int]:
        return [_CHAT_ID]

    scheduler = SpontaneousPingScheduler(manager, candidates, ping_probability=1.0)
    await scheduler._maybe_ping_candidates()

    assert manager.qsize() == 0, "без повода писать не о чем"


async def test_the_spontaneous_ping_respects_the_gate(tmp_path: Path) -> None:
    from efi.behavior.spontaneous_ping import SpontaneousPingScheduler
    from efi.notifications.manager import NotificationManager

    gate, _facts = _gate(tmp_path)
    await gate.record_initiative(_CHAT_ID)
    manager = NotificationManager(worker_count=1)

    async def candidates() -> list[int]:
        return [_CHAT_ID]

    async def thought() -> str | None:
        return "мысль, с которой правда стоило бы прийти"

    scheduler = SpontaneousPingScheduler(
        manager,
        candidates,
        ping_probability=1.0,
        reasons=PingReasonBuilder(incubated_thought_provider=thought),
        initiative=gate,
    )
    await scheduler._maybe_ping_candidates()

    assert manager.qsize() == 0, "на прошлое сообщение не ответили — даже с поводом молчим"


async def test_the_silence_monitor_respects_the_gate(tmp_path: Path) -> None:
    from efi.behavior.silence_monitor import SilenceMonitor
    from efi.notifications.manager import NotificationManager

    gate, _facts = _gate(tmp_path)
    await gate.record_initiative(_CHAT_ID)
    manager = NotificationManager(worker_count=1)
    monitor = SilenceMonitor(manager, silence_threshold=timedelta(seconds=0), initiative=gate)
    monitor.record_activity(_CHAT_ID)

    await monitor._check_silence()

    assert manager.qsize() == 0


async def test_the_silence_ping_never_asks_whether_he_is_alive() -> None:
    """Затишье — бедный повод, поэтому запреты в формулировке приходится называть прямо."""
    from efi.behavior.silence_monitor import _render_plain_silence

    text = _render_plain_silence(timedelta(hours=6))

    assert "ЗАПРЕЩЕНО" in text
    assert "ты там живой?" in text
    assert "проверк" in text


# -- одна реплика вместо «эй» + вопрос ---------------------------------------------


def test_a_proactive_message_is_collapsed_into_one_bubble() -> None:
    from efi.notifications.schemas import Notification, NotificationType
    from efi.tools.base import ToolContext
    from efi.tools.telegram_actions.send_message import _collapse_bubbles_if_proactive

    context = ToolContext(
        notification=Notification(type=NotificationType.SILENCE_PING, chat_id=_CHAT_ID, message="повод")
    )

    assert _collapse_bubbles_if_proactive("эй /// ты там живой?", context) == "эй ты там живой?"


def test_a_normal_reply_keeps_its_bubbles() -> None:
    """В ответе серия коротких реплик — это живая речь, а не нетерпение."""
    from efi.notifications.schemas import Notification, NotificationType
    from efi.tools.base import ToolContext
    from efi.tools.telegram_actions.send_message import _collapse_bubbles_if_proactive

    context = ToolContext(
        notification=Notification(type=NotificationType.USER_MESSAGE, chat_id=_CHAT_ID, message="привет")
    )
    text = "прикинь /// я тока щас узнала /// а ты?"

    assert _collapse_bubbles_if_proactive(text, context) == text


def test_the_prompt_bans_the_exact_phrases_it_used_to_produce() -> None:
    from efi.notifications.schemas import Notification, NotificationType
    from efi.prompts.builder import _build_proactive_brevity_block

    block = _build_proactive_brevity_block(
        Notification(type=NotificationType.SPONTANEOUS_PING, chat_id=_CHAT_ID, message="повод")
    )

    for phrase in ("«эй»", "ты там живой?", "не утонул в коде?"):
        assert phrase in block, f"{phrase} — ровно то, что она писала месяцами"


def test_a_plain_how_are_you_is_not_forbidden() -> None:
    """
    Запрещено допытываться, здесь ли собеседник, — а не спрашивать, как у
    него дела. «Как дела» живые люди пишут постоянно, и запрещать это значило
    бы лечить симптом вместо причины (причина была в отсутствии повода).
    """
    from efi.notifications.schemas import Notification, NotificationType
    from efi.prompts.builder import _build_proactive_brevity_block

    block = _build_proactive_brevity_block(
        Notification(type=NotificationType.SPONTANEOUS_PING, chat_id=_CHAT_ID, message="повод")
    )

    assert "«как дела» под запрет НЕ подпадает" in block


# -- поводов стало больше ---------------------------------------------------------


async def test_an_open_promise_is_the_strongest_reason(tmp_path: Path) -> None:
    """Обещание человек помнит и ждёт — оно важнее и своих находок, и вежливого интереса."""
    from efi.memory.working_memory import WorkingMemory

    memory = WorkingMemory(tmp_path / "wm.json")
    await memory.add_item("скинуть ссылку на тот сканер", chat_id=_CHAT_ID)

    reason = await PingReasonBuilder(working_memory=memory).reason_for(_CHAT_ID)

    assert reason is not None
    assert "скинуть ссылку на тот сканер" in reason


async def test_a_fresh_diary_entry_keeps_her_from_going_mute(tmp_path: Path) -> None:
    """
    Дневник наполняется с первого же разговора, а строгая память — только
    когда прозвучал устойчивый факт. Без этого источника Эфи молчала бы
    неделями после чистой установки.
    """
    from efi.llm.schemas import DiaryEntry, DiaryEntryMetadata
    from efi.memory.diary import Diary

    diary = Diary(tmp_path / "diary")
    await diary.save(
        DiaryEntry(
            id="fresh_1",
            metadata=DiaryEntryMetadata(confidence=0.5),
            body="Читала сегодня про плёночные сканеры и залипла на час.",
        )
    )

    reason = await PingReasonBuilder(diary=diary).reason_for(_CHAT_ID)

    assert reason is not None
    assert "плёночные сканеры" in reason


async def test_a_stale_diary_entry_is_not_a_reason(tmp_path: Path) -> None:
    """«Я на прошлой неделе читала» — уже не разговор, а натянутый повод."""
    from efi.llm.schemas import DiaryEntry, DiaryEntryMetadata
    from efi.memory.diary import Diary

    diary = Diary(tmp_path / "diary")
    await diary.save(
        DiaryEntry(
            id="old_1",
            metadata=DiaryEntryMetadata(confidence=0.5, created_at=datetime.now(UTC) - timedelta(days=9)),
            body="Что-то давнее и уже неактуальное.",
        )
    )

    assert await PingReasonBuilder(diary=diary).reason_for(_CHAT_ID) is None


async def test_a_promise_from_another_chat_is_not_a_reason(tmp_path: Path) -> None:
    from efi.memory.working_memory import WorkingMemory

    memory = WorkingMemory(tmp_path / "wm.json")
    await memory.add_item("это обещание из другого чата", chat_id=-999)

    assert await PingReasonBuilder(working_memory=memory).reason_for(_CHAT_ID) is None
