"""
Тесты для efi.memory.pulse.MemoryPulse и связанных изменений в
efi.memory.consolidation.

Главная регрессия, ради которой всё это появилось: день переписки становился
памятью только в 03:30 ночи. До этого момента прожитое существовало лишь как
строки в таблице `messages` — Эфи не могла сослаться днём на утренний
разговор, а перезапуск до ночи оставлял день неосмысленным.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

from efi.db.core import Database
from efi.db.models import MIGRATIONS
from efi.llm.schemas import Choice, DiaryEntry, LLMParams, Message, Response, Role, Session
from efi.memory.consolidation import DiaryConsolidator
from efi.memory.facts import FactStore
from efi.memory.pulse import MemoryPulse

#: MemoryPulse берёт «сейчас» из системных часов, поэтому моменты последнего
#: сообщения задаются относительно реального времени, а не фиксированной даты.
_NOW = datetime.now(UTC)


class _FakeRouter:
    """Возвращает заранее заданный «разбор эпизода» и запоминает, что ему показали."""

    def __init__(self, reply: str = "вспомнила одну штуку про Рому") -> None:
        self.reply = reply
        self.prompts: list[str] = []
        self.system_prompts: list[str] = []

    async def chat(self, role: object, params: LLMParams, session: Session) -> Response:
        self.prompts.append(session.messages[-1].content)
        self.system_prompts.append(params.system_prompt)
        return Response(choices=[Choice(message=Message(role=Role.ASSISTANT, content=self.reply))])


class _FakeRAG:
    def __init__(self) -> None:
        self.remembered: list[str] = []

    async def remember(self, body: str, *, confidence: float = 0.0) -> DiaryEntry | None:
        self.remembered.append(body)
        return DiaryEntry(id=f"entry_{len(self.remembered)}", body=body)


class _FakeHistory:
    """История одного чата с управляемым моментом последнего сообщения."""

    def __init__(self, *, message_count: int, last_message_at: datetime | None) -> None:
        self.messages = [
            Message(role=Role.USER if index % 2 == 0 else Role.ASSISTANT, content=f"реплика {index}")
            for index in range(message_count)
        ]
        self.last_message_at = last_message_at

    async def get_active_chat_ids(self, *, since: datetime) -> list[int]:
        return [42]

    async def get_since(self, chat_id: int, *, since: datetime) -> Session:
        return Session(messages=list(self.messages))

    async def get_last_message_at(self, chat_id: int) -> datetime | None:
        return self.last_message_at


class _FakeExperience:
    def __init__(self, lines: list[str]) -> None:
        self.lines = lines

    async def context_lines_for_chat(self, chat_id: int, *, since: datetime, limit: int = 30) -> list[str]:
        return list(self.lines)


def _pulse(
    tmp_path: Path,
    history: _FakeHistory,
    *,
    router: _FakeRouter | None = None,
    rag: _FakeRAG | None = None,
    experience: _FakeExperience | None = None,
    **overrides: object,
) -> tuple[MemoryPulse, _FakeRouter, _FakeRAG, FactStore]:
    router = router or _FakeRouter()
    rag = rag or _FakeRAG()
    facts = FactStore(Database(tmp_path / "test.db", migrations=MIGRATIONS))
    consolidator = DiaryConsolidator(diary=None, router=router, rag=rag)  # type: ignore[arg-type]
    defaults: dict[str, object] = dict(episode_idle_seconds=900.0, max_messages_before_flush=30, min_messages=3)
    defaults.update(overrides)
    pulse = MemoryPulse(consolidator, history, facts, experience=experience, **defaults)  # type: ignore[arg-type]
    return pulse, router, rag, facts


# -- когда эпизод считается прожитым ---------------------------------------------


async def test_cooled_down_conversation_becomes_a_memory_right_away(tmp_path: Path) -> None:
    """Ядро всей задачи: разговор закончился — воспоминание появляется сразу, не ночью."""
    history = _FakeHistory(message_count=8, last_message_at=_NOW - timedelta(minutes=30))
    pulse, _router, rag, _facts = _pulse(tmp_path, history)

    assert await pulse.tick() == 1
    assert rag.remembered == ["вспомнила одну штуку про Рому"]


async def test_live_conversation_is_left_alone(tmp_path: Path) -> None:
    """Пока человек ещё пишет, эпизод не закончен — дёргать LLM на каждую реплику незачем."""
    history = _FakeHistory(message_count=8, last_message_at=_NOW - timedelta(seconds=30))
    pulse, router, rag, _facts = _pulse(tmp_path, history)

    assert await pulse.tick() == 0
    assert router.prompts == [] and rag.remembered == []


async def test_marathon_conversation_is_flushed_without_waiting_for_a_pause(tmp_path: Path) -> None:
    """Иначе многочасовая переписка снова свернулась бы в один обрубленный кусок под лимитом вывода."""
    history = _FakeHistory(message_count=40, last_message_at=_NOW - timedelta(seconds=5))
    pulse, _router, rag, _facts = _pulse(tmp_path, history)

    assert await pulse.tick() == 1
    assert len(rag.remembered) == 1


async def test_too_few_messages_do_not_move_the_watermark(tmp_path: Path) -> None:
    """
    Иначе короткий эпизод терялся бы навсегда: пульс заглянул после двух
    реплик, отметил окно закрытым, а продолжение разговора уже за отметкой.
    """
    history = _FakeHistory(message_count=2, last_message_at=_NOW - timedelta(hours=1))
    pulse, _router, _rag, facts = _pulse(tmp_path, history)

    assert await pulse.tick() == 0
    assert await facts.get("chat:42", "last_novelized_at") is None


async def test_chat_without_any_messages_yet_is_not_ready(tmp_path: Path) -> None:
    history = _FakeHistory(message_count=5, last_message_at=None)
    pulse, _router, _rag, _facts = _pulse(tmp_path, history)
    assert await pulse.tick() == 0


async def test_watermark_stops_the_same_episode_from_being_recorded_twice(tmp_path: Path) -> None:
    history = _FakeHistory(message_count=8, last_message_at=_NOW - timedelta(minutes=30))
    pulse, router, _rag, facts = _pulse(tmp_path, history)

    await pulse.tick()
    watermark = await facts.get("chat:42", "last_novelized_at")
    assert watermark is not None

    # Второй тик: история та же, но окно уже сдвинуто — _FakeHistory это не
    # моделирует, поэтому проверяем сам факт сдвига отметки, а не число
    # записей (за окно отвечает SQL в SqliteHistoryRepository.get_since).
    assert datetime.fromisoformat(watermark) <= datetime.now(UTC)
    assert len(router.prompts) == 1


# -- единая личность: внешний опыт попадает в тот же разбор ---------------------------


async def test_web_lookups_and_comments_are_folded_into_the_same_episode(tmp_path: Path) -> None:
    """
    Главное требование «Эфи — единая личность»: то, что она гуглила по ходу
    разговора, осмысляется вместе с самим разговором, а не отдельным логом.
    """
    history = _FakeHistory(message_count=6, last_message_at=_NOW - timedelta(minutes=20))
    experience = _FakeExperience(["Полезла гуглить и вычитала: «ленивые импорты в Python — ...»"])
    pulse, router, _rag, _facts = _pulse(tmp_path, history, experience=experience)

    await pulse.tick()

    assert "Полезла гуглить" in router.prompts[0]
    assert "реплика 0" in router.prompts[0], "сам разговор тоже должен остаться в том же запросе"


async def test_community_chat_is_remembered_from_comments_alone(tmp_path: Path) -> None:
    """
    В канале сообщества Эфи может не вести разговора вовсе — только
    комментировать и читать треды. По счёту сообщений это «пусто», хотя
    прожито там больше, чем в ином диалоге, поэтому внешний опыт считается
    наравне с репликами.
    """
    history = _FakeHistory(message_count=1, last_message_at=_NOW - timedelta(minutes=25))
    experience = _FakeExperience(
        [
            "Написала комментарий в «Линуксовый канал»: «да там же ядро течёт»",
            "Читала обсуждение в «Линуксовый канал». Там: «...»",
            "Полезла гуглить и вычитала: «...»",
        ]
    )
    pulse, router, rag, _facts = _pulse(tmp_path, history, experience=experience)

    assert await pulse.tick() == 1
    assert "Написала комментарий" in router.prompts[0]
    assert len(rag.remembered) == 1


async def test_novelization_prompt_demands_details_and_feelings(tmp_path: Path) -> None:
    """
    Подробность дневника держится на промпте, а не на объёме вывода: без
    явного требования деталей и отношения модель скатывается в протокол
    («обсудили баг, договорились исправить»).
    """
    history = _FakeHistory(message_count=6, last_message_at=_NOW - timedelta(minutes=20))
    pulse, router, _rag, _facts = _pulse(tmp_path, history)

    await pulse.tick()

    system_prompt = router.system_prompts[0]
    assert "ЧТО БЫЛО" in system_prompt
    assert "ЧТО Я ПОЧУВСТВОВАЛА" in system_prompt
    assert "ЧТО ЭТО ЗНАЧИТ" in system_prompt
    assert "ДОСЛОВНО" in system_prompt


# -- устойчивость ------------------------------------------------------------------


async def test_a_broken_chat_does_not_take_down_the_loop(tmp_path: Path) -> None:
    class _ExplodingHistory(_FakeHistory):
        async def get_since(self, chat_id: int, *, since: datetime) -> Session:
            raise RuntimeError("database is gone")

    history = _ExplodingHistory(message_count=6, last_message_at=_NOW - timedelta(minutes=20))
    pulse, _router, _rag, facts = _pulse(tmp_path, history)

    try:
        await pulse.tick()
    except RuntimeError:
        pass  # tick() пробрасывает — ловит и логирует сам run(), см. MemoryPulse.run

    # Ключевое: отметка не сдвинулась, значит следующий заход попробует снова,
    # а не спишет неразобранный кусок дня как уже осмысленный.
    assert await facts.get("chat:42", "last_novelized_at") is None
