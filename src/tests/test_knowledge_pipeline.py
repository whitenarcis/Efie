"""
Сквозные тесты конвейера строгой памяти: прожитый эпизод -> knowledge_facts.

Конвейер (parser -> validator -> разрешение сущностей -> dedup) был собран и
покрыт тестами по частям, но в живом приложении его никто не вызывал:
`MemoryIngestor` создавался в `EfiApp.__init__` и на этом всё. То есть вся
граница доверия существовала как код и не существовала как поведение — факты
в `knowledge_facts` не попадали никогда, а блок проверенных фактов в промпте
был пуст по построению.

Здесь проверяется именно склейка: что эпизод, попавший в дневник, тем же
проходом попадает в строгую память, что разрешение упоминаний получает
каталог известных людей, и что заданный уточняющий вопрос закрывается ответом
и не задаётся во второй раз.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from efi.behavior.ambiguity import PendingClarifications
from efi.db.core import Database
from efi.db.models import MIGRATIONS
from efi.llm.schemas import Choice, LLMParams, Message, Response, Role, Session
from efi.memory.catalog import apply_confirmed_answers, build_people_catalog
from efi.memory.consolidation import DiaryConsolidator
from efi.memory.dedup import KnowledgeStore
from efi.memory.diary import Diary
from efi.memory.facts import FactStore
from efi.memory.ingest import MemoryIngestor
from efi.memory.knowledge_sink import EpisodeKnowledgeSink
from efi.memory.parser import PerceptionParser
from efi.memory.people import PeopleStore, PersonProfile
from efi.memory.router import MemoryDomain
from efi.memory.validator import FactValidator

_OWNER_ID = 2129889949
_CHAT_ID = -400123


class _ScriptedRouter:
    """
    LLM, отвечающая по сценарию: первый вызов — новеллизация, второй —
    восприятие фактов. Оба разбора идут по одному и тому же эпизоду.
    """

    def __init__(self, *, candidates: list[dict[str, object]], diary_text: str = "ПУСТО") -> None:
        self._candidates = candidates
        self._diary_text = diary_text
        self.prompts: list[str] = []

    async def chat(self, role: object, params: LLMParams, session: Session) -> Response:
        self.prompts.append(session.messages[-1].content)
        # Промпт восприятия просит СТРОГО JSON-массив — по этому и различаем.
        is_perception = "JSON" in (params.system_prompt or "")
        text = json.dumps(self._candidates, ensure_ascii=False) if is_perception else self._diary_text
        return Response(choices=[Choice(message=Message(role=Role.ASSISTANT, content=text))])

    async def embedding(self, text: str) -> list[float]:  # pragma: no cover — dedup здесь без векторов
        return []


class _StubHistory:
    def __init__(self, session: Session) -> None:
        self._session = session

    async def get_active_chat_ids(self, *, since: object) -> list[int]:
        return [_CHAT_ID]

    async def get_since(self, chat_id: int, *, since: object) -> Session:
        return self._session


class _StubRag:
    async def remember(self, text: str, *, confidence: float = 0.5) -> None:
        return None


def _now() -> datetime:
    """Окно разбора «с этого момента» — весь эпизод отдаёт подставной источник истории."""
    return datetime.now(UTC)


def _episode() -> Session:
    return Session(
        messages=[
            Message(role=Role.USER, content="Рома: я теперь работаю монтажёром на телеке"),
            Message(role=Role.ASSISTANT, content="о, серьёзно? и как оно?"),
        ]
    )


async def _build(tmp_path: Path, router: _ScriptedRouter, *, people: PeopleStore | None = None):  # type: ignore[no-untyped-def]
    database = Database(tmp_path / "efi.db", migrations=MIGRATIONS)
    knowledge = KnowledgeStore(database)
    pending = PendingClarifications()
    ingestor = MemoryIngestor(
        PerceptionParser(router),  # type: ignore[arg-type]
        FactValidator(owner_id=_OWNER_ID),
        knowledge,
        pending=pending,
    )
    sink = EpisodeKnowledgeSink(ingestor, people or PeopleStore(database))
    consolidator = DiaryConsolidator(
        Diary(tmp_path / "diary"),
        router,  # type: ignore[arg-type]
        rag=_StubRag(),  # type: ignore[arg-type]
        knowledge=sink,
    )
    return consolidator, knowledge, pending, database


# -- сама склейка ---------------------------------------------------------------


async def test_a_lived_episode_reaches_strict_memory(tmp_path: Path) -> None:
    """
    Главная регрессия: конвейер был собран, но не вызывался — факты в
    knowledge_facts не попадали никогда.
    """
    router = _ScriptedRouter(
        candidates=[
            {
                "domain": "P",
                "kind": "attribute",
                "entity": "Рома",
                "attribute": "работа",
                "value": "монтажёр на телевидении",
                "confidence": 0.8,
            }
        ]
    )
    consolidator, knowledge, _pending, database = await _build(tmp_path, router)

    await consolidator.novelize_chat(
        _CHAT_ID,
        history=_StubHistory(_episode()),  # type: ignore[arg-type]
        facts=FactStore(database),
        since=_now(),
        min_messages=1,
    )

    stored = await knowledge.recall(domains=[MemoryDomain.PERSONAL], limit=10)
    assert [(fact.entity_id, fact.attribute, fact.value) for fact in stored] == [
        ("person:рома", "работа", "монтажёр на телевидении")
    ]


async def test_both_passes_see_the_same_episode_text(tmp_path: Path) -> None:
    """
    Дневник и строгая память — два среза одного прожитого куска. Разъедься
    тексты, и расхождение между срезами потом не отследить ничем.
    """
    router = _ScriptedRouter(candidates=[])
    consolidator, _knowledge, _pending, database = await _build(tmp_path, router)

    await consolidator.novelize_chat(
        _CHAT_ID,
        history=_StubHistory(_episode()),  # type: ignore[arg-type]
        facts=FactStore(database),
        since=_now(),
        min_messages=1,
    )

    assert len(router.prompts) == 2, "новеллизация и восприятие — два запроса по одному эпизоду"
    assert router.prompts[0] == router.prompts[1]


async def test_a_broken_knowledge_pass_does_not_cost_us_the_diary(tmp_path: Path) -> None:
    """
    Два независимых среза памяти: потерять оба из-за проблем в одном хуже,
    чем потерять один.
    """

    class _ExplodingSink:
        async def ingest_episode(self, episode_text: str, *, chat_id: int | None = None) -> object:
            raise RuntimeError("строгая память недоступна")

    saved: list[str] = []

    class _RecordingRag:
        async def remember(self, text: str, *, confidence: float = 0.5) -> object:
            saved.append(text)
            return object()

    router = _ScriptedRouter(candidates=[], diary_text="Разговор про новую работу Ромы.")
    consolidator = DiaryConsolidator(
        Diary(tmp_path / "diary"),
        router,  # type: ignore[arg-type]
        rag=_RecordingRag(),  # type: ignore[arg-type]
        knowledge=_ExplodingSink(),  # type: ignore[arg-type]
    )
    database = Database(tmp_path / "efi.db", migrations=MIGRATIONS)

    created = await consolidator.novelize_chat(
        _CHAT_ID,
        history=_StubHistory(_episode()),  # type: ignore[arg-type]
        facts=FactStore(database),
        since=_now(),
        min_messages=1,
    )

    assert created == 1
    assert saved == ["Разговор про новую работу Ромы."]


# -- граница доверия на живом пути ----------------------------------------------


async def test_a_junk_candidate_is_rejected_and_logged(tmp_path: Path) -> None:
    """
    Проверка, что на живом пути стоит именно валидатор, а не доверие к
    модели: служебный ключ приложения записать нельзя ни при какой
    формулировке.
    """
    router = _ScriptedRouter(
        candidates=[
            {
                "domain": "P",
                "kind": "attribute",
                "entity": "Рома",
                "attribute": "last_novelized_at",
                "value": "2020-01-01",
                "confidence": 0.9,
            }
        ]
    )
    consolidator, knowledge, _pending, database = await _build(tmp_path, router)

    await consolidator.novelize_chat(
        _CHAT_ID,
        history=_StubHistory(_episode()),  # type: ignore[arg-type]
        facts=FactStore(database),
        since=_now(),
        min_messages=1,
    )

    assert await knowledge.recall(limit=10) == []
    rejections = await database.fetch_all("SELECT attribute, reason FROM knowledge_rejections", ())
    assert len(rejections) == 1
    assert "служебн" in rejections[0]["reason"]


# -- каталог сущностей ----------------------------------------------------------


def test_two_people_with_one_name_in_the_same_chat_are_ambiguous() -> None:
    catalog = build_people_catalog(
        [
            PersonProfile(user_id=1, display_name="Рома", last_chat_id=_CHAT_ID),
            PersonProfile(user_id=2, display_name="Рома", last_chat_id=_CHAT_ID),
        ],
        chat_id=_CHAT_ID,
    )

    scores = {candidate.score for candidate in catalog["рома"]}
    assert scores == {1.0}, "равные свидетельства — равные оценки, то есть вопрос, а не догадка"


def test_a_participant_of_this_chat_outweighs_a_namesake_from_elsewhere() -> None:
    catalog = build_people_catalog(
        [
            PersonProfile(user_id=1, display_name="Рома", last_chat_id=_CHAT_ID),
            PersonProfile(user_id=2, display_name="Рома", last_chat_id=-999, last_chat_title="Работа"),
        ],
        chat_id=_CHAT_ID,
    )

    ranked = sorted(catalog["рома"], key=lambda candidate: candidate.score, reverse=True)
    assert ranked[0].entity_id == "user:1"
    assert ranked[0].score - ranked[1].score > 0.08, "разрыв должен превышать порог неразличимости"


def test_the_catalog_key_matches_how_the_validator_normalises_entities() -> None:
    """
    Разойдись эти две нормализации — каталог просто перестал бы находиться, и
    разрешение упоминаний тихо выключилось бы без единой ошибки в логах.
    """
    catalog = build_people_catalog([PersonProfile(user_id=1, display_name="Аня Ли")])
    validated = FactValidator(owner_id=_OWNER_ID)

    entity_id = validated.normalize_entity_for_lookup("Аня Ли")

    assert entity_id.split(":", 1)[-1] in catalog


def test_an_answered_question_is_not_asked_again() -> None:
    """
    Вопрос, ответ на который не запомнили, раздражает сильнее, чем молчание:
    следующий эпизод упёрся бы в те же два имени и спросил бы то же самое.
    """
    catalog = build_people_catalog(
        [
            PersonProfile(user_id=1, display_name="Рома", last_chat_id=_CHAT_ID),
            PersonProfile(user_id=2, display_name="Рома", last_chat_id=_CHAT_ID),
        ],
        chat_id=_CHAT_ID,
    )

    narrowed = apply_confirmed_answers(catalog, {"рома": "user:2"})

    assert [candidate.entity_id for candidate in narrowed["рома"]] == ["user:2"]


def test_an_unrelated_answer_leaves_the_catalog_alone() -> None:
    catalog = build_people_catalog([PersonProfile(user_id=1, display_name="Рома")])

    assert apply_confirmed_answers(catalog, {"феникс": "topic:феникс"}) == catalog


def test_people_without_a_name_are_skipped() -> None:
    assert build_people_catalog([PersonProfile(user_id=1, display_name="   ")]) == {}


# -- уточнение доходит до собеседника и закрывается ------------------------------


async def test_an_ambiguous_mention_asks_instead_of_guessing(tmp_path: Path) -> None:
    """
    Ради этого граница доверия и вводилась: угаданное упоминание оседает в
    памяти как факт и всплывает через неделю, когда отличить его от
    настоящего уже нельзя.
    """
    router = _ScriptedRouter(
        candidates=[
            {
                "domain": "P",
                "kind": "attribute",
                "entity": "Рома",
                "attribute": "работа",
                "value": "монтажёр",
                "confidence": 0.8,
            }
        ]
    )
    database = Database(tmp_path / "efi.db", migrations=MIGRATIONS)
    people = PeopleStore(database)
    await people.record_message(1, "привет", display_name="Рома", chat_id=_CHAT_ID)
    await people.record_message(2, "и тебе", display_name="Рома", chat_id=_CHAT_ID)

    knowledge = KnowledgeStore(database)
    pending = PendingClarifications()
    sink = EpisodeKnowledgeSink(
        MemoryIngestor(
            PerceptionParser(router),  # type: ignore[arg-type]
            FactValidator(owner_id=_OWNER_ID),
            knowledge,
            pending=pending,
        ),
        people,
    )

    result = await sink.ingest_episode("Рома: я теперь монтажёр", chat_id=_CHAT_ID)

    assert result.needs_clarification, "два тёзки в одном чате — это вопрос, а не догадка"
    assert await knowledge.recall(limit=10) == [], "пока не выяснила — не записывает"
    assert pending.peek(_CHAT_ID) is not None


async def test_the_question_reaches_the_prompt(tmp_path: Path) -> None:
    """
    Вопрос должен прозвучать её обычной репликой, а не остаться записью в
    логе: иначе конвейер молча отказывается запоминать, и со стороны это
    неотличимо от «память не работает».
    """
    from efi.behavior.ambiguity import EntityCandidate, Resolution, render_clarification
    from efi.prompts.builder import _build_clarification_block

    pending = PendingClarifications()
    candidates = (
        EntityCandidate(entity_id="user:1", label="Рома", score=1.0, hint="из этого чата"),
        EntityCandidate(entity_id="user:2", label="Рома", score=1.0, hint="из «Работа»"),
    )
    pending.remember(
        _CHAT_ID,
        Resolution(clarification=render_clarification("Рома", candidates), candidates=candidates, mention="Рома"),
    )

    block = _build_clarification_block(pending.peek(_CHAT_ID))

    assert "[Надо уточнить]" in block
    assert "Рома" in block
    assert "из этого чата" in block and "из «Работа»" in block
    assert "ничего про это не запоминай" in block


def test_no_pending_question_means_no_block() -> None:
    from efi.prompts.builder import _build_clarification_block

    assert _build_clarification_block(None) == ""


async def test_an_answer_closes_the_question_and_is_remembered(tmp_path: Path) -> None:
    pending = PendingClarifications()
    from efi.behavior.ambiguity import EntityCandidate, Resolution, render_clarification

    candidates = (
        EntityCandidate(entity_id="user:1", label="Рома", score=1.0, hint="из этого чата"),
        EntityCandidate(entity_id="user:2", label="Рома", score=1.0, hint="из «Работа»"),
    )
    pending.remember(
        _CHAT_ID,
        Resolution(clarification=render_clarification("Рома", candidates), candidates=candidates, mention="Рома"),
    )

    chosen = pending.resolve_with_answer(_CHAT_ID, "да тот, который из «Работа»")

    assert chosen is not None and chosen.entity_id == "user:2"
    assert pending.peek(_CHAT_ID) is None, "вопрос закрыт — второй раз спрашивать нечего"
    assert pending.confirmed_entities(_CHAT_ID) == {"рома": "user:2"}


async def test_an_unrecognised_answer_keeps_the_question_open() -> None:
    """Неопознанный ответ ничем не лучше исходной неоднозначности — записывать по нему нельзя."""
    from efi.behavior.ambiguity import EntityCandidate, Resolution, render_clarification

    pending = PendingClarifications()
    candidates = (
        EntityCandidate(entity_id="user:1", label="Рома", score=1.0, hint="из этого чата"),
        EntityCandidate(entity_id="user:2", label="Рома", score=1.0, hint="из «Работа»"),
    )
    pending.remember(
        _CHAT_ID,
        Resolution(clarification=render_clarification("Рома", candidates), candidates=candidates, mention="Рома"),
    )

    assert pending.resolve_with_answer(_CHAT_ID, "да неважно") is None
    assert pending.peek(_CHAT_ID) is not None
    assert pending.confirmed_entities(_CHAT_ID) == {}


# -- ради чего всё это: факт возвращается в промпт --------------------------------


async def test_a_learned_fact_comes_back_in_the_prompt(tmp_path: Path) -> None:
    """
    Замыкание круга. Разрешение сущности по каталогу здесь не украшение:
    без него факт лёг бы как `person:рома`, а промпт ищет по `user:<id>` —
    Эфи узнала бы про человека и не смогла бы этим воспользоваться.
    """
    from efi.memory.dedup import render_facts_block

    router = _ScriptedRouter(
        candidates=[
            {
                "domain": "P",
                "kind": "attribute",
                "entity": "Рома",
                "attribute": "работа",
                "value": "монтажёр на телевидении",
                "confidence": 0.8,
            }
        ]
    )
    database = Database(tmp_path / "efi.db", migrations=MIGRATIONS)
    people = PeopleStore(database)
    await people.record_message(555, "привет", display_name="Рома", chat_id=_CHAT_ID)

    knowledge = KnowledgeStore(database)
    sink = EpisodeKnowledgeSink(
        MemoryIngestor(
            PerceptionParser(router),  # type: ignore[arg-type]
            FactValidator(owner_id=_OWNER_ID),
            knowledge,
            pending=PendingClarifications(),
        ),
        people,
    )

    await sink.ingest_episode("Рома: я теперь монтажёр на телеке", chat_id=_CHAT_ID)

    recalled = await knowledge.recall(entity_ids=["user:555"], domains=[MemoryDomain.PERSONAL], limit=8)
    assert [fact.value for fact in recalled] == ["монтажёр на телевидении"]
    assert "монтажёр на телевидении" in render_facts_block(recalled)


async def test_the_same_fact_twice_is_reinforced_not_duplicated(tmp_path: Path) -> None:
    """occurrence_count — то, чем «упомянуто один раз» отличается от «повторяется постоянно»."""
    router = _ScriptedRouter(
        candidates=[
            {
                "domain": "P",
                "kind": "attribute",
                "entity": "Рома",
                "attribute": "работа",
                "value": "монтажёр",
                "confidence": 0.8,
            }
        ]
    )
    database = Database(tmp_path / "efi.db", migrations=MIGRATIONS)
    knowledge = KnowledgeStore(database)
    sink = EpisodeKnowledgeSink(
        MemoryIngestor(
            PerceptionParser(router),  # type: ignore[arg-type]
            FactValidator(owner_id=_OWNER_ID),
            knowledge,
            pending=PendingClarifications(),
        ),
        PeopleStore(database),
    )

    first = await sink.ingest_episode("Рома: я монтажёр", chat_id=_CHAT_ID)
    second = await sink.ingest_episode("Рома: я всё ещё монтажёр", chat_id=_CHAT_ID)

    assert len(first.created) == 1 and len(second.created) == 0
    assert len(second.reinforced) == 1
    stored = await knowledge.recall(limit=10)
    assert len(stored) == 1
    assert stored[0].occurrence_count == 2
