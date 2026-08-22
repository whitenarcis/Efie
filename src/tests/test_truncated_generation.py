"""
Тесты обработки обрыва на местах, где текст модели попадает в дневник.

Проверяется не парсер, а поведение вызывающих сторон: фоновое исследование
(researcher), движок автономии (life_engine) и ночная консолидация. Именно
они писали в дневник записи, обрывающиеся на полуслове, потому что
`finish_reason` от провайдера никто не читал.

Ключевой инвариант везде один: обрыв — не ошибка запроса. Ответ пришёл
успешно, он просто неполный, поэтому его нужно либо обрезать по последней
законченной фразе, либо не сохранять вовсе — но ни в коем случае не
сохранять как есть.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from efi.behavior.life_engine import BackgroundLifeWorker
from efi.behavior.researcher import BackgroundResearcher
from efi.llm.schemas import Choice, LLMParams, Message, Response, Role, Session
from efi.memory.consolidation import DiaryConsolidator
from efi.memory.diary import Diary

#: Две законченные фразы плюс висящий хвост — ровно то, что приходит с
#: finish_reason="length" (см. пример из дневника в шапке
#: tests/test_text_truncation.py).
_TRUNCATED = (
    "Меня давно занимает, почему старые дома строили именно из кирпича. "
    "Это заметно дороже и медленнее, чем панель. Хотя ещё лет пять назад это"
)
_TRIMMED = (
    "Меня давно занимает, почему старые дома строили именно из кирпича. "
    "Это заметно дороже и медленнее, чем панель."
)

#: Обрубок без единой законченной фразы — спасать нечего.
_HOPELESS = "Мне кажется, что вся эта история на самом деле про то, как мы"


class _FakeRouter:
    """Отдаёт заранее заданные ответы по очереди; последний повторяется."""

    def __init__(self, *responses: tuple[str, str | None]) -> None:
        self._responses = list(responses)
        self.calls: list[LLMParams] = []
        #: Что именно спросили — нужно там, где инструкция уходит в
        #: пользовательскую реплику, а не в системный промпт (просьба дописать).
        self.prompts: list[str] = []

    async def chat(self, role: object, params: LLMParams, session: Session) -> Response:
        self.calls.append(params)
        self.prompts.append(session.messages[-1].content if session.messages else "")
        text, finish_reason = self._responses[min(len(self.calls) - 1, len(self._responses) - 1)]
        return Response(
            choices=[
                Choice(index=0, message=Message(role=Role.ASSISTANT, content=text), finish_reason=finish_reason)
            ]
        )


# -- researcher ---------------------------------------------------------------


def _researcher(router: _FakeRouter) -> BackgroundResearcher:
    return BackgroundResearcher(
        Path("worldview.json"),
        web_search=None,  # type: ignore[arg-type]
        rag=None,  # type: ignore[arg-type]
        router=router,  # type: ignore[arg-type]
        facts=None,  # type: ignore[arg-type]
    )


async def test_truncated_hypothesis_is_trimmed_not_stored_raw() -> None:
    router = _FakeRouter((_TRUNCATED, "length"))

    hypothesis = await _researcher(router)._formulate_hypothesis("кирпич", "результаты поиска")

    assert hypothesis == _TRIMMED


async def test_hopeless_hypothesis_is_dropped_entirely() -> None:
    """Записи не будет вовсе — это лучше, чем запись-обрубок в дневнике."""
    router = _FakeRouter((_HOPELESS, "length"))

    assert await _researcher(router)._formulate_hypothesis("кирпич", "результаты поиска") is None


async def test_complete_hypothesis_passes_through_untouched() -> None:
    router = _FakeRouter(("Кирпич просто дольше живёт, вот и весь секрет", "stop"))

    hypothesis = await _researcher(router)._formulate_hypothesis("кирпич", "результаты поиска")

    assert hypothesis == "Кирпич просто дольше живёт, вот и весь секрет"


async def test_hypothesis_budget_leaves_room_for_cyrillic() -> None:
    """
    Регрессия на первопричину: 256 токенов хватало на 1-3 английские фразы,
    но не на русские — кириллица у бесплатных моделей стоит в 2-3 раза дороже.
    """
    router = _FakeRouter(("что-то", "stop"))

    await _researcher(router)._formulate_hypothesis("кирпич", "результаты поиска")

    assert router.calls[0].max_output_tokens >= 512


# -- life_engine --------------------------------------------------------------


def _worker(router: _FakeRouter) -> BackgroundLifeWorker:
    return BackgroundLifeWorker(
        curiosity=None,  # type: ignore[arg-type]
        web_search=None,  # type: ignore[arg-type]
        rag=None,  # type: ignore[arg-type]
        router=router,  # type: ignore[arg-type]
        organic_ping=None,  # type: ignore[arg-type]
    )


async def test_truncated_finding_is_trimmed() -> None:
    router = _FakeRouter((_TRUNCATED, "length"))

    finding = await _worker(router)._formulate_finding("кирпич", "результаты поиска")

    assert finding == _TRIMMED


async def test_hopeless_finding_is_dropped() -> None:
    router = _FakeRouter((_HOPELESS, "length"))

    assert await _worker(router)._formulate_finding("кирпич", "результаты поиска") is None


async def test_finding_and_reaction_budgets_leave_room_for_cyrillic() -> None:
    router = _FakeRouter(("что-то", "stop"))
    worker = _worker(router)

    await worker._formulate_finding("кирпич", "результаты поиска")
    await worker._formulate_reaction("кирпич", "результаты поиска", "вывод")

    assert router.calls[0].max_output_tokens >= 512, "находка — 1-3 предложения по-русски"
    assert router.calls[1].max_output_tokens >= 256, "реакция — одно предложение по-русски"


# -- консолидация -------------------------------------------------------------


def _episode(text: str) -> str:
    """Текст эпизода в том виде, в каком его собирает novelize_chat."""
    return f"собеседник: {text}"


def _consolidator(tmp_path: Path, router: _FakeRouter) -> DiaryConsolidator:
    return DiaryConsolidator(Diary(tmp_path / "diary"), router, rag=None)  # type: ignore[arg-type]


async def test_truncated_novelization_is_finished_not_cut(tmp_path: Path) -> None:
    """
    Главный сценарий: обрыв по лимиту лечится ДОПИСЫВАНИЕМ, а не обрезанием.

    Промпт новеллизации требует подробностей и нескольких записей за проход,
    поэтому упереться в лимит — норма активного дня, а не исключение. Пока
    единственным лечением было отрезание хвоста, дневник наполнялся записями,
    обрывающимися на полумысли.
    """
    router = _FakeRouter(
        (_TRUNCATED, "length"),
        (" казалось несущественным, а теперь понятно, что дело в теплоёмкости.", "stop"),
    )

    pieces = await _consolidator(tmp_path, router)._extract_memories(_episode("про кирпич"))

    assert len(pieces) == 1
    assert pieces[0].endswith("дело в теплоёмкости.")
    assert pieces[0].startswith("Меня давно занимает")
    assert len(router.calls) == 2, "один запрос на продолжение, а не переписывание с нуля"
    assert "Продолжи РОВНО с этого места" in router.prompts[1]
    assert _TRUNCATED[-40:] in router.prompts[1], "модели показан хвост, по которому она найдёт место обрыва"


async def test_continuation_is_asked_no_more_than_twice(tmp_path: Path) -> None:
    """Модель, которая не умеет останавливаться, не должна жечь лимиты бесконечно."""
    router = _FakeRouter((_TRUNCATED, "length"))

    pieces = await _consolidator(tmp_path, router)._extract_memories(_episode("про кирпич"))

    assert len(router.calls) == 3, "исходный запрос плюс два дописывания"
    assert pieces and pieces[0].endswith(".") , "то, что не удалось дописать, обрезается по фразе"


async def test_only_the_last_novelized_entry_is_repaired(tmp_path: Path) -> None:
    """
    Дописать не удалось — тогда обрыв бьёт по хвосту, а не по всему ответу:
    записи до разделителя модель успела закончить целиком, и терять их из-за
    оборванной последней значит выкинуть всю память за проход.
    """
    complete = "Утро прошло за разговором про кирпич, и это было неожиданно интересно."
    router = _FakeRouter((f"{complete}\n---\n{_TRUNCATED}", "length"), ("", "length"))

    pieces = await _consolidator(tmp_path, router)._extract_memories(
        _episode("про кирпич")
    )

    assert pieces == [complete, _TRIMMED]


async def test_hopeless_last_entry_is_dropped_and_the_rest_survives(tmp_path: Path) -> None:
    complete = "Утро прошло за разговором про кирпич, и это было неожиданно интересно."
    router = _FakeRouter((f"{complete}\n---\n{_HOPELESS}", "length"), ("", "length"))

    pieces = await _consolidator(tmp_path, router)._extract_memories(
        _episode("про кирпич")
    )

    assert pieces == [complete]


async def test_complete_novelization_is_untouched(tmp_path: Path) -> None:
    router = _FakeRouter(("Первая запись целиком.\n---\nВторая запись целиком.", "stop"))

    pieces = await _consolidator(tmp_path, router)._extract_memories(
        _episode("про кирпич")
    )

    assert pieces == ["Первая запись целиком.", "Вторая запись целиком."]


async def test_truncated_memoir_summary_is_trimmed(tmp_path: Path) -> None:
    router = _FakeRouter((_TRUNCATED, "length"))

    summary = await _consolidator(tmp_path, router)._summarize_via_llm([])

    assert summary == _TRIMMED


async def test_hopeless_memoir_summary_keeps_the_originals(tmp_path: Path) -> None:
    """
    None здесь означает «сжатие не удалось» — исходные записи остаются в
    дневнике нетронутыми (см. summarize_stale_entries). Это ровно то, что
    нужно: лучше десять старых записей, чем один обрубок вместо них.
    """
    router = _FakeRouter((_HOPELESS, "length"))

    assert await _consolidator(tmp_path, router)._summarize_via_llm([]) is None


# -- восприятие ---------------------------------------------------------------


@pytest.mark.parametrize("finish_reason", ["length", "stop"])
async def test_truncated_perception_names_the_real_cause(finish_reason: str) -> None:
    """
    Оборванный JSON — это исчерпанный бюджет, а не «модель ответила чушью».
    В логе должно быть написано именно это, иначе чинить нечего.
    """
    from efi.memory.parser import PerceptionParser

    router = _FakeRouter(('[{"entity": "user", "attribute": "город", "value": "Мин', finish_reason))
    batch = await PerceptionParser(router).extract("переписка")  # type: ignore[arg-type]

    assert batch.parse_error
    assert ("оборван лимитом" in batch.parse_error) is (finish_reason == "length")


async def test_a_slow_model_gets_a_smaller_diary_instead_of_none(tmp_path: Path) -> None:
    """
    Медленная модель не успевает написать столько, сколько попросили, — и
    тогда пропадает ВСЯ запись, а не её часть. Короткий эпизод в дневнике
    лучше, чем ещё одна дыра в памяти за этот вечер.
    """
    from efi.llm.errors import LLMTimeoutError

    budgets: list[int] = []

    class _SlowRouter:
        async def chat(self, role: object, params: LLMParams, session: Session) -> Response:
            budgets.append(params.max_output_tokens)
            if len(budgets) == 1:
                raise LLMTimeoutError("request timed out after 15s", provider="test")
            return Response(
                choices=[
                    Choice(message=Message(role=Role.ASSISTANT, content="Короткая, но живая запись."))
                ]
            )

    consolidator = DiaryConsolidator(
        Diary(tmp_path / "diary"),
        _SlowRouter(),  # type: ignore[arg-type]
        rag=None,
        novelization_max_output_tokens=4096,
    )

    written = await consolidator._novelize("Разговор был такой.")

    assert written is not None
    assert budgets == [4096, 2048], "просим не дольше ждать, а написать короче"
    assert "живая запись" in written.body
