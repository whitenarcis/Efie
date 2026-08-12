"""
Тесты атрибуции реплик в плоском тексте переписки (efi.memory.transcript).

Регрессия из жизни: в дневнике Эфи регулярно менялись местами автор и
собеседник — то, что написал человек, записывалось как её собственные слова,
и наоборот.

Причина не в дневнике и не в модели. В обычном запросе к чат-модели роли —
часть протокола: реплики лежат в разных полях, перепутать их нельзя. Но
новеллизация подаёт диалог ОДНИМ куском текста, и вся атрибуция сводится к
тому, что написано в самих строках. А написано там было вот что:

    user: Рома: привет, как оно
    assistant: да норм, я тут весь вечер с плёнкой вожусь

Дальше модель просят записать это в дневник от первого лица — то есть
сообразить, что английское служебное слово `assistant` посреди русского
текста означает «я», а `user` — «не я». На бесплатном тире это не работает
стабильно.

Отдельная деталь: промпт новеллизации ПРЯМО обещал, что «перед каждой
репликой указано имя написавшего», а рендерер подставлял туда `user`/
`assistant`. Обещание не выполнялось, и модель добирала недостающее
догадкой.
"""

from __future__ import annotations

from efi.llm.schemas import Message, Role, Session
from efi.memory.transcript import SELF_MARKER, render_transcript


def _session(*messages: tuple[Role, str]) -> Session:
    return Session(messages=[Message(role=role, content=text) for role, text in messages])


def _render(session: Session, *, char_limit: int = 10_000) -> str:
    return render_transcript(session, self_name="Эфи", char_limit=char_limit)


# -- сам баг ------------------------------------------------------------------


def test_her_own_lines_are_marked_as_hers() -> None:
    """Главная регрессия: её реплику нельзя спутать с чужой."""
    rendered = _render(_session((Role.ASSISTANT, "я тут весь вечер с плёнкой вожусь")))

    assert f"Эфи ({SELF_MARKER}): я тут весь вечер с плёнкой вожусь" in rendered


def test_protocol_role_words_never_reach_the_text() -> None:
    """
    `user`/`assistant` — служебные слова протокола. В тексте, который читает
    модель, им делать нечего: ровно на них и ломалась атрибуция.
    """
    rendered = _render(_session((Role.USER, "Рома: привет"), (Role.ASSISTANT, "привет")))

    assert "user:" not in rendered
    assert "assistant:" not in rendered


def test_the_sender_name_is_not_duplicated() -> None:
    """
    Имя уже стоит в тексте сообщения (efi.telegram.formatting.
    format_user_message). Вторая подпись поверх читалась бы как два разных
    участника.
    """
    rendered = _render(_session((Role.USER, "Рома: привет, как оно")))

    assert "Рома: привет, как оно" in rendered
    assert "собеседник: Рома" not in rendered


def test_reply_context_in_the_name_prefix_survives() -> None:
    """format_user_message умеет вписывать в подпись контекст ответа — он не должен ломать разбор."""
    line = 'Рома (в ответ на: "с плёнкой вожусь"): а какой сканер?'

    assert line in _render(_session((Role.USER, line)))


def test_an_unsigned_user_line_still_gets_an_author() -> None:
    """Старые записи истории (до format_user_message) подписи не имеют — безымянной строка остаться не может."""
    rendered = _render(_session((Role.USER, "старое сообщение без подписи")))

    assert "собеседник: старое сообщение без подписи" in rendered
    assert SELF_MARKER not in rendered.split("\n\n", 1)[1], "чужая реплика не должна быть помечена как её"


def test_a_colon_inside_a_sentence_is_not_mistaken_for_a_signature() -> None:
    """
    Иначе длинная фраза с двоеточием сходила бы за подпись, и её «имя»
    оказалось бы половиной предложения.
    """
    long_line = "слушай, я тут подумал вот о чём и решил всё-таки написать: давай перенесём на завтра"

    assert f"собеседник: {long_line}" in _render(_session((Role.USER, long_line)))


def test_the_convention_is_explained_once_at_the_top() -> None:
    """
    Пометки в строках дублируются легендой намеренно: на длинной переписке
    модель опирается на начало текста.
    """
    rendered = _render(_session((Role.USER, "Рома: привет")))

    assert rendered.startswith("Ниже переписка.")
    assert SELF_MARKER in rendered.split("\n")[0]


# -- поведение вокруг -----------------------------------------------------------


def test_the_dialogue_keeps_its_order() -> None:
    rendered = _render(
        _session(
            (Role.USER, "Рома: первый вопрос"),
            (Role.ASSISTANT, "мой ответ"),
            (Role.USER, "Рома: второй вопрос"),
        )
    )

    body = rendered.split("\n\n", 1)[1].split("\n")
    assert body == ["Рома: первый вопрос", f"Эфи ({SELF_MARKER}): мой ответ", "Рома: второй вопрос"]


def test_empty_and_tool_only_messages_are_skipped() -> None:
    rendered = _render(_session((Role.USER, "Рома: привет"), (Role.ASSISTANT, "   ")))

    assert rendered.count(SELF_MARKER) == 1, "пометка осталась только в легенде — её реплики тут нет"


def test_a_session_without_content_renders_to_nothing() -> None:
    """Пустая строка — сигнал вызывающей стороне, что новеллизировать нечего."""
    assert _render(_session((Role.ASSISTANT, "  "))) == ""
    assert _render(Session()) == ""


def test_the_legend_survives_truncation() -> None:
    """Обрезается начало разговора — но не объяснение того, кто есть кто; на длинном тексте оно нужнее всего."""
    session = _session(*[(Role.USER, f"Рома: реплика номер {index}") for index in range(200)])

    rendered = _render(session, char_limit=200)

    assert rendered.startswith("Ниже переписка.")
    assert "[...начало разговора опущено...]" in rendered
    assert "реплика номер 199" in rendered


def test_the_character_name_comes_from_settings() -> None:
    """Имя не захардкожено: владелец волен назвать её как угодно (Settings.character_name)."""
    rendered = render_transcript(_session((Role.ASSISTANT, "ага")), self_name="Куни", char_limit=10_000)

    assert f"Куни ({SELF_MARKER}): ага" in rendered
    assert "Эфи" not in rendered


# -- сквозь новеллизацию --------------------------------------------------------


async def test_the_novelization_prompt_gets_attributed_lines(tmp_path) -> None:  # type: ignore[no-untyped-def]
    """
    Проверка на том уровне, где баг и проявлялся: что реально уходит в
    LLM при новеллизации эпизода.
    """
    from efi.llm.schemas import Choice, LLMParams, Response
    from efi.memory.consolidation import DiaryConsolidator
    from efi.memory.diary import Diary

    seen: list[str] = []

    class _CapturingRouter:
        async def chat(self, role: object, params: LLMParams, session: Session) -> Response:
            seen.append(session.messages[-1].content)
            return Response(choices=[Choice(message=Message(role=Role.ASSISTANT, content="ПУСТО"))])

    consolidator = DiaryConsolidator(
        Diary(tmp_path / "diary"),
        _CapturingRouter(),  # type: ignore[arg-type]
        rag=None,  # type: ignore[arg-type]
        character_name="Эфи",
    )

    await consolidator._extract_memories(
        consolidator._compose_episode(
            _session((Role.USER, "Рома: я весь вечер чинил парсер"), (Role.ASSISTANT, "а я вожусь с плёнкой"))
        )
    )

    assert len(seen) == 1
    prompt = seen[0]
    assert f"Эфи ({SELF_MARKER}): а я вожусь с плёнкой" in prompt
    assert "Рома: я весь вечер чинил парсер" in prompt
    assert "assistant:" not in prompt


def test_the_system_prompt_explains_the_marker_it_will_actually_see() -> None:
    """
    Промпт обещал, что «перед каждой репликой указано имя написавшего», а
    рендерер подставлял туда `user`/`assistant`. Обещание и разметка должны
    сходиться, иначе модель добирает недостающее догадкой.
    """
    from efi.memory.consolidation import _NOVELIZATION_SYSTEM_PROMPT

    assert SELF_MARKER in _NOVELIZATION_SYSTEM_PROMPT
    assert "имя" in _NOVELIZATION_SYSTEM_PROMPT
