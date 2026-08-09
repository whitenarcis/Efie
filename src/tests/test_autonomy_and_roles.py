"""
Тесты на разнообразие проактивных пингов, дневник личной жизни и строгий
регламент ролей моделей.

Регрессии, которые здесь закрыты:
    - органический пинг всегда начинался одним и тем же шаблоном
      ("Нашла интересное по нашей теме X"), из-за чего инициативные
      сообщения читались как автоматическая рассылка;
    - в дневник о фоновом исследовании попадала выжимка из статьи без
      личного отношения, и сослаться на неё в разговоре было не на что;
    - фоновые задачи ходили в роль FAST наравне со служебными вызовами, без
      явного разделения "живой диалог / фоновая жизнь".
"""

from __future__ import annotations

from efi.behavior.life_engine import AUTONOMOUS_THOUGHT_TAG, InformedThought, _render_diary_entry
from efi.behavior.organic_ping import _PING_ANGLES, _render_ping_message
from efi.config.schema import (
    EndpointConfig,
    LLMRolesSettings,
    PathsSettings,
    RoleRoute,
    Settings,
    TaskRole,
    TelegramSettings,
)


def _thought(**overrides: object) -> InformedThought:
    defaults: dict[str, object] = dict(
        seed_id=1, topic="eBPF", source_chat_id=42, finding="это база для трассировки ядра", weight=0.7
    )
    defaults.update(overrides)
    return InformedThought(**defaults)  # type: ignore[arg-type]


# -- разнообразие инициативных пингов ----------------------------------------


def test_ping_message_carries_topic_and_finding() -> None:
    message = _render_ping_message(_thought())
    assert "eBPF" in message
    assert "это база для трассировки ядра" in message


def test_different_seeds_get_different_angles() -> None:
    """Разные находки не должны начинаться одной и той же фразой."""
    messages = {_render_ping_message(_thought(seed_id=seed)) for seed in range(len(_PING_ANGLES))}
    assert len(messages) == len(_PING_ANGLES)


def test_same_seed_is_stable() -> None:
    """Повторный пинг по той же находке — тот же угол, а не вторая попытка достучаться другими словами."""
    assert _render_ping_message(_thought(seed_id=3)) == _render_ping_message(_thought(seed_id=3))


# -- дневник личной жизни ------------------------------------------------------


def test_diary_entry_reads_as_lived_experience_not_a_summary() -> None:
    entry = _render_diary_entry(_thought(reaction="и меня, честно, взбесило, как это подают"))
    assert AUTONOMOUS_THOUGHT_TAG in entry
    assert "Читала сегодня про eBPF" in entry
    assert "взбесило" in entry


def test_diary_entry_survives_a_missing_reaction() -> None:
    """Сбой второго запроса (личное отношение) не должен терять саму находку."""
    entry = _render_diary_entry(_thought(reaction=""))
    assert "это база для трассировки ядра" in entry


# -- строгий регламент ролей ---------------------------------------------------


def _endpoint(model: str, *, base_url: str = "https://omni.example/v1") -> EndpointConfig:
    return EndpointConfig(base_url=base_url, api_key="secret", model=model)


def _settings(*, background: RoleRoute | None = None, vision_on_groq: bool = True) -> Settings:
    vision_url = "https://api.groq.com/openai/v1" if vision_on_groq else "https://omni.example/v1"
    return Settings(
        telegram=TelegramSettings(api_id=1, api_hash="x", owner_id=1),
        llm_roles=LLMRolesSettings(
            main=RoleRoute(primary=_endpoint("gemma")),
            fast=RoleRoute(primary=_endpoint("fast-model")),
            vision=RoleRoute(primary=_endpoint("vision-model", base_url=vision_url)),
            background=background,
        ),
        paths=PathsSettings(session_name="test"),
    )


def test_background_role_falls_back_to_fast_when_not_configured() -> None:
    routes = _settings().llm_roles.as_routes()
    assert routes[TaskRole.BACKGROUND] is routes[TaskRole.FAST]


def test_background_role_is_independent_when_configured() -> None:
    background = RoleRoute(primary=_endpoint("background-model"))
    routes = _settings(background=background).llm_roles.as_routes()
    assert routes[TaskRole.BACKGROUND].primary.model == "background-model"
    assert routes[TaskRole.FAST].primary.model == "fast-model"


def test_every_role_has_a_route() -> None:
    """LLMRouter требует маршрут на каждую роль — as_routes() обязан покрывать весь TaskRole."""
    assert set(_settings().llm_roles.as_routes()) == set(TaskRole)


# -- унификация ключа Groq ------------------------------------------------------


def test_groq_key_is_taken_from_llm_roles_without_duplication() -> None:
    resolved = _settings(vision_on_groq=True).resolve_groq_api_key()
    assert resolved is not None
    assert resolved.get_secret_value() == "secret"


def test_groq_key_is_none_when_no_endpoint_points_at_groq() -> None:
    assert _settings(vision_on_groq=False).resolve_groq_api_key() is None


def test_explicit_stt_key_overrides_the_one_from_llm_roles() -> None:
    from efi.config.schema import SttSettings

    settings = _settings(vision_on_groq=True).model_copy(
        update={"stt": SttSettings(groq_api_key="explicit-override")}  # type: ignore[arg-type]
    )
    resolved = settings.resolve_groq_api_key()
    assert resolved is not None
    assert resolved.get_secret_value() == "explicit-override"
