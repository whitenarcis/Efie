"""Тесты для блока текущего состояния личности в efi.prompts.builder (mood/social_distance/sycophancy)."""

from __future__ import annotations

from datetime import UTC, datetime

from efi.behavior.affinity import AffinitySnapshot
from efi.memory.beliefs import Belief
from efi.prompts.builder import _build_state_vector_block, _resolve_mood

_SYCOPHANCY_TEXT = "не соглашайся просто чтобы понравиться"


def _belief(topic: str, confidence: float) -> Belief:
    return Belief(
        topic=topic, stance="держусь своего мнения", confidence_score=confidence, origin_date=datetime.now(UTC)
    )


def test_resolve_mood_strong_belief_wins_over_respect() -> None:
    beliefs = [_belief("vim vs vscode", 0.9)]
    affinity = AffinitySnapshot(affinity=0.9, respect_level=0.9)  # высокое уважение не должно перебивать инерцию
    assert _resolve_mood(beliefs, affinity) == "skeptical_focused"


def test_resolve_mood_weak_belief_does_not_trigger_inertia() -> None:
    beliefs = [_belief("случайная тема", 0.3)]
    affinity = AffinitySnapshot(affinity=0.5, respect_level=0.5)
    assert _resolve_mood(beliefs, affinity) == "engaged"


def test_resolve_mood_low_respect_is_ironic() -> None:
    affinity = AffinitySnapshot(affinity=0.5, respect_level=0.1)
    assert _resolve_mood([], affinity) == "ironic"


def test_resolve_mood_high_respect_is_analytical() -> None:
    affinity = AffinitySnapshot(affinity=0.8, respect_level=0.8)
    assert _resolve_mood([], affinity) == "analytical"


def test_resolve_mood_default_is_engaged() -> None:
    affinity = AffinitySnapshot(affinity=0.5, respect_level=0.5)
    assert _resolve_mood([], affinity) == "engaged"


def test_state_vector_block_includes_sycophancy_protection() -> None:
    block = _build_state_vector_block([], AffinitySnapshot(), _SYCOPHANCY_TEXT)
    assert _SYCOPHANCY_TEXT in block
    assert "[Текущее состояние личности]" in block


def test_state_vector_block_lists_relevant_beliefs() -> None:
    beliefs = [_belief("gran turismo 5", 0.85)]
    block = _build_state_vector_block(beliefs, AffinitySnapshot(), _SYCOPHANCY_TEXT)
    assert "gran turismo 5" in block
    assert "0.85" in block


def test_state_vector_block_omits_beliefs_section_when_none_relevant() -> None:
    block = _build_state_vector_block([], AffinitySnapshot(), _SYCOPHANCY_TEXT)
    assert "твои текущие убеждения" not in block
