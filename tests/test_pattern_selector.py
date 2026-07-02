from __future__ import annotations

from math_animation_studio.understanding import AnimationPatternSelector


def test_pattern_selector_returns_requested_pattern() -> None:
    selector = AnimationPatternSelector()

    pattern = selector.select("penalty_curve")

    assert pattern.id == "penalty_curve"


def test_pattern_selector_falls_back_for_unknown_pattern() -> None:
    selector = AnimationPatternSelector()

    pattern = selector.select("missing_pattern")

    assert pattern.id == "generic_symbol_decomposition"
