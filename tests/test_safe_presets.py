from __future__ import annotations

from math_animation_studio.safe_presets import normalize_loss_surface_preset


def test_loss_surface_preset_aliases_are_normalized() -> None:
    assert normalize_loss_surface_preset("hill_surface") == "quadratic_ripple"
    assert normalize_loss_surface_preset("simple_quadratic") == "quadratic_ripple"
    assert normalize_loss_surface_preset("two_valleys") == "double_well_2d"
    assert normalize_loss_surface_preset("double_well_curve") == "double_well_1d"


def test_unknown_loss_surface_preset_falls_back_to_safe_default() -> None:
    assert normalize_loss_surface_preset("made_up_function") == "quadratic_ripple"
