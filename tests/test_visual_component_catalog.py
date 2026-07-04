from __future__ import annotations

from math_animation_studio.understanding.visual_component_catalog import (
    load_visual_component_catalog,
    visual_component_prompt_summary,
)


def test_visual_component_catalog_loads_expected_components() -> None:
    catalog = load_visual_component_catalog()

    assert "formula_focus" in catalog
    assert "probability_bars" in catalog
    assert "gradient_arrow" in catalog
    assert catalog["formula_focus"].params[0].name == "formula_focus"
    assert catalog["gradient_arrow"].visual_type == "vector"


def test_visual_component_prompt_summary_is_llm_readable() -> None:
    summary = visual_component_prompt_summary()

    assert "- formula_focus:" in summary
    assert "visual_type=formula" in summary
    assert "templates=penalty_curve" in summary
    assert "gradient_descent_3d" in summary
    assert "function_preset=['quadratic_ripple', 'double_well_2d']" in summary
