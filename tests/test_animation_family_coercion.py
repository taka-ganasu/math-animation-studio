from __future__ import annotations

from math_animation_studio.understanding.formula_understanding_planner import (
    _coerce_animation_family,
)
from math_animation_studio.understanding.sample_plans import (
    sample_classification,
    sample_explanation_plan,
    sample_formula_analysis,
)


def test_cross_entropy_llm_pattern_is_coerced_to_penalty_curve() -> None:
    formula = r"L = - \sum_i y_i \log(\hat{y}_i)"
    formula_analysis = sample_formula_analysis(formula, "cross_entropy")
    formula_analysis.detected_name = "クロスエントロピー損失関数"
    classification = sample_classification("cross_entropy")
    classification.primary_concept = "クロスエントロピー損失"
    explanation_plan = sample_explanation_plan(
        formula,
        "cross_entropy",
        "high_school_math",
    )
    explanation_plan.target_concept = "クロスエントロピー損失"

    result = _coerce_animation_family(
        requested="distribution_update",
        formula=formula,
        formula_analysis=formula_analysis,
        classification=classification,
        explanation_plan=explanation_plan,
    )

    assert result == "penalty_curve"


def test_unknown_pattern_is_not_coerced_without_known_signals() -> None:
    formula = "a + b = c"
    formula_analysis = sample_formula_analysis(formula, "generic")
    classification = sample_classification("generic")
    explanation_plan = sample_explanation_plan(
        formula,
        "generic",
        "high_school_math",
    )

    result = _coerce_animation_family(
        requested="generic_symbol_decomposition",
        formula=formula,
        formula_analysis=formula_analysis,
        classification=classification,
        explanation_plan=explanation_plan,
    )

    assert result == "generic_symbol_decomposition"
