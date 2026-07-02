from __future__ import annotations

from math_animation_studio.schema import FormulaAnalysis
from math_animation_studio.understanding.sample_plans import sample_formula_analysis


def test_formula_analysis_schema_parses_cross_entropy_sample() -> None:
    analysis = sample_formula_analysis(
        r"L = - \sum_i y_i \log(\hat{y}_i)",
        "cross_entropy",
    )

    parsed = FormulaAnalysis.model_validate_json(analysis.model_dump_json())
    assert parsed.detected_name == "cross_entropy"
    assert parsed.confidence > 0.8
