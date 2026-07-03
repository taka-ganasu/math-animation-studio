from __future__ import annotations

from math_animation_studio.schema import ExplanationPlan
from math_animation_studio.understanding.sample_plans import sample_explanation_plan


def test_explanation_plan_schema_parses_attention_sample() -> None:
    plan = sample_explanation_plan(
        r"Attention(Q, K, V) = softmax(QK^T / \sqrt{d_k})V",
        "scaled_dot_product_attention",
        "engineer_beginner_math",
    )

    parsed = ExplanationPlan.model_validate_json(plan.model_dump_json())
    assert parsed.target_concept == "scaled_dot_product_attention"
    assert len(parsed.explanation_steps) >= 5


def test_explanation_plan_accepts_array_concrete_values() -> None:
    plan = sample_explanation_plan(
        r"L = - \sum_i y_i \log(\hat{y}_i)",
        "cross_entropy",
        "engineer_beginner_math",
    )

    parsed = ExplanationPlan.model_validate_json(plan.model_dump_json())

    assert parsed.recommended_examples[1].concrete_values["y"] == [0, 0, 1, 0, 0, 0]
