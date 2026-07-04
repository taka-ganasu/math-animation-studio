from __future__ import annotations

from math_animation_studio.schema import ExplanationPlan, PlannedAnimationComponent
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


def test_explanation_step_accepts_planned_components() -> None:
    plan = sample_explanation_plan(
        r"L = - \sum_i y_i \log(\hat{y}_i)",
        "cross_entropy",
        "engineer_beginner_math",
    )
    first_step = plan.explanation_steps[0].model_copy(
        update={
            "planned_components": [
                PlannedAnimationComponent(
                    kind="formula_focus",
                    description="y_iを強調する",
                    label="y_i",
                    params={"formula_focus": "y_i", "emphasis": "zoom"},
                )
            ]
        }
    )
    plan = plan.model_copy(update={"explanation_steps": [first_step, *plan.explanation_steps[1:]]})

    parsed = ExplanationPlan.model_validate_json(plan.model_dump_json())

    assert parsed.explanation_steps[0].planned_components[0].kind == "formula_focus"
    assert parsed.explanation_steps[0].planned_components[0].params["emphasis"] == "zoom"
