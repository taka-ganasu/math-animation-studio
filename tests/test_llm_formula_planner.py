from __future__ import annotations

from typing import Any

import pytest

from math_animation_studio.llm import LLMUnavailableError
from math_animation_studio.schema import (
    FormulaPlanConsistencyReview,
    FormulaUnderstandingLLMPlan,
    LLMGenerationBoundary,
)
from math_animation_studio.understanding import LLMFormulaUnderstandingPlanner
from math_animation_studio.understanding.sample_plans import (
    sample_classification,
    sample_explanation_plan,
    sample_formula_analysis,
    sample_prerequisites,
)


class FakeLLMClient:
    def __init__(self) -> None:
        self.last_prompt = ""
        self.last_schema_name = ""
        self.schema_names: list[str] = []
        self.prompts: list[str] = []

    def complete_model(self, **kwargs: Any) -> FormulaUnderstandingLLMPlan | FormulaPlanConsistencyReview:
        self.last_prompt = kwargs["prompt"]
        self.last_schema_name = kwargs["schema_name"]
        self.schema_names.append(kwargs["schema_name"])
        self.prompts.append(kwargs["prompt"])
        if kwargs["response_model"] is FormulaPlanConsistencyReview:
            return FormulaPlanConsistencyReview(
                is_consistent=True,
                target_concept="cross_entropy",
                selected_animation_pattern_id="penalty_curve",
            )
        return FormulaUnderstandingLLMPlan(
            formula_analysis=sample_formula_analysis(
                r"L = - \sum_i y_i \log(\hat{y}_i)",
                "cross_entropy",
            ),
            concept_classification=sample_classification("cross_entropy"),
            prerequisite_map=sample_prerequisites("cross_entropy"),
            explanation_plan=sample_explanation_plan(
                r"L = - \sum_i y_i \log(\hat{y}_i)",
                "cross_entropy",
                "engineer_beginner_math",
            ),
        )


class UnsafeFakeLLMClient(FakeLLMClient):
    def complete_model(self, **kwargs: Any) -> FormulaUnderstandingLLMPlan:
        if kwargs["response_model"] is FormulaPlanConsistencyReview:
            return FormulaPlanConsistencyReview(
                is_consistent=True,
                target_concept="cross_entropy",
                selected_animation_pattern_id="penalty_curve",
            )
        plan = super().complete_model(**kwargs)
        assert isinstance(plan, FormulaUnderstandingLLMPlan)
        return plan.model_copy(
            update={
                "generation_boundary": LLMGenerationBoundary(
                    code_generation_allowed=True,
                )
            }
        )


class InconsistentThenRevisionFakeLLMClient:
    def __init__(self) -> None:
        self.schema_names: list[str] = []
        self.prompts: list[str] = []

    def complete_model(self, **kwargs: Any) -> FormulaUnderstandingLLMPlan | FormulaPlanConsistencyReview:
        self.schema_names.append(kwargs["schema_name"])
        self.prompts.append(kwargs["prompt"])
        if kwargs["response_model"] is FormulaPlanConsistencyReview:
            return FormulaPlanConsistencyReview(
                is_consistent=False,
                target_concept="gradient_descent",
                selected_animation_pattern_id="trajectory_on_surface",
                issues=["数式名に引っ張られてクロスエントロピーが主題になっている。"],
                revision_instructions=[
                    "理解ゴールに合わせ、勾配降下法を主題にする。",
                    "クロスエントロピー式は損失地形の例としてだけ扱う。",
                    "selected_animation_pattern_idはtrajectory_on_surfaceにする。",
                ],
            )
        if kwargs["schema_name"] == "formula_understanding_plan_revision":
            return FormulaUnderstandingLLMPlan(
                formula_analysis=sample_formula_analysis(
                    r"\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)",
                    "gradient_descent",
                ),
                concept_classification=sample_classification("gradient_descent"),
                prerequisite_map=sample_prerequisites("gradient_descent"),
                explanation_plan=sample_explanation_plan(
                    r"\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)",
                    "gradient_descent",
                    "high_school_math",
                ),
            )
        return FormulaUnderstandingLLMPlan(
            formula_analysis=sample_formula_analysis(
                r"L = - \sum_i y_i \log(\hat{y}_i)",
                "cross_entropy",
            ),
            concept_classification=sample_classification("cross_entropy"),
            prerequisite_map=sample_prerequisites("cross_entropy"),
            explanation_plan=sample_explanation_plan(
                r"L = - \sum_i y_i \log(\hat{y}_i)",
                "cross_entropy",
                "high_school_math",
            ),
        )


def test_llm_formula_planner_builds_structured_plan() -> None:
    client = FakeLLMClient()
    planner = LLMFormulaUnderstandingPlanner(client=client)  # type: ignore[arg-type]

    result = planner.plan(
        formula=r"L = - \sum_i y_i \log(\hat{y}_i)",
        goal="初学者向けに理解したい",
        audience="engineer_beginner_math",
        domain_hint="machine_learning",
        animation_pattern_ids=["penalty_curve", "generic_symbol_decomposition"],
        target_duration_seconds=60,
        concept_hint="cross_entropy",
    )

    assert result.formula_analysis.detected_name == "cross_entropy"
    assert result.explanation_plan.selected_animation_pattern_id == "penalty_curve"
    assert result.generation_boundary.llm_role == "education_planner"
    assert result.generation_boundary.code_generation_allowed is False
    assert "60秒" in client.prompts[0]
    assert "penalty_curve" in client.prompts[0]
    assert "cross_entropy" in client.prompts[0]
    assert client.schema_names == [
        "formula_understanding_plan",
        "formula_plan_consistency_review",
    ]


def test_llm_formula_planner_revises_inconsistent_plan_with_llm_review() -> None:
    client = InconsistentThenRevisionFakeLLMClient()
    planner = LLMFormulaUnderstandingPlanner(client=client)  # type: ignore[arg-type]

    result = planner.plan(
        formula=r"L = - \sum_i y_i \log(\hat{y}_i)",
        goal="2変数を変更した場合の勾配降下法の動きを幾何的に理解したい",
        audience="high_school_math",
        domain_hint="machine_learning",
        animation_pattern_ids=["penalty_curve", "trajectory_on_surface"],
        target_duration_seconds=60,
        concept_hint="gradient_descent",
    )

    assert result.concept_classification.primary_concept == "gradient_descent"
    assert result.explanation_plan.selected_animation_pattern_id == "trajectory_on_surface"
    assert client.schema_names == [
        "formula_understanding_plan",
        "formula_plan_consistency_review",
        "formula_understanding_plan_revision",
    ]
    assert "revision_instructions" in client.prompts[2]


def test_llm_formula_planner_rejects_code_generation_boundary() -> None:
    planner = LLMFormulaUnderstandingPlanner(client=UnsafeFakeLLMClient())  # type: ignore[arg-type]

    with pytest.raises(LLMUnavailableError, match="code generation"):
        planner.plan(
            formula=r"L = - \sum_i y_i \log(\hat{y}_i)",
            goal="初学者向けに理解したい",
            audience="engineer_beginner_math",
            domain_hint="machine_learning",
            animation_pattern_ids=["penalty_curve"],
            target_duration_seconds=60,
        )
