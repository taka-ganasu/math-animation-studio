from __future__ import annotations

from typing import Any

import pytest

from math_animation_studio.llm import LLMUnavailableError
from math_animation_studio.schema import FormulaUnderstandingLLMPlan, LLMGenerationBoundary
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

    def complete_model(self, **kwargs: Any) -> FormulaUnderstandingLLMPlan:
        self.last_prompt = kwargs["prompt"]
        self.last_schema_name = kwargs["schema_name"]
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
        plan = super().complete_model(**kwargs)
        return plan.model_copy(
            update={
                "generation_boundary": LLMGenerationBoundary(
                    code_generation_allowed=True,
                )
            }
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
    )

    assert result.formula_analysis.detected_name == "cross_entropy"
    assert result.explanation_plan.selected_animation_pattern_id == "penalty_curve"
    assert result.generation_boundary.llm_role == "education_planner"
    assert result.generation_boundary.code_generation_allowed is False
    assert "60秒" in client.last_prompt
    assert "penalty_curve" in client.last_prompt
    assert client.last_schema_name == "formula_understanding_plan"


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
