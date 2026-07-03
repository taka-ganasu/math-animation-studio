from __future__ import annotations

from pydantic import BaseModel

from math_animation_studio.llm import LLMClient
from math_animation_studio.schema import FormulaUnderstandingLLMPlan
from math_animation_studio.understanding.sample_plans import (
    sample_classification,
    sample_explanation_plan,
    sample_formula_analysis,
    sample_prerequisites,
)


class RetryFakeLLMClient(LLMClient):
    def __init__(self, responses: list[str]) -> None:
        self.responses = responses
        self.prompts: list[str] = []

    def _complete_json_schema(
        self,
        *,
        prompt: str,
        response_model: type[BaseModel],
        schema_name: str,
    ) -> str:
        self.prompts.append(prompt)
        return self.responses.pop(0)


def test_complete_model_retries_schema_validation_error() -> None:
    valid_plan = FormulaUnderstandingLLMPlan(
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
    client = RetryFakeLLMClient(
        responses=[
            '{"formula_analysis": {}}',
            valid_plan.model_dump_json(),
        ]
    )

    result = client.complete_model(
        prompt="original prompt",
        response_model=FormulaUnderstandingLLMPlan,
        schema_name="formula_understanding_plan",
    )

    assert result.formula_analysis.detected_name == "cross_entropy"
    assert len(client.prompts) == 2
    assert "Validation error" in client.prompts[1]
    assert "前回のJSON" in client.prompts[1]
