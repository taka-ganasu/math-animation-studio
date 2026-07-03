from __future__ import annotations

from math_animation_studio.llm import (
    LLMClient,
    LLMUnavailableError,
    build_formula_understanding_plan_prompt,
)
from math_animation_studio.schema import FormulaUnderstandingLLMPlan


class LLMFormulaUnderstandingPlanner:
    def __init__(self, *, client: LLMClient | None = None) -> None:
        self.client = client or LLMClient()

    def plan(
        self,
        *,
        formula: str,
        goal: str | None,
        audience: str,
        domain_hint: str | None,
        animation_pattern_ids: list[str],
        target_duration_seconds: int,
    ) -> FormulaUnderstandingLLMPlan:
        prompt = build_formula_understanding_plan_prompt(
            formula=formula,
            goal=goal,
            audience=audience,
            domain_hint=domain_hint,
            animation_pattern_ids=animation_pattern_ids,
            target_duration_seconds=target_duration_seconds,
        )
        plan = self.client.complete_model(
            prompt=prompt,
            response_model=FormulaUnderstandingLLMPlan,
            schema_name="formula_understanding_plan",
        )
        if plan.generation_boundary.code_generation_allowed:
            raise LLMUnavailableError(
                "LLM output requested code generation, but this renderer only accepts "
                "declarative plans mapped to built-in templates."
            )
        return plan
