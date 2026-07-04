from __future__ import annotations

from math_animation_studio.llm import (
    LLMClient,
    LLMUnavailableError,
    build_formula_plan_consistency_prompt,
    build_formula_understanding_plan_prompt,
    build_formula_understanding_revision_prompt,
)
from math_animation_studio.schema import (
    FormulaPlanConsistencyReview,
    FormulaUnderstandingLLMPlan,
)


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
        concept_hint: str | None = None,
    ) -> FormulaUnderstandingLLMPlan:
        prompt = build_formula_understanding_plan_prompt(
            formula=formula,
            goal=goal,
            audience=audience,
            domain_hint=domain_hint,
            concept_hint=concept_hint,
            animation_pattern_ids=animation_pattern_ids,
            target_duration_seconds=target_duration_seconds,
        )
        plan = self.client.complete_model(
            prompt=prompt,
            response_model=FormulaUnderstandingLLMPlan,
            schema_name="formula_understanding_plan",
        )
        self._reject_code_generation(plan)

        review = self.client.complete_model(
            prompt=build_formula_plan_consistency_prompt(
                formula=formula,
                goal=goal,
                audience=audience,
                domain_hint=domain_hint,
                concept_hint=concept_hint,
                animation_pattern_ids=animation_pattern_ids,
                plan_json=plan.model_dump_json(indent=2),
            ),
            response_model=FormulaPlanConsistencyReview,
            schema_name="formula_plan_consistency_review",
        )
        if not review.is_consistent:
            plan = self.client.complete_model(
                prompt=build_formula_understanding_revision_prompt(
                    original_prompt=prompt,
                    first_plan_json=plan.model_dump_json(indent=2),
                    review_json=review.model_dump_json(indent=2),
                ),
                response_model=FormulaUnderstandingLLMPlan,
                schema_name="formula_understanding_plan_revision",
            )
            self._reject_code_generation(plan)
        return plan

    def _reject_code_generation(self, plan: FormulaUnderstandingLLMPlan) -> None:
        if not plan.generation_boundary.code_generation_allowed:
            return
        raise LLMUnavailableError(
            "LLM output requested code generation, but this renderer only accepts "
            "declarative plans mapped to built-in templates."
        )
