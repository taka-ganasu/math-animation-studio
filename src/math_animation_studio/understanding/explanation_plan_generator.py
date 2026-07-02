from __future__ import annotations

from math_animation_studio.schema import ExplanationPlan

from .sample_plans import sample_explanation_plan


class ExplanationPlanGenerator:
    def generate(self, *, formula: str, sample_key: str, audience: str) -> ExplanationPlan:
        return sample_explanation_plan(formula, sample_key, audience)
