from __future__ import annotations

from typing import Literal

from pydantic import Field

from .concept_classification import ConceptClassification
from .explanation_plan import ExplanationPlan
from .formula_analysis import FormulaAnalysis
from .prerequisite_map import PrerequisiteMap
from .storyboard import StrictModel


class LLMGenerationBoundary(StrictModel):
    llm_role: Literal["education_planner"] = "education_planner"
    renderer_role: Literal["template_renderer"] = "template_renderer"
    code_generation_allowed: bool = False
    allowed_outputs: list[str] = Field(
        default_factory=lambda: [
            "formula_analysis",
            "concept_classification",
            "prerequisite_map",
            "explanation_plan",
            "declarative_visual_intent",
        ]
    )
    forbidden_outputs: list[str] = Field(
        default_factory=lambda: [
            "python_code",
            "manim_code",
            "eval",
            "exec",
            "arbitrary_function_body",
        ]
    )


class FormulaUnderstandingLLMPlan(StrictModel):
    formula_analysis: FormulaAnalysis
    concept_classification: ConceptClassification
    prerequisite_map: PrerequisiteMap
    explanation_plan: ExplanationPlan
    generation_boundary: LLMGenerationBoundary = Field(default_factory=LLMGenerationBoundary)
