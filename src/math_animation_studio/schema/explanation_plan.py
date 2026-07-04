from __future__ import annotations

from typing import Literal

from pydantic import Field

from .storyboard import ExampleValue, SceneRole, StrictModel


class TeachingExample(StrictModel):
    title: str
    description: str
    why_it_works: str
    concrete_values: dict[str, ExampleValue] = Field(default_factory=dict)


class PlannedAnimationComponent(StrictModel):
    kind: str
    description: str
    label: str | None = None
    params: dict[str, ExampleValue] = Field(default_factory=dict)


class ExplanationStep(StrictModel):
    id: str
    scene_role: SceneRole | None = None
    title: str
    learning_goal: str
    explanation: str
    visual_idea: str
    formula_focus: str | None = None
    common_misunderstanding_addressed: str | None = None
    planned_components: list[PlannedAnimationComponent] = Field(default_factory=list)


class ExplanationPlan(StrictModel):
    formula: str
    target_concept: str
    one_sentence_summary: str
    audience: str
    teaching_strategy: Literal[
        "concrete_to_abstract",
        "visual_first",
        "symbol_decomposition",
        "compare_good_bad_examples",
        "geometric_intuition",
        "probabilistic_intuition",
    ]
    recommended_examples: list[TeachingExample]
    selected_animation_pattern_id: str
    explanation_steps: list[ExplanationStep]
    misconceptions: list[str] = Field(default_factory=list)
    next_questions_to_study: list[str] = Field(default_factory=list)
