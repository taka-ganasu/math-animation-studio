from __future__ import annotations

from typing import Literal

from pydantic import Field

from .storyboard import StrictModel


class ConceptClassification(StrictModel):
    primary_domain: Literal[
        "calculus",
        "linear_algebra",
        "optimization",
        "probability",
        "statistics",
        "machine_learning",
        "deep_learning",
        "information_theory",
        "unknown",
    ]
    primary_concept: str
    related_concepts: list[str] = Field(default_factory=list)
    difficulty_level: Literal[
        "high_school",
        "undergraduate_intro",
        "undergraduate_advanced",
        "graduate_intro",
        "advanced",
    ]
    recommended_animation_family: str
    confidence: float = Field(ge=0.0, le=1.0)
