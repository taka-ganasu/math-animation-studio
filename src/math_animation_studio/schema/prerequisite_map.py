from __future__ import annotations

from typing import Literal

from pydantic import Field

from .storyboard import StrictModel


class PrerequisiteItem(StrictModel):
    concept: str
    why_needed: str
    priority: Literal["required", "helpful", "optional"]
    suggested_micro_explanation: str


class PrerequisiteMap(StrictModel):
    target_concept: str
    prerequisites: list[PrerequisiteItem]
    likely_blockers: list[str] = Field(default_factory=list)
