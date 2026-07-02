from __future__ import annotations

from pydantic import Field

from .storyboard import StrictModel


class AnimationPattern(StrictModel):
    id: str
    name: str
    suitable_for: list[str]
    description: str
    visual_metaphor: str
    manim_template: str | None = None
    required_visual_objects: list[str] = Field(default_factory=list)
