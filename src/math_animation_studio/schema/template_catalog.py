from __future__ import annotations

from pydantic import Field

from .storyboard import SceneRole, StrictModel


class TemplateChapterDefinition(StrictModel):
    id: str
    role: SceneRole
    title: str
    intent: str
    segment_ids: list[str] = Field(default_factory=list)
    component_kinds: list[str] = Field(default_factory=list)


class TemplateDefinition(StrictModel):
    id: str
    concept: str
    name: str
    manim_template: str
    description: str
    chapters: list[TemplateChapterDefinition]
