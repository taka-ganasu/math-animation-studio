from __future__ import annotations

from pydantic import Field

from .storyboard import ExampleValue, StrictModel


class VisualComponentParamDefinition(StrictModel):
    name: str
    value_type: str
    description: str
    required: bool = False
    default: ExampleValue | None = None
    allowed_values: list[str] = Field(default_factory=list)


class VisualComponentDefinition(StrictModel):
    id: str
    name: str
    category: str
    description: str
    use_when: list[str]
    avoid_when: list[str] = Field(default_factory=list)
    visual_type: str
    template_support: list[str]
    params: list[VisualComponentParamDefinition] = Field(default_factory=list)
    example: str
