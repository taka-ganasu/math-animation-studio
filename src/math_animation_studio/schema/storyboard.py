from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class SymbolDefinition(StrictModel):
    symbol: str
    meaning: str
    intuition: str | None = None


class Example(StrictModel):
    title: str
    description: str
    values: dict[str, float | int | str] = Field(default_factory=dict)


class VisualObject(StrictModel):
    type: Literal[
        "surface",
        "point",
        "vector",
        "formula",
        "text",
        "axis",
        "contour",
        "curve",
        "camera",
    ]
    name: str
    description: str
    params: dict[str, Any] = Field(default_factory=dict)


class SceneSpec(StrictModel):
    id: str
    title: str
    learning_goal: str
    narration: str
    visual_objects: list[VisualObject]
    duration_seconds: int = Field(default=10, gt=0)


class Storyboard(StrictModel):
    concept: str
    formula: str | None = None
    one_sentence_summary: str
    audience: str
    prerequisites: list[str] = Field(default_factory=list)
    symbol_ledger: list[SymbolDefinition]
    examples: list[Example]
    scenes: list[SceneSpec]
    misconceptions: list[str] = Field(default_factory=list)


def load_storyboard(path: Path) -> Storyboard:
    return Storyboard.model_validate_json(path.read_text(encoding="utf-8"))


def save_storyboard(storyboard: Storyboard, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        storyboard.model_dump_json(indent=2, exclude_none=False),
        encoding="utf-8",
    )
