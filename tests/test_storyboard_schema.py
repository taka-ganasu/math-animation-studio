from __future__ import annotations

import json
from importlib.resources import files

import pytest
from pydantic import ValidationError

from math_animation_studio.schema import AnimationComponent, NarrationCue, Storyboard


def test_sample_storyboard_parses() -> None:
    sample = files("math_animation_studio.samples").joinpath(
        "gradient_descent_storyboard.json"
    )
    storyboard = Storyboard.model_validate_json(sample.read_text(encoding="utf-8"))
    assert storyboard.concept == "gradient_descent"
    assert len(storyboard.scenes) == 6
    assert storyboard.scenes[0].components == []
    assert storyboard.scenes[0].narration_cues == []


def test_storyboard_accepts_animation_components_and_narration_cues() -> None:
    component = AnimationComponent(
        id="formula_focus",
        kind="formula_focus",
        description="式の一部を強調する",
        label=r"\log(\hat{y}_i)",
    )
    cue = NarrationCue(
        segment_id="focus_log",
        text="ここではlogを見ます。",
        duration_seconds=3.0,
        component_id="formula_focus",
        formula_focus=r"\log(\hat{y}_i)",
    )

    assert component.kind == "formula_focus"
    assert cue.component_id == "formula_focus"


def test_storyboard_accepts_metaphor_animation_components() -> None:
    component = AnimationComponent(
        id="terrain",
        kind="terrain_metaphor",
        description="損失を山として見せる",
        label="height = loss",
    )

    assert component.kind == "terrain_metaphor"


def test_missing_required_field_fails() -> None:
    payload = {
        "concept": "gradient_descent",
        "audience": "engineer_beginner_math",
        "symbol_ledger": [],
        "examples": [],
        "scenes": [],
    }

    with pytest.raises(ValidationError):
        Storyboard.model_validate_json(json.dumps(payload))
