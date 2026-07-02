from __future__ import annotations

import json
from importlib.resources import files

import pytest
from pydantic import ValidationError

from math_animation_studio.schema import Storyboard


def test_sample_storyboard_parses() -> None:
    sample = files("math_animation_studio.samples").joinpath(
        "gradient_descent_storyboard.json"
    )
    storyboard = Storyboard.model_validate_json(sample.read_text(encoding="utf-8"))
    assert storyboard.concept == "gradient_descent"
    assert len(storyboard.scenes) == 6


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
