from __future__ import annotations

import json
from importlib.resources import files

from math_animation_studio.artifacts import ArtifactManager
from math_animation_studio.schema import Storyboard


def _sample_storyboard() -> Storyboard:
    sample = files("math_animation_studio.samples").joinpath(
        "gradient_descent_storyboard.json"
    )
    return Storyboard.model_validate_json(sample.read_text(encoding="utf-8"))


def test_artifact_manager_writes_expected_files(tmp_path) -> None:
    storyboard = _sample_storyboard()
    manager = ArtifactManager(tmp_path)
    manager.prepare()
    manager.write_storyboard(storyboard)
    manager.write_symbols(storyboard)
    manager.write_narration(storyboard)
    manager.write_metadata(
        storyboard=storyboard,
        status="generated",
        duration_seconds_target=60,
    )

    assert manager.storyboard_path.exists()
    assert manager.symbols_path.read_text(encoding="utf-8").startswith("# Symbols")
    assert manager.narration_path.read_text(encoding="utf-8").startswith("# Narration")
    metadata = json.loads(manager.metadata_path.read_text(encoding="utf-8"))
    assert metadata["concept"] == "gradient_descent"
    assert metadata["status"] == "generated"
