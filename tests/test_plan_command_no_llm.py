from __future__ import annotations

from typer.testing import CliRunner

from math_animation_studio.cli import app
from math_animation_studio.schema import Storyboard


runner = CliRunner()


def test_plan_help() -> None:
    result = runner.invoke(app, ["plan", "--help"])

    assert result.exit_code == 0
    assert "--formula" in result.output
    assert "--voiceover" in result.output


def test_plan_voiceover_requires_render(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "plan",
            "--formula",
            r"L = - \sum_i y_i \log(\hat{y}_i)",
            "--output-dir",
            str(tmp_path),
            "--no-llm",
            "--voiceover",
        ],
    )

    assert result.exit_code == 1
    assert "--voiceover requires --render" in result.output


def test_plan_no_llm_cross_entropy_outputs_artifacts(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "plan",
            "--formula",
            r"L = - \sum_i y_i \log(\hat{y}_i)",
            "--goal",
            "クロスエントロピー損失を直感的に理解したい",
            "--output-dir",
            str(tmp_path),
            "--no-llm",
        ],
    )

    assert result.exit_code == 0, result.output
    expected = [
        "formula_analysis.json",
        "concept_classification.json",
        "prerequisite_map.json",
        "explanation_plan.json",
        "animation_brief.md",
        "storyboard.json",
        "metadata.json",
    ]
    for filename in expected:
        assert (tmp_path / filename).exists(), filename

    storyboard = Storyboard.model_validate_json(
        (tmp_path / "storyboard.json").read_text(encoding="utf-8")
    )
    brief = (tmp_path / "animation_brief.md").read_text(encoding="utf-8")

    assert storyboard.concept == "cross_entropy"
    assert "一言でいうと" in brief
    assert "$$\nL = - \\sum_i y_i \\log(\\hat{y}_i)\n$$" in brief
    assert "## 記号と操作" in brief
    assert "$\\log(\\hat{y}_i)$" in brief
    assert "注目する式: $y_i$" in brief
    assert "\\[" not in brief
