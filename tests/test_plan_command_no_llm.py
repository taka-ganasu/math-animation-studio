from __future__ import annotations

from typer.testing import CliRunner

from math_animation_studio.cli import app
from math_animation_studio.schema import Storyboard


runner = CliRunner()


def test_plan_help() -> None:
    result = runner.invoke(app, ["plan", "--help"])

    assert result.exit_code == 0
    assert "--formula" in result.output
    assert "--duration" in result.output
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
    assert storyboard.scenes[0].components
    assert storyboard.scenes[0].narration_cues
    assert "一言でいうと" in brief
    assert "$$\nL = - \\sum_i y_i \\log(\\hat{y}_i)\n$$" in brief
    assert "## 記号と操作" in brief
    assert "$\\log(\\hat{y}_i)$" in brief
    assert "注目する式: $y_i$" in brief
    assert "\\[" not in brief


def test_plan_no_llm_gradient_double_well_outputs_storyboard(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "plan",
            "--formula",
            r"\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)",
            "--goal",
            "2次元で谷が2箇所ある時に勾配降下法がどう判断するか知りたい",
            "--output-dir",
            str(tmp_path),
            "--no-llm",
        ],
    )

    assert result.exit_code == 0, result.output
    storyboard = Storyboard.model_validate_json(
        (tmp_path / "storyboard.json").read_text(encoding="utf-8")
    )
    brief = (tmp_path / "animation_brief.md").read_text(encoding="utf-8")

    assert storyboard.concept == "gradient_descent"
    assert storyboard.examples[0].values["function_preset"] == "double_well_2d"
    assert storyboard.scenes[0].components[0].kind == "contour_map"
    assert storyboard.scenes[0].narration_cues[0].component_id == "loss_landscape"
    assert "局所最小" in brief
    assert "SGD" in brief


def test_plan_no_llm_gradient_double_well_1d_outputs_storyboard(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "plan",
            "--formula",
            r"\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)",
            "--goal",
            "1変数の損失曲線で、谷→山→谷がある時に勾配降下法がどう判断するか知りたい",
            "--output-dir",
            str(tmp_path),
            "--no-llm",
        ],
    )

    assert result.exit_code == 0, result.output
    storyboard = Storyboard.model_validate_json(
        (tmp_path / "storyboard.json").read_text(encoding="utf-8")
    )
    brief = (tmp_path / "animation_brief.md").read_text(encoding="utf-8")

    assert storyboard.concept == "gradient_descent"
    assert storyboard.examples[0].values["function_preset"] == "double_well_1d"
    assert storyboard.scenes[0].components[0].kind == "loss_curve"
    assert storyboard.scenes[2].narration_cues[0].formula_focus == r"-\nabla L(\theta_t)"
    assert "損失曲線" in brief
    assert "谷・山・谷" in brief
