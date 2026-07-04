from __future__ import annotations

from typer.testing import CliRunner

from math_animation_studio.cli import app


runner = CliRunner()


def test_cli_help() -> None:
    result = runner.invoke(app, ["--help"])

    assert result.exit_code == 0
    assert "generate" in result.output


def test_generate_help() -> None:
    result = runner.invoke(app, ["generate", "--help"])

    assert result.exit_code == 0
    assert "--no-llm" in result.output


def test_catalog_markdown() -> None:
    result = runner.invoke(app, ["catalog"])

    assert result.exit_code == 0, result.output
    assert "Storyboard DSL" in result.output
    assert "formula_first" in result.output
    assert "Visual Components" in result.output
    assert "formula_focus" in result.output


def test_catalog_json() -> None:
    result = runner.invoke(app, ["catalog", "--format", "json"])

    assert result.exit_code == 0, result.output
    assert '"storyboard_dsl"' in result.output
    assert '"visual_components"' in result.output
    assert '"formula_first"' in result.output


def test_generate_no_llm_no_render(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "generate",
            "--concept",
            "gradient_descent",
            "--output-dir",
            str(tmp_path),
            "--no-llm",
            "--no-render",
        ],
    )

    assert result.exit_code == 0, result.output
    assert (tmp_path / "storyboard.json").exists()
    assert (tmp_path / "symbols.md").exists()
    assert (tmp_path / "narration.md").exists()
    assert (tmp_path / "manim_scene.py").exists()
    assert (tmp_path / "metadata.json").exists()
