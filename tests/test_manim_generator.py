from __future__ import annotations

from importlib.resources import files

from math_animation_studio.generator import ManimGenerator
from math_animation_studio.schema import Storyboard
from math_animation_studio.understanding import FormulaUnderstandingPlanner
from math_animation_studio.validation import validate_python_syntax


def test_generator_writes_compilable_manim_scene(tmp_path) -> None:
    sample = files("math_animation_studio.samples").joinpath(
        "gradient_descent_storyboard.json"
    )
    storyboard = Storyboard.model_validate_json(sample.read_text(encoding="utf-8"))
    output_path = tmp_path / "manim_scene.py"

    ManimGenerator().generate(storyboard, output_path)

    assert output_path.exists()
    assert "class GradientDescent3DScene" in output_path.read_text(encoding="utf-8")
    validate_python_syntax(output_path)


def test_generator_writes_compilable_cross_entropy_scene(tmp_path) -> None:
    artifacts = FormulaUnderstandingPlanner(no_llm=True).plan(
        formula=r"L = - \sum_i y_i \log(\hat{y}_i)",
        goal="クロスエントロピー損失を直感的に理解したい",
        audience="engineer_beginner_math",
    )
    assert artifacts.storyboard is not None

    output_path = tmp_path / "manim_scene.py"
    generator = ManimGenerator()

    generator.generate(artifacts.storyboard, output_path)

    assert output_path.exists()
    assert generator.scene_name_for(artifacts.storyboard) == "CrossEntropyPenaltyScene"
    assert "class CrossEntropyPenaltyScene" in output_path.read_text(encoding="utf-8")
    validate_python_syntax(output_path)
