from __future__ import annotations

from importlib.resources import files

from math_animation_studio.generator import ManimGenerator
from math_animation_studio.schema import Example, SceneSpec, Storyboard, SymbolDefinition, VisualObject
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


def test_cross_entropy_scene_uses_storyboard_example_and_captions(tmp_path) -> None:
    storyboard = Storyboard(
        concept="cross_entropy",
        formula=r"L = - \sum_i y_i \log(\hat{y}_i)",
        one_sentence_summary="リンゴに高い確率を置けるほど、損失は小さくなります。",
        audience="high_school_math",
        symbol_ledger=[
            SymbolDefinition(
                symbol=r"\hat{y}_i",
                meaning="モデルが各クラスへ置いた予測確率",
                intuition="どちらだと思っているかの自信",
            )
        ],
        examples=[
            Example(
                title="イチゴとリンゴの分類例",
                description="リンゴが正解のときの二値分類例。",
                values={
                    "y": "[0, 1]",
                    r"\hat{y}_good": "[0.1, 0.9]",
                    r"\hat{y}_bad": "[0.8, 0.2]",
                },
            )
        ],
        scenes=[
            SceneSpec(
                id="step1",
                title="身近な分類例",
                learning_goal="クロスエントロピーを確率の罰として見る。",
                narration="リンゴが正解なら、リンゴへの予測確率だけを取り出して罰に変えます。",
                visual_objects=[
                    VisualObject(
                        type="formula",
                        name="loss",
                        description="クロスエントロピー",
                        params={"latex": r"L = - \sum_i y_i \log(\hat{y}_i)"},
                    )
                ],
                duration_seconds=10,
            )
        ],
    )
    output_path = tmp_path / "manim_scene.py"

    ManimGenerator(target_duration_seconds=60).generate(storyboard, output_path)
    rendered = output_path.read_text(encoding="utf-8")

    assert "イチゴ" in rendered
    assert "リンゴ" in rendered
    assert "リンゴへの予測確率だけ" in rendered
    assert "SCENE_COMPONENTS" in rendered
    assert "model_pipeline" in rendered
    assert "formula_parts_focus" in rendered
    assert "FORMULA_FOCUS_ITEMS" in rendered
    assert "decomposed_formula" in rendered
    assert "正解クラスだけ1になるone-hot" in rendered
    assert "GOOD_DISTRIBUTION = (0.1, 0.9)" in rendered
    assert "BAD_DISTRIBUTION = (0.8, 0.2)" in rendered
    validate_python_syntax(output_path)


def test_cross_entropy_dice_example_uses_six_class_pipeline(tmp_path) -> None:
    storyboard = Storyboard(
        concept="cross_entropy",
        formula=r"L = - \sum_i y_i \log(\hat{y}_i)",
        one_sentence_summary="サイコロの目が正解なら、6個の候補のうち正解の確率だけを罰に変えます。",
        audience="high_school_math",
        symbol_ledger=[
            SymbolDefinition(symbol="z", meaning="logits", intuition="softmax前の点数")
        ],
        examples=[
            Example(
                title="サイコロの目予測例",
                description="正解が目3のとき、6クラスの確率分布を見る。",
                values={
                    "y": "[0, 0, 1, 0, 0, 0]",
                    r"\hat{y}_good": "[0.03, 0.04, 0.80, 0.05, 0.04, 0.04]",
                    r"\hat{y}_bad": "[0.20, 0.25, 0.05, 0.20, 0.15, 0.15]",
                },
            )
        ],
        scenes=[
            SceneSpec(
                id="step1",
                title="モデル出力を見る",
                learning_goal="logitsからsoftmaxを通じて予測確率になる流れを見る。",
                narration="モデルはサイコロ画像からlogitsを出し、softmaxで6個の確率に変えます。",
                visual_objects=[
                    VisualObject(
                        type="formula",
                        name="pipeline",
                        description="model to softmax",
                        params={"latex": r"\hat{y}=\mathrm{softmax}(f_\theta(x))"},
                    )
                ],
                duration_seconds=10,
            )
        ],
    )
    output_path = tmp_path / "manim_scene.py"

    ManimGenerator(target_duration_seconds=60).generate(storyboard, output_path)
    rendered = output_path.read_text(encoding="utf-8")

    assert "CLASS_LABELS = ('目1', '目2', '目3', '目4', '目5', '目6')" in rendered
    assert "y = [\" + \", \".join(values) + \"]" in rendered
    assert "GOOD_DISTRIBUTION = (0.03, 0.04, 0.8, 0.05, 0.04, 0.04)" in rendered
    assert "BAD_DISTRIBUTION = (0.2, 0.25, 0.05, 0.2, 0.15, 0.15)" in rendered
    assert "softmax" in rendered
    assert "GOOD_LOGITS" in rendered
    validate_python_syntax(output_path)
