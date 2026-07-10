from __future__ import annotations

import json

from typer.testing import CliRunner

from math_animation_studio.cli import app
from math_animation_studio.schema import Storyboard


runner = CliRunner()


def test_plan_help() -> None:
    result = runner.invoke(app, ["plan", "--help"])

    assert result.exit_code == 0
    assert "--formula" in result.output
    assert "--duration" in result.output
    assert "--concept-hint" in result.output
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
    explanation_plan = json.loads((tmp_path / "explanation_plan.json").read_text(encoding="utf-8"))
    brief = (tmp_path / "animation_brief.md").read_text(encoding="utf-8")

    assert storyboard.concept == "cross_entropy"
    assert len(storyboard.examples) == 1
    assert len(explanation_plan["recommended_examples"]) >= 3
    assert storyboard.scenes[0].components
    assert storyboard.scenes[0].narration_cues
    assert "一言でいうと" in brief
    assert "$$\nL = - \\sum_i y_i \\log(\\hat{y}_i)\n$$" in brief
    assert "## 記号と操作" in brief
    assert "$\\log(\\hat{y}_i)$" in brief
    assert "注目する式: $y_i$" in brief
    assert "\\[" not in brief


def test_plan_interactive_example_accepts_recommended_example(tmp_path) -> None:
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
            "--interactive-example",
        ],
        input="\n\n",
    )

    assert result.exit_code == 0, result.output
    assert "Recommended teaching example" in result.output
    storyboard = Storyboard.model_validate_json(
        (tmp_path / "storyboard.json").read_text(encoding="utf-8")
    )

    assert storyboard.examples[0].title == "3クラス分類: 猫・犬・鳥"


def test_plan_interactive_example_can_choose_llm_style_candidate(tmp_path) -> None:
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
            "--interactive-example",
        ],
        input="2\n\n",
    )

    assert result.exit_code == 0, result.output
    assert "2. サイコロの目予測例" in result.output
    storyboard = Storyboard.model_validate_json(
        (tmp_path / "storyboard.json").read_text(encoding="utf-8")
    )
    brief = (tmp_path / "animation_brief.md").read_text(encoding="utf-8")

    assert storyboard.examples[0].title == "サイコロの目予測例"
    assert storyboard.examples[0].values["y"] == [0, 0, 1, 0, 0, 0]
    assert "サイコロの目予測例" in brief


def test_plan_interactive_example_allows_editing_example(tmp_path) -> None:
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
            "--interactive-example",
        ],
        input=(
            "1\n"
            "n\n"
            "サイコロの目予測\n"
            "正解が目3のとき、6個の候補のうち正解の確率だけを罰に変える。\n"
            "分類の候補数が見えやすく、one-hotの意味を確認しやすい。\n"
        ),
    )

    assert result.exit_code == 0, result.output
    storyboard = Storyboard.model_validate_json(
        (tmp_path / "storyboard.json").read_text(encoding="utf-8")
    )
    brief = (tmp_path / "animation_brief.md").read_text(encoding="utf-8")

    assert storyboard.examples[0].title == "サイコロの目予測"
    assert "サイコロの目予測" in brief
    assert "分類の候補数が見えやすく" in brief


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
    component_kinds = {
        component.kind
        for scene in storyboard.scenes
        for component in scene.components
    }
    assert "terrain_metaphor" in component_kinds
    assert "uphill_arrow" in component_kinds
    assert "downhill_arrow" in component_kinds
    assert "formula_bridge" in component_kinds
    assert "局所最小" in brief
    assert "SGD" in brief


def test_plan_no_llm_concept_hint_overrides_cross_entropy_formula(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "plan",
            "--formula",
            r"L = - \sum_i y_i \log(\hat{y}_i)",
            "--goal",
            "2次元で谷が2箇所ある時に勾配降下法がどう判断するか知りたい",
            "--concept-hint",
            "gradient_descent",
            "--output-dir",
            str(tmp_path),
            "--no-llm",
        ],
    )

    assert result.exit_code == 0, result.output
    storyboard = Storyboard.model_validate_json(
        (tmp_path / "storyboard.json").read_text(encoding="utf-8")
    )
    metadata = json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8"))

    assert storyboard.concept == "gradient_descent"
    assert storyboard.examples[0].values["function_preset"] == "double_well_2d"
    assert metadata["concept_hint"] == "gradient_descent"


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


def test_plan_no_llm_chain_rule_outputs_storyboard(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "plan",
            "--formula",
            r"\frac{dy}{dx}=\frac{dy}{du}\frac{du}{dx}",
            "--goal",
            "連鎖律を2階微分ではなく変化率のつながりとして理解したい",
            "--concept-hint",
            "chain_rule",
            "--output-dir",
            str(tmp_path),
            "--no-llm",
        ],
    )

    assert result.exit_code == 0, result.output
    storyboard = Storyboard.model_validate_json(
        (tmp_path / "storyboard.json").read_text(encoding="utf-8")
    )
    explanation_plan = json.loads((tmp_path / "explanation_plan.json").read_text(encoding="utf-8"))
    brief = (tmp_path / "animation_brief.md").read_text(encoding="utf-8")

    assert storyboard.concept == "chain_rule"
    assert storyboard.examples[0].values["dy_dx"] == 12
    assert explanation_plan["selected_animation_pattern_id"] == "chain_rule_flow"
    assert "$$\n\\frac{dy}{dx}=\\frac{dy}{du}\\frac{du}{dx}\n$$" in brief
    assert "2階微分ではない" in brief


def test_plan_no_llm_neural_network_transform_outputs_storyboard(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "plan",
            "--formula",
            r"h=\sigma(Wx+b)",
            "--goal",
            "ニューラルネットワークにおける線形変換・非線形変換・中間表現の意味を直感的に理解したい",
            "--concept-hint",
            "neural_network_transform",
            "--output-dir",
            str(tmp_path),
            "--no-llm",
        ],
    )

    assert result.exit_code == 0, result.output
    storyboard = Storyboard.model_validate_json(
        (tmp_path / "storyboard.json").read_text(encoding="utf-8")
    )
    explanation_plan = json.loads((tmp_path / "explanation_plan.json").read_text(encoding="utf-8"))
    brief = (tmp_path / "animation_brief.md").read_text(encoding="utf-8")

    assert storyboard.concept == "neural_network_transform"
    assert storyboard.examples[0].values["activation"] == "relu"
    assert explanation_plan["selected_animation_pattern_id"] == "neural_network_transform_flow"
    assert "$$\nh=\\sigma(Wx+b)\n$$" in brief
    assert "解きやすい座標系" in brief


def test_plan_no_llm_activation_functions_outputs_storyboard(tmp_path) -> None:
    result = runner.invoke(
        app,
        [
            "plan",
            "--formula",
            r"a=f(z),\quad p=\mathrm{softmax}(o)",
            "--goal",
            "ReLU、sigmoid、tanh、softmaxの違いと、隠れ層・出力層での使い分けを直感的に理解したい",
            "--concept-hint",
            "activation_functions",
            "--output-dir",
            str(tmp_path),
            "--no-llm",
        ],
    )

    assert result.exit_code == 0, result.output
    storyboard = Storyboard.model_validate_json(
        (tmp_path / "storyboard.json").read_text(encoding="utf-8")
    )
    explanation_plan = json.loads((tmp_path / "explanation_plan.json").read_text(encoding="utf-8"))
    brief = (tmp_path / "animation_brief.md").read_text(encoding="utf-8")

    assert storyboard.concept == "activation_functions"
    assert storyboard.examples[0].values["class_labels"] == ["猫", "犬", "鳥"]
    assert explanation_plan["selected_animation_pattern_id"] == "activation_function_comparison"
    assert "$$\na=f(z),\\quad p=\\mathrm{softmax}(o)\n$$" in brief
    assert "Adam" not in brief
    assert "softmaxとクロスエントロピー" in brief
