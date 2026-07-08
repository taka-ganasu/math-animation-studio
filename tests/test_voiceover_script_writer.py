from __future__ import annotations

import pytest

from math_animation_studio.understanding import FormulaUnderstandingPlanner
from math_animation_studio.voiceover import VoiceoverScriptWriter


def test_voiceover_script_writer_creates_cross_entropy_script() -> None:
    artifacts = FormulaUnderstandingPlanner(no_llm=True).plan(
        formula=r"L = - \sum_i y_i \log(\hat{y}_i)",
        goal="クロスエントロピー損失を直感的に理解したい",
        audience="engineer_beginner_math",
    )
    assert artifacts.storyboard is not None

    writer = VoiceoverScriptWriter()
    script = writer.write(artifacts.storyboard)
    slow_script = writer.write(artifacts.storyboard, target_duration_seconds=30)
    segments = writer.write_segments(artifacts.storyboard, target_duration_seconds=30)
    markdown = writer.write_markdown(
        artifacts.storyboard,
        target_duration_seconds=30,
        segments=segments,
    )

    assert "クロスエントロピー損失" in script
    assert "マイナスログ" in script
    assert len(script) < 170
    assert "3クラス分類: 猫・犬・鳥" in slow_script
    assert "式のy_i" in slow_script
    assert "ワイハットi" in slow_script
    assert "logワイハットi" in slow_script
    assert "softmaxで確率" in slow_script
    assert "正解クラスの確率p" in slow_script
    assert len(slow_script) > len(script)
    assert [segment.id for segment in segments][:6] == [
        "intro_formula",
        "focus_y_i",
        "focus_y_hat_i",
        "focus_log",
        "focus_sum",
        "focus_minus",
    ]
    assert "model_pipeline" in {segment.id for segment in segments}
    assert "negative_log_penalty" in {segment.id for segment in segments}
    focus_log = next(segment for segment in segments if segment.id == "focus_log")
    assert focus_log.component_id == "formula_parts_focus"
    assert focus_log.formula_focus == r"\log(\hat{y}_i)"
    assert sum(segment.duration_seconds for segment in segments) == pytest.approx(30.0)
    assert "Target duration: 30 seconds" in markdown
    assert "## Voiceover Segments" in markdown
    assert "### focus_log" in markdown
    assert "Component: `formula_parts_focus`" in markdown
    assert r"Focus: $\log(\hat{y}_i)$" in markdown
    assert "## Voiceover Script" in markdown
    assert "## Source Scenes" in markdown


def test_voiceover_script_writer_segments_gradient_double_well() -> None:
    artifacts = FormulaUnderstandingPlanner(no_llm=True).plan(
        formula=r"\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)",
        goal="2次元で谷が2箇所ある時に勾配降下法がどう判断するか知りたい",
        audience="high_school_math",
    )
    assert artifacts.storyboard is not None

    writer = VoiceoverScriptWriter()
    segments = writer.write_segments(artifacts.storyboard, target_duration_seconds=52)
    script = writer.write(artifacts.storyboard, target_duration_seconds=52)

    assert [segment.id for segment in segments] == [
        "intro_landscape",
        "two_valleys",
        "local_slope",
        "left_descent",
        "right_descent",
        "compare_minima",
        "sgd_bridge",
        "summary",
    ]
    assert segments[2].component_id == "gradient_arrow"
    assert segments[2].formula_focus == r"-\nabla L"
    assert sum(segment.duration_seconds for segment in segments) == pytest.approx(52.0)
    assert "局所最小" in script
    assert "SGD" in script


def test_voiceover_script_writer_segments_gradient_surface_formula_parts() -> None:
    artifacts = FormulaUnderstandingPlanner(no_llm=True).plan(
        formula=r"\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)",
        goal="2変数の勾配降下法を山を下る比喩で理解したい",
        audience="high_school_math",
        concept_hint="gradient_descent",
    )
    assert artifacts.storyboard is not None

    writer = VoiceoverScriptWriter()
    segments = writer.write_segments(artifacts.storyboard, target_duration_seconds=30)
    script = writer.write(artifacts.storyboard, target_duration_seconds=30)

    assert [segment.id for segment in segments] == [
        "title_intro",
        "formula_parts",
        "intro_surface",
        "current_point",
        "local_gradient",
        "descent_path",
        "summary_surface",
    ]
    assert segments[0].component_id == "intro_formula"
    assert segments[1].component_id == "formula_parts_focus"
    assert segments[1].formula_focus == r"\theta_{t+1}=\theta_t-\eta\nabla L(\theta_t)"
    assert sum(segment.duration_seconds for segment in segments) == pytest.approx(30.0)
    assert "勾配降下法について見ていきます" in script
    assert "更新式を分解" in script
    assert "シータt" in script
    assert "イータ" in script


def test_voiceover_script_writer_segments_gradient_double_well_1d() -> None:
    artifacts = FormulaUnderstandingPlanner(no_llm=True).plan(
        formula=r"\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)",
        goal="1変数の損失曲線で、谷→山→谷がある時に勾配降下法がどう判断するか知りたい",
        audience="high_school_math",
    )
    assert artifacts.storyboard is not None

    writer = VoiceoverScriptWriter()
    segments = writer.write_segments(artifacts.storyboard, target_duration_seconds=50)
    script = writer.write(artifacts.storyboard, target_duration_seconds=50)

    assert [segment.id for segment in segments] == [
        "intro_curve",
        "two_valleys_1d",
        "local_slope_1d",
        "left_descent_1d",
        "right_descent_1d",
        "compare_minima_1d",
        "sgd_bridge_1d",
        "summary_1d",
    ]
    assert segments[0].component_id == "loss_curve"
    assert segments[2].component_id == "gradient_arrow"
    assert segments[2].formula_focus == r"-\nabla L(\theta_t)"
    assert sum(segment.duration_seconds for segment in segments) == pytest.approx(50.0)
    assert "損失曲線" in script
    assert "山を越え" in script


def test_voiceover_script_writer_segments_perceptron() -> None:
    artifacts = FormulaUnderstandingPlanner(no_llm=True).plan(
        formula=r"a = \mathrm{step}(w_1x_1 + w_2x_2 + b)",
        goal="単純パーセプトロンの順伝播と決定境界を直感的に理解したい",
        audience="high_school_math",
        concept_hint="perceptron",
    )
    assert artifacts.storyboard is not None

    writer = VoiceoverScriptWriter()
    segments = writer.write_segments(artifacts.storyboard, target_duration_seconds=50)
    script = writer.write(artifacts.storyboard, target_duration_seconds=50)

    assert [segment.id for segment in segments] == [
        "title_intro",
        "formula_weighted_inputs",
        "formula_bias",
        "formula_activation",
        "formula_output",
        "network_diagram",
        "weighted_sum",
        "activation",
        "decision_boundary",
        "summary",
    ]
    assert segments[1].component_id == "formula_parts_focus"
    assert segments[1].formula_focus == r"w_1x_1+w_2x_2"
    assert segments[5].component_id == "perceptron_node"
    assert segments[8].component_id == "decision_boundary"
    assert segments[8].formula_focus == r"w_1x_1+w_2x_2+b=0"
    assert sum(segment.duration_seconds for segment in segments) == pytest.approx(95.0)
    assert "単純パーセプトロン" in script
    assert "決定境界" in script


def test_voiceover_script_writer_shortens_perceptron_timeline_for_faster_voice() -> None:
    artifacts = FormulaUnderstandingPlanner(no_llm=True).plan(
        formula=r"a = \mathrm{step}(w_1x_1 + w_2x_2 + b)",
        goal="単純パーセプトロンの順伝播と決定境界を直感的に理解したい",
        audience="high_school_math",
        concept_hint="perceptron",
    )
    assert artifacts.storyboard is not None

    writer = VoiceoverScriptWriter()
    segments = writer.write_segments(
        artifacts.storyboard,
        target_duration_seconds=95,
        voice_rate=130,
    )

    assert sum(segment.duration_seconds for segment in segments) == pytest.approx(
        95.0 * 120.0 / 130.0,
        abs=0.001,
    )


def test_voiceover_script_writer_segments_fully_connected_network() -> None:
    artifacts = FormulaUnderstandingPlanner(no_llm=True).plan(
        formula=r"\hat{y}=\mathrm{softmax}(W_2\sigma(W_1x+b_1)+b_2),\quad L=-\sum_i y_i\log(\hat{y}_i)",
        goal="全結合ニューラルネットワークの順伝播からクロスエントロピー損失までを直感的に理解したい",
        audience="high_school_math",
        concept_hint="fully_connected_network",
    )
    assert artifacts.storyboard is not None

    writer = VoiceoverScriptWriter()
    segments = writer.write_segments(artifacts.storyboard, target_duration_seconds=114)
    script = writer.write(artifacts.storyboard, target_duration_seconds=114)

    assert [segment.id for segment in segments] == [
        "title_intro",
        "formula_affine",
        "formula_activation",
        "formula_output",
        "layer_stack",
        "full_connections",
        "forward_pass",
        "softmax_output",
        "one_hot_label",
        "correct_probability",
        "cross_entropy_loss",
        "summary",
    ]
    assert segments[4].component_id == "dense_layer"
    assert segments[5].component_id == "fully_connected_edges"
    assert segments[7].component_id == "softmax_output"
    assert segments[8].component_id == "one_hot_vector"
    assert segments[9].component_id == "probability_selector"
    assert segments[10].component_id == "negative_log_curve"
    assert sum(segment.duration_seconds for segment in segments) == pytest.approx(114.0)
    assert "全結合ニューラルネットワーク" in script
    assert "softmax" in script
    assert "マイナスlog" in script
