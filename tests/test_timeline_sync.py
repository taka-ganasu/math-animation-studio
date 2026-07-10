from __future__ import annotations

import pytest

from math_animation_studio.generator import ManimGenerator
from math_animation_studio.understanding import FormulaUnderstandingPlanner
from math_animation_studio.voiceover import VoiceoverScriptWriter


@pytest.mark.parametrize(
    ("formula", "goal", "concept_hint", "duration"),
    [
        (
            r"L = - \sum_i y_i \log(\hat{y}_i)",
            "クロスエントロピー損失を直感的に理解したい",
            "cross_entropy",
            60,
        ),
        (
            r"a = \mathrm{step}(w_1x_1 + w_2x_2 + b)",
            "単純パーセプトロンの順伝播と決定境界を直感的に理解したい",
            "perceptron",
            95,
        ),
        (
            r"\hat{y}=\mathrm{softmax}(W_2\sigma(W_1x+b_1)+b_2)",
            "全結合ニューラルネットワークの順伝播を直感的に理解したい",
            "fully_connected_network",
            114,
        ),
        (
            r"\delta^{(l)}=(W^{(l+1)T}\delta^{(l+1)})\odot\sigma'(z^{(l)})",
            "バックプロパゲーションで誤差信号がどう戻るか理解したい",
            "backpropagation",
            151,
        ),
        (
            r"\frac{dy}{dx}=\frac{dy}{du}\frac{du}{dx}",
            "連鎖律を変化率のつながりとして理解したい",
            "chain_rule",
            88,
        ),
        (
            r"h=\sigma(Wx+b)",
            "ニューラルネットワークにおける変換の意味を理解したい",
            "neural_network_transform",
            100,
        ),
        (
            r"a=f(z),\quad p=\mathrm{softmax}(o)",
            "ReLU、sigmoid、tanh、softmaxの違いを理解したい",
            "activation_functions",
            130,
        ),
    ],
)
def test_render_segments_and_voiceover_segments_are_aligned(
    tmp_path,
    formula: str,
    goal: str,
    concept_hint: str,
    duration: int,
) -> None:
    artifacts = FormulaUnderstandingPlanner(no_llm=True).plan(
        formula=formula,
        goal=goal,
        audience="high_school_math",
        concept_hint=concept_hint,
    )
    assert artifacts.storyboard is not None

    params = ManimGenerator(target_duration_seconds=duration).generate(
        artifacts.storyboard,
        tmp_path / f"{concept_hint}_scene.py",
    )
    voiceover_segments = VoiceoverScriptWriter().write_segments(
        artifacts.storyboard,
        target_duration_seconds=duration,
    )

    render_ids = list(params.segment_durations)
    voiceover_ids = [segment.id for segment in voiceover_segments]

    assert render_ids == voiceover_ids
    assert sum(params.segment_durations.values()) == pytest.approx(duration)
    assert sum(segment.duration_seconds for segment in voiceover_segments) == pytest.approx(
        duration
    )
