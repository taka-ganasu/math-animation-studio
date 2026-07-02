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
    assert sum(segment.duration_seconds for segment in segments) == pytest.approx(30.0)
    assert "Target duration: 30 seconds" in markdown
    assert "## Voiceover Segments" in markdown
    assert "### focus_log" in markdown
    assert "## Voiceover Script" in markdown
    assert "## Source Scenes" in markdown
