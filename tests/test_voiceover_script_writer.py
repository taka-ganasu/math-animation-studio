from __future__ import annotations

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
    markdown = writer.write_markdown(artifacts.storyboard)

    assert "クロスエントロピー損失" in script
    assert "マイナスログ" in script
    assert len(script) < 170
    assert "## Voiceover Script" in markdown
    assert "## Source Scenes" in markdown
