from __future__ import annotations

from typing import Any

from math_animation_studio.schema import VoiceoverScript
from math_animation_studio.understanding import FormulaUnderstandingPlanner
from math_animation_studio.voiceover import LLMVoiceoverScriptWriter


class FakeLLMClient:
    def __init__(self) -> None:
        self.last_prompt = ""
        self.last_schema_name = ""

    def complete_model(self, **kwargs: Any) -> VoiceoverScript:
        self.last_prompt = kwargs["prompt"]
        self.last_schema_name = kwargs["schema_name"]
        return VoiceoverScript(
            script="まず猫が正解だと見ます。猫の確率が低いと、罰が大きくなります。"
        )


def test_llm_voiceover_script_writer_returns_script() -> None:
    artifacts = FormulaUnderstandingPlanner(no_llm=True).plan(
        formula=r"L = - \sum_i y_i \log(\hat{y}_i)",
        goal="クロスエントロピー損失を直感的に理解したい",
        audience="engineer_beginner_math",
    )
    assert artifacts.storyboard is not None
    client = FakeLLMClient()
    writer = LLMVoiceoverScriptWriter(client=client)  # type: ignore[arg-type]

    script = writer.write(
        storyboard=artifacts.storyboard,
        target_duration_seconds=60,
        audience="engineer_beginner_math",
        goal="直感的に理解したい",
    )

    assert "猫の確率" in script
    assert "60秒" in client.last_prompt
    assert client.last_schema_name == "voiceover_script"
