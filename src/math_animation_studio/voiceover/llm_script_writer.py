from __future__ import annotations

from math_animation_studio.llm import LLMClient, build_voiceover_script_prompt
from math_animation_studio.schema import Storyboard, VoiceoverScript


class LLMVoiceoverScriptWriter:
    def __init__(self, *, client: LLMClient | None = None) -> None:
        self.client = client or LLMClient()

    def write(
        self,
        *,
        storyboard: Storyboard,
        target_duration_seconds: int,
        audience: str,
        goal: str | None,
    ) -> str:
        prompt = build_voiceover_script_prompt(
            storyboard=storyboard,
            target_duration_seconds=target_duration_seconds,
            audience=audience,
            goal=goal,
        )
        result = self.client.complete_model(
            prompt=prompt,
            response_model=VoiceoverScript,
            schema_name="voiceover_script",
        )
        return result.script.strip()
