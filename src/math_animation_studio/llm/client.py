from __future__ import annotations

from math_animation_studio.config import Settings, load_settings


class LLMUnavailableError(RuntimeError):
    pass


class LLMClient:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or load_settings()

    def complete_json(self, prompt: str) -> str:
        if not self.settings.openai_api_key:
            raise LLMUnavailableError(
                "OPENAI_API_KEY is not set. Use --no-llm for the bundled sample mode."
            )

        try:
            from openai import OpenAI
        except ImportError as exc:
            raise LLMUnavailableError(
                "openai package is not installed. Install dependencies or use --no-llm."
            ) from exc

        client = OpenAI(api_key=self.settings.openai_api_key)
        response = client.chat.completions.create(
            model=self.settings.openai_model,
            messages=[
                {
                    "role": "system",
                    "content": "Return valid JSON only. Do not include Markdown fences.",
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content
        if not content:
            raise LLMUnavailableError("LLM returned an empty response.")
        return content
