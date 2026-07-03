from __future__ import annotations

import json
from typing import TypeVar

from pydantic import BaseModel, ValidationError

from math_animation_studio.config import Settings, load_settings


class LLMUnavailableError(RuntimeError):
    pass


SchemaModelT = TypeVar("SchemaModelT", bound=BaseModel)


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

    def complete_model(
        self,
        *,
        prompt: str,
        response_model: type[SchemaModelT],
        schema_name: str,
    ) -> SchemaModelT:
        raw = self._complete_json_schema(
            prompt=prompt,
            response_model=response_model,
            schema_name=schema_name,
        )
        try:
            return response_model.model_validate_json(raw)
        except ValidationError as exc:
            retry_prompt = _validation_retry_prompt(
                original_prompt=prompt,
                raw_response=raw,
                error_message=str(exc),
            )
            retry_raw = self._complete_json_schema(
                prompt=retry_prompt,
                response_model=response_model,
                schema_name=schema_name,
            )
            try:
                return response_model.model_validate_json(retry_raw)
            except ValidationError as retry_exc:
                raise LLMUnavailableError(
                    f"LLM output did not match {response_model.__name__}: {retry_exc}"
                ) from retry_exc

    def _complete_json_schema(
        self,
        *,
        prompt: str,
        response_model: type[BaseModel],
        schema_name: str,
    ) -> str:
        if not self.settings.openai_api_key:
            raise LLMUnavailableError(
                "OPENAI_API_KEY is not set. Use --no-llm for the bundled sample mode."
            )

        try:
            from openai import BadRequestError, OpenAI, OpenAIError
        except ImportError as exc:
            raise LLMUnavailableError(
                "openai package is not installed. Install dependencies or use --no-llm."
            ) from exc

        client = OpenAI(api_key=self.settings.openai_api_key)
        messages = [
            {
                "role": "system",
                "content": (
                    "You output JSON only. Do not include Markdown fences. "
                    "Never output Python code, eval, or exec."
                ),
            },
            {"role": "user", "content": prompt},
        ]
        try:
            response = client.chat.completions.create(
                model=self.settings.openai_model,
                messages=messages,
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "schema": response_model.model_json_schema(),
                        "strict": False,
                    },
                },
            )
        except BadRequestError:
            try:
                response = client.chat.completions.create(
                    model=self.settings.openai_model,
                    messages=messages
                    + [
                        {
                            "role": "user",
                            "content": (
                                "If the schema above is too complex for structured outputs, "
                                "still return one JSON object matching it."
                            ),
                        }
                    ],
                    response_format={"type": "json_object"},
                )
            except OpenAIError as exc:
                raise LLMUnavailableError(f"OpenAI API request failed: {exc}") from exc
        except OpenAIError as exc:
            raise LLMUnavailableError(f"OpenAI API request failed: {exc}") from exc

        content = response.choices[0].message.content
        if not content:
            raise LLMUnavailableError("LLM returned an empty response.")
        return _extract_json(content)


def _extract_json(content: str) -> str:
    text = content.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    try:
        json.loads(text)
    except json.JSONDecodeError as exc:
        raise LLMUnavailableError(f"LLM returned invalid JSON: {exc}") from exc
    return text


def _validation_retry_prompt(
    *,
    original_prompt: str,
    raw_response: str,
    error_message: str,
) -> str:
    return f"""{original_prompt}

前回のJSONはschema validationに失敗しました。

Validation error:
{error_message}

前回のJSON:
{raw_response}

上記のエラーを修正し、JSON schemaに合うJSONオブジェクトだけを再出力してください。
Markdown fences、説明文、Pythonコード、Manimコードは出力しないでください。
"""
