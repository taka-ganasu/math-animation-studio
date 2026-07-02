from __future__ import annotations

from importlib.resources import files

from pydantic import ValidationError

from math_animation_studio.llm import LLMClient, build_concept_planner_prompt, build_retry_prompt
from math_animation_studio.schema import Storyboard


class PlannerError(RuntimeError):
    pass


class ConceptPlanner:
    def __init__(
        self,
        *,
        no_llm: bool = False,
        llm_client: LLMClient | None = None,
        max_retries: int = 1,
    ) -> None:
        self.no_llm = no_llm
        self.llm_client = llm_client or LLMClient()
        self.max_retries = max_retries

    def plan(
        self,
        *,
        concept: str,
        formula: str | None,
        goal: str | None,
        audience: str,
    ) -> Storyboard:
        if self.no_llm:
            return self._sample_storyboard(
                concept=concept,
                formula=formula,
                audience=audience,
            )

        prompt = build_concept_planner_prompt(
            concept=concept,
            formula=formula,
            goal=goal,
            audience=audience,
        )
        raw_response = ""
        last_error = ""
        for attempt in range(self.max_retries + 1):
            active_prompt = (
                prompt
                if attempt == 0
                else build_retry_prompt(prompt, raw_response, last_error)
            )
            raw_response = self.llm_client.complete_json(active_prompt)
            try:
                return Storyboard.model_validate_json(raw_response)
            except ValidationError as exc:
                last_error = str(exc)

        raise PlannerError(
            f"LLM response could not be parsed as Storyboard JSON after "
            f"{self.max_retries + 1} attempts: {last_error}"
        )

    def _sample_storyboard(
        self,
        *,
        concept: str,
        formula: str | None,
        audience: str,
    ) -> Storyboard:
        normalized = concept.strip().lower().replace("-", "_")
        if normalized not in {"gradient_descent", "勾配降下法"}:
            raise PlannerError(
                "--no-llm currently supports only concept=gradient_descent."
            )

        sample_path = files("math_animation_studio.samples").joinpath(
            "gradient_descent_storyboard.json"
        )
        storyboard = Storyboard.model_validate_json(
            sample_path.read_text(encoding="utf-8")
        )
        if formula:
            storyboard.formula = formula
        storyboard.audience = audience
        return storyboard
