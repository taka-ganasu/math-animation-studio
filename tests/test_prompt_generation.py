from __future__ import annotations

from math_animation_studio.llm import build_concept_planner_prompt, build_retry_prompt


def test_prompt_contains_schema_and_constraints() -> None:
    prompt = build_concept_planner_prompt(
        concept="gradient_descent",
        formula=r"\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)",
        goal="直感的に理解する",
        audience="engineer_beginner_math",
    )

    assert "JSON schema" in prompt
    assert "Manimコードそのものは生成しない" in prompt
    assert "gradient_descent" in prompt


def test_retry_prompt_includes_previous_error() -> None:
    retry = build_retry_prompt("original", "{bad json", "parse error")

    assert "original" in retry
    assert "{bad json" in retry
    assert "parse error" in retry
