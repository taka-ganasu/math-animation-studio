from __future__ import annotations

from math_animation_studio.llm import (
    build_concept_planner_prompt,
    build_formula_understanding_plan_prompt,
    build_retry_prompt,
    build_voiceover_script_prompt,
)
from math_animation_studio.understanding import FormulaUnderstandingPlanner


def test_prompt_contains_schema_and_constraints() -> None:
    prompt = build_concept_planner_prompt(
        concept="gradient_descent",
        formula=r"\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)",
        goal="直感的に理解する",
        audience="engineer_beginner_math",
    )

    assert "JSON schema" in prompt
    assert "Manimコードそのものは生成しない" in prompt
    assert "LLMの役割" in prompt
    assert "レンダラー" in prompt
    assert "gradient_descent" in prompt


def test_retry_prompt_includes_previous_error() -> None:
    retry = build_retry_prompt("original", "{bad json", "parse error")

    assert "original" in retry
    assert "{bad json" in retry
    assert "parse error" in retry


def test_formula_understanding_plan_prompt_contains_schema_and_patterns() -> None:
    prompt = build_formula_understanding_plan_prompt(
        formula=r"L = - \sum_i y_i \log(\hat{y}_i)",
        goal="噛み砕いて理解したい",
        audience="engineer_beginner_math",
        domain_hint="machine_learning",
        animation_pattern_ids=["penalty_curve", "generic_symbol_decomposition"],
        target_duration_seconds=60,
    )

    assert "FormulaUnderstandingLLMPlan" in prompt or "formula_analysis" in prompt
    assert "penalty_curve" in prompt
    assert "60秒" in prompt
    assert "Pythonコード" in prompt
    assert "generation_boundary.code_generation_allowedは必ずfalse" in prompt
    assert "既存preset名" in prompt
    assert "recommended_examplesは2〜3個" in prompt


def test_voiceover_script_prompt_contains_storyboard_and_constraints() -> None:
    artifacts = FormulaUnderstandingPlanner(no_llm=True).plan(
        formula=r"L = - \sum_i y_i \log(\hat{y}_i)",
        goal="クロスエントロピー損失を直感的に理解したい",
        audience="engineer_beginner_math",
    )
    assert artifacts.storyboard is not None

    prompt = build_voiceover_script_prompt(
        storyboard=artifacts.storyboard,
        target_duration_seconds=60,
        audience="engineer_beginner_math",
        goal="噛み砕いて理解したい",
    )

    assert "VoiceoverScript" in prompt or '"script"' in prompt
    assert "60秒" in prompt
    assert "難しい語は短く噛み砕く" in prompt
