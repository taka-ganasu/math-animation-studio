from __future__ import annotations

from math_animation_studio.llm import (
    build_concept_planner_prompt,
    build_formula_plan_consistency_prompt,
    build_formula_understanding_plan_prompt,
    build_formula_understanding_revision_prompt,
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


def test_formula_understanding_plan_prompt_contains_concept_hint() -> None:
    prompt = build_formula_understanding_plan_prompt(
        formula=r"L = - \sum_i y_i \log(\hat{y}_i)",
        goal="2変数を変更した場合の勾配降下法の動きを理解したい",
        audience="high_school_math",
        domain_hint="machine_learning",
        animation_pattern_ids=["penalty_curve", "trajectory_on_surface"],
        target_duration_seconds=60,
        concept_hint="gradient_descent",
    )

    assert "優先したい概念" in prompt
    assert "gradient_descent" in prompt
    assert "数式名よりも理解ゴールと優先概念を重視する" in prompt
    assert "損失地形を下る更新" in prompt


def test_formula_understanding_plan_prompt_contains_visual_component_catalog() -> None:
    prompt = build_formula_understanding_plan_prompt(
        formula=r"L = - \sum_i y_i \log(\hat{y}_i)",
        goal="数式パーツごとに意味を理解したい",
        audience="high_school_math",
        domain_hint="machine_learning",
        animation_pattern_ids=["penalty_curve"],
        target_duration_seconds=60,
        visual_component_catalog=(
            "- formula_focus: 数式の一部を強調する (category=formula)\n"
            "- terrain_metaphor: 損失を地形として見せる (category=metaphor)"
        ),
    )

    assert "利用可能な視覚部品カタログ" in prompt
    assert "planned_components" in prompt
    assert "formula_focus" in prompt
    assert "視覚部品カタログにあるidだけ" in prompt
    assert "category=metaphor" in prompt
    assert "terrain_metaphor, hiker_marker, uphill_arrow" in prompt
    assert "surface_y_shift, surface_camera_zoom, title_top_buff, caption_bottom_buff" in prompt


def test_formula_plan_consistency_prompt_reviews_goal_alignment() -> None:
    prompt = build_formula_plan_consistency_prompt(
        formula=r"L = - \sum_i y_i \log(\hat{y}_i)",
        goal="勾配降下法の動きを幾何的に理解したい",
        audience="high_school_math",
        domain_hint="machine_learning",
        concept_hint="gradient_descent",
        animation_pattern_ids=["penalty_curve", "trajectory_on_surface"],
        plan_json='{"concept_classification":{"primary_concept":"cross_entropy"}}',
        visual_component_catalog="- gradient_arrow: 勾配方向を矢印で示す",
    )

    assert "一貫しているか" in prompt
    assert "数式名だけに引っ張られず" in prompt
    assert "goalと優先概念を主題" in prompt
    assert "planned_components" in prompt
    assert "地形、現在地、上り勾配、下り更新、足跡、式への橋渡し" in prompt
    assert "gradient_arrow" in prompt
    assert "FormulaPlanConsistencyReview" in prompt or "is_consistent" in prompt


def test_formula_understanding_revision_prompt_contains_review() -> None:
    prompt = build_formula_understanding_revision_prompt(
        original_prompt="original",
        first_plan_json='{"target_concept":"cross_entropy"}',
        review_json='{"revision_instructions":["gradient_descentを主題にする"]}',
    )

    assert "original" in prompt
    assert "一貫性レビュー" in prompt
    assert "gradient_descentを主題にする" in prompt


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
