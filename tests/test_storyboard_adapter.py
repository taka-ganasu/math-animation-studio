from __future__ import annotations

from math_animation_studio.schema import PlannedAnimationComponent
from math_animation_studio.understanding.sample_plans import (
    sample_explanation_plan,
    sample_formula_analysis,
)
from math_animation_studio.understanding.storyboard_adapter import StoryboardAdapter


def test_storyboard_adapter_uses_planned_components_from_explanation_step() -> None:
    formula = r"L = - \sum_i y_i \log(\hat{y}_i)"
    formula_analysis = sample_formula_analysis(formula, "cross_entropy")
    explanation_plan = sample_explanation_plan(
        formula,
        "cross_entropy",
        "high_school_math",
    )
    first_step = explanation_plan.explanation_steps[0].model_copy(
        update={
            "planned_components": [
                PlannedAnimationComponent(
                    kind="formula_focus",
                    description="y_iを拡大して正解スイッチとして見せる",
                    label="y_i",
                    params={"formula_focus": "y_i", "emphasis": "zoom"},
                ),
                PlannedAnimationComponent(
                    kind="unknown_component",
                    description="未定義部品は採用しない",
                ),
            ]
        }
    )
    explanation_plan = explanation_plan.model_copy(
        update={"explanation_steps": [first_step, *explanation_plan.explanation_steps[1:]]}
    )

    storyboard = StoryboardAdapter().convert(
        formula_analysis=formula_analysis,
        explanation_plan=explanation_plan,
    )

    first_scene = storyboard.scenes[0]
    component_kinds = [component.kind for component in first_scene.components]
    visual_names = [visual.name for visual in first_scene.visual_objects]

    assert "formula_focus" in component_kinds
    assert "unknown_component" not in component_kinds
    assert first_scene.components[0].params["formula_focus"] == "y_i"
    assert any(name.endswith("formula_focus_01_visual") for name in visual_names)
