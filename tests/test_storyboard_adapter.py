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
            "scene_role": "formula_structure",
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

    assert storyboard.blueprint is not None
    assert storyboard.blueprint.flow_name == "formula_first"
    assert storyboard.scenes[0].scene_role == "formula_structure"
    assert storyboard.scenes[0].beat_id == "formula_structure"

    first_scene = storyboard.scenes[0]
    component_kinds = [component.kind for component in first_scene.components]
    visual_names = [visual.name for visual in first_scene.visual_objects]

    assert "formula_focus" in component_kinds
    assert "unknown_component" not in component_kinds
    assert first_scene.components[0].params["formula_focus"] == "y_i"
    assert any(name.endswith("formula_focus_01_visual") for name in visual_names)


def test_storyboard_adapter_normalizes_llm_surface_aliases_and_initial_position() -> None:
    formula = r"\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)"
    formula_analysis = sample_formula_analysis(formula, "gradient_descent")
    explanation_plan = sample_explanation_plan(
        formula,
        "gradient_descent",
        "high_school_math",
    )
    example = explanation_plan.recommended_examples[0].model_copy(
        update={
            "concrete_values": {
                "function_preset": "hill_surface",
                "initial_position": [2, -2],
                "learning_rate": 0.05,
                "steps": 6,
            }
        }
    )
    first_step = explanation_plan.explanation_steps[0].model_copy(
        update={
            "planned_components": [
                PlannedAnimationComponent(
                    kind="surface_plot",
                    description="LLMが自然名で指定した曲面",
                    params={"function_preset": "simple_quadratic"},
                )
            ]
        }
    )
    explanation_plan = explanation_plan.model_copy(
        update={
            "recommended_examples": [example],
            "explanation_steps": [first_step, *explanation_plan.explanation_steps[1:]],
        }
    )

    storyboard = StoryboardAdapter().convert(
        formula_analysis=formula_analysis,
        explanation_plan=explanation_plan,
    )

    first_scene = storyboard.scenes[0]
    second_scene = storyboard.scenes[1]

    assert first_scene.visual_objects[0].params["function_preset"] == "quadratic_ripple"
    assert first_scene.visual_objects[0].params["surface_y_shift"] == 2.7
    assert first_scene.visual_objects[0].params["surface_z_length"] == 2.4
    assert first_scene.visual_objects[0].params["surface_camera_zoom"] == 0.52
    assert first_scene.visual_objects[0].params["surface_camera_phi"] == 55.0
    assert first_scene.visual_objects[0].params["surface_camera_theta"] == -48.0
    assert first_scene.visual_objects[0].params["title_top_buff"] == 0.18
    assert first_scene.visual_objects[0].params["caption_bottom_buff"] == 0.32
    assert first_scene.components[1].params["function_preset"] == "quadratic_ripple"
    assert second_scene.visual_objects[0].params["x"] == 2.0
    assert second_scene.visual_objects[0].params["y"] == -2.0


def test_storyboard_adapter_enriches_gradient_metaphor_components() -> None:
    formula = r"\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)"
    formula_analysis = sample_formula_analysis(formula, "gradient_descent")
    explanation_plan = sample_explanation_plan(
        formula,
        "gradient_descent",
        "high_school_math",
    )

    storyboard = StoryboardAdapter().convert(
        formula_analysis=formula_analysis,
        explanation_plan=explanation_plan,
    )

    component_kinds = {
        component.kind
        for scene in storyboard.scenes
        for component in scene.components
    }

    assert "terrain_metaphor" in component_kinds
    assert "hiker_marker" in component_kinds
    assert "uphill_arrow" in component_kinds
    assert "downhill_arrow" in component_kinds
    assert "footstep_path" in component_kinds
    assert "formula_bridge" in component_kinds


def test_storyboard_adapter_converts_perceptron_plan() -> None:
    formula = r"a = \mathrm{step}(w_1x_1 + w_2x_2 + b)"
    formula_analysis = sample_formula_analysis(formula, "perceptron")
    explanation_plan = sample_explanation_plan(
        formula,
        "perceptron",
        "high_school_math",
    )

    storyboard = StoryboardAdapter().convert(
        formula_analysis=formula_analysis,
        explanation_plan=explanation_plan,
    )

    component_kinds = {
        component.kind
        for scene in storyboard.scenes
        for component in scene.components
    }

    assert storyboard.concept == "perceptron"
    assert storyboard.blueprint is not None
    assert storyboard.blueprint.flow_name == "formula_first"
    assert storyboard.scenes[0].scene_role == "formula_structure"
    assert storyboard.scenes[3].scene_role == "visualization"
    assert "perceptron_node" in component_kinds
    assert "weighted_sum" in component_kinds
    assert "activation_function" in component_kinds
    assert "decision_boundary" in component_kinds


def test_storyboard_adapter_converts_fully_connected_plan() -> None:
    formula = r"\hat{y}=\mathrm{softmax}(W_2\sigma(W_1x+b_1)+b_2)"
    formula_analysis = sample_formula_analysis(formula, "fully_connected_network")
    explanation_plan = sample_explanation_plan(
        formula,
        "fully_connected_network",
        "high_school_math",
    )

    storyboard = StoryboardAdapter().convert(
        formula_analysis=formula_analysis,
        explanation_plan=explanation_plan,
    )

    component_kinds = {
        component.kind
        for scene in storyboard.scenes
        for component in scene.components
    }

    assert storyboard.concept == "fully_connected_network"
    assert storyboard.blueprint is not None
    assert storyboard.blueprint.flow_name == "formula_first"
    assert "dense_layer" in component_kinds
    assert "fully_connected_edges" in component_kinds
    assert "layer_activation" in component_kinds
    assert "softmax_output" in component_kinds


def test_storyboard_adapter_converts_neural_network_transform_plan() -> None:
    formula = r"h=\sigma(Wx+b)"
    formula_analysis = sample_formula_analysis(formula, "neural_network_transform")
    explanation_plan = sample_explanation_plan(
        formula,
        "neural_network_transform",
        "high_school_math",
    )

    storyboard = StoryboardAdapter().convert(
        formula_analysis=formula_analysis,
        explanation_plan=explanation_plan,
    )

    component_kinds = {
        component.kind
        for scene in storyboard.scenes
        for component in scene.components
    }

    assert storyboard.concept == "neural_network_transform"
    assert storyboard.blueprint is not None
    assert storyboard.blueprint.flow_name == "formula_first"
    assert "feature_axis_mixing" in component_kinds
    assert "activation_gate" in component_kinds
    assert "representation_space" in component_kinds
    assert "decision_boundary" in component_kinds
