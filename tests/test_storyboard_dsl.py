from __future__ import annotations

from math_animation_studio.schema import ExplanationStep
from math_animation_studio.understanding.storyboard_dsl import (
    FORMULA_FIRST_ROLE_ORDER,
    blueprint_unknown_component_ids,
    default_formula_first_blueprint,
    infer_scene_role,
    storyboard_dsl_prompt_summary,
)


def test_formula_first_blueprint_uses_known_visual_components() -> None:
    blueprint = default_formula_first_blueprint(
        target_concept="gradient_descent",
        animation_pattern_id="trajectory_on_surface",
    )

    assert blueprint.version == "storyboard_dsl_v1"
    assert blueprint.flow_name == "formula_first"
    assert [beat.role for beat in blueprint.beats] == list(FORMULA_FIRST_ROLE_ORDER)
    assert blueprint_unknown_component_ids(blueprint) == set()
    assert "surface_plot" in blueprint.beats[3].preferred_component_kinds
    assert "formula_bridge" in blueprint.beats[-1].preferred_component_kinds


def test_infer_scene_role_respects_explicit_llm_role() -> None:
    step = ExplanationStep(
        id="step_01",
        scene_role="formula_structure",
        title="式を分解する",
        learning_goal="記号の意味を知る",
        explanation="thetaの意味を見る。",
        visual_idea="式の一部を強調する。",
        formula_focus=r"\theta_t",
    )

    assert (
        infer_scene_role(
            step,
            step_index=1,
            step_count=3,
            animation_pattern_id="trajectory_on_surface",
        )
        == "formula_structure"
    )


def test_storyboard_dsl_prompt_summary_is_llm_readable() -> None:
    summary = storyboard_dsl_prompt_summary(
        animation_pattern_ids=["penalty_curve", "trajectory_on_surface"],
    )

    assert "Storyboard DSL v1" in summary
    assert "title_intro -> formula_structure" in summary
    assert "formula_focus" in summary
    assert "surface_plot" in summary
    assert "negative_log_curve" in summary


def test_formula_first_blueprint_supports_perceptron_components() -> None:
    blueprint = default_formula_first_blueprint(
        target_concept="perceptron",
        animation_pattern_id="perceptron_decision_boundary",
    )

    by_role = {beat.role: beat for beat in blueprint.beats}

    assert "weighted_sum" in by_role["formula_structure"].preferred_component_kinds
    assert "perceptron_node" in by_role["concrete_example"].preferred_component_kinds
    assert "decision_boundary" in by_role["visualization"].preferred_component_kinds


def test_formula_first_blueprint_supports_fully_connected_components() -> None:
    blueprint = default_formula_first_blueprint(
        target_concept="fully_connected_network",
        animation_pattern_id="fully_connected_forward_pass",
    )

    by_role = {beat.role: beat for beat in blueprint.beats}

    assert "layer_activation" in by_role["formula_structure"].preferred_component_kinds
    assert "dense_layer" in by_role["concrete_example"].preferred_component_kinds
    assert "fully_connected_edges" in by_role["visualization"].preferred_component_kinds
    assert "softmax_output" in by_role["visualization"].preferred_component_kinds


def test_formula_first_blueprint_supports_backpropagation_components() -> None:
    blueprint = default_formula_first_blueprint(
        target_concept="backpropagation",
        animation_pattern_id="backpropagation_chain_rule",
    )

    by_role = {beat.role: beat for beat in blueprint.beats}

    assert "loss_gradient" in by_role["formula_structure"].preferred_component_kinds
    assert "backward_pass" in by_role["visualization"].preferred_component_kinds
    assert "weight_update" in by_role["visualization"].preferred_component_kinds


def test_formula_first_blueprint_supports_neural_network_transform_components() -> None:
    blueprint = default_formula_first_blueprint(
        target_concept="neural_network_transform",
        animation_pattern_id="neural_network_transform_flow",
    )

    by_role = {beat.role: beat for beat in blueprint.beats}

    assert "feature_axis_mixing" in by_role["formula_structure"].preferred_component_kinds
    assert "activation_gate" in by_role["formula_structure"].preferred_component_kinds
    assert "representation_space" in by_role["concrete_example"].preferred_component_kinds
    assert "decision_boundary" in by_role["visualization"].preferred_component_kinds


def test_formula_first_blueprint_supports_activation_functions_components() -> None:
    blueprint = default_formula_first_blueprint(
        target_concept="activation_functions",
        animation_pattern_id="activation_function_comparison",
    )

    by_role = {beat.role: beat for beat in blueprint.beats}

    assert "activation_comparison" in by_role["formula_structure"].preferred_component_kinds
    assert "activation_curve" in by_role["visualization"].preferred_component_kinds
    assert "softmax_probability_flow" in by_role["visualization"].preferred_component_kinds
    assert "summary" in by_role["summary"].preferred_component_kinds


def test_formula_first_blueprint_supports_chain_rule_components() -> None:
    blueprint = default_formula_first_blueprint(
        target_concept="chain_rule",
        animation_pattern_id="chain_rule_flow",
    )

    by_role = {beat.role: beat for beat in blueprint.beats}

    assert "chain_rule" in by_role["formula_structure"].preferred_component_kinds
    assert "chain_rule" in by_role["concrete_example"].preferred_component_kinds
    assert "backward_pass" in by_role["visualization"].preferred_component_kinds
