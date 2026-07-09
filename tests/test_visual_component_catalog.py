from __future__ import annotations

from math_animation_studio.understanding.visual_component_catalog import (
    load_visual_component_catalog,
    visual_component_prompt_summary,
)


def test_visual_component_catalog_loads_expected_components() -> None:
    catalog = load_visual_component_catalog()

    assert "formula_focus" in catalog
    assert "probability_bars" in catalog
    assert "gradient_arrow" in catalog
    assert "perceptron_node" in catalog
    assert "dense_layer" in catalog
    assert "fully_connected_edges" in catalog
    assert "feature_axis_mixing" in catalog
    assert "activation_gate" in catalog
    assert "representation_space" in catalog
    assert "softmax_output" in catalog
    assert "decision_boundary" in catalog
    assert "backward_pass" in catalog
    assert "weight_update" in catalog
    assert catalog["formula_focus"].params[0].name == "formula_focus"
    assert catalog["gradient_arrow"].visual_type == "vector"
    assert catalog["perceptron_node"].category == "neural_network"
    assert catalog["dense_layer"].category == "neural_network"
    assert catalog["decision_boundary"].visual_type == "axis"
    assert catalog["feature_axis_mixing"].visual_type == "axis"
    assert catalog["activation_gate"].visual_type == "curve"
    assert catalog["representation_space"].category == "neural_network"
    assert catalog["backward_pass"].visual_type == "vector"
    assert catalog["weight_update"].category == "optimization"
    assert catalog["terrain_metaphor"].category == "metaphor"
    assert catalog["formula_bridge"].visual_type == "formula"
    assert {param.name for param in catalog["surface_plot"].params} >= {
        "surface_y_shift",
        "surface_z_length",
        "surface_camera_zoom",
        "surface_camera_phi",
        "surface_camera_theta",
        "title_top_buff",
        "caption_bottom_buff",
    }


def test_visual_component_prompt_summary_is_llm_readable() -> None:
    summary = visual_component_prompt_summary()

    assert "- formula_focus:" in summary
    assert "visual_type=formula" in summary
    assert "category=metaphor" in summary
    assert "- terrain_metaphor:" in summary
    assert "templates=penalty_curve" in summary
    assert "gradient_descent_3d" in summary
    assert "function_preset=['quadratic_ripple', 'double_well_2d']" in summary
    assert "- perceptron_node:" in summary
    assert "templates=perceptron" in summary
    assert "- dense_layer:" in summary
    assert "templates=fully_connected_network" in summary
    assert "- feature_axis_mixing:" in summary
    assert "templates=neural_network_transform" in summary
    assert "- backward_pass:" in summary
    assert "templates=backpropagation" in summary
