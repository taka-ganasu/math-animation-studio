from __future__ import annotations

from math_animation_studio.schema import SceneRole
from math_animation_studio.understanding.template_catalog import (
    load_template_catalog,
    template_prompt_summary,
    template_timeline_segment_ids,
    template_unknown_segment_ids,
)
from math_animation_studio.understanding.visual_component_catalog import (
    visual_component_ids,
)


def test_template_catalog_loads_supported_templates() -> None:
    catalog = load_template_catalog()

    assert "penalty_curve" in catalog
    assert "gradient_descent_3d" in catalog
    assert "perceptron" in catalog
    assert "fully_connected_network" in catalog
    assert "backpropagation" in catalog
    assert "chain_rule" in catalog
    assert "neural_network_transform" in catalog
    assert "activation_functions" in catalog
    assert catalog["perceptron"].chapters[1].role == "formula_structure"
    assert catalog["backpropagation"].manim_template == "backpropagation.py.j2"


def test_template_catalog_segments_match_timelines() -> None:
    for template in load_template_catalog().values():
        assert template_unknown_segment_ids(template) == set()
        covered = {
            segment_id
            for chapter in template.chapters
            for segment_id in chapter.segment_ids
        }
        assert covered == template_timeline_segment_ids(template.id)


def test_template_catalog_uses_known_storyboard_roles_and_components() -> None:
    known_components = visual_component_ids()
    roles = set(SceneRole.__args__)

    for template in load_template_catalog().values():
        assert {chapter.role for chapter in template.chapters} <= roles
        for chapter in template.chapters:
            assert set(chapter.component_kinds) <= known_components


def test_template_prompt_summary_is_llm_readable() -> None:
    summary = template_prompt_summary()

    assert "- perceptron:" in summary
    assert "formula_structure (role=formula_structure)" in summary
    assert "segments=formula_weighted_inputs, formula_bias" in summary
    assert "- backpropagation:" in summary
    assert "components=backward_pass, error_attribution, weight_update" in summary
    assert "- activation_functions:" in summary
