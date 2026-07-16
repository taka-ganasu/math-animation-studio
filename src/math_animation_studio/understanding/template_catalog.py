from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import yaml

from math_animation_studio.schema import TemplateDefinition
from math_animation_studio.timing import (
    ACTIVATION_FUNCTIONS_BASE_TIMELINE,
    BACKPROPAGATION_BASE_TIMELINE,
    CHAIN_RULE_BASE_TIMELINE,
    CROSS_ENTROPY_BASE_TIMELINE,
    FULLY_CONNECTED_BASE_TIMELINE,
    GRADIENT_SURFACE_3D_BASE_TIMELINE,
    NEURAL_NETWORK_TRANSFORM_BASE_TIMELINE,
    PERCEPTRON_BASE_TIMELINE,
    TimelineSegment,
)


TIMELINE_BY_TEMPLATE_ID: dict[str, tuple[TimelineSegment, ...]] = {
    "penalty_curve": CROSS_ENTROPY_BASE_TIMELINE,
    "gradient_descent_3d": GRADIENT_SURFACE_3D_BASE_TIMELINE,
    "perceptron": PERCEPTRON_BASE_TIMELINE,
    "fully_connected_network": FULLY_CONNECTED_BASE_TIMELINE,
    "backpropagation": BACKPROPAGATION_BASE_TIMELINE,
    "chain_rule": CHAIN_RULE_BASE_TIMELINE,
    "neural_network_transform": NEURAL_NETWORK_TRANSFORM_BASE_TIMELINE,
    "activation_functions": ACTIVATION_FUNCTIONS_BASE_TIMELINE,
}


@lru_cache
def load_template_catalog() -> dict[str, TemplateDefinition]:
    path = files("math_animation_studio.knowledge").joinpath("template_chapters.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    templates = [
        TemplateDefinition.model_validate(item)
        for item in payload.get("templates", [])
    ]
    return {template.id: template for template in templates}


def template_prompt_summary() -> str:
    lines: list[str] = []
    for template in load_template_catalog().values():
        lines.append(
            f"- {template.id}: {template.description} "
            f"(concept={template.concept}, manim_template={template.manim_template})"
        )
        for chapter in template.chapters:
            segments = ", ".join(chapter.segment_ids)
            components = ", ".join(chapter.component_kinds)
            lines.append(
                f"  - {chapter.id} (role={chapter.role}): {chapter.title}; "
                f"intent={chapter.intent}; segments={segments}; components={components}"
            )
    return "\n".join(lines)


def template_unknown_segment_ids(template: TemplateDefinition) -> set[str]:
    timeline = TIMELINE_BY_TEMPLATE_ID.get(template.id)
    if timeline is None:
        return {
            segment_id
            for chapter in template.chapters
            for segment_id in chapter.segment_ids
        }

    known = {segment.id for segment in timeline}
    return {
        segment_id
        for chapter in template.chapters
        for segment_id in chapter.segment_ids
        if segment_id not in known
    }


def template_timeline_segment_ids(template_id: str) -> set[str]:
    timeline = TIMELINE_BY_TEMPLATE_ID.get(template_id, ())
    return {segment.id for segment in timeline}
