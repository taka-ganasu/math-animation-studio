from __future__ import annotations

from functools import lru_cache
from importlib.resources import files

import yaml

from math_animation_studio.schema import VisualComponentDefinition


@lru_cache
def load_visual_component_catalog() -> dict[str, VisualComponentDefinition]:
    path = files("math_animation_studio.knowledge").joinpath("visual_components.yaml")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    components = [
        VisualComponentDefinition.model_validate(item)
        for item in payload.get("components", [])
    ]
    return {component.id: component for component in components}


def visual_component_ids() -> set[str]:
    return set(load_visual_component_catalog())


def visual_component_definition(component_id: str) -> VisualComponentDefinition | None:
    return load_visual_component_catalog().get(component_id)


def visual_component_prompt_summary() -> str:
    lines: list[str] = []
    for component in load_visual_component_catalog().values():
        params = ", ".join(
            f"{param.name}{'*' if param.required else ''}" for param in component.params
        )
        params_text = params or "none"
        templates = ", ".join(component.template_support)
        lines.append(
            f"- {component.id}: {component.description} "
            f"(visual_type={component.visual_type}, params={params_text}, templates={templates})"
        )
    return "\n".join(lines)
