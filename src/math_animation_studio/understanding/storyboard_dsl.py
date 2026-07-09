from __future__ import annotations

from collections.abc import Iterable, Sequence

from math_animation_studio.schema import (
    ExplanationPlan,
    ExplanationStep,
    SceneRole,
    StoryboardBeat,
    StoryboardBlueprint,
)

from .visual_component_catalog import visual_component_ids


FORMULA_FIRST_FLOW_NAME = "formula_first"
FORMULA_FIRST_ROLE_ORDER: tuple[SceneRole, ...] = (
    "title_intro",
    "formula_structure",
    "concrete_example",
    "visualization",
    "summary",
)

ROLE_INTENTS: dict[SceneRole, str] = {
    "title_intro": "今回扱う概念と、何を見ればよいかを短く示す。",
    "formula_structure": "数式を意味のあるパーツに分け、記号と操作の役割を視線誘導する。",
    "concrete_example": "身近な値や題材に置き換え、抽象的な式が何を受け取るかを見せる。",
    "visualization": "グラフ、曲面、棒、矢印、軌跡などで、実際にどう使われるかを動かす。",
    "summary": "最後に式へ戻り、直感と数式の対応を短く回収する。",
}

ROLE_DEFAULT_COMPONENTS: dict[SceneRole, tuple[str, ...]] = {
    "title_intro": ("text_caption", "formula_focus"),
    "formula_structure": ("formula_focus", "text_caption"),
    "concrete_example": ("model_pipeline", "one_hot_vector", "probability_bars", "text_caption"),
    "visualization": ("surface_plot", "loss_curve", "contour_map", "descent_path", "negative_log_curve"),
    "summary": ("summary", "formula_bridge", "text_caption"),
}

PATTERN_COMPONENTS: dict[str, dict[SceneRole, tuple[str, ...]]] = {
    "penalty_curve": {
        "concrete_example": (
            "model_pipeline",
            "one_hot_vector",
            "probability_bars",
            "probability_selector",
        ),
        "visualization": ("negative_log_curve", "probability_selector", "probability_bars"),
        "summary": ("summary", "formula_focus"),
    },
    "trajectory_on_surface": {
        "concrete_example": ("terrain_metaphor", "hiker_marker", "text_caption"),
        "visualization": (
            "surface_plot",
            "contour_map",
            "loss_curve",
            "gradient_arrow",
            "uphill_arrow",
            "downhill_arrow",
            "descent_path",
            "footstep_path",
        ),
        "summary": ("formula_bridge", "summary"),
    },
    "perceptron_decision_boundary": {
        "formula_structure": ("formula_focus", "weighted_sum", "activation_function"),
        "concrete_example": ("perceptron_node", "weighted_connection", "forward_pass"),
        "visualization": ("decision_boundary", "activation_function", "forward_pass"),
        "summary": ("summary", "formula_focus"),
    },
    "fully_connected_forward_pass": {
        "formula_structure": ("formula_focus", "weighted_sum", "layer_activation"),
        "concrete_example": ("dense_layer", "fully_connected_edges", "forward_pass"),
        "visualization": ("dense_layer", "fully_connected_edges", "layer_activation", "softmax_output"),
        "summary": ("summary", "formula_focus"),
    },
    "backpropagation_chain_rule": {
        "formula_structure": ("formula_focus", "loss_gradient", "backward_pass", "chain_rule"),
        "concrete_example": ("dense_layer", "fully_connected_edges", "forward_pass", "loss_gradient"),
        "visualization": ("backward_pass", "error_attribution", "weight_update", "chain_rule"),
        "summary": ("summary", "formula_focus"),
    },
    "chain_rule_flow": {
        "formula_structure": ("formula_focus", "chain_rule", "text_caption"),
        "concrete_example": ("chain_rule", "formula_focus", "text_caption"),
        "visualization": ("chain_rule", "backward_pass", "formula_focus"),
        "summary": ("summary", "formula_focus"),
    },
}


def default_formula_first_blueprint(
    *,
    target_concept: str = "target_formula",
    animation_pattern_id: str = "generic_symbol_decomposition",
) -> StoryboardBlueprint:
    beats = [
        StoryboardBeat(
            id=role,
            role=role,
            title=_role_title(role, target_concept),
            intent=ROLE_INTENTS[role],
            preferred_component_kinds=_preferred_components(role, animation_pattern_id),
            duration_weight=_duration_weight(role),
        )
        for role in FORMULA_FIRST_ROLE_ORDER
    ]
    return StoryboardBlueprint(
        version="storyboard_dsl_v1",
        flow_name=FORMULA_FIRST_FLOW_NAME,
        beats=beats,
    )


def formula_first_blueprint(explanation_plan: ExplanationPlan) -> StoryboardBlueprint:
    return default_formula_first_blueprint(
        target_concept=explanation_plan.target_concept,
        animation_pattern_id=explanation_plan.selected_animation_pattern_id,
    )


def infer_scene_role(
    step: ExplanationStep,
    *,
    step_index: int,
    step_count: int,
    animation_pattern_id: str,
) -> SceneRole:
    if step.scene_role is not None:
        return step.scene_role

    text = " ".join(
        part
        for part in [
            step.title,
            step.learning_goal,
            step.explanation,
            step.visual_idea,
            step.formula_focus or "",
            step.common_misunderstanding_addressed or "",
        ]
        if part
    ).lower()

    if step_index == step_count or _contains_any(text, ("まとめ", "最後", "式へ戻る", "式に戻る", "回収")):
        return "summary"
    if _contains_any(text, ("具体", "例", "正解クラス", "one-hot", "予測確率", "入力", "モデル")):
        return "concrete_example"
    if _contains_any(text, ("式を分解", "記号", "項", "演算", "formula", "symbol")):
        return "formula_structure"
    if (
        step_index <= 2
        and step.formula_focus
        and animation_pattern_id != "trajectory_on_surface"
    ):
        return "formula_structure"
    return "visualization"


def storyboard_dsl_prompt_summary(
    *,
    animation_pattern_ids: Sequence[str] | None = None,
) -> str:
    pattern_ids = tuple(animation_pattern_ids or ())
    rows = [
        "Storyboard DSL v1: formula_first",
        "Scene roles must be one of: "
        + ", ".join(FORMULA_FIRST_ROLE_ORDER),
        "Recommended flow: title_intro -> formula_structure -> concrete_example -> visualization -> summary.",
        "title_intro may be supplied by the renderer when the template already has a title sequence.",
        "",
        "Roles:",
    ]
    for role in FORMULA_FIRST_ROLE_ORDER:
        components = _component_summary(role, pattern_ids)
        rows.append(f"- {role}: {ROLE_INTENTS[role]} preferred_components={components}")
    return "\n".join(rows)


def blueprint_unknown_component_ids(blueprint: StoryboardBlueprint) -> set[str]:
    known = visual_component_ids()
    unknown: set[str] = set()
    for beat in blueprint.beats:
        unknown.update(
            component_id
            for component_id in beat.preferred_component_kinds
            if component_id not in known
        )
    return unknown


def _preferred_components(role: SceneRole, animation_pattern_id: str) -> list[str]:
    pattern_components = PATTERN_COMPONENTS.get(animation_pattern_id, {}).get(role, ())
    if pattern_components:
        merged = [*pattern_components, "text_caption"]
    else:
        merged = list(ROLE_DEFAULT_COMPONENTS[role])
    known = visual_component_ids()
    return _dedupe(component for component in merged if component in known)


def _component_summary(role: SceneRole, pattern_ids: Sequence[str]) -> str:
    components = list(ROLE_DEFAULT_COMPONENTS[role])
    for pattern_id in pattern_ids:
        components.extend(PATTERN_COMPONENTS.get(pattern_id, {}).get(role, ()))
    known = visual_component_ids()
    return ", ".join(_dedupe(component for component in components if component in known))


def _role_title(role: SceneRole, target_concept: str) -> str:
    titles: dict[SceneRole, str] = {
        "title_intro": f"{target_concept}を見る",
        "formula_structure": "式の構造を分解する",
        "concrete_example": "具体例に置き換える",
        "visualization": "動きとして可視化する",
        "summary": "式へ戻ってまとめる",
    }
    return titles[role]


def _duration_weight(role: SceneRole) -> float:
    weights: dict[SceneRole, float] = {
        "title_intro": 0.8,
        "formula_structure": 1.4,
        "concrete_example": 1.2,
        "visualization": 2.0,
        "summary": 0.9,
    }
    return weights[role]


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle.lower() in text for needle in needles)


def _dedupe(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result
