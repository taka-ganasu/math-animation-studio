from __future__ import annotations

from typing import get_args

from math_animation_studio.schema import (
    AnimationComponent,
    Example,
    ExplanationPlan,
    ExplanationStep,
    FormulaAnalysis,
    NarrationCue,
    SceneSpec,
    Storyboard,
    SymbolDefinition,
    VisualObject,
)
from math_animation_studio.safe_presets import normalize_loss_surface_preset

from .visual_component_catalog import (
    visual_component_definition,
    visual_component_ids,
)
from .storyboard_dsl import formula_first_blueprint, infer_scene_role


SUPPORTED_COMPONENT_KINDS = set(get_args(AnimationComponent.model_fields["kind"].annotation))
SUPPORTED_VISUAL_TYPES = set(get_args(VisualObject.model_fields["type"].annotation))


class StoryboardAdapter:
    def convert(
        self,
        *,
        formula_analysis: FormulaAnalysis,
        explanation_plan: ExplanationPlan,
    ) -> Storyboard:
        scenes: list[SceneSpec] = []
        example_values = (
            explanation_plan.recommended_examples[0].concrete_values
            if explanation_plan.recommended_examples
            else {}
        )
        step_count = len(explanation_plan.explanation_steps)
        for index, step in enumerate(explanation_plan.explanation_steps, start=1):
            scene_role = infer_scene_role(
                step,
                step_index=index,
                step_count=step_count,
                animation_pattern_id=explanation_plan.selected_animation_pattern_id,
            )
            components = _components_for_step(
                explanation_plan,
                step,
                step_index=index,
                step_count=step_count,
            )
            visual_objects = [
                VisualObject(
                    type="formula",
                    name=f"{step.id}_formula",
                    description=f"このステップで注目する式: {step.formula_focus or explanation_plan.formula}",
                    params={"latex": step.formula_focus or explanation_plan.formula},
                ),
                VisualObject(
                    type="text",
                    name=f"{step.id}_visual_idea",
                    description=step.visual_idea,
                    params={"text": step.visual_idea},
                ),
            ]
            visual_objects.extend(_visual_objects_for_components(components))
            if explanation_plan.selected_animation_pattern_id == "trajectory_on_surface" and index == 1:
                function_preset = normalize_loss_surface_preset(
                    example_values.get("function_preset", "quadratic_ripple")
                )
                surface_params = {
                    "function_preset": function_preset,
                    "function": "0.35*x**2 + y**2 + 0.25*x*y + 0.8*sin(1.5*x)*cos(y)",
                    "x_range": [-3, 3],
                    "y_range": [-3, 3],
                    "surface_y_shift": _safe_float(example_values.get("surface_y_shift"), 2.7),
                    "surface_z_length": _safe_float(example_values.get("surface_z_length"), 2.4),
                    "surface_camera_zoom": _safe_float(
                        example_values.get("surface_camera_zoom"),
                        0.52,
                    ),
                    "surface_camera_phi": _safe_float(
                        example_values.get("surface_camera_phi"),
                        55.0,
                    ),
                    "surface_camera_theta": _safe_float(
                        example_values.get("surface_camera_theta"),
                        -48.0,
                    ),
                    "title_top_buff": _safe_float(example_values.get("title_top_buff"), 0.18),
                    "caption_bottom_buff": _safe_float(
                        example_values.get("caption_bottom_buff"),
                        0.32,
                    ),
                }
                if function_preset == "double_well_2d":
                    surface_params.update(
                        {
                            "function": "builtin_double_well_2d",
                            "x_range": [-3, 3],
                            "y_range": [-2.4, 2.4],
                        }
                    )
                if function_preset == "double_well_1d":
                    surface_params.update(
                        {
                            "function": "builtin_double_well_1d",
                            "x_range": [-1.9, 1.9],
                            "y_range": [-0.5, 5.8],
                        }
                    )
                components.insert(
                    0,
                    AnimationComponent(
                        id="loss_landscape",
                        kind=_loss_landscape_kind(function_preset),
                        description="損失関数の全体像を表示する",
                        label="loss landscape",
                        params={"function_preset": function_preset},
                    ),
                )
                visual_objects.insert(
                    0,
                    VisualObject(
                        type="surface",
                        name="loss_surface",
                        description="損失関数を地形として表示する",
                        params=surface_params,
                    ),
                )
            if explanation_plan.selected_animation_pattern_id == "trajectory_on_surface" and index == 2:
                initial_position = _number_pair(example_values.get("initial_position"))
                point_params = {
                    "x": float(
                        example_values.get(
                            "initial_x",
                            initial_position[0] if initial_position else 2.5,
                        )
                    ),
                    "y": float(
                        example_values.get(
                            "initial_y",
                            initial_position[1] if initial_position else 2.0,
                        )
                    ),
                }
                if "comparison_initial_x" in example_values:
                    point_params["comparison_x"] = float(example_values["comparison_initial_x"])
                if "comparison_initial_y" in example_values:
                    point_params["comparison_y"] = float(example_values["comparison_initial_y"])
                components.insert(
                    0,
                    AnimationComponent(
                        id="descent_start_points",
                        kind="descent_path",
                        description="初期位置と到達する谷を比較する",
                        label="start points",
                        params=point_params,
                    ),
                )
                visual_objects.insert(
                    0,
                    VisualObject(
                        type="point",
                        name="current_position",
                        description="現在のパラメータ位置を点として示す",
                        params=point_params,
                    ),
                )
            if explanation_plan.selected_animation_pattern_id == "trajectory_on_surface" and index == 3:
                components.insert(
                    0,
                    AnimationComponent(
                        id="negative_gradient_arrow",
                        kind="gradient_arrow",
                        description="負の勾配方向を矢印として示す",
                        label="-gradient",
                        params={"learning_rate": float(example_values.get("learning_rate", 0.15))},
                    ),
                )
                visual_objects.insert(
                    0,
                    VisualObject(
                        type="vector",
                        name="negative_gradient",
                        description="負の勾配方向を矢印として示す",
                        params={"learning_rate": float(example_values.get("learning_rate", 0.15))},
                    ),
                )
            if explanation_plan.selected_animation_pattern_id == "trajectory_on_surface" and index == 5:
                components.insert(
                    0,
                    AnimationComponent(
                        id="descent_path",
                        kind="descent_path",
                        description="更新で移動した点の軌跡を示す",
                        label="path",
                        params={"steps": int(example_values.get("steps", 30))},
                    ),
                )
                visual_objects.insert(
                    0,
                    VisualObject(
                        type="curve",
                        name="descent_path",
                        description="更新で移動した点の軌跡を示す",
                        params={"steps": int(example_values.get("steps", 30))},
                    ),
                )
            if explanation_plan.selected_animation_pattern_id == "penalty_curve" and index == 2:
                components.insert(
                    0,
                    AnimationComponent(
                        id="probability_bars",
                        kind="probability_bars",
                        description="各クラスの予測確率を棒として表示する",
                        label="probabilities",
                        params={"correct_index": 0},
                    ),
                )
                visual_objects.insert(
                    0,
                    VisualObject(
                        type="axis",
                        name="probability_bars",
                        description="各クラスの予測確率を棒として表示する",
                        params={"correct_index": 0},
                    ),
                )
            if explanation_plan.selected_animation_pattern_id == "penalty_curve" and index == 4:
                components.insert(
                    0,
                    AnimationComponent(
                        id="negative_log_curve",
                        kind="negative_log_curve",
                        description="正解確率pに対する-log(p)の罰を表示する",
                        label="-log(p)",
                        params={"x_range": [0.05, 1.0], "y_range": [0.0, 3.2]},
                    ),
                )
                visual_objects.insert(
                    0,
                    VisualObject(
                        type="curve",
                        name="negative_log_curve",
                        description="正解確率pに対する-log(p)の罰を表示する",
                        params={"x_range": [0.05, 1.0], "y_range": [0.0, 3.2]},
                    ),
                )
            if explanation_plan.selected_animation_pattern_id == "penalty_curve" and index == 5:
                components.insert(
                    0,
                    AnimationComponent(
                        id="probability_selector",
                        kind="probability_selector",
                        description="one-hotが選ぶ正解確率を表示する",
                        label="selected p",
                        params={"good_probability": 0.9, "bad_probability": 0.1},
                    ),
                )
                visual_objects.insert(
                    0,
                    VisualObject(
                        type="point",
                        name="good_bad_probability_points",
                        description="良い予測と悪い予測を-log(p)曲線上で比較する",
                        params={"good_probability": 0.9, "bad_probability": 0.1},
                    ),
                )

            scenes.append(
                SceneSpec(
                    id=step.id,
                    scene_role=scene_role,
                    beat_id=scene_role,
                    title=step.title,
                    learning_goal=step.learning_goal,
                    narration=step.explanation,
                    visual_objects=visual_objects,
                    components=components,
                    narration_cues=_narration_cues_for_step(step, components),
                    duration_seconds=10,
                )
            )

        examples = [
            Example(
                title=example.title,
                description=example.description,
                values=example.concrete_values,
            )
            for example in explanation_plan.recommended_examples[:1]
        ]

        return Storyboard(
            concept=_render_concept(explanation_plan),
            formula=explanation_plan.formula,
            blueprint=formula_first_blueprint(explanation_plan),
            one_sentence_summary=explanation_plan.one_sentence_summary,
            audience=explanation_plan.audience,
            prerequisites=[],
            symbol_ledger=[
                SymbolDefinition(
                    symbol=symbol.symbol,
                    meaning=symbol.meaning,
                    intuition=symbol.intuition,
                )
                for symbol in formula_analysis.symbols
            ],
            examples=examples,
            scenes=scenes,
            misconceptions=explanation_plan.misconceptions,
        )


def _render_concept(explanation_plan: ExplanationPlan) -> str:
    if explanation_plan.selected_animation_pattern_id == "penalty_curve":
        return "cross_entropy"
    if explanation_plan.selected_animation_pattern_id == "trajectory_on_surface":
        return "gradient_descent"
    if explanation_plan.selected_animation_pattern_id == "perceptron_decision_boundary":
        return "perceptron"
    if explanation_plan.selected_animation_pattern_id == "fully_connected_forward_pass":
        return "fully_connected_network"
    if explanation_plan.selected_animation_pattern_id == "neural_network_transform_flow":
        return "neural_network_transform"
    if explanation_plan.selected_animation_pattern_id == "activation_function_comparison":
        return "activation_functions"
    if explanation_plan.selected_animation_pattern_id == "backpropagation_chain_rule":
        return "backpropagation"
    if explanation_plan.selected_animation_pattern_id == "chain_rule_flow":
        return "chain_rule"
    return explanation_plan.target_concept


def _base_components_for_step(step: ExplanationStep) -> list[AnimationComponent]:
    if step.formula_focus:
        return [_formula_focus_component(step)]
    return [_text_caption_component(step)]


def _components_for_step(
    explanation_plan: ExplanationPlan,
    step: ExplanationStep,
    *,
    step_index: int,
    step_count: int,
) -> list[AnimationComponent]:
    components = _planned_components_for_step(step)
    if not components:
        components = _base_components_for_step(step)
    if step.formula_focus and not any(component.kind == "formula_focus" for component in components):
        components.append(_formula_focus_component(step))
    return _with_gradient_metaphor_components(
        explanation_plan,
        step,
        components,
        step_index=step_index,
        step_count=step_count,
    )


def _planned_components_for_step(step: ExplanationStep) -> list[AnimationComponent]:
    allowed_ids = visual_component_ids()
    components: list[AnimationComponent] = []
    for index, planned in enumerate(step.planned_components, start=1):
        kind = planned.kind.strip()
        if kind not in allowed_ids or kind not in SUPPORTED_COMPONENT_KINDS:
            continue
        definition = visual_component_definition(kind)
        params = dict(planned.params)
        params = _sanitize_component_params(kind, params)
        if kind == "formula_focus" and step.formula_focus:
            params.setdefault("formula_focus", step.formula_focus)
        label = planned.label or str(params.get("formula_focus") or "")
        if not label and definition is not None:
            label = definition.name
        components.append(
            AnimationComponent(
                id=f"{step.id}_{kind}_{index:02d}",
                kind=kind,
                description=planned.description
                or (definition.description if definition is not None else kind),
                label=label or kind,
                params=params,
            )
        )
    return components


def _with_gradient_metaphor_components(
    explanation_plan: ExplanationPlan,
    step: ExplanationStep,
    components: list[AnimationComponent],
    *,
    step_index: int,
    step_count: int,
) -> list[AnimationComponent]:
    if not _uses_gradient_metaphor(explanation_plan):
        return components

    text = _step_intent_text(step)
    is_first_step = step_index == 1
    is_last_step = step_index == step_count

    if is_first_step or _contains_any(
        text,
        ("地形", "山", "谷", "曲面", "損失関数", "loss landscape", "surface", "terrain"),
    ):
        _append_component_if_missing(
            components,
            step=step,
            kind="terrain_metaphor",
            label="高さ = 損失",
            params={"metaphor_label": "高さ = 損失"},
        )

    if _contains_any(
        text,
        ("現在地", "今いる", "パラメータ", "theta_t", r"\theta_t", "current position"),
    ):
        _append_component_if_missing(
            components,
            step=step,
            kind="hiker_marker",
            label="現在地",
            params={"label": "現在地"},
        )

    if _contains_any(text, ("勾配", "gradient", "nabla", r"\nabla", "上り", "下り")):
        _append_component_if_missing(
            components,
            step=step,
            kind="uphill_arrow",
            label="勾配 = 上り方向",
            params={"label": "勾配 = 上り方向"},
        )
        _append_component_if_missing(
            components,
            step=step,
            kind="downhill_arrow",
            label="更新 = 下り方向",
            params={"label": "更新 = 下り方向"},
        )

    if _contains_any(
        text,
        ("学習率", "一歩", "更新", "繰り返", "軌跡", r"\eta", "learning rate", "step", "path"),
    ):
        _append_component_if_missing(
            components,
            step=step,
            kind="footstep_path",
            label="一歩ずつ進む",
            params={"footprint_every": 1},
        )

    if is_last_step or _contains_any(
        text,
        ("式へ戻る", "式に戻る", "更新式", "formula", "対応づけ", "対応付け"),
    ):
        _append_component_if_missing(
            components,
            step=step,
            kind="formula_bridge",
            label="式の中の下る一歩",
            params={
                "formula_focus": r"-\eta\nabla L",
                "caption": "式の中の下る一歩",
            },
        )

    return components


def _uses_gradient_metaphor(explanation_plan: ExplanationPlan) -> bool:
    if explanation_plan.selected_animation_pattern_id != "trajectory_on_surface":
        return False
    concept = explanation_plan.target_concept.lower()
    return (
        explanation_plan.teaching_strategy in {"geometric_intuition", "visual_first"}
        or "gradient" in concept
        or "勾配" in concept
    )


def _step_intent_text(step: ExplanationStep) -> str:
    return " ".join(
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


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle.lower() in text for needle in needles)


def _append_component_if_missing(
    components: list[AnimationComponent],
    *,
    step: ExplanationStep,
    kind: str,
    label: str,
    params: dict[str, object],
) -> None:
    if any(component.kind == kind for component in components):
        return
    definition = visual_component_definition(kind)
    components.append(
        AnimationComponent(
            id=f"{step.id}_{kind}_auto",
            kind=kind,
            description=definition.description if definition is not None else kind,
            label=label,
            params=params,
        )
    )


def _formula_focus_component(step: ExplanationStep) -> AnimationComponent:
    return AnimationComponent(
        id=f"{step.id}_formula_focus",
        kind="formula_focus",
        description=f"数式パーツ {step.formula_focus} に注目する",
        label=step.formula_focus,
        params={"formula_focus": step.formula_focus},
    )


def _text_caption_component(step: ExplanationStep) -> AnimationComponent:
    return AnimationComponent(
        id=f"{step.id}_caption",
        kind="text_caption",
        description=step.visual_idea,
        label=step.title,
        params={},
    )


def _visual_objects_for_components(components: list[AnimationComponent]) -> list[VisualObject]:
    visual_objects: list[VisualObject] = []
    for component in components:
        definition = visual_component_definition(component.kind)
        visual_type = _safe_visual_type(definition.visual_type if definition else "text")
        params = dict(component.params)
        if visual_type == "formula":
            params.setdefault("latex", params.get("formula_focus") or component.label or "")
        if visual_type == "text":
            params.setdefault("text", component.label or component.description)
        visual_objects.append(
            VisualObject(
                type=visual_type,
                name=f"{component.id}_visual",
                description=component.description,
                params=params,
            )
        )
    return visual_objects


def _safe_visual_type(visual_type: str) -> str:
    if visual_type in SUPPORTED_VISUAL_TYPES:
        return visual_type
    return "text"


def _sanitize_component_params(kind: str, params: dict[str, object]) -> dict[str, object]:
    if kind in {"surface_plot", "contour_map", "loss_curve"}:
        default = "double_well_1d" if kind == "loss_curve" else "quadratic_ripple"
        params["function_preset"] = normalize_loss_surface_preset(
            params.get("function_preset"),
            default=default,
        )
    return params


def _number_pair(value: object) -> tuple[float, float] | None:
    if not isinstance(value, list | tuple) or len(value) != 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _narration_cues_for_step(
    step: ExplanationStep,
    components: list[AnimationComponent],
) -> list[NarrationCue]:
    component_id = components[0].id if components else None
    return [
        NarrationCue(
            segment_id=step.id,
            text=step.explanation,
            component_id=component_id,
            formula_focus=step.formula_focus,
        )
    ]


def _loss_landscape_kind(function_preset: str) -> str:
    if function_preset == "double_well_1d":
        return "loss_curve"
    if function_preset == "double_well_2d":
        return "contour_map"
    return "surface_plot"
