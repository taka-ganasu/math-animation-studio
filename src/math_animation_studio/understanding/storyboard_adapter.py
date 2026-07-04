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

from .visual_component_catalog import (
    visual_component_definition,
    visual_component_ids,
)


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
        for index, step in enumerate(explanation_plan.explanation_steps, start=1):
            components = _components_for_step(step)
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
                function_preset = str(example_values.get("function_preset", "quadratic_ripple"))
                surface_params = {
                    "function_preset": function_preset,
                    "function": "0.35*x**2 + y**2 + 0.25*x*y + 0.8*sin(1.5*x)*cos(y)",
                    "x_range": [-3, 3],
                    "y_range": [-3, 3],
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
                point_params = {
                    "x": float(example_values.get("initial_x", 2.5)),
                    "y": float(example_values.get("initial_y", 2.0)),
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
    return explanation_plan.target_concept


def _base_components_for_step(step: ExplanationStep) -> list[AnimationComponent]:
    if step.formula_focus:
        return [_formula_focus_component(step)]
    return [_text_caption_component(step)]


def _components_for_step(step: ExplanationStep) -> list[AnimationComponent]:
    components = _planned_components_for_step(step)
    if not components:
        return _base_components_for_step(step)
    if step.formula_focus and not any(component.kind == "formula_focus" for component in components):
        components.append(_formula_focus_component(step))
    return components


def _planned_components_for_step(step: ExplanationStep) -> list[AnimationComponent]:
    allowed_ids = visual_component_ids()
    components: list[AnimationComponent] = []
    for index, planned in enumerate(step.planned_components, start=1):
        kind = planned.kind.strip()
        if kind not in allowed_ids or kind not in SUPPORTED_COMPONENT_KINDS:
            continue
        definition = visual_component_definition(kind)
        params = dict(planned.params)
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
