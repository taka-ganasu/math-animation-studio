from __future__ import annotations

from math_animation_studio.schema import (
    Example,
    ExplanationPlan,
    FormulaAnalysis,
    SceneSpec,
    Storyboard,
    SymbolDefinition,
    VisualObject,
)


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
                    duration_seconds=10,
                )
            )

        examples = [
            Example(
                title=example.title,
                description=example.description,
                values=example.concrete_values,
            )
            for example in explanation_plan.recommended_examples
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
