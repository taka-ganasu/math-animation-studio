from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Literal

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel, ConfigDict, Field

from math_animation_studio.schema import Storyboard, VisualObject


DEFAULT_FUNCTION_EXPR = (
    "0.35*x**2 + y**2 + 0.25*x*y + 0.8*sin(1.5*x)*cos(y)"
)
DEFAULT_GRADIENT_X = "0.7*x + 0.25*y + 1.2*cos(1.5*x)*cos(y)"
DEFAULT_GRADIENT_Y = "2*y + 0.25*x - 0.8*sin(1.5*x)*sin(y)"


class GeneratorError(RuntimeError):
    pass


class GradientDescentParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = "Gradient Descent"
    function_preset: Literal["quadratic_ripple"] = "quadratic_ripple"
    function_expr: str = DEFAULT_FUNCTION_EXPR
    gradient_expr_x: str = DEFAULT_GRADIENT_X
    gradient_expr_y: str = DEFAULT_GRADIENT_Y
    initial_x: float = 2.5
    initial_y: float = 2.0
    learning_rate: float = Field(default=0.15, gt=0, le=1)
    steps: int = Field(default=30, ge=1, le=80)
    x_range: tuple[float, float] = (-3.0, 3.0)
    y_range: tuple[float, float] = (-3.0, 3.0)
    formula_latex: str = r"\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)"
    narration_lines: list[str] = Field(default_factory=list)


class ManimGenerator:
    def __init__(self, *, template: str = "auto") -> None:
        self.template = template

    def generate(self, storyboard: Storyboard, output_path: Path) -> GradientDescentParams:
        if storyboard.concept.strip().lower().replace("-", "_") != "gradient_descent":
            raise GeneratorError("MVP supports only concept=gradient_descent.")
        if self.template not in {"auto", "gradient_descent_3d"}:
            raise GeneratorError(
                f"Unsupported template '{self.template}'. Use 'auto' or 'gradient_descent_3d'."
            )

        params = self._params_from_storyboard(storyboard)
        template = self._environment().get_template("gradient_descent_3d.py.j2")
        rendered = template.render(params=params)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
        return params

    def _environment(self) -> Environment:
        template_dir = files("math_animation_studio.generator.templates")
        env = Environment(
            loader=FileSystemLoader(str(template_dir)),
            autoescape=False,
            keep_trailing_newline=True,
            undefined=StrictUndefined,
            trim_blocks=False,
            lstrip_blocks=False,
        )
        env.filters["py_repr"] = repr
        return env

    def _params_from_storyboard(self, storyboard: Storyboard) -> GradientDescentParams:
        surface = _first_visual(storyboard, "surface")
        point = _first_visual(storyboard, "point")
        vector = _first_visual(storyboard, "vector")
        curve = _first_visual(storyboard, "curve")

        function_preset = surface.params.get("function_preset", "quadratic_ripple")
        function_expr = surface.params.get("function", DEFAULT_FUNCTION_EXPR)
        if function_preset != "quadratic_ripple" or function_expr != DEFAULT_FUNCTION_EXPR:
            raise GeneratorError(
                "MVP uses the built-in quadratic_ripple loss surface only. "
                "Storyboard function strings are not executed as Python code."
            )

        values = storyboard.examples[0].values if storyboard.examples else {}
        x_range = _range_tuple(surface.params.get("x_range"), (-3.0, 3.0))
        y_range = _range_tuple(surface.params.get("y_range"), (-3.0, 3.0))

        return GradientDescentParams(
            title=_title_from_storyboard(storyboard),
            function_preset="quadratic_ripple",
            function_expr=DEFAULT_FUNCTION_EXPR,
            gradient_expr_x=DEFAULT_GRADIENT_X,
            gradient_expr_y=DEFAULT_GRADIENT_Y,
            initial_x=float(point.params.get("x", values.get("initial_x", 2.5))),
            initial_y=float(point.params.get("y", values.get("initial_y", 2.0))),
            learning_rate=float(
                vector.params.get("learning_rate", values.get("learning_rate", 0.15))
            ),
            steps=int(curve.params.get("steps", values.get("steps", 30))),
            x_range=x_range,
            y_range=y_range,
            formula_latex=storyboard.formula
            or r"\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)",
            narration_lines=[scene.narration for scene in storyboard.scenes],
        )


def _title_from_storyboard(storyboard: Storyboard) -> str:
    if storyboard.concept == "gradient_descent":
        return "Gradient Descent"
    return storyboard.concept.replace("_", " ").title()


def _first_visual(storyboard: Storyboard, visual_type: str) -> VisualObject:
    for scene in storyboard.scenes:
        for visual in scene.visual_objects:
            if visual.type == visual_type:
                return visual
    raise GeneratorError(f"Storyboard is missing a '{visual_type}' visual object.")


def _range_tuple(value: object, fallback: tuple[float, float]) -> tuple[float, float]:
    if not isinstance(value, list | tuple) or len(value) != 2:
        return fallback
    return (float(value[0]), float(value[1]))
