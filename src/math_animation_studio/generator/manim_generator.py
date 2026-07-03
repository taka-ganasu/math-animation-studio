from __future__ import annotations

import json
import math
import re
from importlib.resources import files
from pathlib import Path
from typing import Any, Literal

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from pydantic import BaseModel, ConfigDict, Field

from math_animation_studio.schema import Storyboard, VisualObject
from math_animation_studio.timing import (
    cross_entropy_timeline_segments,
    gradient_double_well_1d_timeline_segments,
    gradient_double_well_timeline_segments,
    segment_duration_map,
)


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
    visualization_style: Literal["surface_3d", "double_well_2d", "double_well_1d"] = "surface_3d"
    function_preset: Literal["quadratic_ripple", "double_well_2d", "double_well_1d"] = "quadratic_ripple"
    function_expr: str = DEFAULT_FUNCTION_EXPR
    gradient_expr_x: str = DEFAULT_GRADIENT_X
    gradient_expr_y: str = DEFAULT_GRADIENT_Y
    initial_x: float = 2.5
    initial_y: float = 2.0
    comparison_initial_x: float = 2.7
    comparison_initial_y: float = 1.8
    learning_rate: float = Field(default=0.15, gt=0, le=1)
    steps: int = Field(default=30, ge=1, le=80)
    x_range: tuple[float, float] = (-3.0, 3.0)
    y_range: tuple[float, float] = (-3.0, 3.0)
    formula_latex: str = r"\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)"
    segment_durations: dict[str, float] = Field(default_factory=dict)
    narration_lines: list[str] = Field(default_factory=list)


class PenaltyCurveParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    base_duration_seconds: float = 29.5
    target_duration_seconds: int = Field(default=30, ge=5, le=180)
    title: str = "Cross Entropy Loss"
    formula_latex: str = r"L = - \sum_i y_i \log(\hat{y}_i)"
    scenario_title: str = "3クラス分類"
    summary_text: str = "正解に高い確率を置くほど、損失は小さくなる。"
    class_labels: tuple[str, ...] = ("猫", "犬", "鳥")
    correct_index: int = Field(default=0, ge=0)
    good_probability: float = Field(default=0.9, gt=0.0, lt=1.0)
    bad_probability: float = Field(default=0.1, gt=0.0, lt=1.0)
    good_distribution: tuple[float, ...] = (0.9, 0.05, 0.05)
    bad_distribution: tuple[float, ...] = (0.1, 0.45, 0.45)
    good_logits: tuple[float, ...] = (2.2, -0.7, -0.7)
    bad_logits: tuple[float, ...] = (-1.4, 0.4, 0.4)
    good_chart_title: str = "良い予測"
    bad_chart_title: str = "悪い予測"
    scene_components: tuple[str, ...] = (
        "intro_formula",
        "formula_parts_focus",
        "model_pipeline",
        "one_hot_vector",
        "softmax_distribution",
        "correct_selector",
        "negative_log_penalty",
        "summary",
    )
    caption_lines: tuple[str, ...] = (
        "クロスエントロピーは、正解クラスの予測確率を罰に変えます。",
        "モデルは入力からlogitsを出し、softmaxで確率分布に変換します。",
        "one-hotラベルでは、正解クラスだけが損失に効きます。",
        "正解クラスの確率が低い予測は、大きく罰されます。",
        "pが0に近いほど、-log(p)は急激に大きくなります。",
        "式の中身は、正解確率を取り出して罰へ変換する操作です。",
    )
    formula_focus_items: tuple[dict[str, str | int], ...] = (
        {
            "segment_id": "focus_y_i",
            "part_start": 4,
            "part_end": 4,
            "label": "y_i",
            "description": "正解クラスだけ1になるone-hotのスイッチです。",
        },
        {
            "segment_id": "focus_y_hat_i",
            "part_start": 7,
            "part_end": 7,
            "label": "ŷ_i",
            "description": "モデルがクラスiへ置いた予測確率です。",
        },
        {
            "segment_id": "focus_log",
            "part_start": 5,
            "part_end": 8,
            "label": "log(ŷ_i)",
            "description": "小さい確率の差を、損失として見えやすくします。",
        },
        {
            "segment_id": "focus_sum",
            "part_start": 3,
            "part_end": 3,
            "label": "Σ_i",
            "description": "全クラスを見る記号ですが、one-hotなので正解クラスだけ残ります。",
        },
        {
            "segment_id": "focus_minus",
            "part_start": 2,
            "part_end": 2,
            "label": "-",
            "description": "log(p)は負になりやすいので、損失を正の罰に変えます。",
        },
    )
    segment_durations: dict[str, float] = Field(
        default_factory=lambda: segment_duration_map(cross_entropy_timeline_segments(30))
    )
    narration_lines: list[str] = Field(default_factory=list)

    @property
    def timing_scale(self) -> float:
        return self.target_duration_seconds / self.base_duration_seconds


class ManimGenerator:
    def __init__(
        self,
        *,
        template: str = "auto",
        target_duration_seconds: int | None = None,
    ) -> None:
        self.template = template
        self.target_duration_seconds = target_duration_seconds

    def generate(
        self,
        storyboard: Storyboard,
        output_path: Path,
    ) -> GradientDescentParams | PenaltyCurveParams:
        template_name = self._select_template(storyboard)
        if template_name == "gradient_descent_3d":
            params = self._gradient_descent_params_from_storyboard(storyboard)
            template = self._environment().get_template("gradient_descent_3d.py.j2")
        else:
            params = self._penalty_curve_params_from_storyboard(storyboard)
            template = self._environment().get_template("penalty_curve.py.j2")

        rendered = template.render(params=params)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
        return params

    def scene_name_for(self, storyboard: Storyboard) -> str:
        template_name = self._select_template(storyboard)
        if template_name == "gradient_descent_3d":
            return "GradientDescent3DScene"
        return "CrossEntropyPenaltyScene"

    def _select_template(self, storyboard: Storyboard) -> str:
        supported_templates = {"auto", "gradient_descent_3d", "penalty_curve"}
        if self.template not in supported_templates:
            raise GeneratorError(
                f"Unsupported template '{self.template}'. "
                "Use 'auto', 'gradient_descent_3d', or 'penalty_curve'."
            )

        concept = _normalized_concept(storyboard.concept)
        if self.template == "auto":
            if concept == "gradient_descent":
                return "gradient_descent_3d"
            if concept == "cross_entropy":
                return "penalty_curve"
            raise GeneratorError(
                "MVP render templates support concept=gradient_descent or concept=cross_entropy."
            )

        expected_concept = {
            "gradient_descent_3d": "gradient_descent",
            "penalty_curve": "cross_entropy",
        }[self.template]
        if concept != expected_concept:
            raise GeneratorError(
                f"Template '{self.template}' requires concept={expected_concept}, "
                f"but storyboard concept is {storyboard.concept}."
            )
        return self.template

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

    def _gradient_descent_params_from_storyboard(self, storyboard: Storyboard) -> GradientDescentParams:
        surface = _first_visual(storyboard, "surface")
        point = _first_visual(storyboard, "point")
        vector = _first_visual(storyboard, "vector")
        curve = _first_visual(storyboard, "curve")

        function_preset = surface.params.get("function_preset", "quadratic_ripple")
        function_expr = surface.params.get("function", DEFAULT_FUNCTION_EXPR)
        if function_preset == "double_well_2d":
            visualization_style: Literal["surface_3d", "double_well_2d", "double_well_1d"] = "double_well_2d"
        elif function_preset == "double_well_1d":
            visualization_style = "double_well_1d"
        elif function_preset == "quadratic_ripple" and function_expr == DEFAULT_FUNCTION_EXPR:
            visualization_style = "surface_3d"
        else:
            raise GeneratorError(
                "MVP uses built-in loss surface presets only. "
                "Storyboard function strings are not executed as Python code."
            )

        values = storyboard.examples[0].values if storyboard.examples else {}
        x_range = _range_tuple(surface.params.get("x_range"), (-3.0, 3.0))
        y_range = _range_tuple(surface.params.get("y_range"), (-3.0, 3.0))

        return GradientDescentParams(
            title=_title_from_storyboard(storyboard),
            visualization_style=visualization_style,
            function_preset=function_preset,
            function_expr=DEFAULT_FUNCTION_EXPR
            if function_preset == "quadratic_ripple"
            else f"builtin_{function_preset}",
            gradient_expr_x=DEFAULT_GRADIENT_X,
            gradient_expr_y=DEFAULT_GRADIENT_Y,
            initial_x=float(point.params.get("x", values.get("initial_x", 2.5))),
            initial_y=float(point.params.get("y", values.get("initial_y", 2.0))),
            comparison_initial_x=float(
                point.params.get(
                    "comparison_x",
                    values.get("comparison_initial_x", 2.7),
                )
            ),
            comparison_initial_y=float(
                point.params.get(
                    "comparison_y",
                    values.get("comparison_initial_y", 1.8),
                )
            ),
            learning_rate=float(
                vector.params.get("learning_rate", values.get("learning_rate", 0.15))
            ),
            steps=int(curve.params.get("steps", values.get("steps", 30))),
            x_range=x_range,
            y_range=y_range,
            formula_latex=storyboard.formula
            or r"\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)",
            segment_durations=_gradient_segment_durations(
                visualization_style,
                self.target_duration_seconds,
            ),
            narration_lines=[scene.narration for scene in storyboard.scenes],
        )

    def _penalty_curve_params_from_storyboard(self, storyboard: Storyboard) -> PenaltyCurveParams:
        example_data = _penalty_curve_example_data(storyboard)
        labels = _class_labels_from_storyboard(storyboard, example_data["class_count"])
        correct_index = min(int(example_data["correct_index"]), len(labels) - 1)
        good_distribution = _fit_distribution(
            example_data["good_distribution"],
            label_count=len(labels),
            correct_index=correct_index,
            fallback_correct_probability=example_data["good_probability"],
        )
        bad_distribution = _fit_distribution(
            example_data["bad_distribution"],
            label_count=len(labels),
            correct_index=correct_index,
            fallback_correct_probability=example_data["bad_probability"],
        )
        scenario_title = _scenario_title_from_storyboard(storyboard)

        focus_items = _formula_focus_items_from_storyboard(storyboard)
        active_focus_ids = tuple(
            str(item["segment_id"]) for item in focus_items if "segment_id" in item
        )
        target_duration_seconds = self.target_duration_seconds or 13

        return PenaltyCurveParams(
            target_duration_seconds=target_duration_seconds,
            title=_title_from_storyboard(storyboard),
            formula_latex=storyboard.formula or r"L = - \sum_i y_i \log(\hat{y}_i)",
            scenario_title=scenario_title,
            summary_text=_shorten_text(storyboard.one_sentence_summary, 62),
            class_labels=tuple(labels),
            correct_index=correct_index,
            good_probability=good_distribution[correct_index],
            bad_probability=bad_distribution[correct_index],
            good_distribution=tuple(good_distribution),
            bad_distribution=tuple(bad_distribution),
            good_logits=tuple(_logits_from_distribution(good_distribution)),
            bad_logits=tuple(_logits_from_distribution(bad_distribution)),
            caption_lines=_caption_lines_from_storyboard(storyboard, scenario_title),
            formula_focus_items=focus_items,
            segment_durations=segment_duration_map(
                cross_entropy_timeline_segments(
                    target_duration_seconds,
                    active_focus_ids=active_focus_ids,
                )
            ),
            narration_lines=[scene.narration for scene in storyboard.scenes],
        )


def _normalized_concept(concept: str) -> str:
    return concept.strip().lower().replace("-", "_")


def _gradient_segment_durations(
    visualization_style: str,
    target_duration_seconds: int | None,
) -> dict[str, float]:
    if visualization_style == "double_well_2d":
        return segment_duration_map(
            gradient_double_well_timeline_segments(target_duration_seconds or 52)
        )
    if visualization_style == "double_well_1d":
        return segment_duration_map(
            gradient_double_well_1d_timeline_segments(target_duration_seconds or 50)
        )
    return {}


def _title_from_storyboard(storyboard: Storyboard) -> str:
    if storyboard.concept == "gradient_descent":
        return "Gradient Descent"
    if storyboard.concept == "cross_entropy":
        return "Cross Entropy Loss"
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


def _penalty_curve_example_data(storyboard: Storyboard) -> dict[str, Any]:
    correct_index = 0
    good_probability = 0.9
    bad_probability = 0.1
    good_distribution: list[float] | None = None
    bad_distribution: list[float] | None = None
    class_count = 0

    for example in storyboard.examples:
        values = example.values

        one_hot = _indexed_values(values, kind="one_hot") or _json_vector_for_keys(
            values, {"y", "label", "one_hot"}
        )
        if one_hot:
            class_count = max(class_count, len(one_hot))
            correct_index = _max_index(one_hot)

        direct_good = _float_for_keys(
            values,
            {
                "cat_probability_good",
                "good_probability",
                "correct_probability_good",
            },
        )
        direct_bad = _float_for_keys(
            values,
            {
                "cat_probability_bad",
                "bad_probability",
                "correct_probability_bad",
            },
        )
        if direct_good is not None:
            good_probability = _clamp_probability(direct_good)
        if direct_bad is not None:
            bad_probability = _clamp_probability(direct_bad)

        example_good = _json_vector_for_role(values, "good")
        example_bad = _json_vector_for_role(values, "bad")
        prediction = _indexed_values(values, kind="prediction") or _json_vector_for_keys(
            values,
            {"\\hat{y}", "hat_y", "y_hat", "prediction", "probabilities"},
        )

        if example_good:
            good_distribution = example_good
        elif prediction and good_distribution is None:
            good_distribution = prediction

        if example_bad:
            bad_distribution = example_bad

        for vector in (good_distribution, bad_distribution, prediction):
            if vector:
                class_count = max(class_count, len(vector))

    if _storyboard_mentions_dice(storyboard):
        class_count = max(class_count, 6)

    if good_distribution:
        correct_index = min(correct_index, len(good_distribution) - 1)
        good_probability = _clamp_probability(good_distribution[correct_index])
    if bad_distribution:
        correct_index = min(correct_index, len(bad_distribution) - 1)
        bad_probability = _clamp_probability(bad_distribution[correct_index])

    return {
        "class_count": max(2, class_count or 3),
        "correct_index": correct_index,
        "good_probability": good_probability,
        "bad_probability": bad_probability,
        "good_distribution": good_distribution,
        "bad_distribution": bad_distribution,
    }


def _indexed_values(values: dict[str, float | int | str], *, kind: str) -> list[float] | None:
    indexed: dict[int, float] = {}
    for key, value in values.items():
        match = re.search(r"_(\d+)$", key.strip())
        if not match:
            continue
        normalized = key.strip().lower()
        if kind == "one_hot":
            if not normalized.startswith("y_"):
                continue
        elif kind == "prediction":
            if not (
                "hat" in normalized
                or "\\hat" in normalized
                or "y_hat" in normalized
                or "prediction" in normalized
            ):
                continue
        else:
            continue
        number = _as_float(value)
        if number is None:
            continue
        indexed[int(match.group(1)) - 1] = number

    if not indexed:
        return None
    max_index = max(indexed)
    return [_clamp_probability(indexed.get(index, 0.0)) for index in range(max_index + 1)]


def _json_vector_for_role(values: dict[str, float | int | str], role: str) -> list[float] | None:
    for key, value in values.items():
        normalized = key.lower()
        if role not in normalized:
            continue
        vector = _parse_json_number_list(value)
        if vector:
            return vector
    return None


def _json_vector_for_keys(
    values: dict[str, float | int | str],
    candidate_keys: set[str],
) -> list[float] | None:
    normalized_candidates = {key.lower() for key in candidate_keys}
    for key, value in values.items():
        if key.lower() not in normalized_candidates:
            continue
        vector = _parse_json_number_list(value)
        if vector:
            return vector
    return None


def _parse_json_number_list(value: object) -> list[float] | None:
    if isinstance(value, list | tuple):
        items = value
    elif isinstance(value, str):
        text = value.strip()
        if not text.startswith("["):
            return None
        try:
            items = json.loads(text)
        except json.JSONDecodeError:
            return None
    else:
        return None

    if not isinstance(items, list) or not items:
        return None
    numbers: list[float] = []
    for item in items:
        number = _as_float(item)
        if number is None:
            return None
        numbers.append(_distribution_probability(number))
    return numbers


def _float_for_keys(values: dict[str, float | int | str], keys: set[str]) -> float | None:
    normalized_keys = {key.lower() for key in keys}
    for key, value in values.items():
        if key.lower() not in normalized_keys:
            continue
        number = _as_float(value)
        if number is not None:
            return number
    return None


def _as_float(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return None
    return None


def _fit_distribution(
    vector: list[float] | None,
    *,
    label_count: int,
    correct_index: int,
    fallback_correct_probability: float,
) -> list[float]:
    if vector:
        fitted = [_distribution_probability(value) for value in vector[:label_count]]
        missing_count = label_count - len(fitted)
        if missing_count > 0:
            remaining = max(0.0, 1.0 - sum(fitted))
            fill_value = remaining / missing_count if remaining > 0 else 0.0
            fitted.extend(fill_value for _ in range(missing_count))
        return fitted

    correct_probability = _clamp_probability(fallback_correct_probability)
    if label_count <= 1:
        return [correct_probability]
    other_probability = (1.0 - correct_probability) / (label_count - 1)
    probabilities = [other_probability for _ in range(label_count)]
    probabilities[correct_index] = correct_probability
    return [_clamp_probability(value) for value in probabilities]


def _clamp_probability(value: float) -> float:
    return min(0.99, max(0.01, float(value)))


def _distribution_probability(value: float) -> float:
    return min(0.99, max(0.0, float(value)))


def _max_index(values: list[float]) -> int:
    return max(range(len(values)), key=lambda index: values[index])


def _logits_from_distribution(distribution: list[float]) -> list[float]:
    return [round(math.log(max(probability, 0.01)), 2) for probability in distribution]


def _storyboard_mentions_dice(storyboard: Storyboard) -> bool:
    texts = [storyboard.one_sentence_summary]
    texts.extend(example.title + " " + example.description for example in storyboard.examples)
    texts.extend(scene.title + " " + scene.narration for scene in storyboard.scenes)
    return any("サイコロ" in text or "dice" in text.lower() for text in texts)


def _class_labels_from_storyboard(storyboard: Storyboard, expected_count: int) -> list[str]:
    for example in storyboard.examples:
        labels = _labels_from_example_title(example.title, expected_count)
        if labels:
            return _fit_labels(labels, expected_count)

    return [f"クラス{index + 1}" for index in range(expected_count)]


def _labels_from_example_title(title: str, expected_count: int) -> list[str] | None:
    if "サイコロ" in title:
        return [f"目{index + 1}" for index in range(expected_count)]

    candidate = title
    if ":" in candidate:
        candidate = candidate.split(":", 1)[1]
    if "：" in candidate:
        candidate = candidate.split("：", 1)[1]
    candidate = re.split(r"[（(]", candidate, maxsplit=1)[0]
    for suffix in ("の分類例", "分類例", "の予測例", "予測例", "分類", "予測"):
        candidate = candidate.replace(suffix, "")
    candidate = candidate.strip(" 　:：。、")

    if "・" in candidate:
        labels = [part.strip() for part in candidate.split("・") if part.strip()]
        if labels:
            return labels
    if expected_count == 2 and "と" in candidate:
        labels = [part.strip() for part in candidate.split("と") if part.strip()]
        if len(labels) == 2:
            return labels
    return None


def _fit_labels(labels: list[str], expected_count: int) -> list[str]:
    fitted = [_shorten_text(label, 8) for label in labels[:expected_count]]
    while len(fitted) < expected_count:
        fitted.append("その他" if len(fitted) == len(labels) else f"クラス{len(fitted) + 1}")
    return fitted


def _scenario_title_from_storyboard(storyboard: Storyboard) -> str:
    if storyboard.examples:
        return _shorten_text(storyboard.examples[0].title, 28)
    return "具体例"


def _caption_lines_from_storyboard(
    storyboard: Storyboard,
    scenario_title: str,
) -> tuple[str, ...]:
    one_hot_caption = _scene_caption_for_keywords(
        storyboard,
        ("one-hot", "ワンホット", "正解"),
        "one-hotラベルは、正解クラスだけを1にするスイッチです。",
    )
    penalty_caption = _scene_caption_for_keywords(
        storyboard,
        ("-log", "log", "罰", "p", "取り出"),
        "pが0に近いほど、-log(p)は急激に大きくなります。",
    )
    lines = [
        f"{scenario_title}で、予測確率が損失へ変わる流れを見ます。",
        "入力xをモデルf_θに通し、logits zをsoftmaxで確率ŷに変えます。",
        one_hot_caption,
        "悪い予測と良い予測で、正解クラスの確率pを比べます。",
        penalty_caption,
        storyboard.one_sentence_summary,
    ]

    return tuple(_shorten_text(_caption_display_text(text), 72) for text in lines[:6])


def _scene_caption_for_keywords(
    storyboard: Storyboard,
    keywords: tuple[str, ...],
    fallback: str,
) -> str:
    for scene in storyboard.scenes:
        text = scene.narration.strip() or scene.title.strip()
        if not text:
            continue
        lowered = text.lower()
        if any(keyword.lower() in lowered for keyword in keywords):
            return text
    return fallback


def _formula_focus_items_from_storyboard(
    storyboard: Storyboard,
) -> tuple[dict[str, str | int], ...]:
    symbol_text = " ".join(symbol.symbol for symbol in storyboard.symbol_ledger)
    scene_text = " ".join(scene.title + " " + scene.narration for scene in storyboard.scenes)
    combined = f"{storyboard.formula or ''} {symbol_text} {scene_text}"

    items: list[dict[str, str | int]] = [
        {
            "segment_id": "focus_y_i",
            "part_start": 4,
            "part_end": 4,
            "label": "y_i",
            "description": "正解クラスだけ1になるone-hotのスイッチです。",
        },
        {
            "segment_id": "focus_y_hat_i",
            "part_start": 7,
            "part_end": 7,
            "label": "ŷ_i",
            "description": "モデルがクラスiへ置いた予測確率です。",
        },
    ]
    if "log" in combined or "対数" in combined or r"\log" in combined:
        items.append(
            {
                "segment_id": "focus_log",
                "part_start": 5,
                "part_end": 8,
                "label": "log(ŷ_i)",
                "description": "小さい確率の差を、損失として見えやすくします。",
            }
        )
    if "sum" in combined or "総和" in combined or r"\sum" in combined:
        items.append(
            {
                "segment_id": "focus_sum",
                "part_start": 3,
                "part_end": 3,
                "label": "Σ_i",
                "description": "全クラスを見る記号ですが、one-hotなので正解クラスだけ残ります。",
            }
        )
    if "-" in (storyboard.formula or "") or "マイナス" in combined:
        items.append(
            {
                "segment_id": "focus_minus",
                "part_start": 2,
                "part_end": 2,
                "label": "-",
                "description": "log(p)は負になりやすいので、損失を正の罰に変えます。",
            }
        )
    return tuple(items[:5])


def _caption_display_text(text: str) -> str:
    cleaned = _single_line(text)
    replacements = {
        r"\hat{y}_i": "ŷ_i",
        r"\hat{y}": "ŷ",
        r"\sum_i": "Σ_i",
        r"\log": "log",
        "{": "",
        "}": "",
    }
    for source, target in replacements.items():
        cleaned = cleaned.replace(source, target)
    return cleaned


def _single_line(text: str) -> str:
    return " ".join(text.split())


def _shorten_text(text: str, max_length: int) -> str:
    cleaned = _single_line(text)
    if len(cleaned) <= max_length:
        return cleaned
    return cleaned[: max_length - 1].rstrip("、。,. ") + "。"
