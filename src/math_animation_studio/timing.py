from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class TimelineSegment:
    id: str
    duration_seconds: float
    component_id: str | None = None
    formula_focus: str | None = None


CROSS_ENTROPY_FORMULA_FOCUS_SEGMENT_IDS = (
    "focus_y_i",
    "focus_y_hat_i",
    "focus_log",
    "focus_sum",
    "focus_minus",
)


CROSS_ENTROPY_BASE_TIMELINE = (
    TimelineSegment("intro_formula", 5.8, "intro_formula"),
    TimelineSegment("focus_y_i", 4.4, "formula_parts_focus", r"y_i"),
    TimelineSegment("focus_y_hat_i", 4.2, "formula_parts_focus", r"\hat{y}_i"),
    TimelineSegment("focus_log", 4.5, "formula_parts_focus", r"\log(\hat{y}_i)"),
    TimelineSegment("focus_sum", 4.8, "formula_parts_focus", r"\sum_i"),
    TimelineSegment("focus_minus", 3.8, "formula_parts_focus", "-"),
    TimelineSegment("model_pipeline", 5.4, "model_pipeline"),
    TimelineSegment("one_hot_vector", 4.7, "one_hot_vector"),
    TimelineSegment("softmax_distribution", 6.5, "softmax_distribution"),
    TimelineSegment("correct_selector", 4.7, "correct_selector"),
    TimelineSegment("negative_log_penalty", 6.2, "negative_log_penalty"),
    TimelineSegment("summary", 5.0, "summary"),
)


GRADIENT_DOUBLE_WELL_BASE_TIMELINE = (
    TimelineSegment("intro_landscape", 5.0, "contour_map"),
    TimelineSegment("two_valleys", 6.0, "contour_map"),
    TimelineSegment("local_slope", 6.0, "gradient_arrow", r"-\nabla L"),
    TimelineSegment("left_descent", 8.0, "descent_path"),
    TimelineSegment("right_descent", 8.0, "descent_path"),
    TimelineSegment("compare_minima", 7.0, "summary"),
    TimelineSegment("sgd_bridge", 7.0, "sgd_jitter"),
    TimelineSegment(
        "summary",
        5.0,
        "summary",
        r"\theta_{t+1}=\theta_t-\eta\nabla L(\theta_t)",
    ),
)


GRADIENT_SURFACE_3D_BASE_TIMELINE = (
    TimelineSegment("title_intro", 5.0, "intro_formula"),
    TimelineSegment(
        "formula_parts",
        12.0,
        "formula_parts_focus",
        r"\theta_{t+1}=\theta_t-\eta\nabla L(\theta_t)",
    ),
    TimelineSegment("intro_surface", 6.0, "surface_plot"),
    TimelineSegment("current_point", 4.0, "descent_path"),
    TimelineSegment("local_gradient", 6.0, "gradient_arrow", r"-\nabla L"),
    TimelineSegment("descent_path", 16.0, "descent_path"),
    TimelineSegment(
        "summary_surface",
        6.0,
        "summary",
        r"\theta_{t+1}=\theta_t-\eta\nabla L(\theta_t)",
    ),
)


GRADIENT_DOUBLE_WELL_1D_BASE_TIMELINE = (
    TimelineSegment("intro_curve", 5.0, "loss_curve"),
    TimelineSegment("two_valleys_1d", 6.0, "loss_curve"),
    TimelineSegment("local_slope_1d", 6.0, "gradient_arrow", r"-\nabla L(\theta_t)"),
    TimelineSegment("left_descent_1d", 7.0, "descent_path"),
    TimelineSegment("right_descent_1d", 7.0, "descent_path"),
    TimelineSegment("compare_minima_1d", 7.0, "summary"),
    TimelineSegment("sgd_bridge_1d", 6.0, "sgd_jitter"),
    TimelineSegment(
        "summary_1d",
        5.0,
        "summary",
        r"\theta_{t+1}=\theta_t-\eta\nabla L(\theta_t)",
    ),
)


PERCEPTRON_BASE_TIMELINE = (
    TimelineSegment("title_intro", 10.0, "intro_formula"),
    TimelineSegment(
        "formula_weighted_inputs",
        12.0,
        "formula_parts_focus",
        r"w_1x_1+w_2x_2",
    ),
    TimelineSegment("formula_bias", 9.0, "formula_parts_focus", "b"),
    TimelineSegment("formula_activation", 8.0, "formula_parts_focus", r"\mathrm{step}"),
    TimelineSegment("formula_output", 6.0, "formula_parts_focus", "a"),
    TimelineSegment("network_diagram", 12.0, "perceptron_node"),
    TimelineSegment("weighted_sum", 9.0, "weighted_sum", r"z=w_1x_1+w_2x_2+b"),
    TimelineSegment("activation", 8.0, "activation_function", r"a=\mathrm{step}(z)"),
    TimelineSegment("decision_boundary", 10.0, "decision_boundary", r"w_1x_1+w_2x_2+b=0"),
    TimelineSegment("summary", 11.0, "summary", r"a=\mathrm{step}(w_1x_1+w_2x_2+b)"),
)

PERCEPTRON_MIN_NARRATION_SECONDS = sum(
    segment.duration_seconds for segment in PERCEPTRON_BASE_TIMELINE
)
PERCEPTRON_REFERENCE_VOICE_RATE = 120.0
PERCEPTRON_MIN_TIMELINE_SECONDS = 65.0


FULLY_CONNECTED_BASE_TIMELINE = (
    TimelineSegment("title_intro", 10.0, "intro_formula"),
    TimelineSegment("formula_affine", 11.0, "formula_parts_focus", r"W_1x+b_1"),
    TimelineSegment("formula_activation", 9.5, "formula_parts_focus", r"\sigma"),
    TimelineSegment("formula_output", 8.0, "formula_parts_focus", r"W_2h+b_2"),
    TimelineSegment("layer_stack", 10.0, "dense_layer"),
    TimelineSegment("full_connections", 10.0, "fully_connected_edges"),
    TimelineSegment("forward_pass", 10.5, "forward_pass"),
    TimelineSegment("softmax_output", 8.0, "softmax_output"),
    TimelineSegment("summary", 11.0, "summary", r"\hat{y}=\mathrm{softmax}(W_2\sigma(W_1x+b_1)+b_2)"),
)


def cross_entropy_timeline_segments(
    target_duration_seconds: int | float | None = None,
    *,
    active_focus_ids: Sequence[str] | None = None,
) -> tuple[TimelineSegment, ...]:
    segments = _select_cross_entropy_segments(active_focus_ids)
    if target_duration_seconds is None:
        return segments
    return scale_timeline(segments, float(target_duration_seconds))


def segment_duration_map(segments: Iterable[TimelineSegment]) -> dict[str, float]:
    return {segment.id: segment.duration_seconds for segment in segments}


def segment_metadata_map(segments: Iterable[TimelineSegment]) -> dict[str, dict[str, str]]:
    metadata: dict[str, dict[str, str]] = {}
    for segment in segments:
        item: dict[str, str] = {}
        if segment.component_id:
            item["component_id"] = segment.component_id
        if segment.formula_focus:
            item["formula_focus"] = segment.formula_focus
        if item:
            metadata[segment.id] = item
    return metadata


def gradient_double_well_timeline_segments(
    target_duration_seconds: int | float | None = None,
) -> tuple[TimelineSegment, ...]:
    if target_duration_seconds is None:
        return GRADIENT_DOUBLE_WELL_BASE_TIMELINE
    return scale_timeline(GRADIENT_DOUBLE_WELL_BASE_TIMELINE, float(target_duration_seconds))


def gradient_surface_3d_timeline_segments(
    target_duration_seconds: int | float | None = None,
) -> tuple[TimelineSegment, ...]:
    if target_duration_seconds is None:
        return GRADIENT_SURFACE_3D_BASE_TIMELINE
    return scale_timeline(GRADIENT_SURFACE_3D_BASE_TIMELINE, float(target_duration_seconds))


def gradient_double_well_1d_timeline_segments(
    target_duration_seconds: int | float | None = None,
) -> tuple[TimelineSegment, ...]:
    if target_duration_seconds is None:
        return GRADIENT_DOUBLE_WELL_1D_BASE_TIMELINE
    return scale_timeline(GRADIENT_DOUBLE_WELL_1D_BASE_TIMELINE, float(target_duration_seconds))


def perceptron_timeline_segments(
    target_duration_seconds: int | float | None = None,
    *,
    voice_rate: int | float | None = None,
) -> tuple[TimelineSegment, ...]:
    target = perceptron_target_duration_seconds(
        target_duration_seconds,
        voice_rate=voice_rate,
    )
    return scale_timeline(PERCEPTRON_BASE_TIMELINE, target)


def perceptron_target_duration_seconds(
    target_duration_seconds: int | float | None = None,
    *,
    voice_rate: int | float | None = None,
) -> float:
    voice_floor = perceptron_voice_adjusted_min_seconds(voice_rate)
    if target_duration_seconds is None:
        return voice_floor

    requested = float(target_duration_seconds)
    if voice_rate is not None and requested <= PERCEPTRON_MIN_NARRATION_SECONDS:
        return voice_floor
    return max(requested, voice_floor)


def perceptron_voice_adjusted_min_seconds(voice_rate: int | float | None = None) -> float:
    if voice_rate is None:
        return float(PERCEPTRON_MIN_NARRATION_SECONDS)

    rate = max(1.0, float(voice_rate))
    adjusted = PERCEPTRON_MIN_NARRATION_SECONDS * PERCEPTRON_REFERENCE_VOICE_RATE / rate
    return max(PERCEPTRON_MIN_TIMELINE_SECONDS, adjusted)


def fully_connected_timeline_segments(
    target_duration_seconds: int | float | None = None,
) -> tuple[TimelineSegment, ...]:
    if target_duration_seconds is None:
        return FULLY_CONNECTED_BASE_TIMELINE
    return scale_timeline(FULLY_CONNECTED_BASE_TIMELINE, float(target_duration_seconds))


def scale_timeline(
    segments: Sequence[TimelineSegment],
    target_duration_seconds: float,
) -> tuple[TimelineSegment, ...]:
    if not segments:
        return ()
    target = max(0.1, float(target_duration_seconds))
    base_total = sum(segment.duration_seconds for segment in segments)
    if base_total <= 0:
        even_duration = target / len(segments)
        return tuple(
            TimelineSegment(
                segment.id,
                even_duration,
                segment.component_id,
                segment.formula_focus,
            )
            for segment in segments
        )

    scale = target / base_total
    scaled = [
        TimelineSegment(
            segment.id,
            round(max(0.05, segment.duration_seconds * scale), 3),
            segment.component_id,
            segment.formula_focus,
        )
        for segment in segments
    ]
    rounded_total = sum(segment.duration_seconds for segment in scaled)
    correction = round(target - rounded_total, 3)
    last = scaled[-1]
    scaled[-1] = TimelineSegment(
        last.id,
        round(max(0.05, last.duration_seconds + correction), 3),
        last.component_id,
        last.formula_focus,
    )
    return tuple(scaled)


def _select_cross_entropy_segments(
    active_focus_ids: Sequence[str] | None,
) -> tuple[TimelineSegment, ...]:
    if active_focus_ids is None:
        return CROSS_ENTROPY_BASE_TIMELINE

    active = set(active_focus_ids)
    return tuple(
        segment
        for segment in CROSS_ENTROPY_BASE_TIMELINE
        if segment.id not in CROSS_ENTROPY_FORMULA_FOCUS_SEGMENT_IDS
        or segment.id in active
    )
