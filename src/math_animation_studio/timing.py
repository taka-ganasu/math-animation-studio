from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence


@dataclass(frozen=True)
class TimelineSegment:
    id: str
    duration_seconds: float


CROSS_ENTROPY_FORMULA_FOCUS_SEGMENT_IDS = (
    "focus_y_i",
    "focus_y_hat_i",
    "focus_log",
    "focus_sum",
    "focus_minus",
)


CROSS_ENTROPY_BASE_TIMELINE = (
    TimelineSegment("intro_formula", 5.8),
    TimelineSegment("focus_y_i", 4.4),
    TimelineSegment("focus_y_hat_i", 4.2),
    TimelineSegment("focus_log", 4.5),
    TimelineSegment("focus_sum", 4.8),
    TimelineSegment("focus_minus", 3.8),
    TimelineSegment("model_pipeline", 5.4),
    TimelineSegment("one_hot_vector", 4.7),
    TimelineSegment("softmax_distribution", 6.5),
    TimelineSegment("correct_selector", 4.7),
    TimelineSegment("negative_log_penalty", 6.2),
    TimelineSegment("summary", 5.0),
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
        return tuple(TimelineSegment(segment.id, even_duration) for segment in segments)

    scale = target / base_total
    scaled = [
        TimelineSegment(segment.id, round(max(0.05, segment.duration_seconds * scale), 3))
        for segment in segments
    ]
    rounded_total = sum(segment.duration_seconds for segment in scaled)
    correction = round(target - rounded_total, 3)
    last = scaled[-1]
    scaled[-1] = TimelineSegment(
        last.id,
        round(max(0.05, last.duration_seconds + correction), 3),
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
