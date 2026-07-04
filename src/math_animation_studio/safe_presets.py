from __future__ import annotations


SUPPORTED_LOSS_SURFACE_PRESETS = {
    "quadratic_ripple",
    "double_well_2d",
    "double_well_1d",
}

LOSS_SURFACE_PRESET_ALIASES = {
    "bowl": "quadratic_ripple",
    "convex": "quadratic_ripple",
    "convex_bowl": "quadratic_ripple",
    "hill": "quadratic_ripple",
    "hill_surface": "quadratic_ripple",
    "loss_surface": "quadratic_ripple",
    "paraboloid": "quadratic_ripple",
    "quadratic": "quadratic_ripple",
    "simple_quadratic": "quadratic_ripple",
    "smooth_hill": "quadratic_ripple",
    "surface_3d": "quadratic_ripple",
    "double_well": "double_well_2d",
    "double_well_surface": "double_well_2d",
    "local_minima": "double_well_2d",
    "multi_minima": "double_well_2d",
    "two_valley": "double_well_2d",
    "two_valleys": "double_well_2d",
    "1d_double_well": "double_well_1d",
    "double_well_curve": "double_well_1d",
    "loss_curve": "double_well_1d",
    "one_dimensional_double_well": "double_well_1d",
}


def normalize_loss_surface_preset(value: object, *, default: str = "quadratic_ripple") -> str:
    raw = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    normalized = LOSS_SURFACE_PRESET_ALIASES.get(raw, raw)
    if normalized in SUPPORTED_LOSS_SURFACE_PRESETS:
        return normalized
    return default
