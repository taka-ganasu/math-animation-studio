from __future__ import annotations

from math_animation_studio.schema import PrerequisiteMap

from .sample_plans import sample_prerequisites


class PrerequisiteMapper:
    def map(self, *, sample_key: str) -> PrerequisiteMap:
        return sample_prerequisites(sample_key)
