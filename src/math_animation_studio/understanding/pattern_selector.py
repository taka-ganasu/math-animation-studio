from __future__ import annotations

from importlib.resources import files

import yaml

from math_animation_studio.schema import AnimationPattern


class PatternSelectionError(RuntimeError):
    pass


class AnimationPatternSelector:
    fallback_pattern_id = "generic_symbol_decomposition"

    def __init__(self) -> None:
        self.patterns = self._load_patterns()

    def select(self, requested_pattern_id: str | None, keywords: list[str] | None = None) -> AnimationPattern:
        if requested_pattern_id and requested_pattern_id in self.patterns:
            return self.patterns[requested_pattern_id]

        keywords = [keyword.lower() for keyword in (keywords or [])]
        for pattern in self.patterns.values():
            suitable_for = [item.lower() for item in pattern.suitable_for]
            if any(keyword in suitable_for for keyword in keywords):
                return pattern

        return self.patterns[self.fallback_pattern_id]

    def get(self, pattern_id: str) -> AnimationPattern:
        if pattern_id not in self.patterns:
            return self.patterns[self.fallback_pattern_id]
        return self.patterns[pattern_id]

    def _load_patterns(self) -> dict[str, AnimationPattern]:
        path = files("math_animation_studio.knowledge").joinpath("animation_patterns.yaml")
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        patterns = {
            item["id"]: AnimationPattern.model_validate(item)
            for item in payload.get("patterns", [])
        }
        if self.fallback_pattern_id not in patterns:
            raise PatternSelectionError(
                f"Fallback animation pattern '{self.fallback_pattern_id}' is missing."
            )
        return patterns
