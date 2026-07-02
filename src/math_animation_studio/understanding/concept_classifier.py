from __future__ import annotations

from math_animation_studio.schema import ConceptClassification

from .sample_plans import sample_classification


class ConceptClassifier:
    def classify(self, *, sample_key: str) -> ConceptClassification:
        return sample_classification(sample_key)
