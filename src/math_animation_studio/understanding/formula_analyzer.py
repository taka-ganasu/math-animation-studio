from __future__ import annotations

from math_animation_studio.schema import FormulaAnalysis

from .sample_plans import sample_formula_analysis


class FormulaAnalyzer:
    def analyze(self, *, formula: str, sample_key: str) -> FormulaAnalysis:
        return sample_formula_analysis(formula, sample_key)
