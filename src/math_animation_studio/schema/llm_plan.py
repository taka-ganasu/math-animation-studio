from __future__ import annotations

from .concept_classification import ConceptClassification
from .explanation_plan import ExplanationPlan
from .formula_analysis import FormulaAnalysis
from .prerequisite_map import PrerequisiteMap
from .storyboard import StrictModel


class FormulaUnderstandingLLMPlan(StrictModel):
    formula_analysis: FormulaAnalysis
    concept_classification: ConceptClassification
    prerequisite_map: PrerequisiteMap
    explanation_plan: ExplanationPlan
