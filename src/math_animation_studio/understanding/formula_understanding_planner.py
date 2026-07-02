from __future__ import annotations

from dataclasses import dataclass

from math_animation_studio.schema import (
    AnimationPattern,
    ConceptClassification,
    ExplanationPlan,
    FormulaAnalysis,
    PrerequisiteMap,
    Storyboard,
)

from .brief_writer import AnimationBriefWriter
from .concept_classifier import ConceptClassifier
from .explanation_plan_generator import ExplanationPlanGenerator
from .formula_analyzer import FormulaAnalyzer
from .pattern_selector import AnimationPatternSelector
from .prerequisite_mapper import PrerequisiteMapper
from .sample_plans import detect_sample_key
from .storyboard_adapter import StoryboardAdapter


class FormulaUnderstandingPlannerError(RuntimeError):
    pass


@dataclass(frozen=True)
class PlanArtifacts:
    formula_analysis: FormulaAnalysis
    concept_classification: ConceptClassification
    prerequisite_map: PrerequisiteMap
    explanation_plan: ExplanationPlan
    selected_pattern: AnimationPattern
    animation_brief: str
    storyboard: Storyboard | None
    llm_used: bool


class FormulaUnderstandingPlanner:
    def __init__(self, *, no_llm: bool = False) -> None:
        self.no_llm = no_llm
        self.formula_analyzer = FormulaAnalyzer()
        self.concept_classifier = ConceptClassifier()
        self.prerequisite_mapper = PrerequisiteMapper()
        self.explanation_plan_generator = ExplanationPlanGenerator()
        self.pattern_selector = AnimationPatternSelector()
        self.brief_writer = AnimationBriefWriter()
        self.storyboard_adapter = StoryboardAdapter()

    def plan(
        self,
        *,
        formula: str,
        goal: str | None,
        audience: str,
        domain_hint: str | None = None,
        to_storyboard: bool = True,
    ) -> PlanArtifacts:
        if not self.no_llm:
            raise FormulaUnderstandingPlannerError(
                "LLM-backed plan mode is not implemented yet. Use --no-llm."
            )

        key = detect_sample_key(" ".join(part for part in [formula, goal or "", domain_hint or ""] if part))
        formula_analysis = self.formula_analyzer.analyze(formula=formula, sample_key=key)
        classification = self.concept_classifier.classify(sample_key=key)
        prerequisite_map = self.prerequisite_mapper.map(sample_key=key)
        explanation_plan = self.explanation_plan_generator.generate(
            formula=formula,
            sample_key=key,
            audience=audience,
        )
        selected_pattern = self.pattern_selector.select(
            classification.recommended_animation_family,
            keywords=[classification.primary_concept, classification.primary_domain],
        )
        explanation_plan.selected_animation_pattern_id = selected_pattern.id
        animation_brief = self.brief_writer.write(
            formula_analysis=formula_analysis,
            classification=classification,
            explanation_plan=explanation_plan,
            selected_pattern=selected_pattern,
        )
        storyboard = (
            self.storyboard_adapter.convert(
                formula_analysis=formula_analysis,
                explanation_plan=explanation_plan,
            )
            if to_storyboard
            else None
        )

        return PlanArtifacts(
            formula_analysis=formula_analysis,
            concept_classification=classification,
            prerequisite_map=prerequisite_map,
            explanation_plan=explanation_plan,
            selected_pattern=selected_pattern,
            animation_brief=animation_brief,
            storyboard=storyboard,
            llm_used=False,
        )
