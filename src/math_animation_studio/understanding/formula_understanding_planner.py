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
from .llm_formula_planner import LLMFormulaUnderstandingPlanner
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
        concept_hint: str | None = None,
        to_storyboard: bool = True,
        target_duration_seconds: int = 30,
    ) -> PlanArtifacts:
        if self.no_llm:
            key = detect_sample_key(
                " ".join(
                    part
                    for part in [
                        f"concept_hint:{concept_hint}" if concept_hint else "",
                        formula,
                        goal or "",
                        domain_hint or "",
                    ]
                    if part
                )
            )
            formula_analysis = self.formula_analyzer.analyze(formula=formula, sample_key=key)
            classification = self.concept_classifier.classify(sample_key=key)
            prerequisite_map = self.prerequisite_mapper.map(sample_key=key)
            explanation_plan = self.explanation_plan_generator.generate(
                formula=formula,
                sample_key=key,
                audience=audience,
            )
            llm_used = False
        else:
            llm_plan = LLMFormulaUnderstandingPlanner().plan(
                formula=formula,
                goal=goal,
                audience=audience,
                domain_hint=domain_hint,
                concept_hint=concept_hint,
                animation_pattern_ids=list(self.pattern_selector.patterns.keys()),
                target_duration_seconds=target_duration_seconds,
            )
            formula_analysis = llm_plan.formula_analysis
            classification = llm_plan.concept_classification
            prerequisite_map = llm_plan.prerequisite_map
            explanation_plan = llm_plan.explanation_plan
            llm_used = True

        requested_animation_family = _coerce_animation_family(
            requested=(
                explanation_plan.selected_animation_pattern_id
                if llm_used
                else classification.recommended_animation_family
            ),
            formula=formula,
            formula_analysis=formula_analysis,
            classification=classification,
            explanation_plan=explanation_plan,
            apply_formula_heuristics=not llm_used and concept_hint is None,
        )
        selected_pattern = self.pattern_selector.select(
            requested_animation_family,
            keywords=[classification.primary_concept, classification.primary_domain],
        )
        classification.recommended_animation_family = selected_pattern.id
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
            llm_used=llm_used,
        )


def _coerce_animation_family(
    *,
    requested: str,
    formula: str,
    formula_analysis: FormulaAnalysis,
    classification: ConceptClassification,
    explanation_plan: ExplanationPlan,
    apply_formula_heuristics: bool = True,
) -> str:
    if not apply_formula_heuristics:
        return requested

    text = " ".join(
        [
            requested,
            formula,
            formula_analysis.detected_name or "",
            formula_analysis.normalized_formula_latex,
            classification.primary_concept,
            classification.primary_domain,
            explanation_plan.target_concept,
        ]
    ).lower()

    explicitly_chain_rule = (
        requested == "chain_rule_flow"
        or _normalize_concept_name(classification.primary_concept) == "chain_rule"
        or _normalize_concept_name(explanation_plan.target_concept) == "chain_rule"
    )
    if explicitly_chain_rule:
        return "chain_rule_flow"

    explicitly_neural_network_transform = (
        requested == "neural_network_transform_flow"
        or _normalize_concept_name(classification.primary_concept)
        == "neural_network_transform"
        or _normalize_concept_name(explanation_plan.target_concept)
        == "neural_network_transform"
    )
    if explicitly_neural_network_transform:
        return "neural_network_transform_flow"

    if (
        "backpropagation" in text
        or "backprop" in text
        or "誤差逆伝播" in text
        or "逆伝播" in text
        or "\\delta" in text
        or "δ" in text
        or "\\partial l" in text
        or "∂l" in text
    ):
        return "backpropagation_chain_rule"
    if "chain_rule" in text or "chain rule" in text or "連鎖律" in text:
        return "chain_rule_flow"
    if (
        "neural_network_transform" in text
        or "nn_transform" in text
        or "representation_learning" in text
        or "中間表現" in text
        or "表現学習" in text
        or "線形変換" in text and "非線形変換" in text
    ):
        return "neural_network_transform_flow"
    if (
        "cross_entropy" in text
        or "cross entropy" in text
        or "クロスエントロピー" in text
        or ("\\log" in text and "\\sum" in text)
        or ("log" in text and "sum" in text)
    ):
        return "penalty_curve"
    if "gradient_descent" in text or "勾配降下" in text or "\\nabla" in text or "∇" in text:
        return "trajectory_on_surface"
    if (
        "fully_connected" in text
        or "fully connected" in text
        or "dense" in text
        or "multilayer" in text
        or "neural_network" in text
        or "neural network" in text
        or "全結合" in text
        or "ニューラルネットワーク" in text
        or "多層" in text
        or "softmax" in text and "\\sigma" in text
    ):
        return "fully_connected_forward_pass"
    if (
        "perceptron" in text
        or "パーセプトロン" in text
        or "単純パーセプトロン" in text
        or "decision boundary" in text
        or "決定境界" in text
        or "activation" in text
        or "活性化" in text
        or "w^t" in text
        or "w_1" in text
    ):
        return "perceptron_decision_boundary"
    if "attention" in text or ("q" in formula.lower() and "k" in formula.lower() and "v" in formula.lower()):
        return "matrix_similarity_heatmap"
    return requested


def _normalize_concept_name(value: str) -> str:
    return value.lower().replace("-", "_").replace(" ", "_")
