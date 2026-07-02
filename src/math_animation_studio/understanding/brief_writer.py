from __future__ import annotations

from math_animation_studio.schema import (
    AnimationPattern,
    ConceptClassification,
    ExplanationPlan,
    FormulaAnalysis,
)


class AnimationBriefWriter:
    def write(
        self,
        *,
        formula_analysis: FormulaAnalysis,
        classification: ConceptClassification,
        explanation_plan: ExplanationPlan,
        selected_pattern: AnimationPattern,
    ) -> str:
        rows: list[str] = [
            f"# Animation Brief: {explanation_plan.target_concept}",
            "",
        ]

        if formula_analysis.confidence < 0.5 or classification.confidence < 0.5:
            rows.extend(
                [
                    "> 注意: この数式の概念判定には不確実性があります。分野ヒントや文脈を追加すると、より正確な企画になります。",
                    "",
                ]
            )

        rows.extend(
            [
                "## 一言でいうと",
                "",
                explanation_plan.one_sentence_summary,
                "",
                "## この数式で理解すべきこと",
                "",
                "対象の数式:",
                "",
                f"\\[{explanation_plan.formula}\\]",
                "",
                formula_analysis.short_description,
                "",
                "## 使う題材",
                "",
            ]
        )
        for example in explanation_plan.recommended_examples:
            rows.extend(
                [
                    f"- {example.title}: {example.description}",
                ]
            )

        rows.extend(["", "## なぜこの題材がよいか", ""])
        for example in explanation_plan.recommended_examples:
            rows.append(f"- {example.why_it_works}")

        rows.extend(
            [
                "",
                "## シーン構成",
                "",
            ]
        )
        for index, step in enumerate(explanation_plan.explanation_steps, start=1):
            rows.extend(
                [
                    f"{index}. {step.title}",
                    f"   - 学習ゴール: {step.learning_goal}",
                    f"   - 視覚化: {step.visual_idea}",
                ]
            )
            if step.formula_focus:
                rows.append(f"   - 注目する式: `{step.formula_focus}`")

        rows.extend(
            [
                "",
                "## アニメーション方針",
                "",
                f"- パターン: {selected_pattern.name}",
                f"- 方針: {selected_pattern.description}",
                f"- 比喩: {selected_pattern.visual_metaphor}",
                "",
                "## 誤解しやすいポイント",
                "",
            ]
        )
        for misconception in explanation_plan.misconceptions:
            rows.append(f"- {misconception}")

        rows.extend(["", "## 次に学ぶとよいこと", ""])
        for question in explanation_plan.next_questions_to_study:
            rows.append(f"- {question}")

        rows.append("")
        return "\n".join(rows)
