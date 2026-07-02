from __future__ import annotations

from math_animation_studio.schema import (
    AnimationPattern,
    ConceptClassification,
    ExplanationPlan,
    FormulaAnalysis,
)


def _strip_math_delimiters(value: str) -> str:
    text = value.strip()
    delimiter_pairs = [
        ("$$", "$$"),
        ("$", "$"),
        (r"\[", r"\]"),
        (r"\(", r"\)"),
    ]
    for left, right in delimiter_pairs:
        if text.startswith(left) and text.endswith(right) and len(text) > len(left) + len(right):
            return text[len(left) : -len(right)].strip()
    return text


def _inline_math(value: str) -> str:
    return f"${_strip_math_delimiters(value)}$"


def _block_math(value: str) -> str:
    return f"$$\n{_strip_math_delimiters(value)}\n$$"


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
                _block_math(explanation_plan.formula),
                "",
                formula_analysis.short_description,
                "",
                "## 記号と操作",
                "",
            ]
        )

        if formula_analysis.symbols:
            rows.append("### 記号")
            rows.append("")
            for symbol in formula_analysis.symbols:
                rows.append(f"- {_inline_math(symbol.symbol)}: {symbol.meaning}")
                if symbol.intuition:
                    rows.append(f"  - 直感: {symbol.intuition}")
            rows.append("")

        if formula_analysis.operations:
            rows.append("### 操作")
            rows.append("")
            for operation in formula_analysis.operations:
                rows.append(f"- {_inline_math(operation.operation)}: {operation.meaning}")
                if operation.intuition:
                    rows.append(f"  - 直感: {operation.intuition}")
                if operation.visual_hint:
                    rows.append(f"  - 視覚化: {operation.visual_hint}")
            rows.append("")

        rows.extend(
            [
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
                rows.append(f"   - 注目する式: {_inline_math(step.formula_focus)}")

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
