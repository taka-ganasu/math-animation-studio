from __future__ import annotations

from math_animation_studio.schema import (
    AnimationPattern,
    ConceptClassification,
    ExplanationPlan,
    FormulaAnalysis,
)

from .storyboard_dsl import FORMULA_FIRST_ROLE_ORDER, infer_scene_role


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
                "## Storyboard DSL",
                "",
                f"- フロー: `formula_first`",
                "- 章立て: "
                + " -> ".join(f"`{role}`" for role in FORMULA_FIRST_ROLE_ORDER),
                "- 方針: タイトル導入、式の構造、具体例、実際の可視化、まとめに分けて設計する。",
                "",
                "## シーン構成",
                "",
            ]
        )
        step_count = len(explanation_plan.explanation_steps)
        for index, step in enumerate(explanation_plan.explanation_steps, start=1):
            scene_role = infer_scene_role(
                step,
                step_index=index,
                step_count=step_count,
                animation_pattern_id=explanation_plan.selected_animation_pattern_id,
            )
            rows.extend(
                [
                    f"{index}. {step.title}",
                    f"   - 役割: `{scene_role}`",
                    f"   - 学習ゴール: {step.learning_goal}",
                    f"   - 視覚化: {step.visual_idea}",
                ]
            )
            if step.formula_focus:
                rows.append(f"   - 注目する式: {_inline_math(step.formula_focus)}")

        rows.extend(["", "## アニメーション方針", ""])
        rows.extend(_animation_policy_rows(explanation_plan, selected_pattern))
        rows.extend(["", "## 誤解しやすいポイント", ""])
        for misconception in explanation_plan.misconceptions:
            rows.append(f"- {misconception}")

        rows.extend(["", "## 次に学ぶとよいこと", ""])
        for question in explanation_plan.next_questions_to_study:
            rows.append(f"- {question}")

        rows.append("")
        return "\n".join(rows)


def _animation_policy_rows(
    explanation_plan: ExplanationPlan,
    selected_pattern: AnimationPattern,
) -> list[str]:
    preset = ""
    if explanation_plan.recommended_examples:
        preset = str(
            explanation_plan.recommended_examples[0].concrete_values.get(
                "function_preset",
                "",
            )
        )
    if preset == "double_well_1d":
        return [
            "- パターン: Trajectory on 1D Loss Curve",
            "- 方針: 1変数の損失曲線を2Dグラフとして描き、点が左右どちらの谷へ下るかを比較する。",
            "- 比喩: 曲線上の現在地から、傾きだけを見て近くの谷へ下る。",
        ]
    if preset == "double_well_2d":
        return [
            "- パターン: Trajectory on 2D Contour Map",
            "- 方針: 2変数の損失を等高線として描き、初期位置によって到達する谷が変わる様子を比較する。",
            "- 比喩: 地図上の現在地から、局所的な斜面だけを頼りに谷へ下る。",
        ]
    if selected_pattern.id == "perceptron_decision_boundary":
        return [
            "- パターン: Perceptron Decision Boundary",
            "- 方針: 1つのニューロンで入力、重み付き和、活性化、出力の順伝播を見せ、2D平面の決定境界へ接続する。",
            "- 比喩: 判断材料を重み付きで足し合わせ、境界線のどちら側かで0/1を決める。",
        ]
    return [
        f"- パターン: {selected_pattern.name}",
        f"- 方針: {selected_pattern.description}",
        f"- 比喩: {selected_pattern.visual_metaphor}",
    ]
