from __future__ import annotations

import json

from math_animation_studio.schema import (
    FormulaUnderstandingLLMPlan,
    Storyboard,
    VoiceoverScript,
)


def build_concept_planner_prompt(
    *,
    concept: str,
    formula: str | None,
    goal: str | None,
    audience: str,
) -> str:
    schema_json = json.dumps(
        Storyboard.model_json_schema(),
        ensure_ascii=False,
        indent=2,
    )
    return f"""あなたは数学アニメーション教材の設計者です。
以下の数式または概念について、初学者が直感的に理解できる短尺アニメーション教材を設計してください。

対象概念:
{concept}

対象数式:
{formula or ""}

理解ゴール:
{goal or ""}

想定読者:
{audience}

制約:
- 出力は必ずJSONのみ
- 指定されたJSON schemaに従う
- 数式はLaTeXで表現する
- 説明は日本語
- Manimで表現しやすいvisual_objectsに分解する
- LLMの役割は教育設計と宣言的な視覚意図の作成に限定する
- レンダラーは既存の安全なテンプレートだけを使って動画化する
- いきなり抽象説明をせず、具体例から始める
- 誤解しやすいポイントを必ず含める
- 1シーンあたりのナレーションは短くする
- 全体で5〜7シーンにする
- Manimコードそのものは生成しない
- Pythonコード、Manimコード、疑似コード、関数本体、eval/exec前提の表現は出力しない

重視すること:
- 数式を意味のある部品に分解する
- 何を入力し、何が出力されるかを明確にする
- 幾何学的な直感を優先する
- 最後に数式へ戻す

JSON schema:
{schema_json}
"""


def build_retry_prompt(original_prompt: str, raw_response: str, error_message: str) -> str:
    return f"""{original_prompt}

前回の出力はJSON schemaに合いませんでした。
エラー:
{error_message}

前回の出力:
{raw_response}

JSONのみを再出力してください。
"""


def build_formula_understanding_plan_prompt(
    *,
    formula: str,
    goal: str | None,
    audience: str,
    domain_hint: str | None,
    animation_pattern_ids: list[str],
    target_duration_seconds: int,
) -> str:
    schema_json = json.dumps(
        FormulaUnderstandingLLMPlan.model_json_schema(),
        ensure_ascii=False,
        indent=2,
    )
    pattern_list = "\n".join(f"- {pattern_id}" for pattern_id in animation_pattern_ids)
    return f"""あなたは数学・機械学習・統計・最適化の数式を、初学者向け教材へ変換する設計者です。

入力された数式を解析し、教材企画・前提知識・説明順・アニメーション方針をJSONで作ってください。

対象数式:
{formula}

理解ゴール:
{goal or ""}

分野ヒント:
{domain_hint or ""}

想定読者:
{audience}

目標動画尺:
{target_duration_seconds}秒

利用可能なアニメーションパターンID:
{pattern_list}

重要な制約:
- 出力は必ずJSONのみ。Markdown fencesや説明文をJSONの外に書かない
- JSON schemaに完全に従う
- 数式はLaTeXとして扱う
- 説明は日本語
- LLMの役割は、教育設計、シーン分割、記号の意味づけ、宣言的な視覚意図の作成に限定する
- レンダラーの役割は、既存の安全なJinja2/Manimテンプレートへ写像すること
- concrete_valuesには、数値、短い文字列、既存preset名、JSON配列だけを入れる
- 損失関数や描画処理が必要な場合は、Python式ではなく既存preset名を選ぶ
- 初学者が「何を見るべきか」が分かるように、具体例から始める
- 記号の数学的意味と直感的意味を分ける
- 操作ごとに「何をしているか」と「視覚化ヒント」を出す
- prerequisite_mapでは、理解に必要な前提知識と詰まりやすい点を出す
- explanation_planは5〜7 stepsにする
- explanation_stepsのexplanationは、ナレーション素材として短く具体的にする
- selected_animation_pattern_idは、利用可能なアニメーションパターンIDから選ぶ
- Pythonコード、Manimコード、疑似コード、関数本体、eval、exec前提の表現は出力しない
- generation_boundary.code_generation_allowedは必ずfalseにする
- 不確実な場合はformula_analysis.ambiguity_notesに明記し、confidenceを低めにする

JSON schema:
{schema_json}
"""


def build_voiceover_script_prompt(
    *,
    storyboard: Storyboard,
    target_duration_seconds: int,
    audience: str,
    goal: str | None,
) -> str:
    schema_json = json.dumps(
        VoiceoverScript.model_json_schema(),
        ensure_ascii=False,
        indent=2,
    )
    storyboard_json = storyboard.model_dump_json(indent=2)
    return f"""あなたは数学アニメーション教材のナレーション作家です。

次のStoryboardに合わせて、初学者が聞いて理解できる日本語ナレーション台本を作ってください。

想定読者:
{audience}

理解ゴール:
{goal or ""}

目標動画尺:
{target_duration_seconds}秒

Storyboard:
{storyboard_json}

重要な制約:
- 出力は必ずJSONのみ
- scriptは読み上げ用の自然な日本語にする
- 難しい語は短く噛み砕く
- 1文を短くする
- 視聴者が今どこを見ればよいかを言葉で誘導する
- 目標動画尺の8割〜9割程度で読み終わる分量にする
- 数式をそのまま長く読み上げすぎない
- Markdown、箇条書き、コード、記号だけの説明は避ける

JSON schema:
{schema_json}
"""
