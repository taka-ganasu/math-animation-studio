from __future__ import annotations

import json

from math_animation_studio.schema import (
    FormulaPlanConsistencyReview,
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
    concept_hint: str | None = None,
    visual_component_catalog: str | None = None,
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

優先したい概念:
{concept_hint or ""}

想定読者:
{audience}

目標動画尺:
{target_duration_seconds}秒

利用可能なアニメーションパターンID:
{pattern_list}

利用可能な視覚部品カタログ:
{visual_component_catalog or ""}

重要な制約:
- 出力は必ずJSONのみ。Markdown fencesや説明文をJSONの外に書かない
- JSON schemaに完全に従う
- 数式はLaTeXとして扱う
- 優先したい概念が指定されている場合は、数式名よりも理解ゴールと優先概念を重視する
- 数式と優先概念がずれる場合は、数式を関連する題材として扱い、優先概念の直感理解に必要な範囲だけ説明する
- 例えば損失関数の式とgradient_descentが同時に与えられたら、損失式そのものではなく、損失地形を下る更新の理解を主題にする
- 説明は日本語
- LLMの役割は、教育設計、シーン分割、記号の意味づけ、宣言的な視覚意図の作成に限定する
- レンダラーの役割は、既存の安全なJinja2/Manimテンプレートへ写像すること
- concrete_valuesには、数値、短い文字列、既存preset名、JSON配列だけを入れる
- 損失関数や描画処理が必要な場合は、Python式ではなく既存preset名を選ぶ
- 初学者が「何を見るべきか」が分かるように、具体例から始める
- 記号の数学的意味と直感的意味を分ける
- recommended_examplesは2〜3個出す。身近さ、動画化しやすさ、誤解の解消しやすさが異なる候補にする
- 1番目のrecommended_examplesをデフォルト採用候補にする
- explanation_stepsは、1番目の候補に自然に合い、かつ他候補を選んでも大きく破綻しない表現にする
- 具体例候補が複数ある場合でも、Pythonコードや新しい描画処理ではなく、既存テンプレートで表現できる値だけをconcrete_valuesへ入れる
- 操作ごとに「何をしているか」と「視覚化ヒント」を出す
- explanation_steps[].planned_componentsには、視覚部品カタログにあるidだけをkindとして入れる
- 1つのexplanation_stepにplanned_componentsを1〜3個入れ、ナレーションで見るべき対象と対応させる
- 理解ゴールが直感・比喩・アナロジーを求めている場合は、category=metaphorの視覚部品を優先的に使う
- 勾配降下法を幾何的・比喩的に説明する場合は、terrain_metaphor, hiker_marker, uphill_arrow, downhill_arrow, footstep_path, formula_bridgeを説明順に応じて使う
- 3D曲面がタイトル、式、字幕と重なりそうな場合は、surface_plotのsurface_y_shift, surface_camera_zoom, title_top_buff, caption_bottom_buffで位置調整する
- 3D曲面の傾斜が急すぎる、または視点が分かりにくい場合は、surface_z_length, surface_camera_phi, surface_camera_thetaで高さ圧縮や俯瞰角度を調整する
- formula_focusを説明する場面では、planned_componentsにformula_focusを入れ、params.formula_focusに強調したいLaTeX部分を入れる
- 視覚部品カタログにない部品名、新しい描画処理、Python式、Manimコードはplanned_componentsに入れない
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


def build_formula_plan_consistency_prompt(
    *,
    formula: str,
    goal: str | None,
    audience: str,
    domain_hint: str | None,
    concept_hint: str | None,
    animation_pattern_ids: list[str],
    plan_json: str,
    visual_component_catalog: str | None = None,
) -> str:
    schema_json = json.dumps(
        FormulaPlanConsistencyReview.model_json_schema(),
        ensure_ascii=False,
        indent=2,
    )
    pattern_list = "\n".join(f"- {pattern_id}" for pattern_id in animation_pattern_ids)
    return f"""あなたは数学アニメーション教材のレビュアーです。

次の入力意図と、生成済みの教材企画JSONが一貫しているかを判定してください。
特に、数式名だけに引っ張られず、理解ゴールと優先したい概念を主題にできているかを確認してください。

対象数式:
{formula}

理解ゴール:
{goal or ""}

分野ヒント:
{domain_hint or ""}

優先したい概念:
{concept_hint or ""}

想定読者:
{audience}

利用可能なアニメーションパターンID:
{pattern_list}

利用可能な視覚部品カタログ:
{visual_component_catalog or ""}

生成済み教材企画JSON:
{plan_json}

判定ルール:
- 出力は必ずJSONのみ
- JSON schemaに完全に従う
- is_consistentは、生成済み教材企画が理解ゴールと優先概念に合っている場合だけtrueにする
- formulaとgoalが別概念を指す場合は、goalと優先概念を主題として扱う
- selected_animation_pattern_idは、利用可能なアニメーションパターンIDから最も適切なものを選ぶ
- explanation_steps[].planned_componentsが、視覚部品カタログ内のidだけを使っているか確認する
- ナレーションで説明する数式パーツとplanned_componentsが対応しているか確認する
- 勾配降下法を幾何的・比喩的に説明しているのに、地形、現在地、上り勾配、下り更新、足跡、式への橋渡しに対応する部品が不足していないか確認する
- 修正が必要な場合は、revision_instructionsに「どの概念を主題にするか」「どの説明を削る/残すか」「どのパターンを使うか」を具体的に書く
- Pythonコード、Manimコード、疑似コード、eval、exec前提の表現は出力しない

JSON schema:
{schema_json}
"""


def build_formula_understanding_revision_prompt(
    *,
    original_prompt: str,
    first_plan_json: str,
    review_json: str,
) -> str:
    return f"""{original_prompt}

上記の入力に対して、最初の教材企画は一貫性レビューで修正が必要と判定されました。

最初の教材企画JSON:
{first_plan_json}

一貫性レビューJSON:
{review_json}

一貫性レビューの revision_instructions に従い、FormulaUnderstandingLLMPlan schemaに合うJSONオブジェクトだけを再出力してください。
理解ゴールと優先したい概念を主題にし、selected_animation_pattern_idは利用可能なアニメーションパターンIDから選んでください。
Markdown fences、説明文、Pythonコード、Manimコード、疑似コード、eval、execは出力しないでください。
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
