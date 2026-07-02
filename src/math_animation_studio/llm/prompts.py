from __future__ import annotations

import json

from math_animation_studio.schema import Storyboard


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
- いきなり抽象説明をせず、具体例から始める
- 誤解しやすいポイントを必ず含める
- 1シーンあたりのナレーションは短くする
- 全体で5〜7シーンにする
- Manimコードそのものは生成しない
- Pythonコード、疑似コード、eval/exec前提の表現は出力しない

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
