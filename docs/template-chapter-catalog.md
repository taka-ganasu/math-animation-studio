# Template Chapter Catalog

このカタログは、各Manimテンプレートがどの章構成で動画を作るかをまとめたものです。

正本は `src/math_animation_studio/knowledge/template_chapters.yaml` です。このMarkdownは人間が読むための説明版です。

## 目的

Math Animation Studio は、LLMに自由なManimコードを書かせず、既存テンプレートへ安全に写像します。そのため、LLMや利用者が「このテンプレートでは何をどの順番で表現できるか」を事前に知る必要があります。

テンプレート章カタログは、次を明示します。

- `role`: `title_intro` / `formula_structure` / `concrete_example` / `visualization` / `summary`
- `segment_ids`: 実際のタイムライン区間
- `component_kinds`: その章で使う視覚部品
- `intent`: その章で理解させたいこと

## CLI

```bash
math-anim catalog
math-anim catalog --format json
```

`catalog` のMarkdown出力には `Template Chapters` セクションが含まれます。JSON出力では `template_chapters` として取得できます。

## 現在の対象

- `penalty_curve`: クロスエントロピーを、式分解、モデル出力、one-hot、正解確率、`-log` の罰として説明する。
- `gradient_descent_3d`: 損失地形、現在地、勾配、下る方向、更新式の対応を説明する。
- `perceptron`: 重み付き和、活性化、2D決定境界として説明する。
- `fully_connected_network`: 層、全接続、順伝播、softmax、損失への接続として説明する。
- `backpropagation`: 損失から誤差信号を戻し、隠れ層の責任配分と重み更新へつなげる。
- `chain_rule`: 合成関数の途中の変化率を掛けて、端から端の変化率を求める。
- `neural_network_transform`: 線形変換、非線形変換、中間表現を、特徴空間の変換として説明する。
- `activation_functions`: ReLU、sigmoid、tanh、softmaxを比較し、隠れ層と出力層での使い分けを説明する。

## 運用ルール

テンプレートを追加・変更したら、`template_chapters.yaml` も更新します。

章の `segment_ids` は `src/math_animation_studio/timing.py` のタイムライン定義と一致している必要があります。テストでこの対応を検証します。

`component_kinds` は `src/math_animation_studio/knowledge/visual_components.yaml` にある既知の視覚部品IDだけを使います。
