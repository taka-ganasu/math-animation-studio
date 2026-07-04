# Visual Component Catalog

このカタログは、Math Animation StudioでLLMが安全に組み合わせられる視覚部品の一覧です。

LLMはここにある部品IDを `explanation_steps[].planned_components[].kind` に入れます。レンダラーは未知の部品を採用せず、既存のJinja2/Manimテンプレートで表現できる部品だけを動画化します。

実行時に使う正本は `src/math_animation_studio/knowledge/visual_components.yaml` です。このMarkdownは人間が読むための説明版です。

## 読み方

- **id**: LLM/Storyboardで使う部品名
- **使う場面**: その部品が自然に効く説明シーン
- **主なparams**: LLMが指定できる値
- **対応テンプレート**: 現時点で動画へ反映されやすいテンプレート

## 数式・説明

### formula_focus

用途:
数式の一部分を拡大・ハイライトして、ナレーションが今どこを説明しているかを示します。

使う場面:
- $\hat{y}_i$ の意味を説明する
- $\log(\hat{y}_i)$ を強調する
- $-\nabla L(\theta_t)$ を大きく見せる

主なparams:
- `formula_focus`: 強調したいLaTeX部分
- `emphasis`: `highlight`, `zoom`, `underline`
- `caption`: 短い補足

対応テンプレート:
`penalty_curve`, `gradient_descent_3d`, `storyboard_only`

### text_caption

用途:
画面上に短い補足を置き、図の読み方を一言で伝えます。

使う場面:
- シーンの切り替わり
- 図の読み方の補助
- 最後のまとめ前の短い確認

長文の説明はナレーションへ寄せ、画面上のテキストは短くします。

## 分類・確率

### model_pipeline

用途:
入力、モデル、logits、softmax、確率分布などの流れを段階的に見せます。

使う場面:
- クロスエントロピーで「予測確率はどこから来るのか」を説明する
- モデル出力と損失関数の接続を見る

主なparams:
- `stages`: 表示する段階名
- `caption`: 流れの要点

### one_hot_vector

用途:
正解クラスだけが1、それ以外が0になるラベルをベクトルとして見せます。

使う場面:
- $y_i$ の意味を説明する
- $\sum_i$ の中で正解クラスだけが残ることを説明する

主なparams:
- `correct_index`: 正解クラスのindex
- `labels`: クラス名

### probability_bars

用途:
クラスごとの予測確率を棒グラフで比較します。

使う場面:
- 良い予測と悪い予測を並べる
- 正解クラスの確率を視覚的に取り出す

主なparams:
- `correct_index`
- `good_distribution`
- `bad_distribution`

### probability_selector

用途:
one-hotラベルが予測分布から正解クラスの確率だけを選ぶ様子を見せます。

使う場面:
- $y_i \log(\hat{y}_i)$ の意味を説明する
- 正解以外の項が0倍されることを示す

主なparams:
- `selected_probability`
- `correct_index`

### negative_log_curve

用途:
$p$ が小さいほど $-\log(p)$ が急に大きくなる罰の形を曲線で見せます。

使う場面:
- なぜlogを使うのかを説明する
- 正解確率が低いと損失が大きいことを見せる

主なparams:
- `x_range`
- `y_range`
- `highlighted_points`

## 最適化・幾何

### surface_plot

用途:
2変数の損失関数を3D曲面として表示します。

使う場面:
- 2つのパラメータと損失の関係を地形として見せる
- 勾配降下法で点が谷へ下る様子を見せる

主なparams:
- `function_preset`: `quadratic_ripple`, `double_well_2d`
- `x_range`
- `y_range`

### contour_map

用途:
2変数の損失を上から見た等高線として表示します。

使う場面:
- 谷が複数ある地形を比較する
- 初期位置で到達先が変わることを説明する

主なparams:
- `function_preset`: `double_well_2d`
- `caption`

### loss_curve

用途:
1変数の損失を横軸 $\theta$、縦軸 $L(\theta)$ の曲線として表示します。

使う場面:
- 谷、山、谷を1枚の曲線で説明する
- 1変数の勾配を傾きとして説明する

主なparams:
- `function_preset`: `double_well_1d`

### gradient_arrow

用途:
勾配または負の勾配方向を矢印で示します。

使う場面:
- 勾配は上り方向であることを説明する
- 更新は $-\nabla L(\theta_t)$ 方向へ進むことを示す

主なparams:
- `direction`: `gradient`, `negative_gradient`
- `learning_rate`

### descent_path

用途:
更新を繰り返した点の軌跡を表示します。

使う場面:
- 勾配降下法の反復を見せる
- 初期位置の違いで到達先が変わることを示す

主なparams:
- `steps`
- `initial_x`
- `initial_y`

### sgd_jitter

用途:
確率的勾配降下法のミニバッチ由来の揺れを点線や小さな横揺れで示します。

使う場面:
- 通常の勾配降下法とSGDの違いを軽く説明する
- ノイズが探索に影響する直感を補足する

主なparams:
- `jitter_strength`

## まとめ

### summary

用途:
最後に式、直感、誤解しやすい点を短く整理します。

使う場面:
- 動画の最後
- 次に学ぶ概念への橋渡し

主なparams:
- `takeaway`
