# Visual Component Catalog

このカタログは、Math Animation StudioでLLMが安全に組み合わせられる視覚部品の一覧です。

LLMはここにある部品IDを `explanation_steps[].planned_components[].kind` に入れます。レンダラーは未知の部品を採用せず、既存のJinja2/Manimテンプレートで表現できる部品だけを動画化します。

実行時に使う正本は `src/math_animation_studio/knowledge/visual_components.yaml` です。このMarkdownは人間が読むための説明版です。

Storyboardの章立ては `docs/storyboard-dsl.md` を参照してください。現在のDSLと部品一覧はCLIでも確認できます。

```bash
math-anim catalog
math-anim catalog --format json
```

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

### chain_rule

用途:
合成関数やバックプロパゲーションで、途中の変化率を掛けて端から端の変化率にする流れを示します。

使う場面:
- $\frac{dy}{dx}=\frac{dy}{du}\frac{du}{dx}$ を説明する
- $W \rightarrow \hat{y} \rightarrow L$ の影響をつなぐ
- 連鎖律が2階微分ではないことを整理する

主なparams:
- `formula_focus`: 強調する連鎖律の式

対応テンプレート:
`backpropagation`, `chain_rule`, `storyboard_only`

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

対応テンプレート:
`penalty_curve`, `fully_connected_network`, `storyboard_only`

### one_hot_vector

用途:
正解クラスだけが1、それ以外が0になるラベルをベクトルとして見せます。

使う場面:
- $y_i$ の意味を説明する
- $\sum_i$ の中で正解クラスだけが残ることを説明する

主なparams:
- `correct_index`: 正解クラスのindex
- `labels`: クラス名

対応テンプレート:
`penalty_curve`, `fully_connected_network`, `storyboard_only`

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

対応テンプレート:
`penalty_curve`, `fully_connected_network`, `storyboard_only`

### probability_selector

用途:
one-hotラベルが予測分布から正解クラスの確率だけを選ぶ様子を見せます。

使う場面:
- $y_i \log(\hat{y}_i)$ の意味を説明する
- 正解以外の項が0倍されることを示す

主なparams:
- `selected_probability`
- `correct_index`

対応テンプレート:
`penalty_curve`, `fully_connected_network`, `storyboard_only`

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

対応テンプレート:
`penalty_curve`, `fully_connected_network`, `storyboard_only`

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
- `surface_y_shift`: 3Dグラフ全体を下方向へずらす量
- `surface_z_length`: 3Dグラフの高さ方向のスケール
- `surface_camera_zoom`: 3Dカメラの引き具合
- `surface_camera_phi`: 3Dカメラの縦方向角度
- `surface_camera_theta`: 3Dカメラの横方向角度
- `title_top_buff`: タイトルと画面上端の余白
- `caption_bottom_buff`: 字幕やまとめ文と画面下端の余白

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

## ニューラルネットワーク

### perceptron_node

用途:
入力、重み付き和、活性化、出力をノードとして並べ、単純パーセプトロンの構造を見せます。

使う場面:
- 単純パーセプトロンの順伝播を説明する
- 入力、重み、バイアス、出力の関係を図にする

主なparams:
- `input_labels`
- `output_label`

### weighted_connection

用途:
入力からニューロンへ向かう矢印に重みを添え、各入力がどれだけ効くかを示します。

使う場面:
- $w_i x_i$ の意味を説明する
- 全結合層へ拡張する前に「接続には重みがある」と見せる

主なparams:
- `weights`

### weighted_sum

用途:
$z = w_1x_1 + w_2x_2 + b$ のように、入力を重み付きで足し合わせる計算を見せます。

使う場面:
- ニューロンが最初に線形結合を作ることを説明する
- バイアスの役割を説明する

主なparams:
- `formula_focus`

### activation_function

用途:
重み付き和 $z$ を、0/1や確率のような出力へ変換する関数として見せます。

使う場面:
- step関数やsigmoidの役割を説明する
- 線形結合だけでは分類結果にならないことを示す

主なparams:
- `activation`: `step`, `sigmoid`

### decision_boundary

用途:
$w_1x_1 + w_2x_2 + b = 0$ を2D平面上の境界線として表示します。

使う場面:
- 単純パーセプトロンが2クラスを直線で分けることを説明する
- 重みとバイアスが境界線の向きと位置を決めることを示す

主なparams:
- `weights`
- `bias`

### forward_pass

用途:
入力から重み付き和、活性化、出力へ値が流れる順伝播をハイライトします。

使う場面:
- 学習ではなく推論時の計算の流れを説明する
- 全結合ニューラルネットワークへ拡張する前に1ニューロンの流れを固定する

主なparams:
- `input_values`

### dense_layer

用途:
複数のニューロンを縦に並べ、入力層、隠れ層、出力層として表示します。

使う場面:
- 全結合ニューラルネットワークの層構造を説明する
- 単純パーセプトロンが層として複数並ぶことを見せる

主なparams:
- `layer_sizes`

対応テンプレート:
`fully_connected_network`, `backpropagation`, `storyboard_only`

### fully_connected_edges

用途:
前の層のすべてのノードから、次の層のすべてのノードへ接続線を引きます。

使う場面:
- 「全結合」が全ノード間の接続であることを説明する
- $W$ が多数の接続重みをまとめた行列であることを見せる

主なparams:
- `connection_opacity`

対応テンプレート:
`fully_connected_network`, `backpropagation`, `storyboard_only`

### layer_activation

用途:
$h = \sigma(Wx + b)$ のように、線形和を活性化関数へ通して次の層へ渡す値に変換することを見せます。

使う場面:
- 隠れ層の出力がどのように作られるかを説明する
- 線形変換だけではなく非線形変換が入ることを見せる

主なparams:
- `activation`: `relu`, `sigmoid`, `tanh`

対応テンプレート:
`fully_connected_network`, `backpropagation`, `storyboard_only`

### feature_axis_mixing

用途:
線形変換で元の特徴を混ぜ直し、次の層が見やすい特徴軸へ変える様子を見せます。

使う場面:
- $Wx+b$ が特徴の見方を変えることを説明する
- $W$ を新しい特徴軸の作り方として見せる

主なparams:
- `input_feature_labels`
- `transformed_feature_labels`

対応テンプレート:
`neural_network_transform`, `storyboard_only`

### activation_gate

用途:
ReLUなどの活性化関数が、値を通す、切る、曲げる効果を持つことを見せます。

使う場面:
- 非線形変換がなぜ必要か説明する
- ReLUが負の値を0にすることを見せる

主なparams:
- `activation`: `relu`, `sigmoid`, `tanh`

対応テンプレート:
`neural_network_transform`, `storyboard_only`

### activation_curve

用途:
ReLU、sigmoid、tanhなどの活性化関数を、横軸 $z$、縦軸 $a$ の曲線として表示します。

使う場面:
- 活性化関数の入力と出力の対応を説明する
- ReLU、sigmoid、tanhの違いを曲線で比較する

主なparams:
- `activation`: `relu`, `sigmoid`, `tanh`

対応テンプレート:
`activation_functions`, `storyboard_only`

### activation_comparison

用途:
活性化関数の役割や使い分けを、カードや簡易フローで比較します。

使う場面:
- 隠れ層と出力層で活性化関数の選び方が違うことを説明する
- ReLU、sigmoid、softmax、Adamのカテゴリ違いを整理する

主なparams:
- `comparison_mode`: `linear_vs_nonlinear`, `hidden_vs_output`, `optimizer_boundary`

対応テンプレート:
`activation_functions`, `storyboard_only`

### softmax_probability_flow

用途:
クラスごとのlogitsをsoftmaxに通し、合計1の確率バーへ変換します。

使う場面:
- softmaxが複数クラス分類の出力で使われることを説明する
- logitsと確率分布の違いを見せる

主なparams:
- `class_labels`
- `logits`
- `probabilities`

対応テンプレート:
`activation_functions`, `storyboard_only`

### representation_space

用途:
入力が層を通って、予測しやすい中間表現へ変わる様子を2D点群で見せます。

使う場面:
- 中間表現を説明する
- データを解きやすい座標系へ変換する直感を見せる

主なparams:
- `feature_labels`

対応テンプレート:
`neural_network_transform`, `storyboard_only`

### softmax_output

用途:
出力層のスコアをクラスごとの確率分布へ変換して表示します。

使う場面:
- 分類タスクの出力を確率として読む
- クロスエントロピーへ接続する前段を説明する

主なparams:
- `class_labels`

対応テンプレート:
`fully_connected_network`, `backpropagation`, `storyboard_only`

### loss_gradient

用途:
損失から出力層の誤差信号を作る様子を表示します。

使う場面:
- softmaxとクロスエントロピーの後に、予測と正解の差を扱う
- 逆伝播の最初の信号を説明する

主なparams:
- `output_probabilities`
- `correct_index`

対応テンプレート:
`backpropagation`, `storyboard_only`

### backward_pass

用途:
出力層から隠れ層、入力側へ誤差信号が戻る流れを矢印で見せます。

使う場面:
- 逆伝播の向きを説明する
- 順伝播との違いを見せる

主なparams:
- `layer_sizes`

対応テンプレート:
`backpropagation`, `storyboard_only`

### chain_rule

用途:
遠い重みの影響を途中の微分の積としてつなげることを示します。

使う場面:
- 逆伝播が単なる逆再生ではないことを説明する
- 重み、活性化、誤差信号の積を見せる

主なparams:
- `formula_focus`

対応テンプレート:
`backpropagation`, `storyboard_only`

### error_attribution

用途:
隠れ層の各ノードや接続へ誤差信号が割り当てられる様子を表示します。

使う場面:
- 隠れ層にも損失への責任が配られることを説明する
- 各重みの勾配へ接続する

主なparams:
- `hidden_deltas`

対応テンプレート:
`backpropagation`, `storyboard_only`

### weight_update

用途:
勾配を使って重みを更新する式と数値例を表示します。

使う場面:
- 逆伝播で計算した勾配を学習へつなげる
- 勾配降下法との接続を説明する

主なparams:
- `learning_rate`
- `selected_weight_gradient`

対応テンプレート:
`backpropagation`, `storyboard_only`

## メタファー

### terrain_metaphor

用途:
損失関数を山や谷の地形として読み替え、高さが損失であることを見せます。

使う場面:
- 勾配降下法を山を下る比喩で説明する
- 損失が高さとして読めることを導入する

主なparams:
- `metaphor_label`: 地形としての読み替えを示す短いラベル

### hiker_marker

用途:
現在のパラメータ位置を、山道上の現在地マーカーとして見せます。

使う場面:
- パラメータを抽象的な数値ではなく現在地として説明する
- 反復更新で現在地が動くことを見せる

主なparams:
- `label`: 現在地マーカーに添える短いラベル

### footstep_path

用途:
更新の軌跡を足跡や一歩ずつ進む印として残します。

使う場面:
- 勾配降下法が一回で答えに飛ぶのではなく、一歩ずつ進むことを示す
- 学習率を一歩の大きさとして説明する

主なparams:
- `footprint_every`: 何ステップごとに足跡を置くか

### uphill_arrow

用途:
勾配そのものが損失の増える上り方向を指すことを示します。

使う場面:
- 勾配は下り方向ではないという誤解を解消する
- gradientとnegative gradientを対比する

主なparams:
- `label`: 上り方向を示す短いラベル

### downhill_arrow

用途:
更新で実際に進む負の勾配方向を、下り方向として示します。

使う場面:
- 勾配の逆向きへ進むことを直感化する
- 式のマイナス記号と下り方向を対応づける

主なparams:
- `label`: 下り方向を示す短いラベル

### formula_bridge

用途:
比喩で見せた動きから、対応する数式パーツへ戻します。

使う場面:
- 山を下る比喩を更新式の $-\eta\nabla L$ に対応づける
- 直感説明のあとに数式を再確認する

主なparams:
- `formula_focus`: 比喩から戻したいLaTeX部分
- `caption`: 比喩と数式をつなぐ短い説明

## まとめ

### summary

用途:
最後に式、直感、誤解しやすい点を短く整理します。

使う場面:
- 動画の最後
- 次に学ぶ概念への橋渡し

主なparams:
- `takeaway`
