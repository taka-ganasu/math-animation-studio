# math-animation-studio

数式・モデルを、Storyboard JSON と Jinja2 ベースの Manim テンプレートから短尺動画にするMVPです。

Phase 1〜5 の実装範囲では、LLMなしの固定サンプル `--no-llm` で勾配降下法の動画生成まで動かします。Manimコードは自由生成せず、`gradient_descent_3d.py.j2` テンプレートから生成します。任意のPythonコード実行、`eval`、`exec` は使いません。

## セットアップ

macOSの場合、先にManimの外部依存を入れます。

```bash
brew install ffmpeg
brew install --cask mactex
```

Python環境を作成します。

```bash
cd math-animation-studio
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools
python -m pip install -e ".[render,dev]"
```

ネットワーク制限のある環境でbuild isolationが失敗する場合は、依存を入れたうえで `python -m pip install -e ".[render,dev]" --no-build-isolation` を使います。

CLIヘルプを確認します。

```bash
math-anim --help
math-anim generate --help
math-anim catalog
```

## LLMなしでStoryboardとManimコードを生成

```bash
math-anim generate \
  --concept gradient_descent \
  --formula "\\theta_{t+1} = \\theta_t - \\eta \\nabla L(\\theta_t)" \
  --goal "勾配降下法を3D曲面上の点の移動として直感的に理解する" \
  --output-dir outputs/gradient_descent_demo \
  --no-llm \
  --no-render
```

生成物:

- `storyboard.json`
- `symbols.md`
- `narration.md`
- `manim_scene.py`
- `metadata.json`

## 動画まで生成

Manim、ffmpeg、LaTeXが使える環境で以下を実行します。

```bash
math-anim generate \
  --concept gradient_descent \
  --formula "\\theta_{t+1} = \\theta_t - \\eta \\nabla L(\\theta_t)" \
  --goal "勾配降下法を3D曲面上の点の移動として直感的に理解する" \
  --output-dir outputs/gradient_descent_demo \
  --no-llm
```

成功すると `outputs/gradient_descent_demo/video.mp4` が作成されます。Manimが未インストールの場合やレンダリングに失敗した場合でも、`render.log` にログが保存されます。

## 数式の教材企画を生成

MVP2では `plan` コマンドで、数式を理解するための教材企画を生成できます。最初はLLMなしの固定サンプルで動作します。

```bash
math-anim plan \
  --formula "L = - \\sum_i y_i \\log(\\hat{y}_i)" \
  --goal "クロスエントロピー損失を直感的に理解したい" \
  --output-dir outputs/cross_entropy_plan \
  --no-llm
```

生成物:

- `formula_analysis.json`
- `concept_classification.json`
- `prerequisite_map.json`
- `explanation_plan.json`
- `animation_brief.md`
- `storyboard.json`
- `metadata.json`

`--no-llm` の固定サンプルは、Cross Entropy、Gradient Descent、Attention、Perceptronに対応しています。`storyboard.json` は既存のStoryboard schemaでparseできる形式です。

Cross Entropy と Gradient Descent は、`--render` を付けるとManim動画まで生成できます。

入力数式と見たい概念がずれる場合は `--concept-hint` で主題のヒントを渡せます。LLMモードでは、初回プラン生成後にLLMが数式・ゴール・ヒントの一貫性をレビューし、必要ならプランを再生成します。たとえば数式は損失関数でも、「その損失地形を勾配降下法でどう下るか」を見たい場合は `--concept-hint gradient_descent` を付けます。

```bash
math-anim plan \
  --formula "L = - \\sum_i y_i \\log(\\hat{y}_i)" \
  --goal "クロスエントロピー損失を直感的に理解したい" \
  --output-dir outputs/cross_entropy_plan \
  --no-llm \
  --duration 30 \
  --render
```

成功すると `outputs/cross_entropy_plan/video.mp4` が作成されます。

```bash
math-anim plan \
  --formula "L = - \\sum_i y_i \\log(\\hat{y}_i)" \
  --goal "2次元で谷が2箇所ある時に勾配降下法がどう判断するか知りたい" \
  --concept-hint gradient_descent \
  --audience high_school_math \
  --domain-hint optimization \
  --duration 52 \
  --output-dir outputs/gradient_descent_from_loss \
  --render \
  --voiceover
```

具体例を自動提案させたあと、人間が確認してから進めたい場合は `--interactive-example` を付けます。候補が複数ある場合は番号で選び、Enterで採用します。`n` を入力するとタイトル、説明、採用理由を書き換えられます。LLMモードでは2〜3個の候補を出すように促します。`--no-llm` では固定候補で同じUIを試せます。

```bash
math-anim plan \
  --formula "L = - \\sum_i y_i \\log(\\hat{y}_i)" \
  --goal "クロスエントロピー損失を直感的に理解したい" \
  --output-dir outputs/cross_entropy_plan \
  --no-llm \
  --interactive-example
```

macOSでは、`--voiceover` を付けると `say` コマンドで日本語ナレーションを生成し、`ffmpeg` で動画に合成します。

```bash
math-anim plan \
  --formula "L = - \\sum_i y_i \\log(\\hat{y}_i)" \
  --goal "クロスエントロピー損失を直感的に理解したい" \
  --output-dir outputs/cross_entropy_plan \
  --no-llm \
  --duration 30 \
  --render \
  --voiceover
```

生成物:

- `narration.md`
- `narration.aiff`
- `voiceover.log`
- `video_with_voice.mp4`

`--duration` は対応済みテンプレートの目標動画尺です。Cross Entropyでは、25秒以上を指定するとナレーション台本も少し噛み砕いた版になります。

### 勾配降下法: 2Dで谷が2つある例

`--no-llm` でも、ゴール文に「2次元」「谷が2箇所」「局所最小」などを含めると、2D等高線で浅い局所最小と深い大域最小を比べる動画を生成できます。

```bash
math-anim plan \
  --formula "\\theta_{t+1} = \\theta_t - \\eta \\nabla L(\\theta_t)" \
  --goal "2次元で谷が2箇所ある時に勾配降下法がどう判断するか知りたい" \
  --audience high_school_math \
  --domain-hint optimization \
  --duration 52 \
  --output-dir outputs/gradient_descent_double_well \
  --no-llm \
  --render \
  --voiceover
```

成功すると、`outputs/gradient_descent_double_well/video_with_voice.mp4` が作成されます。

### 勾配降下法: 1D損失曲線で谷が2つある例

ゴール文に「1変数」「損失曲線」「谷→山→谷」などを含めると、1Dの損失曲線上で、現在地の傾きだけを見て左右どちらの谷へ下るかを説明する動画を生成できます。

```bash
math-anim plan \
  --formula "\\theta_{t+1} = \\theta_t - \\eta \\nabla L(\\theta_t)" \
  --goal "1変数の損失曲線で、谷→山→谷がある時に勾配降下法がどう判断するか知りたい" \
  --audience high_school_math \
  --domain-hint optimization \
  --duration 54 \
  --output-dir outputs/gradient_descent_1d_double_well \
  --no-llm \
  --render \
  --voiceover
```

成功すると、`outputs/gradient_descent_1d_double_well/video_with_voice.mp4` が作成されます。谷・山・谷を持ち、下に有界な損失の例として、3次関数ではなく4次関数の曲線を使います。

### 単純パーセプトロン: 順伝播と決定境界

`--no-llm` でも、`--concept-hint perceptron` を指定すると、入力、重み付き和、活性化関数、2D決定境界を順に説明する動画を生成できます。
ナレーションを聞き取りやすくするため、このテンプレートは `--voice-rate 120` を基準に最低 95 秒のシーン尺を確保します。`--voice-rate` を上げると、発話速度に合わせてシーン尺も少し短くなります。

```bash
math-anim plan \
  --formula "a = \\mathrm{step}(w_1x_1 + w_2x_2 + b)" \
  --goal "単純パーセプトロンの順伝播と決定境界を直感的に理解したい" \
  --concept-hint perceptron \
  --audience high_school_math \
  --domain-hint machine_learning \
  --duration 95 \
  --output-dir outputs/perceptron_phase1 \
  --no-llm \
  --render \
  --voiceover \
  --voice-rate 130
```

成功すると、`outputs/perceptron_phase1/video_with_voice.mp4` が作成されます。学習や誤差逆伝播ではなく、まずは学習済み重みでの順伝播に絞っています。

### 全結合ニューラルネットワーク: 層と順伝播

`--concept-hint fully_connected_network` を指定すると、入力層、隠れ層、出力層、全結合の接続、softmax出力、one-hot正解ラベル、クロスエントロピー損失までを順に説明する動画を生成できます。学習や誤差逆伝播ではなく、まずは推論時の順伝播と損失への接続に絞っています。

```bash
math-anim plan \
  --formula "\\hat{y}=\\mathrm{softmax}(W_2\\sigma(W_1x+b_1)+b_2),\\quad L=-\\sum_i y_i\\log(\\hat{y}_i)" \
  --goal "全結合ニューラルネットワークの順伝播からクロスエントロピー損失までを直感的に理解したい" \
  --concept-hint fully_connected_network \
  --audience high_school_math \
  --domain-hint deep_learning \
  --duration 114 \
  --output-dir outputs/fully_connected_phase1 \
  --no-llm \
  --render \
  --voiceover \
  --voice-rate 130
```

成功すると、`outputs/fully_connected_phase1/video_with_voice.mp4` が作成されます。

### バックプロパゲーション: 誤差信号と重み更新

`--concept-hint backpropagation` を指定すると、損失から出力層、隠れ層、重みへ誤差信号を戻し、最後に勾配降下法で1本の重みを更新する流れを説明する動画を生成できます。自由なPythonコード生成ではなく、固定のJinja2テンプレートで描画します。

```bash
math-anim plan \
  --formula "\\delta^{(l)}=(W^{(l+1)T}\\delta^{(l+1)})\\odot\\sigma'(z^{(l)})" \
  --goal "バックプロパゲーションで誤差信号がどう戻り、重み更新につながるか理解したい" \
  --concept-hint backpropagation \
  --audience high_school_math \
  --domain-hint deep_learning \
  --duration 151 \
  --output-dir outputs/backpropagation_phase1 \
  --no-llm \
  --render \
  --voiceover \
  --voice-rate 130
```

成功すると、`outputs/backpropagation_phase1/video_with_voice.mp4` が作成されます。

### 連鎖律: 途中の変化率を掛けてつなぐ

`--concept-hint chain_rule` を指定すると、合成関数 `x -> u -> y` の流れから、`dy/dx = dy/du * du/dx` を説明する動画を生成できます。最後に `dL/dW = dL/dy_hat * dy_hat/dW` として、バックプロパゲーションへの接続も見せます。

```bash
math-anim plan \
  --formula "\\frac{dy}{dx}=\\frac{dy}{du}\\frac{du}{dx}" \
  --goal "連鎖律を2階微分ではなく変化率のつながりとして理解したい" \
  --concept-hint chain_rule \
  --audience high_school_math \
  --domain-hint calculus \
  --duration 88 \
  --output-dir outputs/chain_rule_intro \
  --no-llm \
  --render \
  --voiceover \
  --voice-rate 130
```

成功すると、`outputs/chain_rule_intro/video_with_voice.mp4` が作成されます。

### NN変換: 線形変換・非線形変換・中間表現

`--concept-hint neural_network_transform` を指定すると、`h = sigma(Wx + b)` を「特徴を混ぜ直す線形変換」「通す/切る/曲げる非線形変換」「予測しやすい中間表現」へ分解して説明する動画を生成できます。

```bash
math-anim plan \
  --formula "h=\\sigma(Wx+b)" \
  --goal "ニューラルネットワークにおける線形変換・非線形変換・中間表現の意味を直感的に理解したい" \
  --concept-hint neural_network_transform \
  --audience high_school_math \
  --domain-hint deep_learning \
  --duration 100 \
  --output-dir outputs/nn_transform_intro \
  --no-llm \
  --render \
  --voiceover \
  --voice-rate 130
```

成功すると、`outputs/nn_transform_intro/video_with_voice.mp4` が作成されます。

### 活性化関数: ReLU・sigmoid・tanh・softmax

`--concept-hint activation_functions` を指定すると、活性化関数を「線形和を次へ渡す値に変える部品」として説明し、ReLU、sigmoid、tanh、softmaxの違いと使い分けを動画化できます。

```bash
math-anim plan \
  --formula "a=f(z),\\quad p=\\mathrm{softmax}(o)" \
  --goal "ReLU、sigmoid、tanh、softmaxの違いと、隠れ層・出力層での使い分けを直感的に理解したい" \
  --concept-hint activation_functions \
  --audience high_school_math \
  --domain-hint deep_learning \
  --duration 130 \
  --output-dir outputs/activation_functions_intro \
  --no-llm \
  --render \
  --voiceover \
  --voice-rate 130
```

成功すると、`outputs/activation_functions_intro/video_with_voice.mp4` が作成されます。

## LLMで動的に教材企画を生成

OpenAI APIを使う場合は、APIキーを環境変数で渡します。キーはリポジトリに保存しません。

```bash
export OPENAI_API_KEY="..."
export OPENAI_MODEL="gpt-4.1-mini"
```

`--no-llm` を外すと、入力数式から `formula_analysis.json`、`explanation_plan.json`、`animation_brief.md`、`narration.md` をLLMで生成します。

```bash
math-anim plan \
  --formula "L = - \\sum_i y_i \\log(\\hat{y}_i)" \
  --goal "初学者にクロスエントロピーを噛み砕いて説明したい" \
  --audience high_school_math \
  --domain-hint machine_learning \
  --duration 60 \
  --output-dir outputs/llm_cross_entropy_plan
```

`--render --voiceover` を付けると、対応済みテンプレートに変換できる場合だけ動画と音声付き動画まで生成します。

LLM出力がschemaに少し合わない場合は、validation errorと前回JSONを渡して1回だけ自動修正を試みます。具体例の `concrete_values` では、数値、文字列、既存preset名に加えて、確率分布やone-hotのようなJSON配列も受け付けます。

## 安全な拡張方針

MVP2以降は、完全自由なManimコード生成ではなく、Storyboard内の宣言的な部品を既存テンプレートへ写像します。

- `components`: 数式パーツ強調、確率バー、損失曲線、勾配矢印、活性化曲線などの再利用可能なアニメーション部品
- `narration_cues`: ナレーションの区間、対応する部品、注目する数式パーツ
- `generation_boundary`: LLMの役割を教材企画に限定し、コード生成を禁止する安全境界

利用可能な視覚部品は [docs/visual-component-catalog.md](docs/visual-component-catalog.md) にまとめています。LLMには同じ内容を `src/math_animation_studio/knowledge/visual_components.yaml` から渡します。

各Manimテンプレートの章構成は [docs/template-chapter-catalog.md](docs/template-chapter-catalog.md) にまとめています。正本は `src/math_animation_studio/knowledge/template_chapters.yaml` です。

Storyboardの共通章立ては [docs/storyboard-dsl.md](docs/storyboard-dsl.md) にまとめています。現在のDSLと視覚部品カタログはCLIでも確認できます。

```bash
math-anim catalog
math-anim catalog --format json
```

LLMに任せることは、数式の分解、説明順、具体例、ナレーション素材、どの部品に注目させるかの選択です。Pythonコード、Manimコード、関数本体、`eval`、`exec` 前提の出力は受け付けません。

## テスト

```bash
python -m pytest
```

## 現在の制約

- `generate --no-llm` は `concept=gradient_descent` の固定サンプルのみ対応
- `plan --no-llm` は Cross Entropy、Gradient Descent、Attention、Perceptron、Fully Connected Network、Backpropagation、Chain Rule、Neural Network Transform の固定サンプルに対応
- Manimテンプレートは Gradient Descent の3D曲面、2D等高線、1D損失曲線、Cross Entropy のPenalty Curve、Perceptron の順伝播・決定境界、Fully Connected Network、Backpropagation、Chain Rule、Neural Network Transform に対応
- Attentionは企画とStoryboard生成まで対応しており、動画テンプレートは未実装
- 音声合成はmacOSの `say` コマンドを使う簡易版で、シーン単位の厳密な同期は未実装
- LLMモードでも動画生成は既存Manimテンプレートに合う数式・パターンのみ対応
- Storyboard内の関数文字列はPythonコードとして評価しません
- LLM出力は `generation_boundary.code_generation_allowed=false` を要求します
