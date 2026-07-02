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

`--no-llm` の固定サンプルは、Cross Entropy、Gradient Descent、Attentionに対応しています。`storyboard.json` は既存のStoryboard schemaでparseできる形式です。

Cross Entropy と Gradient Descent は、`--render` を付けるとManim動画まで生成できます。

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

## テスト

```bash
python -m pytest
```

## 現在の制約

- `generate --no-llm` は `concept=gradient_descent` の固定サンプルのみ対応
- `plan --no-llm` は Cross Entropy、Gradient Descent、Attention の固定サンプルに対応
- Manimテンプレートは Gradient Descent の3D曲面と Cross Entropy のPenalty Curve に対応
- Attentionは企画とStoryboard生成まで対応しており、動画テンプレートは未実装
- 音声合成はmacOSの `say` コマンドを使う簡易版で、シーン単位の厳密な同期は未実装
- LLMモードでも動画生成は既存Manimテンプレートに合う数式・パターンのみ対応
- Storyboard内の関数文字列はPythonコードとして評価しません
