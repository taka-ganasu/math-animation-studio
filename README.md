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

## テスト

```bash
python -m pytest
```

## 現在の制約

- `--no-llm` は `concept=gradient_descent` の固定サンプルのみ対応
- Manimテンプレートは勾配降下法の3D曲面プリセットのみ対応
- Storyboard内の関数文字列はPythonコードとして評価しません
- 自動音声合成は未実装で、ナレーション原稿をMarkdownとして出力します
