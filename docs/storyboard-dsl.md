# Storyboard DSL v1

Math Animation Studio は、LLMに自由なManimコードを書かせず、宣言的なStoryboardを安全なJinja2/Manimテンプレートへ写像する。

Storyboard DSL v1 は、そのStoryboardを「どの順番で何を見せるか」という章立てで制御するための最小仕様である。

## Flow: formula_first

基本の流れは次の5章。

1. `title_intro`
   - 今回扱う概念と、何を見ればよいかを短く示す。
   - テンプレート側が自動生成できるため、必ず独立シーンにしなくてもよい。

2. `formula_structure`
   - 数式を意味のあるパーツに分ける。
   - `formula_focus` で、ナレーション中の記号や演算を強調する。

3. `concrete_example`
   - 身近な値や題材に置き換える。
   - クロスエントロピーなら、one-hot、確率バー、モデル出力の流れなどを使う。
   - 勾配降下法なら、地形、現在地、一歩の大きさなどの比喩部品を使う。
   - パーセプトロンなら、入力ノード、重み付き接続、順伝播の流れを使う。
   - 連鎖律なら、`x -> u -> y` の流れと隣同士の変化率を使う。
   - NN変換なら、入力空間、特徴軸の混ぜ直し、活性化を使う。
   - 活性化関数なら、隠れ層と出力層での使い分け比較を使う。

4. `visualization`
   - グラフ、曲面、矢印、軌跡などで実際の動きを見せる。
   - パーセプトロンなら、step関数と2D決定境界を見せる。
   - 連鎖律なら、数値例と `W -> \hat{y} -> L` の接続を見せる。
   - NN変換なら、中間表現と単純な決定境界を見せる。
   - 活性化関数なら、ReLU、sigmoid、tanhの曲線とsoftmaxの確率変換を見せる。
   - ここが動画として一番長い主部になる。

5. `summary`
   - 最後に式へ戻り、直感と数式の対応を短く回収する。

## LLM Output

LLMは `explanation_steps[].scene_role` に上記roleのいずれかを入れる。

`planned_components[].kind` には `src/math_animation_studio/knowledge/visual_components.yaml` にある部品IDだけを入れる。Pythonコード、Manimコード、関数本体、任意の式評価は出力しない。

## Runtime Mapping

`StoryboardAdapter` は `ExplanationPlan` から `Storyboard` を作るときに次を行う。

- `storyboard.blueprint` に `formula_first` の章立てを入れる。
- 各 `SceneSpec` に `scene_role` と `beat_id` を付与する。
- LLMが `scene_role` を出していない場合は、タイトル、説明文、formula_focus、step位置から保守的に推定する。
- 未知の視覚部品は採用しない。

## Inspect

現在のDSLと視覚部品カタログはCLIで確認できる。

```bash
math-anim catalog
math-anim catalog --format json
```
