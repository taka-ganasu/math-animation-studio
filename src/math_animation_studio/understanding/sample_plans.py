from __future__ import annotations

from math_animation_studio.schema import (
    ConceptClassification,
    ExplanationPlan,
    ExplanationStep,
    FormulaAnalysis,
    OperationAnalysis,
    PlannedAnimationComponent,
    PrerequisiteItem,
    PrerequisiteMap,
    SymbolRole,
    TeachingExample,
)


def detect_sample_key(formula: str) -> str:
    normalized = formula.lower().replace(" ", "")
    wants_1d_double_well = any(
        keyword in normalized
        for keyword in (
            "1次元",
            "一次元",
            "1変数",
            "一変数",
            "1d",
            "曲線",
            "3次関数",
            "三次関数",
            "4次関数",
            "四次関数",
            "谷→山→谷",
            "谷山谷",
            "損失曲線",
        )
    )
    wants_double_well = any(
        keyword in normalized
        for keyword in (
            "2次元",
            "二次元",
            "2d",
            "谷2",
            "谷が2",
            "谷が二",
            "局所",
            "localminimum",
            "localminima",
            "doublewell",
        )
    )
    wants_gradient_descent = any(
        keyword in normalized
        for keyword in (
            "concept_hint:gradient_descent",
            "concept_hint:gradientdescent",
            "concept_hint:optimization",
            "concept_hint:最適化",
            "concept_hint:勾配降下法",
        )
    )
    wants_perceptron = any(
        keyword in normalized
        for keyword in (
            "concept_hint:perceptron",
            "concept_hint:simple_perceptron",
            "concept_hint:neural_network",
            "perceptron",
            "パーセプトロン",
            "単純パーセプトロン",
            "w^tx",
            "w_1x_1",
            "w1x1",
            "活性化",
            "決定境界",
            "decisionboundary",
        )
    )
    if wants_perceptron:
        return "perceptron"
    if wants_gradient_descent:
        if wants_1d_double_well:
            return "gradient_descent_double_well_1d"
        if wants_double_well:
            return "gradient_descent_double_well"
        return "gradient_descent"
    if "concept_hint:cross_entropy" in normalized or "concept_hint:crossentropy" in normalized:
        return "cross_entropy"
    if ("log" in normalized and "sum" in normalized) or "cross" in normalized:
        return "cross_entropy"
    if "nabla" in normalized or "grad" in normalized or "∇" in normalized:
        if wants_1d_double_well:
            return "gradient_descent_double_well_1d"
        if wants_double_well:
            return "gradient_descent_double_well"
        return "gradient_descent"
    if "attention" in normalized or ("q" in normalized and "k" in normalized and "v" in normalized):
        return "scaled_dot_product_attention"
    return "generic"


def sample_formula_analysis(formula: str, key: str) -> FormulaAnalysis:
    if key == "cross_entropy":
        return FormulaAnalysis(
            raw_formula=formula,
            normalized_formula_latex=r"L = - \sum_i y_i \log(\hat{y}_i)",
            detected_name="cross_entropy",
            short_description="クロスエントロピー損失は、正解クラスに低い確率を置いたときに大きな罰を与える損失関数である。",
            symbols=[
                SymbolRole(symbol="L", normalized_symbol="loss", role="output", meaning="損失値", intuition="モデルの予測の悪さ", confidence=0.95),
                SymbolRole(symbol="y_i", normalized_symbol="label_i", role="input", meaning="クラスiが正解かどうかを表すラベル", intuition="正解クラスだけが1になるスイッチ", confidence=0.9),
                SymbolRole(symbol=r"\hat{y}_i", normalized_symbol="predicted_probability_i", role="input", meaning="モデルがクラスiに割り当てた予測確率", intuition="モデルの自信", confidence=0.9),
            ],
            operations=[
                OperationAnalysis(operation=r"\log(\hat{y}_i)", meaning="予測確率を対数スケールに変換する", intuition="低い確率を強く罰する", visual_hint="-log(p)曲線で見せる"),
                OperationAnalysis(operation=r"\sum_i", meaning="クラスごとの項を合計する", intuition="one-hotなら正解クラスの項だけが残る", visual_hint="正解クラスの棒だけをハイライトする"),
            ],
            inputs=["y_i", r"\hat{y}_i"],
            outputs=["L"],
            assumptions=["分類問題", "y_i はone-hotラベルとして扱う"],
            ambiguity_notes=["ラベルがone-hotか確率分布かは文脈に依存する。"],
            confidence=0.9,
        )
    if key in {"gradient_descent", "gradient_descent_double_well", "gradient_descent_double_well_1d"}:
        return FormulaAnalysis(
            raw_formula=formula,
            normalized_formula_latex=r"\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)",
            detected_name="gradient_descent",
            short_description="勾配降下法は、損失が最も増える方向の逆へ少しずつ進む更新則である。",
            symbols=[
                SymbolRole(symbol=r"\theta_t", normalized_symbol="theta_t", role="state", meaning="現在のパラメータ", intuition="今いる場所", confidence=0.95),
                SymbolRole(symbol=r"\eta", normalized_symbol="learning_rate", role="hyperparameter", meaning="学習率", intuition="一歩の大きさ", confidence=0.95),
                SymbolRole(symbol=r"\nabla L(\theta_t)", normalized_symbol="gradient", role="direction", meaning="損失が最も増える方向", intuition="一番急な上り坂", confidence=0.95),
            ],
            operations=[
                OperationAnalysis(operation="-", meaning="勾配の逆方向へ進む", intuition="上り坂の反対へ下る", visual_hint="曲面上の矢印で示す"),
            ],
            inputs=[r"\theta_t", r"\eta", r"\nabla L(\theta_t)"],
            outputs=[r"\theta_{t+1}"],
            assumptions=["損失関数が微分可能"],
            ambiguity_notes=[],
            confidence=0.95,
        )
    if key == "scaled_dot_product_attention":
        return FormulaAnalysis(
            raw_formula=formula,
            normalized_formula_latex=r"\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V",
            detected_name="scaled_dot_product_attention",
            short_description="Attentionは、QueryとKeyの類似度から重みを作り、Valueを重み付き平均する操作である。",
            symbols=[
                SymbolRole(symbol="Q", normalized_symbol="query", role="input", meaning="探したい情報を表すベクトル群", intuition="何を見たいか", confidence=0.85),
                SymbolRole(symbol="K", normalized_symbol="key", role="input", meaning="各情報の見出しを表すベクトル群", intuition="照合されるラベル", confidence=0.85),
                SymbolRole(symbol="V", normalized_symbol="value", role="input", meaning="実際に取り出す情報", intuition="混ぜ合わせる中身", confidence=0.85),
            ],
            operations=[
                OperationAnalysis(operation=r"QK^T", meaning="QueryとKeyの類似度を計算する", intuition="どれを見るべきかの相性表", visual_hint="ヒートマップで表示する"),
                OperationAnalysis(operation="softmax", meaning="類似度を重みに変換する", intuition="注目度の配分", visual_hint="行ごとに重みへ変わる様子を見せる"),
            ],
            inputs=["Q", "K", "V"],
            outputs=["Attention(Q,K,V)"],
            assumptions=["Q, K, V は同じ系列または対応する系列から作られる"],
            ambiguity_notes=["self-attentionかcross-attentionかは式だけでは断定できない。"],
            confidence=0.85,
        )
    if key == "perceptron":
        return FormulaAnalysis(
            raw_formula=formula,
            normalized_formula_latex=r"a = \mathrm{step}(w_1x_1 + w_2x_2 + b)",
            detected_name="perceptron",
            short_description="単純パーセプトロンは、入力を重み付きで足し合わせ、活性化関数で2クラスの出力へ変換するモデルである。",
            symbols=[
                SymbolRole(symbol=r"x_1, x_2", normalized_symbol="inputs", role="input", meaning="入力特徴量", intuition="判断材料", confidence=0.9),
                SymbolRole(symbol=r"w_1, w_2", normalized_symbol="weights", role="parameter", meaning="各入力の重要度", intuition="どの材料をどれくらい重く見るか", confidence=0.9),
                SymbolRole(symbol="b", normalized_symbol="bias", role="parameter", meaning="判断基準をずらす値", intuition="境界線の位置調整", confidence=0.9),
                SymbolRole(symbol="z", normalized_symbol="weighted_sum", role="intermediate", meaning="重み付き和", intuition="材料を足し合わせた判定スコア", confidence=0.9),
                SymbolRole(symbol="a", normalized_symbol="activation_output", role="output", meaning="活性化後の出力", intuition="0か1の判断", confidence=0.9),
            ],
            operations=[
                OperationAnalysis(operation=r"w_1x_1 + w_2x_2 + b", meaning="入力を重み付きで足し合わせる", intuition="判断材料に重みを付けて合計点を作る", visual_hint="入力ノードからsumノードへ矢印を流す"),
                OperationAnalysis(operation=r"\mathrm{step}(z)", meaning="スコアを0/1の分類結果へ変換する", intuition="境界を超えたら1、超えなければ0", visual_hint="step関数と決定境界で見せる"),
            ],
            inputs=[r"x_1", r"x_2"],
            outputs=["a"],
            assumptions=["2値分類", "入力特徴量が2つ", "活性化関数はstep関数として扱う"],
            ambiguity_notes=["学習方法ではなく、まず順伝播と決定境界に絞る。"],
            confidence=0.9,
        )
    return FormulaAnalysis(
        raw_formula=formula,
        normalized_formula_latex=formula,
        detected_name=None,
        short_description="入力された数式を記号と操作に分けて理解する必要がある。",
        symbols=[],
        operations=[],
        ambiguity_notes=["固定サンプルではこの数式を既知パターンに分類できなかった。"],
        confidence=0.35,
    )


def sample_classification(key: str) -> ConceptClassification:
    if key == "cross_entropy":
        return ConceptClassification(primary_domain="machine_learning", primary_concept="cross_entropy", related_concepts=["log_loss", "negative_log_likelihood", "softmax"], difficulty_level="undergraduate_intro", recommended_animation_family="penalty_curve", confidence=0.9)
    if key in {"gradient_descent", "gradient_descent_double_well", "gradient_descent_double_well_1d"}:
        return ConceptClassification(primary_domain="optimization", primary_concept="gradient_descent", related_concepts=["gradient", "learning_rate", "loss_minimization"], difficulty_level="undergraduate_intro", recommended_animation_family="trajectory_on_surface", confidence=0.95)
    if key == "scaled_dot_product_attention":
        return ConceptClassification(primary_domain="deep_learning", primary_concept="scaled_dot_product_attention", related_concepts=["matrix_multiplication", "softmax", "weighted_sum"], difficulty_level="undergraduate_advanced", recommended_animation_family="matrix_similarity_heatmap", confidence=0.85)
    if key == "perceptron":
        return ConceptClassification(primary_domain="machine_learning", primary_concept="perceptron", related_concepts=["linear_classifier", "activation_function", "decision_boundary", "neural_network"], difficulty_level="undergraduate_intro", recommended_animation_family="perceptron_decision_boundary", confidence=0.9)
    return ConceptClassification(primary_domain="unknown", primary_concept="unknown_formula", related_concepts=[], difficulty_level="undergraduate_intro", recommended_animation_family="generic_symbol_decomposition", confidence=0.35)


def sample_prerequisites(key: str) -> PrerequisiteMap:
    if key == "cross_entropy":
        return PrerequisiteMap(
            target_concept="cross_entropy",
            prerequisites=[
                PrerequisiteItem(concept="確率分布", why_needed="予測をクラスごとの確率として読むため", priority="required", suggested_micro_explanation="各クラスに割り当てた確率の合計は1になる。"),
                PrerequisiteItem(concept="対数", why_needed="-log(p)の罰の形を理解するため", priority="required", suggested_micro_explanation="pが0に近いほど-log(p)は急激に大きくなる。"),
                PrerequisiteItem(concept="one-hotラベル", why_needed="正解クラスだけが損失に残る理由を理解するため", priority="helpful", suggested_micro_explanation="正解だけ1、他は0のラベル表現。"),
            ],
            likely_blockers=["logがなぜ出るのか", "sumの全項が効くと誤解すること"],
        )
    if key in {"gradient_descent", "gradient_descent_double_well", "gradient_descent_double_well_1d"}:
        return PrerequisiteMap(
            target_concept="gradient_descent",
            prerequisites=[
                PrerequisiteItem(concept="微分", why_needed="勾配が増加方向を表すため", priority="required", suggested_micro_explanation="微分は関数がどちらへ増えるかを教える。"),
                PrerequisiteItem(concept="ベクトル", why_needed="複数パラメータの方向を扱うため", priority="helpful", suggested_micro_explanation="ベクトルは向きと大きさを持つ。"),
            ],
            likely_blockers=["勾配は下り方向だと誤解すること", "学習率を大きくすれば常に良いと思うこと"],
        )
    if key == "scaled_dot_product_attention":
        return PrerequisiteMap(
            target_concept="scaled_dot_product_attention",
            prerequisites=[
                PrerequisiteItem(concept="内積", why_needed="QueryとKeyの類似度を測るため", priority="required", suggested_micro_explanation="向きが近いベクトルほど内積が大きくなる。"),
                PrerequisiteItem(concept="softmax", why_needed="類似度を重みに変換するため", priority="required", suggested_micro_explanation="数値列を合計1の重みに変える。"),
                PrerequisiteItem(concept="重み付き平均", why_needed="Valueを混ぜる最終操作を理解するため", priority="helpful", suggested_micro_explanation="重要なものを大きな重みで足し合わせる。"),
            ],
            likely_blockers=["Q/K/Vの役割が混ざること", "softmax後の重みがValueを混ぜることを見落とすこと"],
        )
    if key == "perceptron":
        return PrerequisiteMap(
            target_concept="perceptron",
            prerequisites=[
                PrerequisiteItem(concept="一次関数", why_needed="決定境界が直線として表れるため", priority="required", suggested_micro_explanation="2変数の一次式を0に等しいと置くと、平面上の直線になる。"),
                PrerequisiteItem(concept="ベクトルと内積", why_needed="w^T x が重み付き和を表すため", priority="helpful", suggested_micro_explanation="各成分を掛けて足すと、入力と重みの相性スコアになる。"),
                PrerequisiteItem(concept="2値分類", why_needed="出力が0か1の判断になるため", priority="required", suggested_micro_explanation="境界線のどちら側かでクラスを分ける。"),
            ],
            likely_blockers=["重みが接続の強さであること", "バイアスが境界線をずらすこと", "学習ではなく順伝播だけを見ていること"],
        )
    return PrerequisiteMap(target_concept="unknown_formula", prerequisites=[], likely_blockers=["式だけでは文脈が不足している可能性"])


def sample_explanation_plan(formula: str, key: str, audience: str) -> ExplanationPlan:
    if key == "cross_entropy":
        return ExplanationPlan(
            formula=r"L = - \sum_i y_i \log(\hat{y}_i)",
            target_concept="cross_entropy",
            one_sentence_summary="クロスエントロピー損失は、正解クラスに低い確率を置いたときに大きな罰を与える。",
            audience=audience,
            teaching_strategy="concrete_to_abstract",
            recommended_examples=[
                TeachingExample(
                    title="3クラス分類: 猫・犬・鳥",
                    description="正解が猫のとき、モデルの予測確率を棒グラフで見る。",
                    why_it_works="正解クラスの確率だけを取り出して-log(p)へ通す流れを視覚化しやすい。",
                    concrete_values={"cat_probability_good": 0.9, "cat_probability_bad": 0.1},
                ),
                TeachingExample(
                    title="サイコロの目予測例",
                    description="正解が目3のとき、6個の候補のうち正解の確率だけを罰に変える。",
                    why_it_works="one-hotが6次元になるため、正解以外の項が0倍されることを確認しやすい。",
                    concrete_values={
                        "y": [0, 0, 1, 0, 0, 0],
                        r"\hat{y}_good": [0.03, 0.04, 0.80, 0.05, 0.04, 0.04],
                        r"\hat{y}_bad": [0.20, 0.25, 0.05, 0.20, 0.15, 0.15],
                    },
                ),
                TeachingExample(
                    title="天気予測: 晴れ・雨・曇り",
                    description="正解が雨のとき、雨に置いた確率が高い予測と低い予測を比べる。",
                    why_it_works="確率予測の良し悪しを日常的な例で捉えやすい。",
                    concrete_values={
                        "y": [0, 1, 0],
                        r"\hat{y}_good": [0.08, 0.84, 0.08],
                        r"\hat{y}_bad": [0.62, 0.12, 0.26],
                    },
                ),
            ],
            selected_animation_pattern_id="penalty_curve",
            explanation_steps=[
                ExplanationStep(id="step_01", title="正解クラスを決める", learning_goal="one-hotラベルの役割を理解する", explanation="まず、どのクラスが正解かを決めます。one-hotラベルでは正解だけが1になります。", visual_idea="候補クラスを並べ、正解クラスだけをハイライトする。", formula_focus="y_i"),
                ExplanationStep(id="step_02", title="予測確率を見る", learning_goal="モデル出力を確率分布として読む", explanation="モデルは各クラスに確率を割り当てます。正解クラスに高い確率を置けるほど良い予測です。", visual_idea="各候補の予測確率を棒グラフで表示する。", formula_focus=r"\hat{y}_i"),
                ExplanationStep(id="step_03", title="正解クラスだけが残る", learning_goal="sumの中でどの項が効くかを理解する", explanation="one-hotラベルでは、正解クラス以外の項は0倍されます。", visual_idea="sumの各項を並べ、0になる項を薄くする。", formula_focus=r"\sum_i y_i \log(\hat{y}_i)", common_misunderstanding_addressed="全クラスが同じ強さで損失に効くわけではない。"),
                ExplanationStep(id="step_04", title="-log(p)の罰を見る", learning_goal="低い正解確率が強く罰される理由を理解する", explanation="正解確率pが小さいほど-log(p)は急激に大きくなります。", visual_idea="-log(p)曲線上でp=0.9とp=0.1を比較する。", formula_focus=r"-\log(p)"),
                ExplanationStep(id="step_05", title="良い予測と悪い予測を比べる", learning_goal="損失値の直感を得る", explanation="正解に高い確率を置く予測は損失が小さく、低い確率しか置かない予測は損失が大きくなります。", visual_idea="2つの予測バーと損失値を左右に並べる。"),
                ExplanationStep(id="step_06", title="最後に式へ戻る", learning_goal="各記号と直感を結びつける", explanation="式全体は、正解クラスの予測確率を取り出し、それを-logで罰に変換する操作です。", visual_idea="元の式の各部分へラベルを付ける。", formula_focus=r"L = - \sum_i y_i \log(\hat{y}_i)"),
            ],
            misconceptions=["全クラスの確率が同じように損失へ効くわけではない", "logは小さい確率を強く罰するために効いている", "損失は確率そのものではなく罰の大きさである"],
            next_questions_to_study=["softmax", "negative log likelihood", "KL divergence"],
        )
    if key == "gradient_descent":
        return ExplanationPlan(
            formula=r"\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)",
            target_concept="gradient_descent",
            one_sentence_summary="勾配降下法は、損失が最も増える方向の逆向きに少しずつ進む方法である。",
            audience=audience,
            teaching_strategy="geometric_intuition",
            recommended_examples=[TeachingExample(title="山の斜面を下る例", description="損失関数を地形として見て、点が谷へ下る様子を追う。", why_it_works="勾配の向きと学習率を空間上の移動として直感化できる。", concrete_values={"learning_rate": 0.15, "steps": 30})],
            selected_animation_pattern_id="trajectory_on_surface",
            explanation_steps=[
                ExplanationStep(id="step_01", title="損失関数を地形として見る", learning_goal="損失を高さとして捉える", explanation="高さが高いほど損失が大きく、低いほど損失が小さいと考えます。", visual_idea="3D曲面を表示する。", formula_focus="L(\\theta)"),
                ExplanationStep(id="step_02", title="現在地を点として置く", learning_goal="パラメータを曲面上の点として見る", explanation="現在のパラメータは曲面上の一点です。", visual_idea="曲面上に赤い点を置く。", formula_focus=r"\theta_t"),
                ExplanationStep(id="step_03", title="勾配は上り方向", learning_goal="勾配の向きを理解する", explanation="勾配は損失が最も増える方向を指します。下るには逆向きへ進みます。", visual_idea="負の勾配方向を矢印で示す。", formula_focus=r"-\nabla L(\theta_t)", common_misunderstanding_addressed="勾配そのものは下り方向ではない。"),
                ExplanationStep(id="step_04", title="学習率は一歩の大きさ", learning_goal="ηの役割を理解する", explanation="学習率は一回の更新でどれだけ進むかを決めます。", visual_idea="矢印の長さとして学習率を表示する。", formula_focus=r"\eta"),
                ExplanationStep(id="step_05", title="更新を繰り返す", learning_goal="軌跡として最適化を見る", explanation="更新を繰り返すと点は谷へ近づいていきます。", visual_idea="点の軌跡を線で残す。"),
                ExplanationStep(id="step_06", title="最後に式へ戻る", learning_goal="式の各項を直感と対応づける", explanation="今いる場所から、負の勾配方向へ、学習率の分だけ進む式です。", visual_idea="更新式の各項にラベルを付ける。", formula_focus=r"\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)"),
            ],
            misconceptions=["勾配は下る方向ではなく上る方向を示す", "学習率が大きいほど常に早く収束するわけではない"],
            next_questions_to_study=["凸最適化", "Momentum", "Adam"],
        )
    if key == "gradient_descent_double_well":
        return ExplanationPlan(
            formula=r"\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)",
            target_concept="gradient_descent",
            one_sentence_summary="谷が複数ある損失では、勾配降下法は現在地の斜面に従うため、初期位置によって到達する谷が変わる。",
            audience=audience,
            teaching_strategy="geometric_intuition",
            recommended_examples=[
                TeachingExample(
                    title="2つの谷がある損失地形",
                    description="2次元の等高線上で、浅い谷と深い谷へ向かう2本の軌跡を比べる。",
                    why_it_works="勾配降下法が全体を見比べず、現在地周辺の傾きだけで進むことを直感化できる。",
                    concrete_values={
                        "function_preset": "double_well_2d",
                        "learning_rate": 0.18,
                        "steps": 36,
                        "initial_x": -2.7,
                        "initial_y": 1.9,
                        "comparison_initial_x": 2.7,
                        "comparison_initial_y": 1.8,
                    },
                )
            ],
            selected_animation_pattern_id="trajectory_on_surface",
            explanation_steps=[
                ExplanationStep(id="step_01", title="2次元の損失地形を見る", learning_goal="等高線を損失の高さとして読む", explanation="2次元の地図で、線が囲む低い場所を谷として見ます。", visual_idea="等高線と2つの谷を表示する。", formula_focus="L(\\theta)"),
                ExplanationStep(id="step_02", title="谷は2つある", learning_goal="局所最小と大域最小の違いを知る", explanation="左の谷は浅い局所最小、右の谷はより低い大域最小です。", visual_idea="浅い谷と深い谷を別色でラベル付けする。"),
                ExplanationStep(id="step_03", title="現在地の斜面だけを見る", learning_goal="勾配が局所情報であることを理解する", explanation="勾配は今いる場所の一番急な上り方向です。下るにはその逆へ進みます。", visual_idea="現在地に負の勾配矢印を出す。", formula_focus=r"-\nabla L(\theta_t)"),
                ExplanationStep(id="step_04", title="左側から始める", learning_goal="初期位置で到達先が変わることを見る", explanation="左側から始めると、近くの浅い谷へ吸い込まれます。", visual_idea="左の初期点から局所最小へ進む軌跡を描く。"),
                ExplanationStep(id="step_05", title="右側から始める", learning_goal="別の初期位置では深い谷へ向かうことを見る", explanation="右側から始めると、より深い谷へ進みます。", visual_idea="右の初期点から大域最小へ進む軌跡を描く。"),
                ExplanationStep(id="step_06", title="どう判断しているか", learning_goal="勾配降下法単体の限界を理解する", explanation="勾配降下法は2つの谷を比較して選ぶわけではありません。初期位置と局所的な傾きで進む谷が決まります。", visual_idea="2本の軌跡と最終損失を比較する。"),
                ExplanationStep(id="step_07", title="SGDへのつながり", learning_goal="確率的勾配降下法との関係を知る", explanation="確率的勾配降下法では、ミニバッチの揺れで軌跡が少しノイズを持ちます。ただし、それだけで常に深い谷を選べるわけではありません。", visual_idea="点線の揺れた軌跡を補足として表示する。"),
                ExplanationStep(id="step_08", title="式へ戻る", learning_goal="更新式と直感を結びつける", explanation="式は、今いる場所から負の勾配方向へ、学習率の分だけ進むことを表します。", visual_idea="更新式の各部分に意味ラベルを付ける。", formula_focus=r"\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)"),
            ],
            misconceptions=["勾配降下法が常に一番深い谷を見つけるわけではない", "勾配は全体地図ではなく現在地の局所情報である", "SGDのノイズは助けになることもあるが万能ではない"],
            next_questions_to_study=["局所最小", "大域最小", "ランダム初期化", "確率的勾配降下法", "Momentum"],
        )
    if key == "gradient_descent_double_well_1d":
        return ExplanationPlan(
            formula=r"\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)",
            target_concept="gradient_descent",
            one_sentence_summary="1変数の損失曲線では、勾配降下法は曲線上の現在地の傾きだけを見て、近い谷へ一歩ずつ下る。",
            audience=audience,
            teaching_strategy="geometric_intuition",
            recommended_examples=[
                TeachingExample(
                    title="谷・山・谷がある1変数の損失曲線",
                    description="横軸をパラメータθ、縦軸を損失L(θ)として、左右の谷へ落ちる2本の軌跡を比べる。",
                    why_it_works="局所最小、大域最小、現在地の傾きだけで進むことを1枚の曲線で直感化できる。",
                    concrete_values={
                        "function_preset": "double_well_1d",
                        "learning_rate": 0.012,
                        "steps": 60,
                        "initial_x": -1.8,
                        "initial_y": 0.0,
                        "comparison_initial_x": 1.8,
                        "comparison_initial_y": 0.0,
                    },
                )
            ],
            selected_animation_pattern_id="trajectory_on_surface",
            explanation_steps=[
                ExplanationStep(id="step_01", title="1変数の損失曲線を見る", learning_goal="横軸θと縦軸L(θ)の関係を読む", explanation="横軸をパラメータθ、縦軸を損失L(θ)として見ます。曲線が低い場所ほど良い状態です。", visual_idea="谷・山・谷を持つ損失曲線を描く。", formula_focus=r"L(\theta)"),
                ExplanationStep(id="step_02", title="谷・山・谷を確認する", learning_goal="局所最小と大域最小を見分ける", explanation="右の谷は近くでは低い局所最小ですが、左の谷の方がさらに低い大域最小です。", visual_idea="左の大域最小、中央の山、右の局所最小にラベルを付ける。"),
                ExplanationStep(id="step_03", title="傾きだけを見て進む", learning_goal="1変数の勾配が傾きであることを理解する", explanation="1変数では、勾配は曲線の傾きです。傾きが上りなら、その逆向きへ進みます。", visual_idea="現在地の接線と負の勾配方向を表示する。", formula_focus=r"-\nabla L(\theta_t)"),
                ExplanationStep(id="step_04", title="左側から始める", learning_goal="大域最小へ向かう例を見る", explanation="左側から始めると、点は左の深い谷へ下っていきます。", visual_idea="左の初期点から大域最小へ進む軌跡を描く。"),
                ExplanationStep(id="step_05", title="右側から始める", learning_goal="局所最小へ止まる例を見る", explanation="右側から始めると、中央の山を越えて左を探すのではなく、近くの右の谷へ下ります。", visual_idea="右の初期点から局所最小へ進む軌跡を描く。"),
                ExplanationStep(id="step_06", title="どう判断しているか", learning_goal="勾配降下法が全体比較をしないことを理解する", explanation="勾配降下法は左右の谷を見比べて選びません。現在地の傾きだけで、次の一歩を決めます。", visual_idea="2本の軌跡と最終損失を比較する。"),
                ExplanationStep(id="step_07", title="SGDへのつながり", learning_goal="ノイズが入る場合の直感を持つ", explanation="SGDでは勾配に揺れが入るので、軌跡が少しぶれます。ただし、常に山を越えて大域最小へ行けるわけではありません。", visual_idea="右側の軌跡に小さな揺れを点線で重ねる。"),
                ExplanationStep(id="step_08", title="式へ戻る", learning_goal="更新式と曲線上の移動を結びつける", explanation="式は、今いる位置から、損失が下がる向きへ、学習率の分だけ進むことを表します。", visual_idea="更新式のθ_t、η、∇Lをラベル付けする。", formula_focus=r"\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)"),
            ],
            misconceptions=["勾配降下法が常に一番低い谷を見つけるわけではない", "1変数の勾配は曲線の傾きとして読める", "3次関数より下に有界な4次関数の方が損失曲線として自然である"],
            next_questions_to_study=["局所最小", "大域最小", "学習率", "確率的勾配降下法", "ランダム初期化"],
        )
    if key == "perceptron":
        return ExplanationPlan(
            formula=r"a = \mathrm{step}(w_1x_1 + w_2x_2 + b)",
            target_concept="perceptron",
            one_sentence_summary="単純パーセプトロンは、入力を重み付きで足し合わせ、境界線のどちら側かで0/1を判断する。",
            audience=audience,
            teaching_strategy="visual_first",
            recommended_examples=[
                TeachingExample(
                    title="2つの特徴量で合格/不合格を分ける例",
                    description="x1を勉強時間、x2を復習回数として、重み付き和とstep関数で0/1を出す。",
                    why_it_works="入力、重み、バイアス、決定境界を2D平面で同時に見せやすい。",
                    concrete_values={
                        "input_labels": ["x_1", "x_2"],
                        "input_values": [1.0, 0.4],
                        "weights": [1.2, -0.8],
                        "bias": -0.1,
                        "activation": "step",
                        "class_labels": ["0", "1"],
                    },
                ),
                TeachingExample(
                    title="メールを迷惑メールか分類する例",
                    description="単語の出現数など2つの特徴から、境界線のどちら側かで判定する。",
                    why_it_works="分類という用途は身近だが、特徴量を2つに絞れば図示しやすい。",
                    concrete_values={
                        "input_labels": ["word_count", "link_count"],
                        "input_values": [0.8, 1.1],
                        "weights": [0.9, 1.1],
                        "bias": -1.0,
                        "activation": "step",
                        "class_labels": ["通常", "迷惑"],
                    },
                ),
            ],
            selected_animation_pattern_id="perceptron_decision_boundary",
            explanation_steps=[
                ExplanationStep(
                    id="step_01",
                    scene_role="formula_structure",
                    title="入力と重みを見る",
                    learning_goal="xとwが何を表すかを理解する",
                    explanation="まず、入力x1とx2を判断材料、重みw1とw2を材料の重要度として見ます。",
                    visual_idea="入力ノードから重み付き矢印を出す。",
                    formula_focus=r"w_1x_1 + w_2x_2",
                    planned_components=[
                        PlannedAnimationComponent(kind="perceptron_node", description="入力と1つのニューロンを並べる"),
                        PlannedAnimationComponent(kind="weighted_connection", description="入力からニューロンへの重み付き接続を示す"),
                    ],
                ),
                ExplanationStep(
                    id="step_02",
                    scene_role="formula_structure",
                    title="重み付き和を作る",
                    learning_goal="zが判定スコアであることを理解する",
                    explanation="次に、入力に重みを掛けて足し、バイアスbを加えます。これが判定前のスコアzです。",
                    visual_idea="z = w1x1 + w2x2 + b を式として強調する。",
                    formula_focus=r"z = w_1x_1 + w_2x_2 + b",
                    planned_components=[
                        PlannedAnimationComponent(kind="weighted_sum", description="重み付き和の式と代入例を表示する"),
                        PlannedAnimationComponent(kind="formula_focus", description="zの式を強調する", params={"formula_focus": r"z = w_1x_1 + w_2x_2 + b"}),
                    ],
                ),
                ExplanationStep(
                    id="step_03",
                    scene_role="concrete_example",
                    title="活性化関数で0/1にする",
                    learning_goal="step関数の役割を理解する",
                    explanation="スコアzをそのまま出すのではなく、step関数で0か1の判断に変えます。",
                    visual_idea="zが0以上なら1、0未満なら0になる階段型の関数を表示する。",
                    formula_focus=r"a = \mathrm{step}(z)",
                    planned_components=[
                        PlannedAnimationComponent(kind="activation_function", description="step関数を表示する", params={"activation": "step"}),
                        PlannedAnimationComponent(kind="forward_pass", description="zからaへ値が流れる様子を示す"),
                    ],
                ),
                ExplanationStep(
                    id="step_04",
                    scene_role="visualization",
                    title="決定境界を見る",
                    learning_goal="2D平面で分類の境目を見る",
                    explanation="2つの入力を平面に置くと、w1x1 + w2x2 + b = 0 が境界線になります。",
                    visual_idea="2D平面に点と直線の決定境界を表示する。",
                    formula_focus=r"w_1x_1 + w_2x_2 + b = 0",
                    planned_components=[
                        PlannedAnimationComponent(kind="decision_boundary", description="決定境界を2D平面上の直線として表示する"),
                    ],
                ),
                ExplanationStep(
                    id="step_05",
                    scene_role="visualization",
                    title="順伝播として値を流す",
                    learning_goal="推論時の計算順序を理解する",
                    explanation="順伝播では、入力から重み付き和、活性化、出力の順に値が流れます。",
                    visual_idea="xからz、a、出力へ順にハイライトする。",
                    planned_components=[
                        PlannedAnimationComponent(kind="forward_pass", description="入力から出力への流れをハイライトする"),
                    ],
                ),
                ExplanationStep(
                    id="step_06",
                    scene_role="summary",
                    title="最後に式へ戻る",
                    learning_goal="式と図を対応づける",
                    explanation="まとめると、パーセプトロンは重み付き和でスコアを作り、活性化関数で0/1の判断へ変えるモデルです。",
                    visual_idea="式の重み付き和、バイアス、step関数に意味ラベルを付ける。",
                    formula_focus=r"a = \mathrm{step}(w_1x_1 + w_2x_2 + b)",
                    planned_components=[
                        PlannedAnimationComponent(kind="summary", description="式と直感をまとめる"),
                        PlannedAnimationComponent(kind="formula_focus", description="式全体を強調する", params={"formula_focus": r"a = \mathrm{step}(w_1x_1 + w_2x_2 + b)"}),
                    ],
                ),
            ],
            misconceptions=[
                "単純パーセプトロンは非線形な境界を直接作るわけではない",
                "重みは入力の重要度、バイアスは境界線の位置調整として読める",
                "ここでは学習ではなく、学習済み重みでの順伝播を見ている",
            ],
            next_questions_to_study=["多層パーセプトロン", "全結合層", "誤差逆伝播", "活性化関数"],
        )
    if key == "scaled_dot_product_attention":
        return ExplanationPlan(
            formula=r"\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V",
            target_concept="scaled_dot_product_attention",
            one_sentence_summary="Attentionは、どの情報を見るべきかを類似度で決め、Valueを重み付き平均する仕組みである。",
            audience=audience,
            teaching_strategy="visual_first",
            recommended_examples=[TeachingExample(title="単語列の参照", description="ある単語が文中のどの単語を参照するかを重みとして見る。", why_it_works="Q/K/Vの役割を、探す・照合する・取り出すという流れで分けられる。", concrete_values={"tokens": "I like cats"})],
            selected_animation_pattern_id="matrix_similarity_heatmap",
            explanation_steps=[
                ExplanationStep(id="step_01", title="Q/K/Vの役割を分ける", learning_goal="3つの入力の意味を理解する", explanation="Queryは何を探すか、Keyは照合される見出し、Valueは取り出す中身です。", visual_idea="Q/K/Vを3列に並べる。", formula_focus="Q, K, V"),
                ExplanationStep(id="step_02", title="類似度を作る", learning_goal="QK^Tの意味を理解する", explanation="QueryとKeyの内積で、どれを見るべきかの相性表を作ります。", visual_idea="類似度行列をヒートマップで表示する。", formula_focus=r"QK^T"),
                ExplanationStep(id="step_03", title="スケールする", learning_goal="sqrt(d_k)で割る理由を知る", explanation="値が大きくなりすぎるとsoftmaxが尖りすぎるので、次元で割って調整します。", visual_idea="割る前後のスコア分布を比べる。", formula_focus=r"\sqrt{d_k}"),
                ExplanationStep(id="step_04", title="softmaxで重みにする", learning_goal="スコアから注目重みへの変換を理解する", explanation="softmaxはスコアを合計1の重みに変えます。", visual_idea="スコア行を確率バーへ変換する。", formula_focus="softmax"),
                ExplanationStep(id="step_05", title="Valueを混ぜる", learning_goal="重み付き平均として出力を見る", explanation="注目重みに従ってValueを足し合わせると出力ができます。", visual_idea="重み付き矢印でValueを合成する。", formula_focus=r"\mathrm{softmax}(...)V"),
                ExplanationStep(id="step_06", title="式全体へ戻る", learning_goal="Attentionの流れを式に対応づける", explanation="類似度を作り、重みに変え、Valueを混ぜるという3段階の式です。", visual_idea="式を3ブロックに色分けする。"),
            ],
            misconceptions=["Q/K/Vは同じものではなく役割が違う", "softmax後の重みがValueを混ぜる", "QK^Tだけではまだ出力ではない"],
            next_questions_to_study=["multi-head attention", "self-attention", "positional encoding"],
        )
    return ExplanationPlan(
        formula=formula,
        target_concept="unknown_formula",
        one_sentence_summary="この数式は、まず記号と操作に分けて読む必要がある。",
        audience=audience,
        teaching_strategy="symbol_decomposition",
        recommended_examples=[],
        selected_animation_pattern_id="generic_symbol_decomposition",
        explanation_steps=[
            ExplanationStep(id="step_01", title="記号を洗い出す", learning_goal="式に出てくる部品を把握する", explanation="まず記号ごとに意味を整理します。", visual_idea="数式を記号単位に分割する。"),
            ExplanationStep(id="step_02", title="操作を読む", learning_goal="どんな変換が行われているかを把握する", explanation="次に、和・積・関数適用などの操作を確認します。", visual_idea="演算子ごとに意味ラベルを付ける。"),
        ],
        misconceptions=["文脈なしで意味を断定しすぎない"],
        next_questions_to_study=["分野ヒントを追加する", "記号の定義を確認する"],
    )
