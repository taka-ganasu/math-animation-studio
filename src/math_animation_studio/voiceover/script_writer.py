from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from math_animation_studio.schema import Storyboard
from math_animation_studio.timing import cross_entropy_timeline_segments


@dataclass(frozen=True)
class VoiceoverSegment:
    id: str
    text: str
    duration_seconds: float


class VoiceoverScriptWriter:
    def write(self, storyboard: Storyboard, *, target_duration_seconds: int | None = None) -> str:
        concept = storyboard.concept.strip().lower().replace("-", "_")
        if concept == "cross_entropy":
            if target_duration_seconds is not None and target_duration_seconds >= 25:
                return "".join(
                    segment.text
                    for segment in self.write_segments(
                        storyboard,
                        target_duration_seconds=target_duration_seconds,
                    )
                )
            return (
                "クロスエントロピー損失は、正解クラスに低い確率を置いたときに、"
                "大きな罰を与える損失です。"
                "ワンホットラベルでは、正解クラスだけが損失に効きます。"
                "正解確率が低いほど、マイナスログの値は急激に大きくなります。"
                "つまり、正解に高い確率を置くほど、損失は小さくなります。"
            )
        if concept == "gradient_descent":
            return (
                "勾配降下法は、損失が増える方向の逆向きに少しずつ進む方法です。"
                "曲面の高さを損失と見ると、点は谷へ向かって移動します。"
                "学習率は一歩の大きさを決めます。"
                "更新を繰り返すことで、損失の小さい場所へ近づいていきます。"
            )

        sentences = [storyboard.one_sentence_summary]
        for scene in storyboard.scenes[:3]:
            sentences.append(scene.narration)
        return "".join(_shorten(sentence, 80) for sentence in sentences)

    def write_segments(
        self,
        storyboard: Storyboard,
        *,
        target_duration_seconds: int | None = None,
    ) -> list[VoiceoverSegment]:
        concept = storyboard.concept.strip().lower().replace("-", "_")
        if concept != "cross_entropy":
            return []

        focus_ids = _cross_entropy_focus_ids_from_storyboard(storyboard)
        timeline = cross_entropy_timeline_segments(
            target_duration_seconds,
            active_focus_ids=focus_ids,
        )
        text_by_id = _cross_entropy_segment_text(storyboard)
        return [
            VoiceoverSegment(
                id=segment.id,
                text=text_by_id.get(segment.id, ""),
                duration_seconds=segment.duration_seconds,
            )
            for segment in timeline
            if text_by_id.get(segment.id, "").strip()
        ]

    def write_markdown(
        self,
        storyboard: Storyboard,
        *,
        target_duration_seconds: int | None = None,
        script: str | None = None,
        segments: Sequence[VoiceoverSegment] | None = None,
    ) -> str:
        if segments is not None:
            script = script or "".join(segment.text for segment in segments)
        else:
            script = script or self.write(
                storyboard,
                target_duration_seconds=target_duration_seconds,
            )
        rows = [
            "# Narration",
            "",
            f"Target duration: {target_duration_seconds or 'auto'} seconds",
            "",
        ]
        if segments is not None:
            rows.extend(["## Voiceover Segments", ""])
            for segment in segments:
                rows.extend(
                    [
                        f"### {segment.id} ({segment.duration_seconds:.2f}s)",
                        "",
                        segment.text,
                        "",
                    ]
                )
            rows.append("")
        rows.extend(
            [
                "## Voiceover Script",
                "",
                script,
                "",
                "## Source Scenes",
                "",
            ]
        )
        for scene in storyboard.scenes:
            rows.extend(
                [
                    f"### {scene.id}: {scene.title}",
                    "",
                    scene.narration,
                    "",
                ]
            )
        return "\n".join(rows)


def _shorten(value: str, max_length: int) -> str:
    text = value.strip()
    if len(text) <= max_length:
        return text
    return text[: max_length - 1].rstrip("、。") + "。"


def _cross_entropy_focus_ids_from_storyboard(storyboard: Storyboard) -> tuple[str, ...]:
    symbol_text = " ".join(symbol.symbol for symbol in storyboard.symbol_ledger)
    scene_text = " ".join(scene.title + " " + scene.narration for scene in storyboard.scenes)
    formula = storyboard.formula or ""
    combined = f"{formula} {symbol_text} {scene_text}"

    ids = ["focus_y_i", "focus_y_hat_i"]
    if "log" in combined or "対数" in combined or r"\log" in combined:
        ids.append("focus_log")
    if "sum" in combined or "総和" in combined or r"\sum" in combined:
        ids.append("focus_sum")
    if "-" in formula or "マイナス" in combined:
        ids.append("focus_minus")
    return tuple(ids)


def _cross_entropy_segment_text(storyboard: Storyboard) -> dict[str, str]:
    example_title = storyboard.examples[0].title if storyboard.examples else "画面の例"
    example_label = _shorten(example_title, 22)
    return {
        "intro_formula": (
            f"今回はクロスエントロピー損失を、{example_label}で見ます。"
        ),
        "focus_y_i": (
            "まず式のy_i。正解だけ一、ほかはゼロのスイッチです。"
        ),
        "focus_y_hat_i": (
            "次にワイハットi。モデルがクラスiに置いた予測確率です。"
        ),
        "focus_log": (
            "logワイハットiは、低い確率を大きな損失に変えます。"
        ),
        "focus_sum": (
            "シグマは全クラスを見る記号です。one-hotなので正解だけが残ります。"
        ),
        "focus_minus": (
            "マイナス符号は、負になりやすいlogを正の罰に変えます。"
        ),
        "model_pipeline": (
            "モデルは入力からlogitsという点数を出し、softmaxで確率に変えます。"
        ),
        "one_hot_vector": (
            "one-hotは正解の場所だけ一、ほかはゼロです。"
        ),
        "softmax_distribution": (
            "悪い予測は正解の棒が低く、良い予測は高いです。損失はこの高さを見ます。"
        ),
        "correct_selector": (
            "one-hotが正解クラスの確率pだけを選びます。ほかはゼロ倍です。"
        ),
        "negative_log_penalty": (
            "最後にpをマイナスlogへ通します。pが小さいほど、大きな罰になります。"
        ),
        "summary": "まとめると、正解に高い確率を置けるほど、損失は小さくなります。",
    }


def _storyboard_driven_script(
    storyboard: Storyboard,
    *,
    intro: str,
    max_scene_count: int,
) -> str:
    sentences = [intro]
    if storyboard.examples:
        sentences.append(f"例は、{storyboard.examples[0].title}です。")
    if storyboard.concept.strip().lower().replace("-", "_") == "cross_entropy":
        sentences.extend(
            [
                "まず式のy_i。正解クラスだけ一になるスイッチです。",
                "次にワイハットi。モデルがクラスiに置いた予測確率です。",
                "logワイハットiは、小さい確率を大きな損失として見せます。",
                "シグマは全クラスを見る記号です。ただしone-hotなので正解クラスだけが残ります。",
                "マイナス符号は、負になりやすいlogを正の罰に変えます。",
                "次に、モデルの出力をlogitsからsoftmaxで確率に変えます。",
                "one-hotが正解クラスの確率pだけを選びます。",
                "最後にpをマイナスlogへ通すと、pが小さいほど罰が大きくなります。",
                _shorten(storyboard.one_sentence_summary, 64),
            ]
        )
        return "".join(sentences)
    for scene in storyboard.scenes[:max_scene_count]:
        sentences.append(_shorten(scene.narration, 72))
    sentences.append(_shorten(storyboard.one_sentence_summary, 72))
    return "".join(sentences)
