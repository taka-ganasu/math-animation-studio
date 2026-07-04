from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from math_animation_studio.schema import Storyboard
from math_animation_studio.timing import (
    TimelineSegment,
    cross_entropy_timeline_segments,
    gradient_double_well_1d_timeline_segments,
    gradient_double_well_timeline_segments,
    gradient_surface_3d_timeline_segments,
)


@dataclass(frozen=True)
class VoiceoverSegment:
    id: str
    text: str
    duration_seconds: float
    component_id: str | None = None
    formula_focus: str | None = None


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
            if target_duration_seconds is not None and target_duration_seconds >= 25:
                return "".join(
                    segment.text
                    for segment in self.write_segments(
                        storyboard,
                        target_duration_seconds=target_duration_seconds,
                    )
                )
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
        if concept == "gradient_descent" and _is_gradient_double_well_storyboard(storyboard):
            if _is_gradient_double_well_1d_storyboard(storyboard):
                timeline = gradient_double_well_1d_timeline_segments(target_duration_seconds)
                text_by_id = _gradient_double_well_1d_segment_text()
            else:
                timeline = gradient_double_well_timeline_segments(target_duration_seconds)
                text_by_id = _gradient_double_well_segment_text()
            return _segments_from_timeline(timeline, text_by_id)

        if concept == "gradient_descent":
            timeline = gradient_surface_3d_timeline_segments(target_duration_seconds)
            text_by_id = _gradient_surface_3d_segment_text()
            return _segments_from_timeline(timeline, text_by_id)

        if concept != "cross_entropy":
            return _segments_from_storyboard_cues(
                storyboard,
                target_duration_seconds=target_duration_seconds,
            )

        focus_ids = _cross_entropy_focus_ids_from_storyboard(storyboard)
        timeline = cross_entropy_timeline_segments(
            target_duration_seconds,
            active_focus_ids=focus_ids,
        )
        text_by_id = _cross_entropy_segment_text(storyboard)
        return _segments_from_timeline(timeline, text_by_id)

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
                metadata = []
                if segment.component_id:
                    metadata.append(f"- Component: `{segment.component_id}`")
                if segment.formula_focus:
                    metadata.append(f"- Focus: ${segment.formula_focus}$")
                rows.extend(
                    [
                        f"### {segment.id} ({segment.duration_seconds:.2f}s)",
                        "",
                        *metadata,
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


def _segments_from_timeline(
    timeline: Sequence[TimelineSegment],
    text_by_id: dict[str, str],
) -> list[VoiceoverSegment]:
    return [
        VoiceoverSegment(
            id=segment.id,
            text=text_by_id.get(segment.id, ""),
            duration_seconds=segment.duration_seconds,
            component_id=segment.component_id,
            formula_focus=segment.formula_focus,
        )
        for segment in timeline
        if text_by_id.get(segment.id, "").strip()
    ]


def _segments_from_storyboard_cues(
    storyboard: Storyboard,
    *,
    target_duration_seconds: int | None,
) -> list[VoiceoverSegment]:
    segments: list[VoiceoverSegment] = []
    for scene in storyboard.scenes:
        if scene.narration_cues:
            for cue in scene.narration_cues:
                text = cue.text or scene.narration
                if not text.strip():
                    continue
                segments.append(
                    VoiceoverSegment(
                        id=cue.segment_id,
                        text=text,
                        duration_seconds=cue.duration_seconds or scene.duration_seconds,
                        component_id=cue.component_id,
                        formula_focus=cue.formula_focus,
                    )
                )
            continue
        if scene.narration.strip():
            segments.append(
                VoiceoverSegment(
                    id=scene.id,
                    text=scene.narration,
                    duration_seconds=scene.duration_seconds,
                )
            )

    if not segments or target_duration_seconds is None:
        return segments

    total = sum(segment.duration_seconds for segment in segments)
    if total <= 0:
        return segments
    scale = target_duration_seconds / total
    scaled = [
        VoiceoverSegment(
            id=segment.id,
            text=segment.text,
            duration_seconds=round(max(0.05, segment.duration_seconds * scale), 3),
            component_id=segment.component_id,
            formula_focus=segment.formula_focus,
        )
        for segment in segments
    ]
    correction = round(
        target_duration_seconds - sum(segment.duration_seconds for segment in scaled),
        3,
    )
    last = scaled[-1]
    scaled[-1] = VoiceoverSegment(
        id=last.id,
        text=last.text,
        duration_seconds=round(max(0.05, last.duration_seconds + correction), 3),
        component_id=last.component_id,
        formula_focus=last.formula_focus,
    )
    return scaled


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


def _is_gradient_double_well_storyboard(storyboard: Storyboard) -> bool:
    for example in storyboard.examples:
        if str(example.values.get("function_preset", "")).lower() in {
            "double_well_2d",
            "double_well_1d",
        }:
            return True
    text = " ".join(
        [storyboard.one_sentence_summary]
        + [example.title + " " + example.description for example in storyboard.examples]
        + [scene.title + " " + scene.narration for scene in storyboard.scenes]
    )
    return any(keyword in text for keyword in ("2つの谷", "二つの谷", "谷が複数", "局所最小"))


def _is_gradient_double_well_1d_storyboard(storyboard: Storyboard) -> bool:
    for example in storyboard.examples:
        if str(example.values.get("function_preset", "")).lower() == "double_well_1d":
            return True
    text = " ".join(
        [storyboard.one_sentence_summary]
        + [example.title + " " + example.description for example in storyboard.examples]
        + [scene.title + " " + scene.narration for scene in storyboard.scenes]
    )
    return any(keyword in text for keyword in ("1変数", "損失曲線", "谷・山・谷"))


def _gradient_double_well_segment_text() -> dict[str, str]:
    return {
        "intro_landscape": "今回は、2つの谷がある損失地形で勾配降下法を見ます。",
        "two_valleys": "左は浅い局所最小、右はより低い大域最小です。",
        "local_slope": "勾配は今いる場所の上り方向です。下るには、その逆へ進みます。",
        "left_descent": "左側から始めると、近くの浅い谷へ吸い込まれます。",
        "right_descent": "右側から始めると、より深い谷へ向かいます。",
        "compare_minima": "勾配降下法は谷を見比べません。初期位置と局所的な傾きで行き先が決まります。",
        "sgd_bridge": "SGDでは勾配に揺れが入ります。ただし、常に深い谷を選べるわけではありません。",
        "summary": "まとめると、現在地の斜面だけを見て、一歩ずつ下る方法です。",
    }


def _gradient_double_well_1d_segment_text() -> dict[str, str]:
    return {
        "intro_curve": "今回は、1変数の損失曲線で勾配降下法を見ます。",
        "two_valleys_1d": "曲線には、低い谷、山、浅い谷があります。",
        "local_slope_1d": "1変数では、勾配は曲線の傾きです。下る向きへ一歩進みます。",
        "left_descent_1d": "左側から始めると、点は左の低い谷へ下っていきます。",
        "right_descent_1d": "右側から始めると、山を越えずに近くの浅い谷へ下ります。",
        "compare_minima_1d": "勾配降下法は左右の谷を見比べません。現在地の傾きだけで次の一歩を決めます。",
        "sgd_bridge_1d": "SGDでは軌跡が少し揺れます。ただし、いつも山を越えられるわけではありません。",
        "summary_1d": "まとめると、現在地の傾きだけを見て、一歩ずつ損失を下げる方法です。",
    }


def _gradient_surface_3d_segment_text() -> dict[str, str]:
    return {
        "formula_parts": (
            "まず更新式を分解します。シータtは今いる場所、イータは一歩の大きさ、"
            "ナブラLは損失が増える上り方向です。マイナスは、その逆へ進む意味です。"
        ),
        "intro_surface": "次に、曲面の高さを損失として見ます。低い場所ほどよい状態です。",
        "current_point": "現在のパラメータは、この曲面上の一点として表せます。",
        "local_gradient": "勾配は上り方向です。更新では、その逆の下り方向へ進みます。",
        "descent_path": "一歩ずつ更新を繰り返すと、点は谷へ近づいていきます。",
        "summary_surface": "まとめると、今いる場所から、負の勾配方向へ、学習率の分だけ進む式です。",
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
