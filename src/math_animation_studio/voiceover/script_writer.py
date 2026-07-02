from __future__ import annotations

from math_animation_studio.schema import Storyboard


class VoiceoverScriptWriter:
    def write(self, storyboard: Storyboard, *, target_duration_seconds: int | None = None) -> str:
        concept = storyboard.concept.strip().lower().replace("-", "_")
        if concept == "cross_entropy":
            if target_duration_seconds is not None and target_duration_seconds >= 25:
                return (
                    "これは、クロスエントロピー損失です。"
                    "まず、正解は猫だとします。"
                    "ワンホットラベルでは、猫だけが一になります。"
                    "モデルが猫に低い確率しか出していないと、悪い予測です。"
                    "このとき、損失は大きくなります。"
                    "右の曲線は、マイナスログです。"
                    "確率pが小さいほど、罰は急に大きくなります。"
                    "逆に、正解に高い確率を出せていれば、損失は小さくなります。"
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

    def write_markdown(
        self,
        storyboard: Storyboard,
        *,
        target_duration_seconds: int | None = None,
        script: str | None = None,
    ) -> str:
        script = script or self.write(storyboard, target_duration_seconds=target_duration_seconds)
        rows = [
            "# Narration",
            "",
            f"Target duration: {target_duration_seconds or 'auto'} seconds",
            "",
            "## Voiceover Script",
            "",
            script,
            "",
            "## Source Scenes",
            "",
        ]
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
