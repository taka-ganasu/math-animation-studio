from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from math_animation_studio.schema import (
    ConceptClassification,
    ExplanationPlan,
    FormulaAnalysis,
    PrerequisiteMap,
)
from math_animation_studio.schema import Storyboard, save_storyboard


class ArtifactManager:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def prepare(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def storyboard_path(self) -> Path:
        return self.output_dir / "storyboard.json"

    @property
    def symbols_path(self) -> Path:
        return self.output_dir / "symbols.md"

    @property
    def narration_path(self) -> Path:
        return self.output_dir / "narration.md"

    @property
    def manim_scene_path(self) -> Path:
        return self.output_dir / "manim_scene.py"

    @property
    def render_log_path(self) -> Path:
        return self.output_dir / "render.log"

    @property
    def metadata_path(self) -> Path:
        return self.output_dir / "metadata.json"

    def write_storyboard(self, storyboard: Storyboard) -> None:
        save_storyboard(storyboard, self.storyboard_path)

    def write_symbols(self, storyboard: Storyboard) -> None:
        rows = ["# Symbols", ""]
        for item in storyboard.symbol_ledger:
            rows.append(f"## {item.symbol}")
            rows.append("")
            rows.append(f"- Meaning: {item.meaning}")
            if item.intuition:
                rows.append(f"- Intuition: {item.intuition}")
            rows.append("")
        self.symbols_path.write_text("\n".join(rows), encoding="utf-8")

    def write_narration(self, storyboard: Storyboard) -> None:
        rows = ["# Narration", ""]
        for scene in storyboard.scenes:
            rows.append(f"## {scene.id}: {scene.title}")
            rows.append("")
            rows.append(scene.narration)
            rows.append("")
        self.narration_path.write_text("\n".join(rows), encoding="utf-8")

    def write_metadata(
        self,
        *,
        storyboard: Storyboard,
        status: str,
        duration_seconds_target: int | float,
        renderer: str = "manim",
        video_path: Path | None = None,
        error: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "concept": storyboard.concept,
            "created_at": datetime.now().astimezone().isoformat(),
            "formula": storyboard.formula,
            "renderer": renderer,
            "status": status,
            "duration_seconds_target": duration_seconds_target,
        }
        if video_path is not None:
            payload["video_path"] = str(video_path)
        if error is not None:
            payload["error"] = error
        self.metadata_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


class PlanArtifactManager:
    def __init__(self, output_dir: Path) -> None:
        self.output_dir = output_dir

    def prepare(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    @property
    def formula_analysis_path(self) -> Path:
        return self.output_dir / "formula_analysis.json"

    @property
    def concept_classification_path(self) -> Path:
        return self.output_dir / "concept_classification.json"

    @property
    def prerequisite_map_path(self) -> Path:
        return self.output_dir / "prerequisite_map.json"

    @property
    def explanation_plan_path(self) -> Path:
        return self.output_dir / "explanation_plan.json"

    @property
    def animation_brief_path(self) -> Path:
        return self.output_dir / "animation_brief.md"

    @property
    def storyboard_path(self) -> Path:
        return self.output_dir / "storyboard.json"

    @property
    def metadata_path(self) -> Path:
        return self.output_dir / "metadata.json"

    @property
    def narration_path(self) -> Path:
        return self.output_dir / "narration.md"

    @property
    def narration_audio_path(self) -> Path:
        return self.output_dir / "narration.aiff"

    @property
    def voiceover_log_path(self) -> Path:
        return self.output_dir / "voiceover.log"

    @property
    def video_with_voice_path(self) -> Path:
        return self.output_dir / "video_with_voice.mp4"

    def write_formula_analysis(self, formula_analysis: FormulaAnalysis) -> None:
        self.formula_analysis_path.write_text(
            formula_analysis.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def write_concept_classification(self, classification: ConceptClassification) -> None:
        self.concept_classification_path.write_text(
            classification.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def write_prerequisite_map(self, prerequisite_map: PrerequisiteMap) -> None:
        self.prerequisite_map_path.write_text(
            prerequisite_map.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def write_explanation_plan(self, explanation_plan: ExplanationPlan) -> None:
        self.explanation_plan_path.write_text(
            explanation_plan.model_dump_json(indent=2),
            encoding="utf-8",
        )

    def write_animation_brief(self, content: str) -> None:
        self.animation_brief_path.write_text(content, encoding="utf-8")

    def write_narration(self, content: str) -> None:
        self.narration_path.write_text(content, encoding="utf-8")

    def write_storyboard(self, storyboard: Storyboard) -> None:
        save_storyboard(storyboard, self.storyboard_path)

    def write_metadata(
        self,
        *,
        formula: str,
        goal: str | None,
        audience: str,
        status: str,
        llm_used: bool,
        concept_hint: str | None = None,
        video_path: Path | None = None,
        video_with_voice_path: Path | None = None,
        voiceover_audio_path: Path | None = None,
        duration_seconds_target: int | float | None = None,
        error: str | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "command": "plan",
            "formula": formula,
            "goal": goal,
            "audience": audience,
            "concept_hint": concept_hint,
            "created_at": datetime.now().astimezone().isoformat(),
            "status": status,
            "llm_used": llm_used,
        }
        if video_path is not None:
            payload["video_path"] = str(video_path)
        if video_with_voice_path is not None:
            payload["video_with_voice_path"] = str(video_with_voice_path)
        if voiceover_audio_path is not None:
            payload["voiceover_audio_path"] = str(voiceover_audio_path)
        if duration_seconds_target is not None:
            payload["duration_seconds_target"] = duration_seconds_target
        if error:
            payload["error"] = error
        self.metadata_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
