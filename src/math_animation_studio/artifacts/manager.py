from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

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
        duration_seconds_target: int,
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
