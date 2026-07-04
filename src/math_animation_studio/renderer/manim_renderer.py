from __future__ import annotations

import os
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


class RenderError(RuntimeError):
    pass


@dataclass(frozen=True)
class RenderResult:
    video_path: Path
    log_path: Path


class ManimRenderer:
    def __init__(self, *, scene_name: str = "GradientDescent3DScene", quality: str = "l") -> None:
        self.scene_name = scene_name
        self.quality = quality

    def render(self, *, scene_path: Path, output_dir: Path, log_path: Path) -> RenderResult:
        scene_path = scene_path.resolve()
        output_dir = output_dir.resolve()
        log_path = log_path.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        manim_bin = shutil.which("manim")
        if not manim_bin:
            message = (
                "Manim CLI was not found. Install render dependencies with "
                "`python -m pip install -e \".[render]\"` and ensure ffmpeg/LaTeX are available."
            )
            log_path.write_text(message + "\n", encoding="utf-8")
            raise RenderError(message)
        manim_bin = str(Path(manim_bin).resolve())

        media_dir = output_dir / ".manim_media"
        command = [
            manim_bin,
            f"-q{self.quality}",
            str(scene_path),
            self.scene_name,
            "--media_dir",
            str(media_dir),
            "-o",
            "video",
        ]

        process = subprocess.run(
            command,
            cwd=output_dir,
            env=_render_environment(),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_path.write_text(
            "$ " + " ".join(command) + "\n\n" + (process.stdout or ""),
            encoding="utf-8",
        )
        if process.returncode != 0:
            raise RenderError(
                f"Manim rendering failed with exit code {process.returncode}. "
                f"See {log_path}."
            )

        rendered_video = self._find_rendered_video(media_dir)
        final_video = output_dir / "video.mp4"
        shutil.copy2(rendered_video, final_video)
        return RenderResult(video_path=final_video, log_path=log_path)

    def _find_rendered_video(self, media_dir: Path) -> Path:
        candidates = sorted(
            media_dir.rglob("video*.mp4"),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise RenderError(f"Manim finished but no video file was found under {media_dir}.")
        return candidates[0]


def _render_environment() -> dict[str, str]:
    env = os.environ.copy()
    path_items = env.get("PATH", "").split(os.pathsep) if env.get("PATH") else []
    for tex_path in _candidate_tex_paths():
        if tex_path.exists() and str(tex_path) not in path_items:
            path_items.insert(0, str(tex_path))
    env["PATH"] = os.pathsep.join(path_items)
    return env


def _candidate_tex_paths() -> tuple[Path, ...]:
    return (
        Path("/Library/TeX/texbin"),
        Path("/usr/local/texlive/2026/bin/universal-darwin"),
        Path("/usr/local/texlive/2025/bin/universal-darwin"),
    )
