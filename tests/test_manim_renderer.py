from __future__ import annotations

from math_animation_studio.renderer import manim_renderer


def test_render_environment_prepends_available_tex_paths(tmp_path, monkeypatch) -> None:
    tex_dir = tmp_path / "texbin"
    tex_dir.mkdir()
    monkeypatch.setenv("PATH", "/usr/bin")
    monkeypatch.setattr(manim_renderer, "_candidate_tex_paths", lambda: (tex_dir,))

    env = manim_renderer._render_environment()

    assert env["PATH"].split(":")[:2] == [str(tex_dir), "/usr/bin"]
