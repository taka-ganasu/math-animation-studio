from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from math_animation_studio import __version__
from math_animation_studio.artifacts import ArtifactManager
from math_animation_studio.generator import GeneratorError, ManimGenerator
from math_animation_studio.llm import LLMUnavailableError
from math_animation_studio.planner import ConceptPlanner, PlannerError
from math_animation_studio.renderer import ManimRenderer, RenderError
from math_animation_studio.validation import ValidationError, validate_python_syntax


app = typer.Typer(
    no_args_is_help=True,
    help="Generate Manim math intuition videos from storyboard JSON.",
)


def _version_callback(value: bool) -> None:
    if value:
        typer.echo(f"math-animation-studio {__version__}")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        False,
        "--version",
        callback=_version_callback,
        is_eager=True,
        help="Show version and exit.",
    ),
) -> None:
    pass


@app.command()
def generate(
    concept: str = typer.Option(..., "--concept", help="Concept name, e.g. gradient_descent."),
    formula: Optional[str] = typer.Option(None, "--formula", help="Target formula in LaTeX."),
    goal: Optional[str] = typer.Option(None, "--goal", help="Learning goal."),
    audience: str = typer.Option(
        "engineer_beginner_math",
        "--audience",
        help="Target audience.",
    ),
    duration: int = typer.Option(60, "--duration", min=1, help="Target duration in seconds."),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        file_okay=False,
        dir_okay=True,
        help="Directory for generated artifacts.",
    ),
    no_render: bool = typer.Option(
        False,
        "--no-render",
        help="Stop after generating Manim code.",
    ),
    template: str = typer.Option(
        "auto",
        "--template",
        help="Template name. Use auto for MVP.",
    ),
    no_llm: bool = typer.Option(
        False,
        "--no-llm",
        help="Use bundled sample storyboard instead of calling an LLM.",
    ),
) -> None:
    manager = ArtifactManager(output_dir)
    manager.prepare()

    try:
        planner = ConceptPlanner(no_llm=no_llm)
        storyboard = planner.plan(
            concept=concept,
            formula=formula,
            goal=goal,
            audience=audience,
        )
        manager.write_storyboard(storyboard)
        manager.write_symbols(storyboard)
        manager.write_narration(storyboard)

        generator = ManimGenerator(template=template)
        generator.generate(storyboard, manager.manim_scene_path)
        validate_python_syntax(manager.manim_scene_path)

        if no_render:
            manager.write_metadata(
                storyboard=storyboard,
                status="generated",
                duration_seconds_target=duration,
            )
            typer.echo(f"Generated artifacts in {output_dir}")
            typer.echo(f"Manim scene: {manager.manim_scene_path}")
            return

        renderer = ManimRenderer()
        result = renderer.render(
            scene_path=manager.manim_scene_path,
            output_dir=output_dir,
            log_path=manager.render_log_path,
        )
        manager.write_metadata(
            storyboard=storyboard,
            status="success",
            duration_seconds_target=duration,
            video_path=result.video_path,
        )
        typer.echo(f"Generated video: {result.video_path}")
    except (
        GeneratorError,
        LLMUnavailableError,
        PlannerError,
        RenderError,
        ValidationError,
    ) as exc:
        status = "render_failed" if isinstance(exc, RenderError) else "failed"
        if "storyboard" in locals():
            manager.write_metadata(
                storyboard=storyboard,
                status=status,
                duration_seconds_target=duration,
                error=str(exc),
            )
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
