from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from math_animation_studio import __version__
from math_animation_studio.artifacts import ArtifactManager, PlanArtifactManager
from math_animation_studio.generator import GeneratorError, ManimGenerator
from math_animation_studio.llm import LLMUnavailableError
from math_animation_studio.planner import ConceptPlanner, PlannerError
from math_animation_studio.renderer import ManimRenderer, RenderError
from math_animation_studio.understanding.formula_understanding_planner import (
    FormulaUnderstandingPlanner,
    FormulaUnderstandingPlannerError,
)
from math_animation_studio.validation import ValidationError, validate_python_syntax
from math_animation_studio.voiceover import (
    MacOSSayVoiceover,
    VoiceoverError,
    VoiceoverScriptWriter,
)


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

        renderer = ManimRenderer(scene_name=generator.scene_name_for(storyboard))
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


@app.command()
def plan(
    formula: str = typer.Option(..., "--formula", help="Formula to understand."),
    goal: Optional[str] = typer.Option(None, "--goal", help="Learning goal."),
    audience: str = typer.Option(
        "engineer_beginner_math",
        "--audience",
        help="Target audience.",
    ),
    domain_hint: Optional[str] = typer.Option(
        None,
        "--domain-hint",
        help="Optional domain hint such as machine_learning or statistics.",
    ),
    output_dir: Path = typer.Option(
        ...,
        "--output-dir",
        file_okay=False,
        dir_okay=True,
        help="Directory for generated planning artifacts.",
    ),
    no_llm: bool = typer.Option(
        False,
        "--no-llm",
        help="Use deterministic bundled samples instead of calling an LLM.",
    ),
    to_storyboard: bool = typer.Option(
        True,
        "--to-storyboard/--no-storyboard",
        help="Generate storyboard.json compatible with the MVP1 schema.",
    ),
    render: bool = typer.Option(
        False,
        "--render",
        help="If possible, generate and render a Manim video from the storyboard.",
    ),
    voiceover: bool = typer.Option(
        False,
        "--voiceover",
        help="Generate macOS say narration and mux it into the rendered video.",
    ),
    voice: Optional[str] = typer.Option(
        None,
        "--voice",
        help="Optional macOS say voice name. Auto-detects a Japanese voice when omitted.",
    ),
    voice_rate: int = typer.Option(
        220,
        "--voice-rate",
        min=80,
        max=360,
        help="macOS say speaking rate.",
    ),
    output_format: str = typer.Option(
        "all",
        "--format",
        help="Output format: json, markdown, or all.",
    ),
) -> None:
    if output_format not in {"json", "markdown", "all"}:
        typer.echo("Error: --format must be one of json, markdown, all.", err=True)
        raise typer.Exit(code=1)
    if voiceover and not render:
        typer.echo("Error: --voiceover requires --render.", err=True)
        raise typer.Exit(code=1)

    manager = PlanArtifactManager(output_dir)
    manager.prepare()

    try:
        planner = FormulaUnderstandingPlanner(no_llm=no_llm)
        artifacts = planner.plan(
            formula=formula,
            goal=goal,
            audience=audience,
            domain_hint=domain_hint,
            to_storyboard=to_storyboard,
        )

        if output_format in {"json", "all"}:
            manager.write_formula_analysis(artifacts.formula_analysis)
            manager.write_concept_classification(artifacts.concept_classification)
            manager.write_prerequisite_map(artifacts.prerequisite_map)
            manager.write_explanation_plan(artifacts.explanation_plan)
            if artifacts.storyboard is not None:
                manager.write_storyboard(artifacts.storyboard)

        if output_format in {"markdown", "all"}:
            manager.write_animation_brief(artifacts.animation_brief)

        rendered_video_path: Path | None = None
        voiceover_video_path: Path | None = None
        voiceover_audio_path: Path | None = None

        if render:
            if artifacts.storyboard is None:
                raise FormulaUnderstandingPlannerError(
                    "--render requires storyboard generation. Remove --no-storyboard."
                )
            artifact_manager = ArtifactManager(output_dir)
            generator = ManimGenerator()
            generator.generate(artifacts.storyboard, artifact_manager.manim_scene_path)
            validate_python_syntax(artifact_manager.manim_scene_path)
            renderer = ManimRenderer(scene_name=generator.scene_name_for(artifacts.storyboard))
            render_result = renderer.render(
                scene_path=artifact_manager.manim_scene_path,
                output_dir=output_dir,
                log_path=artifact_manager.render_log_path,
            )
            rendered_video_path = render_result.video_path

            if voiceover:
                script_writer = VoiceoverScriptWriter()
                script = script_writer.write(artifacts.storyboard)
                voiceover_result = MacOSSayVoiceover().create(
                    video_path=render_result.video_path,
                    script=script,
                    script_path=manager.narration_path,
                    audio_path=manager.narration_audio_path,
                    output_video_path=manager.video_with_voice_path,
                    log_path=manager.voiceover_log_path,
                    voice=voice,
                    rate=voice_rate,
                )
                manager.write_narration(script_writer.write_markdown(artifacts.storyboard))
                voiceover_video_path = voiceover_result.video_path
                voiceover_audio_path = voiceover_result.audio_path

        manager.write_metadata(
            formula=formula,
            goal=goal,
            audience=audience,
            status="success",
            llm_used=artifacts.llm_used,
            video_path=rendered_video_path,
            video_with_voice_path=voiceover_video_path,
            voiceover_audio_path=voiceover_audio_path,
        )

        typer.echo(f"Generated planning artifacts in {output_dir}")
    except (
        FormulaUnderstandingPlannerError,
        GeneratorError,
        RenderError,
        ValidationError,
        VoiceoverError,
    ) as exc:
        manager.write_metadata(
            formula=formula,
            goal=goal,
            audience=audience,
            status="failed",
            llm_used=False,
            error=str(exc),
        )
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc
