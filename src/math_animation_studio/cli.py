from __future__ import annotations

import json
from dataclasses import replace
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
    PlanArtifacts,
)
from math_animation_studio.understanding.storyboard_dsl import (
    default_formula_first_blueprint,
    storyboard_dsl_prompt_summary,
)
from math_animation_studio.understanding.visual_component_catalog import (
    load_visual_component_catalog,
    visual_component_prompt_summary,
)
from math_animation_studio.schema import TeachingExample
from math_animation_studio.timing import perceptron_target_duration_seconds
from math_animation_studio.validation import ValidationError, validate_python_syntax
from math_animation_studio.voiceover import (
    LLMVoiceoverScriptWriter,
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


def _effective_render_duration_seconds(
    concept: str,
    requested_duration_seconds: int | float,
    *,
    voiceover: bool,
    voice_rate: int | float,
) -> float:
    normalized_concept = concept.strip().lower().replace("-", "_")
    if voiceover and normalized_concept == "perceptron":
        return perceptron_target_duration_seconds(
            requested_duration_seconds,
            voice_rate=voice_rate,
        )
    return float(requested_duration_seconds)


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
def catalog(
    output_format: str = typer.Option(
        "markdown",
        "--format",
        help="Output format: markdown or json.",
    ),
) -> None:
    if output_format not in {"markdown", "json"}:
        typer.echo("Error: --format must be one of markdown, json.", err=True)
        raise typer.Exit(code=1)

    blueprint = default_formula_first_blueprint()
    components = load_visual_component_catalog()
    if output_format == "json":
        typer.echo(
            json.dumps(
                {
                    "storyboard_dsl": blueprint.model_dump(mode="json"),
                    "visual_components": [
                        component.model_dump(mode="json")
                        for component in components.values()
                    ],
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    typer.echo("# Math Animation Studio Catalog")
    typer.echo("")
    typer.echo("## Storyboard DSL")
    typer.echo("")
    typer.echo(storyboard_dsl_prompt_summary())
    typer.echo("")
    typer.echo("## Visual Components")
    typer.echo("")
    typer.echo(visual_component_prompt_summary())


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
    concept_hint: Optional[str] = typer.Option(
        None,
        "--concept-hint",
        help="Optional learning concept hint such as gradient_descent or cross_entropy.",
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
    duration: int = typer.Option(
        30,
        "--duration",
        min=5,
        max=180,
        help="Target rendered video duration in seconds for supported templates.",
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
        130,
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
    interactive_example: bool = typer.Option(
        False,
        "--interactive-example",
        help="Review and optionally edit the recommended teaching example before writing artifacts.",
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
            concept_hint=concept_hint,
            to_storyboard=to_storyboard,
            target_duration_seconds=duration,
        )
        if interactive_example:
            artifacts = _review_teaching_example(
                artifacts,
                planner=planner,
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
            render_duration = _effective_render_duration_seconds(
                artifacts.storyboard.concept,
                duration,
                voiceover=voiceover,
                voice_rate=voice_rate,
            )
            artifact_manager = ArtifactManager(output_dir)
            generator = ManimGenerator(
                target_duration_seconds=render_duration,
                voice_rate=voice_rate if voiceover else None,
            )
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
                segments = script_writer.write_segments(
                    artifacts.storyboard,
                    target_duration_seconds=render_duration,
                    voice_rate=voice_rate,
                )
                if segments:
                    voiceover_result = MacOSSayVoiceover().create_segmented(
                        video_path=render_result.video_path,
                        segments=segments,
                        script_path=manager.narration_path,
                        audio_path=manager.narration_audio_path,
                        output_video_path=manager.video_with_voice_path,
                        log_path=manager.voiceover_log_path,
                        voice=voice,
                        rate=voice_rate,
                    )
                    manager.write_narration(
                        script_writer.write_markdown(
                            artifacts.storyboard,
                            target_duration_seconds=render_duration,
                            voice_rate=voice_rate,
                            segments=segments,
                        )
                    )
                else:
                    if artifacts.llm_used:
                        try:
                            script = LLMVoiceoverScriptWriter().write(
                                storyboard=artifacts.storyboard,
                                target_duration_seconds=int(round(render_duration)),
                                audience=audience,
                                goal=goal,
                            )
                        except LLMUnavailableError:
                            script = script_writer.write(
                                artifacts.storyboard,
                                target_duration_seconds=render_duration,
                                voice_rate=voice_rate,
                            )
                    else:
                        script = script_writer.write(
                            artifacts.storyboard,
                            target_duration_seconds=render_duration,
                            voice_rate=voice_rate,
                        )
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
                    manager.write_narration(
                        script_writer.write_markdown(
                            artifacts.storyboard,
                            target_duration_seconds=render_duration,
                            voice_rate=voice_rate,
                            script=script,
                        )
                    )
                voiceover_video_path = voiceover_result.video_path
                voiceover_audio_path = voiceover_result.audio_path

        manager.write_metadata(
            formula=formula,
            goal=goal,
            audience=audience,
            concept_hint=concept_hint,
            status="success",
            llm_used=artifacts.llm_used,
            video_path=rendered_video_path,
            video_with_voice_path=voiceover_video_path,
            voiceover_audio_path=voiceover_audio_path,
            duration_seconds_target=round(render_duration, 3) if render else None,
        )

        typer.echo(f"Generated planning artifacts in {output_dir}")
        if rendered_video_path is not None:
            typer.echo(f"Generated video: {rendered_video_path}")
        if voiceover_video_path is not None:
            typer.echo(f"Generated narrated video: {voiceover_video_path}")
    except (
        FormulaUnderstandingPlannerError,
        GeneratorError,
        LLMUnavailableError,
        RenderError,
        ValidationError,
        VoiceoverError,
    ) as exc:
        manager.write_metadata(
            formula=formula,
            goal=goal,
            audience=audience,
            concept_hint=concept_hint,
            status="failed",
            llm_used=False,
            error=str(exc),
        )
        typer.echo(f"Error: {exc}", err=True)
        raise typer.Exit(code=1) from exc


def _review_teaching_example(
    artifacts: PlanArtifacts,
    *,
    planner: FormulaUnderstandingPlanner,
    to_storyboard: bool,
) -> PlanArtifacts:
    examples = artifacts.explanation_plan.recommended_examples
    if not examples:
        typer.echo("No recommended teaching examples were generated.")
        return artifacts

    typer.echo("")
    typer.echo("Recommended teaching example")
    typer.echo("----------------------------")
    for index, example in enumerate(examples, start=1):
        typer.echo(f"{index}. {example.title}")
        typer.echo(f"   {example.description}")
        typer.echo(f"   Why: {example.why_it_works}")

    selected = examples[0]
    if len(examples) > 1:
        selected = examples[_prompt_example_index(len(examples)) - 1]
    if typer.confirm("Use this example?", default=True):
        return _replace_teaching_example(
            artifacts,
            planner=planner,
            example=selected,
            to_storyboard=to_storyboard,
        )

    typer.echo("Edit the example. Press Enter to keep the current value.")
    title = typer.prompt("Title", default=selected.title)
    description = typer.prompt("Description", default=selected.description)
    why_it_works = typer.prompt("Why it works", default=selected.why_it_works)
    edited = TeachingExample(
        title=title,
        description=description,
        why_it_works=why_it_works,
        concrete_values=selected.concrete_values,
    )
    return _replace_teaching_example(
        artifacts,
        planner=planner,
        example=edited,
        to_storyboard=to_storyboard,
    )


def _prompt_example_index(example_count: int) -> int:
    while True:
        raw_choice = typer.prompt(
            f"Choose example number [1-{example_count}]",
            default="1",
        )
        try:
            choice = int(raw_choice)
        except ValueError:
            typer.echo("Please enter a number.")
            continue
        if 1 <= choice <= example_count:
            return choice
        typer.echo(f"Please enter a number from 1 to {example_count}.")


def _replace_teaching_example(
    artifacts: PlanArtifacts,
    *,
    planner: FormulaUnderstandingPlanner,
    example: TeachingExample,
    to_storyboard: bool,
) -> PlanArtifacts:
    explanation_plan = artifacts.explanation_plan.model_copy(
        update={"recommended_examples": [example]},
    )
    animation_brief = planner.brief_writer.write(
        formula_analysis=artifacts.formula_analysis,
        classification=artifacts.concept_classification,
        explanation_plan=explanation_plan,
        selected_pattern=artifacts.selected_pattern,
    )
    storyboard = (
        planner.storyboard_adapter.convert(
            formula_analysis=artifacts.formula_analysis,
            explanation_plan=explanation_plan,
        )
        if to_storyboard
        else None
    )
    return replace(
        artifacts,
        explanation_plan=explanation_plan,
        animation_brief=animation_brief,
        storyboard=storyboard,
    )
