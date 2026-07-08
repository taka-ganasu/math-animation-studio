from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from math_animation_studio.voiceover.script_writer import VoiceoverSegment


class VoiceoverError(RuntimeError):
    pass


@dataclass(frozen=True)
class VoiceoverResult:
    script_path: Path
    audio_path: Path
    video_path: Path
    log_path: Path
    voice: str | None


class MacOSSayVoiceover:
    def create(
        self,
        *,
        video_path: Path,
        script: str,
        script_path: Path,
        audio_path: Path,
        output_video_path: Path,
        log_path: Path,
        voice: str | None = None,
        rate: int = 130,
    ) -> VoiceoverResult:
        say_bin = shutil.which("say")
        ffmpeg_bin = shutil.which("ffmpeg")
        ffprobe_bin = shutil.which("ffprobe")
        if say_bin is None:
            raise VoiceoverError("macOS say command was not found.")
        if ffmpeg_bin is None:
            raise VoiceoverError("ffmpeg command was not found.")
        if ffprobe_bin is None:
            raise VoiceoverError("ffprobe command was not found.")

        video_path = video_path.resolve()
        script_path = script_path.resolve()
        audio_path = audio_path.resolve()
        output_video_path = output_video_path.resolve()
        log_path = log_path.resolve()

        script_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(script.strip() + "\n", encoding="utf-8")
        video_duration = self._probe_duration(ffprobe_bin, video_path)

        selected_voice = voice or self._detect_japanese_voice(say_bin)
        say_command = self._say_command(
            say_bin=say_bin,
            rate=rate,
            audio_path=audio_path,
            script=script,
            voice=selected_voice,
        )

        say_process = subprocess.run(
            say_command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if say_process.returncode != 0 and selected_voice:
            fallback_command = self._say_command(
                say_bin=say_bin,
                rate=rate,
                audio_path=audio_path,
                script=script,
                voice=None,
            )
            fallback_process = subprocess.run(
                fallback_command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            say_output = (
                "$ " + " ".join(say_command) + "\n"
                + (say_process.stdout or "")
                + "\n$ " + " ".join(fallback_command) + "\n"
                + (fallback_process.stdout or "")
            )
            say_process = fallback_process
            selected_voice = None
        else:
            say_output = "$ " + " ".join(say_command) + "\n" + (say_process.stdout or "")

        if say_process.returncode != 0:
            log_path.write_text(say_output, encoding="utf-8")
            raise VoiceoverError(f"Voice synthesis failed. See {log_path}.")

        audio_duration = self._probe_duration(ffprobe_bin, audio_path)
        mux_duration = max(video_duration, audio_duration)
        if audio_duration > video_duration:
            say_output += (
                "\n# Preserved requested voice rate; audio is longer than video."
                f"\n# audio_duration={audio_duration:.3f}, video_duration={video_duration:.3f}\n"
            )

        ffmpeg_command = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-filter_complex",
            f"[1:a]volume=1.0,apad=pad_dur={mux_duration:.3f}[a]",
            "-map",
            "0:v:0",
            "-map",
            "[a]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-t",
            f"{mux_duration:.3f}",
            str(output_video_path),
        ]
        ffmpeg_process = subprocess.run(
            ffmpeg_command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_path.write_text(
            say_output
            + "\n\n$ "
            + " ".join(ffmpeg_command)
            + "\n"
            + (ffmpeg_process.stdout or ""),
            encoding="utf-8",
        )
        if ffmpeg_process.returncode != 0:
            raise VoiceoverError(f"Voiceover muxing failed. See {log_path}.")

        return VoiceoverResult(
            script_path=script_path,
            audio_path=audio_path,
            video_path=output_video_path,
            log_path=log_path,
            voice=selected_voice,
        )

    def create_segmented(
        self,
        *,
        video_path: Path,
        segments: Sequence[VoiceoverSegment],
        script_path: Path,
        audio_path: Path,
        output_video_path: Path,
        log_path: Path,
        voice: str | None = None,
        rate: int = 130,
    ) -> VoiceoverResult:
        if not segments:
            raise VoiceoverError("Segmented voiceover requires at least one segment.")

        say_bin = shutil.which("say")
        ffmpeg_bin = shutil.which("ffmpeg")
        ffprobe_bin = shutil.which("ffprobe")
        if say_bin is None:
            raise VoiceoverError("macOS say command was not found.")
        if ffmpeg_bin is None:
            raise VoiceoverError("ffmpeg command was not found.")
        if ffprobe_bin is None:
            raise VoiceoverError("ffprobe command was not found.")

        video_path = video_path.resolve()
        script_path = script_path.resolve()
        audio_path = audio_path.resolve()
        output_video_path = output_video_path.resolve()
        log_path = log_path.resolve()

        script_path.parent.mkdir(parents=True, exist_ok=True)
        audio_path.parent.mkdir(parents=True, exist_ok=True)
        output_video_path.parent.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        script_path.write_text(self._segmented_script_markdown(segments), encoding="utf-8")

        segment_dir = audio_path.parent / ".voiceover_segments"
        segment_dir.mkdir(parents=True, exist_ok=True)
        selected_voice = voice or self._detect_japanese_voice(say_bin)
        log_sections = ["# Segmented voiceover synthesis"]
        padded_paths: list[Path] = []
        padded_durations: list[float] = []

        for index, segment in enumerate(segments):
            stem = f"{index:02d}_{self._safe_file_stem(segment.id)}"
            raw_path = segment_dir / f"{stem}.aiff"
            padded_path = segment_dir / f"{stem}.wav"
            command = self._say_command(
                say_bin=say_bin,
                rate=rate,
                audio_path=raw_path,
                script=segment.text,
                voice=selected_voice,
            )
            process = subprocess.run(
                command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            log_sections.append(
                self._segment_log_header(index, segment)
                + "$ "
                + " ".join(command)
                + "\n"
                + (process.stdout or "")
            )
            if process.returncode != 0 and selected_voice:
                fallback_command = self._say_command(
                    say_bin=say_bin,
                    rate=rate,
                    audio_path=raw_path,
                    script=segment.text,
                    voice=None,
                )
                fallback_process = subprocess.run(
                    fallback_command,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
                log_sections.append(
                    "# Fallback without selected voice.\n$ "
                    + " ".join(fallback_command)
                    + "\n"
                    + (fallback_process.stdout or "")
                )
                process = fallback_process
                selected_voice = None
            if process.returncode != 0:
                log_path.write_text("\n\n".join(log_sections), encoding="utf-8")
                raise VoiceoverError(f"Voice synthesis failed. See {log_path}.")

            audio_duration = self._probe_duration(ffprobe_bin, raw_path)
            padded_duration = max(segment.duration_seconds, audio_duration + 0.25)
            if audio_duration > segment.duration_seconds:
                log_sections.append(
                    "# Preserved requested voice rate; segment audio is longer than "
                    f"allocated duration for {segment.id}.\n"
                    f"# audio_duration={audio_duration:.3f}, "
                    f"segment_duration={segment.duration_seconds:.3f}, "
                    f"padded_duration={padded_duration:.3f}"
                )

            pad_command = [
                ffmpeg_bin,
                "-y",
                "-i",
                str(raw_path),
                "-filter:a",
                (
                    f"apad=pad_dur={padded_duration:.3f},"
                    f"atrim=0:{padded_duration:.3f},"
                    "asetpts=N/SR/TB"
                ),
                "-ar",
                "44100",
                "-ac",
                "2",
                "-c:a",
                "pcm_s16le",
                str(padded_path),
            ]
            pad_process = subprocess.run(
                pad_command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            log_sections.append("$ " + " ".join(pad_command) + "\n" + (pad_process.stdout or ""))
            if pad_process.returncode != 0:
                log_path.write_text("\n\n".join(log_sections), encoding="utf-8")
                raise VoiceoverError(f"Segment audio padding failed. See {log_path}.")
            padded_paths.append(padded_path)
            padded_durations.append(padded_duration)

        concat_list_path = segment_dir / "concat.txt"
        concat_list_path.write_text(
            "\n".join(f"file '{path.name}'" for path in padded_paths) + "\n",
            encoding="utf-8",
        )
        concat_command = [
            ffmpeg_bin,
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            concat_list_path.name,
            "-c:a",
            "pcm_s16be",
            str(audio_path),
        ]
        concat_process = subprocess.run(
            concat_command,
            cwd=segment_dir,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_sections.append("$ " + " ".join(concat_command) + "\n" + (concat_process.stdout or ""))
        if concat_process.returncode != 0:
            log_path.write_text("\n\n".join(log_sections), encoding="utf-8")
            raise VoiceoverError(f"Segment audio concat failed. See {log_path}.")

        video_duration = self._probe_duration(ffprobe_bin, video_path)
        audio_duration = sum(padded_durations)
        mux_duration = max(video_duration, audio_duration)
        ffmpeg_command = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-filter_complex",
            f"[1:a]volume=1.0,apad=pad_dur={mux_duration:.3f}[a]",
            "-map",
            "0:v:0",
            "-map",
            "[a]",
            "-c:v",
            "copy",
            "-c:a",
            "aac",
            "-b:a",
            "160k",
            "-t",
            f"{mux_duration:.3f}",
            str(output_video_path),
        ]
        ffmpeg_process = subprocess.run(
            ffmpeg_command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_sections.append("$ " + " ".join(ffmpeg_command) + "\n" + (ffmpeg_process.stdout or ""))
        log_path.write_text("\n\n".join(log_sections), encoding="utf-8")
        if ffmpeg_process.returncode != 0:
            raise VoiceoverError(f"Voiceover muxing failed. See {log_path}.")

        return VoiceoverResult(
            script_path=script_path,
            audio_path=audio_path,
            video_path=output_video_path,
            log_path=log_path,
            voice=selected_voice,
        )

    def _say_command(
        self,
        *,
        say_bin: str,
        rate: int,
        audio_path: Path,
        script: str,
        voice: str | None,
    ) -> list[str]:
        command = [
            say_bin,
            "-r",
            str(rate),
            "-o",
            str(audio_path),
        ]
        if voice:
            command.extend(["-v", voice])
        command.append(script)
        return command

    def _segmented_script_markdown(self, segments: Sequence[VoiceoverSegment]) -> str:
        rows = [
            "# Narration",
            "",
            "## Voiceover Segments",
            "",
        ]
        for segment in segments:
            rows.extend(
                [
                    f"### {segment.id} ({segment.duration_seconds:.2f}s)",
                    "",
                    segment.text,
                    "",
                ]
            )
        rows.extend(
            [
                "## Voiceover Script",
                "",
                "".join(segment.text for segment in segments),
                "",
            ]
        )
        return "\n".join(rows)

    def _segment_log_header(self, index: int, segment: VoiceoverSegment) -> str:
        return f"## Segment {index:02d}: {segment.id} ({segment.duration_seconds:.3f}s)\n"

    def _safe_file_stem(self, value: str) -> str:
        safe = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
        return safe or "segment"

    def _probe_duration(self, ffprobe_bin: str, video_path: Path) -> float:
        process = subprocess.run(
            [
                ffprobe_bin,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                str(video_path),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if process.returncode != 0:
            raise VoiceoverError(f"Could not inspect video duration: {process.stdout}")
        try:
            duration = float(process.stdout.strip())
        except ValueError as exc:
            raise VoiceoverError(
                f"Could not parse video duration from ffprobe output: {process.stdout}"
            ) from exc
        if duration <= 0:
            raise VoiceoverError(f"Video duration must be positive, got {duration}.")
        return duration

    def _detect_japanese_voice(self, say_bin: str) -> str | None:
        process = subprocess.run(
            [say_bin, "-v", "?"],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        if process.returncode != 0:
            return None

        lines = process.stdout.splitlines()
        for line in lines:
            if re.match(r"^Kyoko\s{2,}ja_JP\b", line):
                return "Kyoko"

        for line in lines:
            match = re.match(r"^(.+?)\s{2,}ja_JP\b", line)
            if match:
                return match.group(1).strip()
        return None
