from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path


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
        rate: int = 220,
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
        if audio_duration > video_duration:
            adjusted_rate = min(
                360,
                max(rate + 1, int(rate * audio_duration / (video_duration * 0.96))),
            )
            adjusted_command = self._say_command(
                say_bin=say_bin,
                rate=adjusted_rate,
                audio_path=audio_path,
                script=script,
                voice=selected_voice,
            )
            adjusted_process = subprocess.run(
                adjusted_command,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                check=False,
            )
            say_output += (
                "\n# Adjusted voice rate to fit video duration.\n$ "
                + " ".join(adjusted_command)
                + "\n"
                + (adjusted_process.stdout or "")
            )
            if adjusted_process.returncode != 0:
                log_path.write_text(say_output, encoding="utf-8")
                raise VoiceoverError(f"Voice synthesis failed. See {log_path}.")

        ffmpeg_command = [
            ffmpeg_bin,
            "-y",
            "-i",
            str(video_path),
            "-i",
            str(audio_path),
            "-filter_complex",
            f"[1:a]volume=1.0,apad=pad_dur={video_duration:.3f}[a]",
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
            f"{video_duration:.3f}",
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
