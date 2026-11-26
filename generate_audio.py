#!/usr/bin/env python3
"""Narrate a UTF-8 text file or stdin as an M4B audiobook."""

import argparse
import contextlib
import math
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

TTS_MODEL = "mlx-community/Ming-3.0-tts-0.5B-bf16"
DEFAULT_VOICE = "UK_George"
# Limit the CLI to documented English presets; Ming silently ignores unknown voices.
VOICES = ("UK_George", "UK_Charlotte", "US_James", "US_Emma", "US_Stella")
PARAGRAPH_PAUSE_SECONDS = 0.8
DEFAULT_BACKGROUND_VOLUME = 0.2
TARGET_LOUDNESS = -23
MAX_TRUE_PEAK = -2
LOUDNESS_RANGE = 7
# Leave extra room for AAC reconstruction peaks after limiting a loud mix.
MIX_HEADROOM_DB = 4


def background_volume(value: str) -> float:
    volume = float(value)
    if not math.isfinite(volume) or volume < 0:
        raise argparse.ArgumentTypeError("Background volume must be finite and nonnegative.")
    return volume


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input", nargs="?", type=Path, help="UTF-8 text file; omit for stdin"
    )
    parser.add_argument(
        "-o", "--output", type=Path, help="Exact output path; must not already exist"
    )
    parser.add_argument(
        "--voice", choices=VOICES, default=DEFAULT_VOICE,
        help=f"Ming English voice (default: {DEFAULT_VOICE})"
    )
    parser.add_argument(
        "--background", type=Path,
        help="Optional audio file, looped to the narration length",
    )
    parser.add_argument(
        "--background-volume", type=background_volume, default=DEFAULT_BACKGROUND_VOLUME,
        help="Background amplitude multiplier (default: %(default)s; 0 mutes it)",
    )
    return parser


def read_text(input_path: Path | None) -> str:
    if input_path is not None:
        if not sys.stdin.isatty() and sys.stdin.read().strip():
            raise ValueError("Provide either an input file or piped text, not both.")
        text = input_path.read_text(encoding="utf-8")
    else:
        if sys.stdin.isatty():
            raise ValueError("Provide an input file or pipe text into the command.")
        text = sys.stdin.read()
    if not text.strip():
        raise ValueError("The input text is empty.")
    return text


def reserve_output(input_path: Path | None, output_path: Path | None) -> Path:
    """Reserve a name exclusively so existing recordings cannot be overwritten."""
    base = output_path or (input_path.with_suffix(".m4b") if input_path else Path("narration.m4b"))
    counter = 0
    while True:
        candidate = base if counter == 0 else base.with_name(f"{base.stem}_{counter}{base.suffix}")
        try:
            with candidate.open("xb"):
                pass
            return candidate
        except FileExistsError:
            if output_path is not None:
                raise ValueError(f"Output file already exists: {output_path}") from None
            counter += 1


def load_engine() -> Any:
    # Keep --help and input validation independent of MLX and its GPU requirements.
    from mlx_audio.tts.utils import load_model

    print(f"Loading {TTS_MODEL}...", file=sys.stderr)
    with contextlib.redirect_stdout(sys.stderr):
        model = load_model(TTS_MODEL)
    if not getattr(model, "ming3", False):
        raise RuntimeError(
            "This MLX-Audio installation lacks Ming 3 dense support. "
            "Install the dependencies from requirements.txt."
        )
    return model

# fmt: off
def ffmpeg_command(
    executable: str, filename: Path, sample_rate: int,
    background: Path | None, volume: float,
) -> list[str]:
    loudnorm = (
        f"loudnorm=I={TARGET_LOUDNESS}:TP={MAX_TRUE_PEAK}:"
        f"LRA={LOUDNESS_RANGE}:linear=true"
    )
    command = [
        executable, "-hide_banner", "-loglevel", "error", "-nostdin", "-y",
        "-f", "s16le", "-ar", str(sample_rate), "-ac", "1", "-i", "pipe:0",
    ]
    if background is not None:
        limit = 10 ** ((MAX_TRUE_PEAK - MIX_HEADROOM_DB) / 20)
        filters = (
            f"[0:a:0]{loudnorm}[voice];"
            f"[1:a:0]aformat=channel_layouts=mono,volume={volume}[background];"
            "[voice][background]amix=inputs=2:duration=first:normalize=0,"
            f"alimiter=limit={limit}:level=false:latency=true[mixed]"
        )
        command += [
            "-stream_loop", "-1", "-i", str(background),
            "-filter_complex", filters, "-map", "[mixed]",
        ]
    else:
        command += ["-map", "0:a:0", "-af", loudnorm]
    command += [
        "-ar", str(sample_rate), "-ac", "1", "-c:a", "aac", "-b:a", "96k",
        "-map_metadata", "-1", "-metadata", f"title={filename.stem}",
        "-metadata", "genre=Audiobook", "-f", "ipod", str(filename.resolve()),
    ]
    return command
# fmt: on

def stop_process(process: subprocess.Popen) -> None:
    if process.poll() is None:
        with contextlib.suppress(ProcessLookupError):
            process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()


def generation_options(voice: str) -> dict[str, Any]:
    from mlx_audio.tts.models.bailingmm.bailingmm import ming3_instruction

    return {
        "instruct": ming3_instruction(
            ip=voice, speed="慢速", pitch="低", volume="中",
        ),
        "use_zero_spk_emb": True,
        "lang_code": "en",
        "stream": True,
    }


def write_audio(
    filename: Path, text: str, model: Any, voice: str,
    background: Path | None, volume: float, executable: str,
) -> float:
    import numpy as np

    options = generation_options(voice)
    sample_rate = model.sample_rate
    command = ffmpeg_command(executable, filename, sample_rate, background, volume)
    started = time.perf_counter()
    total_samples = 0

    # A file-backed error stream cannot fill up and deadlock the PCM pipe.
    with tempfile.TemporaryFile() as errors:
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=errors
        )
        assert process.stdin is not None
        try:
            with contextlib.redirect_stdout(sys.stderr):
                # Ming splits paragraphs into bounded fragments and yields each when ready.
                paragraphs = re.split(r"\n[ \t]*\n", text.strip())
                pause = bytes(round(sample_rate * PARAGRAPH_PAUSE_SECONDS) * 2)
                for index, paragraph in enumerate(p for p in paragraphs if p.strip()):
                    if index:
                        process.stdin.write(pause)
                        total_samples += len(pause) // 2
                    paragraph_samples = 0
                    for result in model.generate(paragraph, **options):
                        if result.sample_rate != sample_rate:
                            raise RuntimeError("TTS changed its sample rate during generation.")
                        samples = np.asarray(result.audio)
                        if samples.ndim != 1 or not np.isfinite(samples).all():
                            raise RuntimeError("TTS returned invalid mono audio samples.")
                        pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
                        process.stdin.write(pcm)
                        paragraph_samples += samples.size
                    if paragraph_samples == 0:
                        raise RuntimeError("TTS returned no audio samples for a paragraph.")
                    total_samples += paragraph_samples
            if total_samples == 0:
                raise RuntimeError("TTS returned no audio samples.")
        except BaseException as error:
            stop_process(process)
            if isinstance(error, BrokenPipeError):
                errors.seek(0)
                detail = errors.read().decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"FFmpeg stopped accepting audio: {detail}") from error
            raise
        finally:
            with contextlib.suppress(OSError):
                process.stdin.close()
        try:
            returncode = process.wait()
        except BaseException:
            stop_process(process)
            raise
        if returncode != 0:
            errors.seek(0)
            detail = errors.read().decode("utf-8", errors="replace").strip()
            raise RuntimeError(f"FFmpeg failed (exit {returncode}): {detail}")

    return (total_samples / sample_rate) / (time.perf_counter() - started)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    output = None
    complete = False
    try:
        input_path = args.input
        text = read_text(input_path)
        if args.background is not None and not args.background.is_file():
            raise ValueError(f"Background audio file does not exist: {args.background}")
        executable = shutil.which("ffmpeg")
        if executable is None:
            raise RuntimeError("FFmpeg is required. Install it with: brew install ffmpeg")
        output = reserve_output(input_path, args.output)
        model = load_engine()
        print(f"Narrating with {args.voice}...", file=sys.stderr)
        realtime = write_audio(
            output, text, model, args.voice,
            args.background, args.background_volume, executable,
        )
        complete = True
        print(f"Saved {output} ({realtime:.2f}x realtime).", file=sys.stderr)
        return 0
    except KeyboardInterrupt:
        print("Audio generation cancelled.", file=sys.stderr)
        return 130
    except Exception as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    finally:
        if output is not None and not complete:
            output.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
