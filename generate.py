#! python3

import contextlib
import io
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import espeakng_loader
import numpy as np
from mlx_audio.tts.utils import load_model as load_tts
from mlx_lm import generate, load
from mlx_lm.sample_utils import make_logits_processors, make_sampler

# Due to some internal errors with misaki originally had to do some
# monkey-patching of the phonemizer to fix AttributeError in
# misaki/espeak.py Now the current version seems to be fixed.
# from phonemizer.backend.espeak.wrapper import EspeakWrapper
# if not hasattr(EspeakWrapper, "set_data_path"):
#    setattr(EspeakWrapper, "set_data_path", lambda _: None)

# Configuration
BACKGROUND_SOUND = Path("rain.m4a")
BACKGROUND_VOLUME = 0.3
VOICE = "bm_george"
OUTPUT_FOLDER = Path("stories")
PROMPT_FILE = Path("PROMPT.md")

# Speech normalization
TARGET_LOUDNESS = -16
MAX_TRUE_PEAK = -1.5
LOUDNESS_RANGE = 11

# Models
#LLM_PATH = "mlx-community/gemma-4-e4b-it-OptiQ-4bit"
LLM_PATH = "mlx-community/Qwen3-14B-4bit"
TTS_PATH = "mlx-community/Kokoro-82M-bf16"

os.environ["ESPEAK_DATA_PATH"] = str(Path(espeakng_loader.get_data_path()).parent)

print("--- Loading Engines ---")
with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
    model, tokenizer, *_ = load(LLM_PATH)
    tokenizer: Any
    tts_model: Any = load_tts(TTS_PATH)  # type: ignore[arg-type]

print("✅ Engines Ready.")


def parse_prompt(file_path: Path) -> list[dict[str, str]]:
    content = file_path.read_text(encoding="utf-8")
    sections = re.split(r"^#\s+", content, flags=re.MULTILINE)

    system_content = ""
    user_content = []

    for section in sections:
        if not section.strip():
            continue
        lines = section.split("\n", 1)
        header = lines[0].strip().lower()
        body = lines[1].strip() if len(lines) > 1 else ""

        if "system prompt" in header:
            system_content = body
        else:
            user_content.append(f"# {section.strip()}")

    messages = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    if user_content:
        messages.append({"role": "user", "content": "\n\n".join(user_content)})
    return messages


def strip_thinking(text: str) -> str:
    text = re.sub(r"<\|channel>.*?<channel\|>", "", text, flags=re.DOTALL)
    text = re.sub(r"<\|channel>.*", "", text, flags=re.DOTALL)
    return text.strip()


def extract_title(story_raw: str) -> tuple[str, str]:
    lines = [line.strip() for line in story_raw.splitlines() if line.strip()]

    for i, line in enumerate(lines):
        if m := re.match(r"^(?:#+\s*|\*{2,})(.+?)\*{0,2}$", line):
            extracted_title = m.group(1).strip()
            story_body = "\n".join(lines[i + 1 :]).strip()
            return extracted_title, story_body

    return "Untitled Cyberpunk Story", "\n".join(lines).strip()


def generate_story() -> dict[str, Any]:
    messages = parse_prompt(PROMPT_FILE)
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    sampler = make_sampler(temp=0.7, top_p=0.9)
    logits_processors = make_logits_processors(
        repetition_penalty=1.03,
        repetition_context_size=1024,
    )

    print("--- Generating Story (this may take a minute) ---")
    start_text = time.perf_counter()
    response_raw = generate(
        model,
        tokenizer,
        prompt=prompt,
        max_tokens=2000,
        sampler=sampler,
        logits_processors=logits_processors,
    )
    story_raw = strip_thinking(response_raw)
    end_text = time.perf_counter()

    title, body = extract_title(story_raw)
    tokens = len(tokenizer.encode(story_raw))

    return {
        "title": title,
        "body": body,
        "token_per_sec": tokens / (end_text - start_text),
    }


def write_audio(filename: Path, title: str, story: str) -> float:
    loudnorm_filter = f"loudnorm=I={TARGET_LOUDNESS}:TP={MAX_TRUE_PEAK}:LRA={LOUDNESS_RANGE}:linear=true"

    if BACKGROUND_SOUND and BACKGROUND_SOUND.exists():
        filter_str = (
            f"[0:a]{loudnorm_filter}[v]; "
            f"[1:a]volume={BACKGROUND_VOLUME}[r]; "
            f"[v][r]amix=inputs=2:duration=first:dropout_transition=2[mixed]"
        )
        command = [
            "ffmpeg", "-y",
            "-f", "s16le", "-ar", "24000", "-ac", "1",
            "-i", "pipe:0",
            "-stream_loop", "-1", "-i", str(BACKGROUND_SOUND),
            "-filter_complex", filter_str,
            "-map", "[mixed]",
            "-ac", "1", "-c:a", "aac", "-b:a", "64k",
            "-metadata", f"title={title}",
            "-metadata", "genre=Audiobook",
            "-metadata:s:a:0", "media_type=10",
            str(filename),
        ]
    else:
        command = [
            "ffmpeg", "-y",
            "-f", "s16le", "-ar", "24000", "-ac", "1",
            "-i", "pipe:0",
            "-af", loudnorm_filter,
            "-ac", "1", "-c:a", "aac", "-b:a", "64k",
            "-metadata", f"title={title}",
            "-metadata", "genre=Audiobook",
            "-metadata:s:a:0", "media_type=10",
            str(filename),
        ]

    process = subprocess.Popen(
        command, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL
    )

    start_audio = time.perf_counter()
    total_samples = 0.0

    print(f"--- Generating Audio: {title} ---")
    try:
        assert process.stdin is not None
        for result in tts_model.generate(story, voice=VOICE, lang_code="b", speed=0.8):
            audio_np = np.array(result.audio)
            samples = (audio_np * 32767).astype(np.int16).tobytes()
            total_samples += len(samples) / 2
            process.stdin.write(samples)
    finally:
        if process.stdin:
            process.stdin.close()
        process.wait()

    if process.returncode != 0:
        raise RuntimeError(f"FFmpeg failed with exit code {process.returncode}")

    end_audio = time.perf_counter()
    return (total_samples / 24000) / (end_audio - start_audio)


def main():
    if BACKGROUND_SOUND and not BACKGROUND_SOUND.exists():
        print(f"Error: Background sound sound file '{BACKGROUND_SOUND}' missing.")
        sys.exit(1)

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    import_story_path: Path | None = None

    if import_story_path and import_story_path.exists():
        raw = import_story_path.read_text(encoding="utf-8")
        title, body = extract_title(raw)
        story = {"title": title, "body": body, "token_per_sec": None}
    else:
        story = generate_story()

    safe_base = re.sub(r"[^\w\s-]", "", story["title"]).replace(" ", "_").lower()

    if not import_story_path:
        text_filename = OUTPUT_FOLDER / f"{safe_base}.md"
        text_filename.write_text(f"{story['title']}\n\n{story['body']}", encoding="utf-8")

    audio_filename = OUTPUT_FOLDER / f"{safe_base}.m4b"
    audio_speed = write_audio(audio_filename, story["title"], story["body"])

    tps_val = story['token_per_sec']
    tps = f"{tps_val:.2f} tokens/sec" if isinstance(tps_val, float) else "N/A"
    audio_x = f"{audio_speed:.2f}x realtime"

    print("\n" + "=" * 40)
    print(f"📖 STORY: {story['title']}")
    print("=" * 40)
    print(f"LLM:   {tps}")
    print(f"Audio: {audio_x}")
    print("=" * 40 + "\n")


if __name__ == "__main__":
    main()
