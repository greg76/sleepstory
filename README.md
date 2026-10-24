# Audio sleep stories

Create sleep stories and narrated audiobooks locally on your Mac. Generate a
story from a prompt, narrate existing text, or combine both, with optional
background audio.

![The AI dreams of Virtual Reality](cover.png)

## Setup

Audio generation and MLX story models require Apple silicon. Install the Python
packages and FFmpeg:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
brew install ffmpeg
```

## Generate story text

`generate_story.py` writes a calm sleep story from a request, aiming for about
800 words unless you specify a length. Supply the request directly, in a UTF-8
file, or through stdin:

```bash
python3 generate_story.py "A lighthouse keeper tends a seaside garden" -o story.txt
python3 generate_story.py --prompt-file request.txt
cat request.txt | python3 generate_story.py --yes
```

The script tries Apple's on-device model through `fm` (macOS 27+ with Apple
Intelligence enabled). If unavailable, it falls back to
[`mlx-community/Qwen3-4B-Instruct-2507-4bit`](https://huggingface.co/mlx-community/Qwen3-4B-Instruct-2507-4bit),
about 2.3 GB. The Apple backend needs no Python packages.

- `--model` accepts `apple`, an MLX-compatible Hugging Face repo ID or URL, or a
  local model directory. `apple` requires Apple's model to be available.
- `--system-prompt FILE` replaces the default sleep-story instructions.
- `--yes` allows uncached model downloads without confirmation. Required for
  downloads when reading stdin or running without a terminal.
- Output goes to stdout unless you use `-o FILE`.

## Generate audio

`generate_audio.py` narrates a UTF-8 text file or stdin using
`mlx-community/Kokoro-82M-bf16` and saves an M4B audiobook. It reads all text,
including headings, so use plain text for narration.

```bash
python3 generate_audio.py story.txt
python3 generate_audio.py story.txt --voice af_heart --speed 1.0
python3 generate_audio.py story.txt --background rain.m4a --background-volume 0.2
python3 generate_story.py "A peaceful woodland walk" --yes | python3 generate_audio.py -o woodland.m4b
```

- Defaults: British English voice `bm_george`, speed `0.8`, no background audio.
  Use `af_heart` for American English or `ff_siwis` for French. Japanese and
  Mandarin voices require `misaki[ja]` and `misaki[zh]`, respectively.
- Background audio loops to match the narration. `--background-volume` scales
  its amplitude, defaults to `0.2`, and accepts `0` to mute it. Adjust by ear.
- File input saves an M4B beside the source with the same filename stem; stdin
  saves as `narration.m4b`. Automatic names add `_1`, `_2`, etc. to avoid overwriting.
  Use `-o FILE` for an exact path.
- Output is mono AAC at 96 kbit/s, with narration loudness normalized.

Both generators refuse to overwrite paths supplied with `-o`; destination
directories must exist. Status messages and errors go to stderr.

## Tests

Run without loading models. Audio mixing tests require FFmpeg:

```bash
python3 -m unittest test_generate_story.py test_generate_audio.py
```

## Learning from model experiments

- `Llama-3.2-3B-Instruct-4bit`: seemed better suited to (very) short stories.
- `Gemma-4-e4b-it-4bit`: followed requested story lengths more closely; 1,000
  words produced about 8 minutes of audio. Reasoning traces appeared in the
  output and needed filtering before narration.
- `Qwen3-14B-4bit`: Larger model size impacted the generation speed negatively, but improved the creative writing.
- `Kokoro-82M-bf16`: its quality is preferred over the built-in macOS voices, including premium
  voices, for narration. (Also note: the new SiriAI voice is not exposed in macOS for TTS purposes as of macOS 27.0.1)
