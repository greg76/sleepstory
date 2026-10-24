# Audio sleep stories

![The AI dreams of Virtual Reality](cover.png)

## Generate audio

`generate_audio.py` narrates all the text in a UTF-8 file or stdin. Titles,
headings, and paragraph breaks are passed to TTS without content extraction.
Use plain text when you want to avoid pronunciation of formatting characters.

```bash
python3 generate_audio.py story.txt
python3 generate_audio.py article.txt --voice af_heart --speed 1.0
python3 generate_audio.py story.txt --background rain.m4a --background-volume 0.2
cat story.txt | python3 generate_audio.py
python3 generate_story.py "A peaceful woodland walk" | python3 generate_audio.py -o woodland.m4b
```

Defaults are `bm_george`, speed `0.8`, and no background audio. The voice name
selects the language automatically, for example `bm_george` for British English,
`af_heart` for American English, or `ff_siwis` for French. Japanese and Mandarin
voices also require their corresponding `misaki[ja]` or `misaki[zh]` dependencies.
Only individual named Kokoro voices are accepted, not voice blends or voice files.

File input produces an M4B beside the source, with the same filename stem. Piped
input produces `narration.m4b` in the current directory. Automatic names gain
`_1`, `_2`, etc. when needed. `-o FILE` selects an exact path and refuses to
overwrite it. The destination directory must exist. Status messages go to
stderr; generated audio goes into the output file. Failed runs remove their
partial output.

Install the packages in `requirements.txt` and FFmpeg (`brew install ffmpeg` on
macOS). Kokoro runs through MLX on Apple silicon. Model and voice downloads use
the default Hugging Face cache, respecting any existing cache configuration.

TTS chunks stream as PCM directly into FFmpeg, without an intermediate speech
file. Narration uses the original single-pass normalization targets: -23 LUFS
(perceived loudness), -2 dBTP (true peak), and a loudness range of 7 LU. Optional
background audio loops for the narration's duration and is mixed into mono.
The mixer preserves the normalized narration level. A limiter caps mixed sample
peaks at -3 dBFS, including 1 dB of extra headroom for AAC encoding, with automatic
gain disabled. There is no second loudness normalization. The final format is
mono AAC at 64 kbit/s in an M4B container, at Kokoro's sample rate.

`--background-volume` is a nonnegative amplitude multiplier, not a percentage
of perceived loudness. `0` mutes the background; the default is `0.2`. Recordings
vary in loudness, so adjust this for the source. Tests with the local rain and
campfire recordings at `0.05`, `0.1`, and `0.2` kept the final loudness near
-23.3 LUFS. This supports retaining `0.2` as a conservative starting value;
listening is still needed to choose the preferred balance. Samples and
measurements from development are in the ignored `stories/audio_checks/` folder.
AAC encoding can introduce peaks above the pre-encoding limiter ceiling.

Run tests without loading models (FFmpeg mixing tests are skipped if it is not
installed):

```bash
python3 -m unittest test_generate_audio.py
```

## Model experiments

These notes describe the earlier combined generator.

* Using **mlx-lm** & **mlx-audio** frameworks to generate spoken words.
* **Llama-3.2-3B-Instruct-4bit** was used first for generating the story based on the instructions in PROMPT.md
  * It seems to work best with smaller stories. This results ~4 min audio, with the slightly slower speed cadence.
* Moved to **Gemma-4-e4b-it-4bit** for story generation, to see how this more recent, similarly sized model might make a difference. 
  * Needed the latest (v0.31.2) `mlx-lm` version, because previous ones did not support Gemma 4 yet.
  * It's chain-of-thoughts reasoning model will include the though process in the responses that has to be filtered out.
  * Follows the length constraits much better. 1000 words results ~8 min audio.
* Later used **Qwen3-14B-4bit** for speed and quality comparison of creative writing.
* **Kokoro-82M-bf16** is used as the text to speech model (using even the "premium" built-in macOS voices produced weaker results with the `say` command)
