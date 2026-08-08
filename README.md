# Audio sleep stories

![The AI dreams of Virtual Reality](cover.png)

* Using **mlx-lm** & **mlx-audio** frameworks to generate spoken words.
* **Llama-3.2-3B-Instruct-4bit** was used first for generating the story based on the instructions in PROMPT.md
  * It seems to work best with smaller stories. This results ~4 min audio, with the slightly slower speed cadence.
* Moved to **Gemma-4-e4b-it-4bit** for story generation, to see how this more recent, similarly sized model might make a difference. 
  * Needed the latest (v0.31.2) `mlx-lm` version, because previous ones did not support Gemma 4 yet.
  * It's chain-of-thoughts reasoning model will include the though process in the responses that has to be filtered out.
  * Follows the length constraits much better. 1000 words results ~8 min audio.
* Later used **Qwen3-14B-4bit** for speed and quality comparison of creative writing.
* **Kokoro-82M-bf16** is used as the text to speech model (using even the "premium" built-in macOS voices produced weaker results with the `say` command)
* Add your own sound file (eg. `BACKGROUND_SOUND = "rain.m4a"`) for background soundscape.
* There were some weird type annotation bugs in some of the underlying packages, hence the `# type: ignore` comments for the linter.
* Both the `markdown` file and the `m4b` (aac audio book format) files are saved to a `stories` subfolder.
