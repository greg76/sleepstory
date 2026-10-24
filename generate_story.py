#!/usr/bin/env python3
"""Generate story text with Apple's on-device model or a Hugging Face MLX model."""

import argparse
import contextlib
import shutil
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_HF_MODEL = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
MAX_GENERATION_TOKENS = 4096
DEFAULT_SYSTEM_PROMPT = """You write soothing sleep stories for adults.
Follow the user's requested setting, characters, and length. Unless requested
otherwise, aim for about 800 words. Use atmospheric sensory detail, gentle
activity, and a slow, flowing rhythm. Avoid conflict, suspense, frightening
events, and sudden changes. Keep the setting and events consistent. Let the
story settle into a clear, peaceful conclusion.
Write plain text in paragraphs, ready to be read aloud. Output only the story,
without a preamble, commentary, stage directions, or Markdown formatting."""


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("prompt", nargs="?", help="Story request; otherwise read stdin")
    source.add_argument(
        "--prompt-file", type=Path, help="Read the request from a UTF-8 file"
    )
    parser.add_argument(
        "--system-prompt",
        type=Path,
        metavar="FILE",
        help="Replace the default sleep-story instructions with a UTF-8 file",
    )
    parser.add_argument(
        "--model",
        default="auto",
        metavar="MODEL",
        help="auto (default), apple, a Hugging Face MLX repo ID/URL, or a local model directory",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        help="Save text to a new file instead of stdout",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Allow a Hugging Face model download without confirmation",
    )
    return parser


def read_request(args: argparse.Namespace) -> tuple[str, str]:
    piped = not sys.stdin.isatty()
    if args.prompt is not None or args.prompt_file is not None:
        # An empty redirected stdin is common in scripts; only nonempty input conflicts.
        if piped and sys.stdin.read().strip():
            raise ValueError(
                "Use only one prompt source: command line, --prompt-file, or stdin."
            )
        prompt = (
            args.prompt
            if args.prompt is not None
            else args.prompt_file.read_text(encoding="utf-8")
        )
    elif piped:
        prompt = sys.stdin.read()
    else:
        raise ValueError("Provide a prompt, --prompt-file, or piped text.")

    if not prompt.strip():
        raise ValueError("The prompt is empty.")
    system_prompt = (
        args.system_prompt.read_text(encoding="utf-8")
        if args.system_prompt is not None
        else DEFAULT_SYSTEM_PROMPT
    )
    if not system_prompt.strip():
        raise ValueError("The system prompt is empty.")
    return prompt, system_prompt


def find_apple_model() -> tuple[str | None, str]:
    command = shutil.which("fm") if sys.platform == "darwin" else None
    if command is None:
        return None, "Apple's fm command is unavailable (requires macOS 27+)."
    try:
        result = subprocess.run(
            [command, "available", "--model", "system"],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return None, str(error)
    if result.returncode == 0 and result.stdout.strip() == "System model available":
        return command, ""
    reason = (result.stderr or result.stdout).strip()
    return None, reason or "Apple's system model is unavailable."


def generate_apple(command: str, prompt: str, system_prompt: str) -> str:
    print("Using Apple's on-device model.", file=sys.stderr)
    result = subprocess.run(
        [command, "respond", "--no-stream", "--instructions", system_prompt],
        input=prompt,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        reason = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Apple generation failed: {reason}")
    return result.stdout


def model_reference(value: str) -> str:
    if Path(value).expanduser().is_dir():
        return str(Path(value).expanduser())
    if "://" not in value:
        return value
    url = urlparse(value)
    parts = url.path.strip("/").split("/")
    if (
        url.scheme != "https"
        or url.netloc != "huggingface.co"
        or len(parts) != 2
        or not all(parts)
        or url.query
        or url.fragment
    ):
        raise ValueError(
            "Use a model repo ID or its https://huggingface.co/owner/model URL."
        )
    return "/".join(parts)


def resolve_hf_model(reference: str, allow_download: bool) -> Path:
    if Path(reference).is_dir():
        return Path(reference)
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError
    from mlx_lm.utils import DEFAULT_ALLOW_PATTERNS

    options = {"repo_id": reference, "allow_patterns": DEFAULT_ALLOW_PATTERNS}
    try:
        return Path(snapshot_download(**options, local_files_only=True))
    except LocalEntryNotFoundError:
        pass

    size = " (about 2.3 GB)" if reference == DEFAULT_HF_MODEL else ""
    if not allow_download:
        if not sys.stdin.isatty():
            raise ValueError(
                f"Model {reference} is not cached. "
                f"Run again with --yes to download it{size}."
            )
        print(
            f"Download {reference}{size}? [y/N] ", end="", file=sys.stderr, flush=True
        )
        if sys.stdin.readline().strip().lower() not in {"y", "yes"}:
            raise ValueError("Model download cancelled.")
    print(f"Downloading {reference}{size}...", file=sys.stderr)
    return Path(snapshot_download(**options))


def generate_hf(
    reference: str, prompt: str, system_prompt: str, allow_download: bool
) -> str:
    # Keep imports lazy: the Apple backend and --help need only the standard library.
    # Some dependencies print to stdout, which must contain only the generated text.
    with contextlib.redirect_stdout(sys.stderr):
        from mlx_lm import load, stream_generate
        from mlx_lm.sample_utils import make_logits_processors, make_sampler

        path = resolve_hf_model(reference, allow_download)
        print(f"Using {reference}.", file=sys.stderr)
        model, tokenizer = load(str(path))
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        formatted = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        pieces = []
        for response in stream_generate(
            model,
            tokenizer,
            prompt=formatted,
            max_tokens=MAX_GENERATION_TOKENS,
            sampler=make_sampler(temp=0.7, top_p=0.9),
            logits_processors=make_logits_processors(
                repetition_penalty=1.05, repetition_context_size=1024
            ),
        ):
            pieces.append(response.text)
            if response.finish_reason == "length":
                print(
                    f"Warning: reached the {MAX_GENERATION_TOKENS}-token output "
                    "limit; the story may be incomplete.",
                    file=sys.stderr,
                )
        return "".join(pieces)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        prompt, system_prompt = read_request(args)
        if args.output is not None:
            if args.output.exists():
                raise ValueError(f"Output file already exists: {args.output}")
            if not args.output.parent.is_dir():
                raise ValueError(
                    f"Output directory does not exist: {args.output.parent}"
                )

        command, reason = (
            find_apple_model() if args.model in {"auto", "apple"} else (None, "")
        )
        if command is not None:
            story = generate_apple(command, prompt, system_prompt)
        else:
            if args.model == "apple":
                raise RuntimeError(reason)
            if args.model == "auto":
                print(f"{reason} Falling back to {DEFAULT_HF_MODEL}.", file=sys.stderr)
            reference = model_reference(
                DEFAULT_HF_MODEL if args.model == "auto" else args.model
            )
            story = generate_hf(reference, prompt, system_prompt, args.yes)

        if not story.strip():
            raise RuntimeError("The model returned no story text.")
        text = story.rstrip() + "\n"
        if args.output is not None:
            with args.output.open("x", encoding="utf-8") as output:
                output.write(text)
        else:
            sys.stdout.write(text)
        return 0
    except BrokenPipeError:
        # Prevent a second BrokenPipeError when Python flushes stdout at shutdown.
        sys.stdout = open("/dev/null", "w")
        return 0
    except Exception as error:
        # Model libraries use different error types; keep CLI failures on stderr.
        print(f"Error: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    sys.exit(main())
