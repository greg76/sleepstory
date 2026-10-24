import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import generate_story as story


class StoryCommandTests(unittest.TestCase):
    def invoke(self, args, stdin=""):
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch("sys.stdin", io.StringIO(stdin)),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            status = story.main(args)
        return status, stdout.getvalue(), stderr.getvalue()

    def test_all_prompt_sources_and_system_prompt_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            prompt_file = Path(directory) / "request.txt"
            prompt_file.write_text("A quiet garden", encoding="utf-8")
            system_file = Path(directory) / "instructions.txt"
            system_file.write_text("Write a haiku.", encoding="utf-8")
            sources = [
                (["A quiet garden"], ""),
                (["--prompt-file", str(prompt_file)], ""),
                ([], "A quiet garden"),
            ]
            for args, stdin in sources:
                with (
                    self.subTest(args=args),
                    patch.object(story, "find_apple_model", return_value=("fm", "")),
                    patch.object(
                        story, "generate_apple",
                        return_value="A story.\n\nA peaceful ending.\n",
                    ) as generate,
                ):
                    status, stdout, _ = self.invoke(
                        args + ["--system-prompt", str(system_file)], stdin
                    )
                    self.assertEqual(status, 0)
                    self.assertEqual(stdout, "A story.\n\nA peaceful ending.\n")
                    generate.assert_called_once_with(
                        "fm", "A quiet garden", "Write a haiku."
                    )

    def test_invalid_input_fails_before_model_detection(self):
        cases = [
            ([], ""), ([""], ""), (["request"], "another request"),
            (["--prompt-file", "/missing/prompt.txt"], ""),
        ]
        for args, stdin in cases:
            with (
                self.subTest(args=args, stdin=stdin),
                patch.object(story, "find_apple_model") as detect,
            ):
                status, stdout, stderr = self.invoke(args, stdin)
                self.assertEqual(status, 1)
                self.assertEqual(stdout, "")
                self.assertIn("Error:", stderr)
                detect.assert_not_called()

    def test_command_line_and_file_are_mutually_exclusive(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            story.build_parser().parse_args(["request", "--prompt-file", "prompt.txt"])
        self.assertEqual(error.exception.code, 2)

    def test_unavailable_apple_falls_back_to_default_hf_model(self):
        with (
            patch.object(story, "find_apple_model", return_value=(None, "Not available.")),
            patch.object(story, "generate_hf", return_value="Story") as generate,
        ):
            status, stdout, stderr = self.invoke(["request", "--yes"])
        self.assertEqual(status, 0)
        self.assertEqual(stdout, "Story\n")
        self.assertIn("Falling back", stderr)
        generate.assert_called_once_with(
            story.DEFAULT_HF_MODEL, "request", story.DEFAULT_SYSTEM_PROMPT, True
        )

    def test_explicit_hf_model_bypasses_apple(self):
        with (
            patch.object(story, "find_apple_model") as detect,
            patch.object(story, "generate_hf", return_value="Story") as generate,
        ):
            status, _, _ = self.invoke(
                ["request", "--model", "https://huggingface.co/owner/model/"]
            )
        self.assertEqual(status, 0)
        detect.assert_not_called()
        self.assertEqual(generate.call_args.args[0], "owner/model")

    def test_apple_failure_does_not_silently_switch_models(self):
        with (
            patch.object(story, "find_apple_model", return_value=("fm", "")),
            patch.object(
                story, "generate_apple", side_effect=RuntimeError("Context exceeded")
            ),
            patch.object(story, "generate_hf") as generate,
        ):
            status, stdout, stderr = self.invoke(["request"])
        self.assertEqual(status, 1)
        self.assertEqual(stdout, "")
        self.assertIn("Context exceeded", stderr)
        generate.assert_not_called()

    def test_forced_apple_does_not_fall_back(self):
        with (
            patch.object(story, "find_apple_model", return_value=(None, "Unavailable")),
            patch.object(story, "generate_hf") as generate,
        ):
            status, _, stderr = self.invoke(["request", "--model", "apple"])
        self.assertEqual(status, 1)
        self.assertIn("Unavailable", stderr)
        generate.assert_not_called()

    def test_output_file_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "story.txt"
            with (
                patch.object(story, "find_apple_model", return_value=("fm", "")),
                patch.object(story, "generate_apple", return_value="The story.") as generate,
            ):
                status, stdout, _ = self.invoke(["request", "-o", str(output)])
                self.assertEqual((status, stdout), (0, ""))
                self.assertEqual(output.read_text(), "The story.\n")
                status, stdout, stderr = self.invoke(["request", "-o", str(output)])
                self.assertEqual((status, stdout), (1, ""))
                self.assertIn("already exists", stderr)
                generate.assert_called_once()

    def test_empty_generation_creates_no_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "story.txt"
            with (
                patch.object(story, "find_apple_model", return_value=("fm", "")),
                patch.object(story, "generate_apple", return_value="  \n"),
            ):
                status, stdout, _ = self.invoke(["request", "-o", str(output)])
            self.assertEqual((status, stdout), (1, ""))
            self.assertFalse(output.exists())

    def test_apple_subprocess_uses_stdin_and_captures_diagnostics(self):
        with (
            patch.object(story.subprocess, "run") as run,
            contextlib.redirect_stderr(io.StringIO()),
        ):
            run.return_value.returncode = 0
            run.return_value.stdout = "Complete story."
            self.assertEqual(
                story.generate_apple("fm", "request", "instructions"), "Complete story."
            )
            self.assertEqual(run.call_args.kwargs["input"], "request")
            self.assertIn("--no-stream", run.call_args.args[0])
            run.return_value.returncode = 1
            run.return_value.stderr = "Model failed"
            with self.assertRaisesRegex(RuntimeError, "Model failed"):
                story.generate_apple("fm", "request", "instructions")


if __name__ == "__main__":
    unittest.main()
