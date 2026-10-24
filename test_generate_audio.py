import array
import contextlib
import io
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import patch

import generate_audio as audio


class AudioCommandTests(unittest.TestCase):
    def invoke(self, args, stdin=""):
        stdout, stderr = io.StringIO(), io.StringIO()
        with (
            patch("sys.stdin", io.StringIO(stdin)),
            contextlib.redirect_stdout(stdout),
            contextlib.redirect_stderr(stderr),
        ):
            status = audio.main(args)
        return status, stdout.getvalue(), stderr.getvalue()

    def test_voice_language(self):
        for voice, language in [("bm_george", "b"), ("af_heart", "a"), ("ff_siwis", "f")]:
            self.assertEqual(audio.voice_language(voice), language)
        for invalid in ["", "unknown", "xm_unknown", "../../bm_george"]:
            with self.assertRaises(ValueError):
                audio.voice_language(invalid)

    def test_speed_and_volume_validation(self):
        for option, values in [
            ("--speed", ["0", "-1", "nan", "inf"]),
            ("--background-volume", ["-1", "nan", "inf"]),
        ]:
            for value in values:
                with (
                    self.subTest(option=option, value=value),
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit),
                ):
                    audio.build_parser().parse_args([option, value])
        self.assertEqual(audio.background_volume("0"), 0)

    def test_file_input_preserves_all_text_and_passes_options(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "story.txt"
            text = "# Title\n\nOpening paragraph.\n\n## A heading\n\nThe ending.\n"
            source.write_text(text)
            background = Path(directory) / "background.wav"
            background.touch()
            model = object()
            with (
                patch.object(audio, "load_engine", return_value=model),
                patch.object(audio, "write_audio", return_value=2.0) as write,
                patch.object(audio.shutil, "which", return_value="ffmpeg"),
            ):
                status, stdout, _ = self.invoke([
                    str(source), "--voice", "af_heart", "--speed", "1.1",
                    "--background", str(background), "--background-volume", "0.1",
                ])
            self.assertEqual((status, stdout), (0, ""))
            write.assert_called_once_with(
                source.with_suffix(".m4b"), text, model, "af_heart", 1.1,
                background, 0.1, "ffmpeg",
            )

    def test_piped_input(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out.m4b"
            with (
                patch.object(audio, "load_engine"),
                patch.object(audio, "write_audio", return_value=1.0) as write,
                patch.object(audio.shutil, "which", return_value="ffmpeg"),
            ):
                status, stdout, _ = self.invoke(
                    ["-o", str(output)], "Entire text\n\nEnd."
                )
            self.assertEqual((status, stdout), (0, ""))
            self.assertEqual(write.call_args.args[1], "Entire text\n\nEnd.")

    def test_invalid_input_does_not_load_model(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = str(Path(directory) / "missing.txt")
            for args, stdin in [
                ([], ""), ([missing], ""),
                ([missing], "Also piped"),
                (["--background", missing], "A story"),
                (["--voice", "xx_invalid"], "A story"),
            ]:
                with patch.object(audio, "load_engine") as load:
                    status, stdout, stderr = self.invoke(args, stdin)
                self.assertEqual((status, stdout), (1, ""))
                self.assertIn("Error:", stderr)
                load.assert_not_called()

    def test_automatic_names_and_explicit_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            original = Path.cwd()
            try:
                os.chdir(directory)
                for suffix in ["", "_1", "_2"]:
                    self.assertEqual(
                        audio.reserve_output(None, None), Path(f"narration{suffix}.m4b")
                    )
                source = Path(directory) / "story.txt"
                self.assertEqual(audio.reserve_output(source, None), source.with_suffix(".m4b"))
                self.assertEqual(
                    audio.reserve_output(source, None), source.with_name("story_1.m4b")
                )
                with self.assertRaises(ValueError):
                    audio.reserve_output(None, source.with_suffix(".m4b"))
            finally:
                os.chdir(original)

    def test_failure_removes_partial_output(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out.m4b"
            with (
                patch.object(audio, "load_engine"),
                patch.object(audio, "write_audio", side_effect=RuntimeError("TTS failed")),
                patch.object(audio.shutil, "which", return_value="ffmpeg"),
            ):
                status, _, stderr = self.invoke(["-o", str(output)], "A story")
            self.assertEqual(status, 1)
            self.assertIn("TTS failed", stderr)
            self.assertFalse(output.exists())

    def test_existing_explicit_output_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "out.m4b"
            output.write_bytes(b"existing audio")
            with (
                patch.object(audio, "load_engine") as load,
                patch.object(audio.shutil, "which", return_value="ffmpeg"),
            ):
                status, _, stderr = self.invoke(["-o", str(output)], "A story")
            self.assertEqual(status, 1)
            self.assertIn("already exists", stderr)
            self.assertEqual(output.read_bytes(), b"existing audio")
            load.assert_not_called()


@unittest.skipUnless(shutil.which("ffmpeg"), "FFmpeg is not installed")
class MixingTests(unittest.TestCase):
    sample_rate = 24000

    def sine_pcm(self, seconds, amplitude, frequency):
        samples = array.array("h", (
            round(amplitude * 32767 * math.sin(2 * math.pi * frequency * i / self.sample_rate))
            for i in range(round(seconds * self.sample_rate))
        ))
        if sys.byteorder != "little":
            samples.byteswap()
        return samples.tobytes()

    def encode_and_decode(self, output, pcm, background=None, volume=0.2):
        command = audio.ffmpeg_command("ffmpeg", output, self.sample_rate, background, volume)
        subprocess.run(command, input=pcm, capture_output=True, check=True, timeout=20)
        result = subprocess.run([
            "ffmpeg", "-v", "error", "-nostdin", "-i", str(output),
            "-f", "f32le", "-c:a", "pcm_f32le", "pipe:1",
        ], capture_output=True, check=True, timeout=20)
        samples = array.array("f")
        samples.frombytes(result.stdout)
        if sys.byteorder != "little":
            samples.byteswap()
        return samples

    def make_background(self, path, amplitude):
        with wave.open(str(path), "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(self.sample_rate)
            output.writeframes(self.sine_pcm(0.25, amplitude, 220))

    def test_normalization_and_zero_background_preserve_narration_level_and_length(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pcm = self.sine_pcm(3, 0.005, 440)
            background = root / "short.wav"
            self.make_background(background, 0.5)
            dry = self.encode_and_decode(root / "dry.m4b", pcm)
            mixed = self.encode_and_decode(root / "mixed.m4b", pcm, background, 0)
            rms = lambda samples: math.sqrt(sum(x*x for x in samples) / len(samples))
            self.assertGreater(rms(dry), 0.005 * 3)
            self.assertAlmostEqual(rms(mixed) / rms(dry), 1, delta=0.02)
            self.assertAlmostEqual(len(mixed) / self.sample_rate, 3, delta=0.1)
            self.assertEqual(len(mixed), len(dry))

    def test_limiter_contains_loud_background_peaks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            background = root / "loud.wav"
            self.make_background(background, 0.8)
            mixed = self.encode_and_decode(
                root / "mixed.m4b", self.sine_pcm(3, 0.005, 440), background, 2
            )
            peak = max(abs(x) for x in mixed)
            self.assertGreater(peak, 0.7)
            self.assertLess(peak, 1)


if __name__ == "__main__":
    unittest.main()
