import sounddevice as sd
from kokoro import KPipeline

SAMPLE_RATE = 24000

pipeline = KPipeline(lang_code="b")

text = """
[Kokoro](/kˈOkəɹO/) is an open-weight TTS model with 82 million parameters. Despite its lightweight architecture, it delivers comparable quality to larger models while being significantly faster and more cost-efficient.
With Apache-licensed weights, [Kokoro](/kˈOkəɹO/) can be deployed anywhere from production environments to personal projects.
"""
generator = pipeline(text, voice="bm_george", speed=0.8)
for i, result in enumerate(generator):
    print(i, result.graphemes, result.phonemes)
    audio = result.audio
    if audio is not None:
        sd.play(audio.cpu().numpy(), samplerate=SAMPLE_RATE, blocking=True)
