import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

from TTS.api import TTS
import sounddevice as sd

print("Loading XTTS model...")

tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

text = "Hello Ashutosh. This voice should sound like your recorded sample."

audio = tts.tts(
    text=text,
    speaker_wav="sample.wav",
    language="en"
)

sd.play(audio, 24000)
sd.wait()
