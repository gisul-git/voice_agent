import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import sounddevice as sd
import numpy as np
import whisperx
import ollama
import librosa
from TTS.api import TTS

print("Loading models...")

# Speech-to-text model
stt = whisperx.load_model("base", device="cpu")

# Text-to-speech model
tts = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2")

# Mac microphone runs at 48000
samplerate = 48000

# Recording duration
duration = 5

print("Voice agent ready")

def speak(text):

    audio = tts.tts(
        text=text,
        speaker_wav="sample.wav",   # remove this line if not cloning voice
        language="en",
        temperature=0.6
    )

    sd.play(audio, 24000)
    sd.wait()


while True:

    print("\nListening...")

    # record audio
    audio = sd.rec(
        int(duration * samplerate),
        samplerate=samplerate,
        channels=1,
        dtype="float32"
    )

    sd.wait()

    audio = audio.flatten()

    # convert 48000 → 16000 for Whisper
    audio = librosa.resample(audio, orig_sr=48000, target_sr=16000)

    # speech recognition
    result = stt.transcribe(audio, language="en")

    user_text = ""

    for seg in result["segments"]:
        user_text += seg["text"] + " "

    user_text = user_text.strip()

    if user_text == "":
        continue

    print("User:", user_text)

    # streaming response from LLM
    response = ollama.chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "system",
                "content": "Reply in maximum two short sentences."
            },
            {
                "role": "user",
                "content": user_text
            }
        ],
        stream=True
    )

    buffer = ""

    print("AI:", end=" ", flush=True)

    for chunk in response:

        token = chunk["message"]["content"]

        print(token, end="", flush=True)

        buffer += token

        # speak when sentence finishes
        if "." in buffer or "!" in buffer or "?" in buffer:

            speak(buffer.strip())

            buffer = ""

    print()
