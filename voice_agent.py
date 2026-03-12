import os
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import queue
import re
import sounddevice as sd
import numpy as np
import requests
import torch
import json
import shutil
import subprocess

from faster_whisper import WhisperModel
from TTS.api import TTS
from silero_vad import load_silero_vad, get_speech_timestamps

# -------------------------
# CONFIG
# -------------------------

sample_rate = 16000
audio_queue = queue.Queue()

ollama_model = os.getenv("OLLAMA_MODEL", "llama3.2:3b")
ollama_url = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434/api/chat")

speaker_wav_path = os.getenv("XTTS_SPEAKER_WAV", "sample.wav")
speaker_name = os.getenv("XTTS_SPEAKER")
max_tts_sentences = int(os.getenv("MAX_TTS_SENTENCES", "2"))
max_tts_chars = int(os.getenv("MAX_TTS_CHARS", "140"))
system_prompt = os.getenv(
    "SYSTEM_PROMPT",
    "Reply in at most two short sentences. Keep answers concise and easy to speak aloud.",
)
audio_player = os.getenv("AUDIO_PLAYER", "auto")

# -------------------------
# LOAD MODELS
# -------------------------

print("Loading Faster-Whisper...")
whisper_model = WhisperModel(
    "base",
    device="cpu",
    compute_type="int8"
)

print("Loading Silero VAD...")
vad_model = load_silero_vad()

print("Loading XTTS...")
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to("cpu")

if speaker_wav_path and not os.path.isfile(speaker_wav_path):
    speaker_wav_path = None

if not speaker_wav_path and not speaker_name:
    available_speakers = getattr(tts, "speakers", None)
    if isinstance(available_speakers, dict) and available_speakers:
        speaker_name = next(iter(available_speakers.keys()))
    elif isinstance(available_speakers, list) and available_speakers:
        speaker_name = available_speakers[0]
    else:
        raise RuntimeError(
            "XTTS needs `speaker` or `speaker_wav`. Set XTTS_SPEAKER_WAV to a reference WAV."
        )

print("Voice agent ready")

# -------------------------
# MICROPHONE STREAM
# -------------------------

def audio_callback(indata, frames, time, status):
    audio_queue.put(indata.copy())

stream = sd.InputStream(
    samplerate=sample_rate,
    channels=1,
    callback=audio_callback
)

stream.start()

# -------------------------
# SPEECH DETECTION
# -------------------------

def listen_for_speech():

    buffer = np.array([], dtype=np.float32)

    while True:

        chunk = audio_queue.get().flatten()
        buffer = np.concatenate((buffer, chunk))

        if len(buffer) < sample_rate * 2:
            continue

        audio_tensor = torch.from_numpy(buffer)

        timestamps = get_speech_timestamps(
            audio_tensor,
            vad_model,
            sampling_rate=sample_rate
        )

        if timestamps:

            start = timestamps[0]["start"]
            end = timestamps[-1]["end"]

            speech = buffer[start:end]

            buffer = np.array([], dtype=np.float32)

            return speech

        buffer = np.array([], dtype=np.float32)

# -------------------------
# SPEECH TO TEXT
# -------------------------

def speech_to_text(audio):

    segments, _ = whisper_model.transcribe(audio)

    text = ""

    for segment in segments:
        text += segment.text

    return text.strip()

# -------------------------
# STREAM LLM TOKENS
# -------------------------

def stream_llm(prompt):

    response = requests.post(
        ollama_url,
        json={
            "model": ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "stream": True,
            "options": {
                "num_predict": 64,
                "temperature": 0.5,
            },
        },
        stream=True,
        timeout=120,
    )
    response.raise_for_status()

    for line in response.iter_lines():

        if not line:
            continue

        try:
            data = json.loads(line.decode("utf-8"))
        except json.JSONDecodeError:
            continue

        token = data.get("message", {}).get("content", "")

        if token:
            yield token

# -------------------------
# SENTENCE BUFFER
# -------------------------

def clamp_text_for_tts(text):
    cleaned = " ".join(text.split()).strip()
    if not cleaned:
        return ""
    if len(cleaned) > max_tts_chars:
        cleaned = cleaned[:max_tts_chars].rstrip() + "..."
    return cleaned

def sentence_stream(token_stream):

    buffer = ""
    yielded = 0

    for token in token_stream:

        buffer += token

        if re.search(r"[.!?]", buffer):

            sentence = clamp_text_for_tts(buffer.strip())

            buffer = ""

            if sentence:
                yield sentence
                yielded += 1
                if yielded >= max_tts_sentences:
                    return

    if buffer and yielded < max_tts_sentences:
        sentence = clamp_text_for_tts(buffer.strip())
        if sentence:
            yield sentence

# -------------------------
# TEXT TO SPEECH
# -------------------------

def speak(text):

    if not text:
        return

    print("AI:", text)

    tts_kwargs = {
        "text": text,
        "language": "en",
        "file_path": "speech.wav"
    }

    if speaker_wav_path:
        tts_kwargs["speaker_wav"] = speaker_wav_path
    else:
        tts_kwargs["speaker"] = speaker_name

    tts.tts_to_file(**tts_kwargs)

    play_speech_file("speech.wav")


def play_speech_file(path):
    if audio_player.lower() == "none":
        return

    if audio_player.lower() != "auto":
        subprocess.run(audio_player.split() + [path], check=False)
        return

    if shutil.which("ffplay"):
        subprocess.run(
            ["ffplay", "-nodisp", "-autoexit", "-loglevel", "error", path],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        return

    if shutil.which("aplay"):
        subprocess.run(["aplay", path], check=False)
        return

    if shutil.which("afplay"):
        subprocess.run(["afplay", path], check=False)

# -------------------------
# MAIN LOOP
# -------------------------

while True:

    print("Listening...")

    audio = listen_for_speech()

    user_text = speech_to_text(audio)

    if not user_text:
        continue

    print("User:", user_text)

    token_stream = stream_llm(user_text)

    for sentence in sentence_stream(token_stream):

        speak(sentence)
