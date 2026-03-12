FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1 \
    OLLAMA_MODEL=llama3.2:3b \
    OLLAMA_URL=http://host.docker.internal:11434/api/chat \
    MAX_TTS_SENTENCES=2 \
    MAX_TTS_CHARS=140

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libsndfile1 \
    libportaudio2 \
    portaudio19-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt /app/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/requirements.txt

COPY voice_agent.py /app/voice_agent.py

CMD ["python", "voice_agent.py"]
