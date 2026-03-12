# Voice Agent Docker + GitHub CI/CD

This project is configured to:
- Build a Docker image from `Dockerfile`
- Push the image to Docker Hub on every push to `main`
- Also support manual trigger via GitHub Actions

## 1) Required GitHub Secrets

In your GitHub repo, add these secrets:
- `DOCKERHUB_USERNAME`
- `DOCKERHUB_TOKEN` (Docker Hub access token, not password)

## 2) Docker Hub Image Name

The workflow publishes to:
- `${DOCKERHUB_USERNAME}/voice-agent`

## 3) GitHub Action

Workflow file:
- `.github/workflows/docker-publish.yml`

Triggers:
- Push to `main`
- Manual run (`workflow_dispatch`)

## 4) Build/Run Locally

Build:

```bash
docker build -t voice-agent:local .
```

Run:

```bash
docker run --rm -it \
  --device /dev/snd \
  -e OLLAMA_URL=http://host.docker.internal:11434/api/chat \
  -e OLLAMA_MODEL=llama3.2:3b \
  -e MAX_TTS_SENTENCES=2 \
  -e MAX_TTS_CHARS=140 \
  voice-agent:local
```

Notes:
- `--device /dev/snd` is for Linux hosts with audio pass-through.
- If you want voice cloning, mount your WAV and set `XTTS_SPEAKER_WAV`.

Example:

```bash
docker run --rm -it \
  --device /dev/snd \
  -v "$(pwd)/sample.wav:/app/sample.wav:ro" \
  -e XTTS_SPEAKER_WAV=/app/sample.wav \
  -e OLLAMA_URL=http://host.docker.internal:11434/api/chat \
  voice-agent:local
```

## 5) Office Deployment with Docker Compose

1) Create `.env` from template:

```bash
cp .env.example .env
```

2) Edit `.env` and set your published image:

```bash
VOICE_AGENT_IMAGE=<your-dockerhub-username>/voice-agent:latest
```

3) Start Ollama + Voice Agent:

```bash
docker compose up -d
```

4) Pull model once (first time only):

```bash
docker compose --profile init up ollama-init
```

5) Check logs:

```bash
docker compose logs -f voice-agent
```

6) Stop services:

```bash
docker compose down
```

Important:
- `docker-compose.yml` uses `/dev/snd` for microphone/speaker access, so this setup is for Linux office machines.
