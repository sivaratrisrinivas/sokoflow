# Deploy SokoFlow

SokoFlow is a Flask + PyTorch CPU app with a 3.7MB weight file. A **persistent process** (Docker, Railway, Render, Fly, HF Spaces Docker) is the reliable demo. Serverless Flask on Vercel is included as a best-effort option; expect a long cold start while `torch` imports.

`GET /health` is the probe. First `/api/new_game` after a cold boot can take tens of seconds.

## Docker (local or any VM)

```bash
docker build -t sokoflow .
docker run --rm -p 5000:5000 sokoflow
```

Compose: `docker compose up --build`

The image installs **CPU** torch. No GPU required.

## Railway

1. New project → Deploy from GitHub → `sivaratrisrinivas/sokoflow`.
2. Railway reads `Procfile` / `railway.json`. Start command:
   `gunicorn -b 0.0.0.0:${PORT:-5000} --workers 1 --threads 4 --timeout 120 app:app`
3. Confirm `sokoban_diffusion.pth` is in the repo (it is tracked).
4. Health path: `/health`.
5. Open the public `*.up.railway.app` URL. Cold start: first puzzle request loads the model.

## Render

`render.yaml` describes a Docker web service.

1. New Web Service → this repo → runtime Docker.
2. Health check: `/health`.
3. Instance: at least 512MB RAM (1GB preferred; torch is the bulk).

## Fly.io

```bash
fly launch --name sokoflow --no-deploy
# dockerfile: Dockerfile, internal port 5000
fly deploy
```

Set `min_machines_running = 0` if you accept cold starts; keep 1 worker equivalent (this image already uses gunicorn workers=1).

## Hugging Face Spaces (Docker)

1. Create a Docker Space.
2. Point it at this repo, or copy `Dockerfile`, `app.py`, engine/diffusion modules, `templates/`, and `sokoban_diffusion.pth`.
3. HF sets `PORT=7860`; the image honors `$PORT`.
4. Space URL is the live demo.

## Vercel (not a live demo on this Hobby account)

Vercel Flask is configured (`vercel.json`, project `sokoflow` on team Srini, Deployment Protection off). The first production deploy **failed** with `LAMBDA_SIZE_EXCEEDED`: default PyPI torch bundled to **5306 MB** vs a **500 MB** function cap.

`requirements.txt` now pins `torch==2.5.1+cpu`. If a later build still exceeds 500 MB, use Docker / Railway / Render / HF Spaces. Do not publish a Vercel URL unless `/health` returns `"model_loaded": true` without auth.

## Env vars

| Name | Default | Purpose |
|---|---|---|
| `PORT` | `5000` | Bind port |
| `SOKOFLOW_MODEL_PATH` | `sokoban_diffusion.pth` | Weights |
| `CORS_ORIGINS` | `*` | Comma-separated origins |
| `RATE_LIMIT_PER_MINUTE` | `30` | `/api/solve` cap per client IP |
| `NEW_GAME_RATE_LIMIT_PER_MINUTE` | `120` | Demo `/api/new_game` + `/api/solve_step` |
| `MAX_CONTENT_LENGTH` | `16384` | Request body cap (bytes) |

## GitHub homepage

Maintainer-only: Settings → GitHub Pages/Homepage field → paste the public demo URL. This PR cannot set it without admin rights.
