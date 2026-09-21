# Deploy SokoFlow

SokoFlow is a Flask + PyTorch CPU app with a 3.7MB weight file. A **persistent process** (Docker, Railway, Render, Fly, HF Spaces Docker) is the reliable demo.

`GET /health` is the probe and must return `"model_loaded": true` (the Docker HEALTHCHECK asserts that, not merely HTTP 200). First `/api/new_game` after a cold boot can take tens of seconds.

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

## Hugging Face Spaces (Gradio)

Live: https://huggingface.co/spaces/Srini410/sokoflow

Repo source of truth is `gradio_app/` (not the Space-only tree). The Space must stay **1-click**: a puzzle is on the board at load; **Play** runs diffusion. Colorful pieces, cream chrome.

Push (maintainer):

```bash
# from repo root — Space root = gradio_app files + shared modules + weights
cp sokoban_engine.py sokoban_diffusion.py sokoban_solve.py sokoban_render.py sokoban_diffusion.pth gradio_app/
# huggingface-cli upload, or clone Srini410/sokoflow Space and copy:
#   gradio_app/app.py          → Space app.py
#   gradio_app/requirements.txt
#   gradio_app/README.md       → Space README.md (YAML header)
#   sokoban_engine.py sokoban_diffusion.py sokoban_solve.py sokoban_render.py sokoban_diffusion.pth
```

ZeroGPU Spaces still need `@spaces.GPU` on the solve wrapper (already in `gradio_app/app.py`) even though inference is CPU.

## Hugging Face Spaces (Docker, Flask)

1. Create a Docker Space.
2. Point it at this repo, or copy `Dockerfile`, `app.py`, engine/diffusion/solve modules, `templates/`, and `sokoban_diffusion.pth`.
3. HF sets `PORT=7860`; the image honors `$PORT`.
4. Space URL is the live Flask demo. A puzzle loads; Play is the one primary action.

## Vercel (unsupported)

Hobby cannot ship torch for this app. Measured bundle sizes:

| Attempt | Torch source | Bundle | Hobby cap |
|---|---|---:|---:|
| CUDA default from pyproject `torch>=2` | PyPI | **5306.44 MB** | 500 MB |
| `torch==2.5.1+cpu` via uv CPU index | download.pytorch.org/whl/cpu | **731.17 MB** | 500 MB |

Both failed `LAMBDA_SIZE_EXCEEDED`. There is no `vercel.json`. **Do not publish a Vercel URL.** Use Docker / Railway / Render / HF Spaces Docker.

## Env vars

| Name | Default | Purpose |
|---|---|---|
| `PORT` | `5000` | Bind port |
| `SOKOFLOW_MODEL_PATH` | `sokoban_diffusion.pth` | Weights |
| `CORS_ORIGINS` | `*` | Comma-separated origins |
| `RATE_LIMIT_PER_MINUTE` | `30` | `/api/solve` cap per client IP |
| `NEW_GAME_RATE_LIMIT_PER_MINUTE` | `120` | Demo `/api/new_game` + `/api/solve_step` (separate bucket from `/api/solve`) |
| `TRUST_PROXY` | unset | If `1`/`true`, rate-limit keys use `X-Forwarded-For`; default is `request.remote_addr` only |
| `MAX_CONTENT_LENGTH` | `16384` | Request body cap (bytes) |

## GitHub homepage

Maintainer-only: Settings → GitHub Pages/Homepage field → paste the public demo URL. This PR cannot set it without admin rights.
