# SokoFlow

A small **CPU** diffusion model that emits Sokoban action sequences on **8×8** boards. It is a research demo, not a warehouse robot and not SOTA.

**Flow** here means the denoising process: random moves → a candidate solution.

## What

SokoFlow trains a CNN + transformer denoiser on reverse-scrambled Sokoban trajectories, then samples action sequences with DDIM. A Flask UI can generate a scrambled board, ask the model for a path, and play it back.

It does **not** search like BFS. Measured solve rate is far below a 30k-node BFS baseline.

## Why

Image diffusion maps noise to pixels. This project asks whether the same idea can map noise to Sokoban moves, on a board small enough to train and serve on CPU.

## How

1. **Data.** `sokoban_data_gen.py` reverse-scrambles a solved 8×8 board, then BFS-solves it. Training keeps trajectories with `len(traj) >= 5`.
2. **Train.** `sokoban_diffusion.py` learns to denoise length-20 action sequences conditioned on a 6-channel board encoding.
3. **Infer.** `app.diffusion_solve_fast` runs 10-step DDIM, 4 samples per iteration, up to 20 iterations, and executes only legal moves.
4. **Eval.** `eval/measure_solve_rate.py` is the GS-T5 protocol. **Eval does not apply the training `len >= 5` filter.**

## Quick Start

Python 3.10+ (3.12 used for the published numbers). GPU is optional and unused in CI/Docker.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open http://localhost:5000. First `/api/new_game` loads PyTorch + 3.7MB weights; cold start can take tens of seconds on a tiny dyno.

### Docker

```bash
docker build -t sokoflow .
docker run --rm -p 5000:5000 sokoflow
```

Or: `docker compose up --build`

`GET /health` should report `"model_loaded": true`.

### Train / eval (optional)

Weights `sokoban_diffusion.pth` are already in git. You do **not** need to train to run the demo.

```bash
python sokoban_data_gen.py          # writes sokoban_dataset.npy (untracked)
python sokoban_diffusion.py         # overwrites sokoban_diffusion.pth
python eval/measure_solve_rate.py   # full GS-T5, 240 puzzles; writes eval/gs_t5_solve_rate.json
python eval/measure_solve_rate.py --smoke   # 1 puzzle, does not overwrite the GS-T5 table
```

Reproduce the published table with the one-liner `python eval/measure_solve_rate.py`. That run will take on the order of a minute on a 4-core CPU (historical elapsed: 57.5s).

## Results (historical GS-T5)

**Do not treat 20.8% as a scramble-hard rate.**

Date: **2026-08-24**. Model: `SokobanDiffusion` (`sokoban_diffusion.pth`). Dataset: 240 reverse-scrambled 8×8 puzzles (20 per box-count / scramble config from `sokoban_data_gen.py`). Hardware: Intel Xeon, 4 cores, 16 GB RAM, **CPU**, Python 3.12.3, PyTorch 2.13.0. Protocol: BFS `max_nodes=30000`; diffusion `app.diffusion_solve_fast` `max_iters=20`. Seed 42.

| Difficulty | N | BFS | Diffusion | Diffusion given BFS |
|---|---:|---|---|---|
| easy (2 boxes) | 60 | 100.0% (60/60) | 33.3% (20/60) | 33.3% (20/60) |
| medium (3 boxes) | 80 | 95.0% (76/80) | 18.8% (15/80) | 19.7% (15/76) |
| hard (4 boxes) | 100 | 93.0% (93/100) | 15.0% (15/100) | 16.1% (15/93) |
| overall | 240 | 95.4% (229/240) | **20.8% (50/240)** | 21.8% (50/229) |

Per-puzzle JSON: `eval/gs_t5_solve_rate.json`.

No new training run was done for the production packaging work. If a later commit retrains, add a **new** dated table and keep this one labeled historical.

## Limitations

- **20.8% is not scramble-hard.** Training drops trajectories shorter than 5 moves. GS-T5 eval does **not**. 111/240 eval puzzles have `boxes_off_target=1`, and **43 of 50** diffusion successes are those boards. Mean BFS length of diffusion solves is **3.7**.
- BFS is the honest baseline: **95.4%** (229/240) with 11 BFS failures at `max_nodes=30000`. Diffusion failed on 190/240.
- 8×8, 2–4 boxes, reverse scramble — not standard Sokoban levels, not Microban, not warehouse robotics.
- Demo solver uses in-memory sessions (`/api/solve_step`) plus a client-side path from `/api/new_game`. Gunicorn is **1 worker**. Stateless `/api/solve` is the production-shaped endpoint (rate-limited, 16KB body cap).
- CPU inference only in Docker/CI. Cold start is dominated by importing torch.

## API

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | Liveness + whether weights loaded |
| GET | `/` | UI |
| POST | `/api/new_game` | Scramble + solve. JSON body optional `{difficulty: int}` |
| POST | `/api/solve` | Stateless solve. Body `{grid, targets}` 8×8 |
| POST | `/api/solve_step` | Playback using `sokoflow_sid` cookie |

Guards: `MAX_CONTENT_LENGTH` default 16KiB; `RATE_LIMIT_PER_MINUTE` default 30 on `/api/solve`; `NEW_GAME_RATE_LIMIT_PER_MINUTE` default 120 on the demo generate/playback routes; `CORS_ORIGINS` default `*`.

## Layout

| Path | Role |
|---|---|
| `app.py` | Flask app, health, CORS, rate limit, solver |
| `sokoban_diffusion.py` | Model + training |
| `sokoban_engine.py` | Game rules |
| `sokoban_data_gen.py` | Dataset + BFS |
| `sokoban_diffusion.pth` | Committed CPU weights (3.7MB) |
| `eval/measure_solve_rate.py` | GS-T5 harness |
| `tests/` | Engine, actions, weight load, eval smoke, API guards |
| `Dockerfile` / `docker-compose.yml` | One-command UI |

Install as a package with `pip install -e ".[dev]"` (`pyproject.toml`).

## Deploy

See [DEPLOY.md](DEPLOY.md). Persistent Docker/Railway/Render/HF Spaces is the realistic host. A Vercel Flask project was linked and **failed** on this Hobby account (`LAMBDA_SIZE_EXCEEDED` 5306 MB vs 500 MB). Do not set the GitHub homepage to a Vercel URL unless `/health` is actually public and healthy.
