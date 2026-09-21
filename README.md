# SokoFlow

A small **CPU** diffusion model that emits Sokoban action sequences on **8×8** boards. It is a research demo, not a warehouse robot and not SOTA.

**Flow** here means the denoising process: random moves → a candidate solution. **Play** makes that visible (Denoise Theater) and puts a BFS twin next to it.

The live board is **colorful** (wall / floor / box / goal / player / box-on-goal). Chrome is cream, not grayscale.

## What

SokoFlow trains a CNN + transformer denoiser on reverse-scrambled Sokoban trajectories, then samples action sequences with DDIM. A Flask UI and a Gradio Space show a scrambled board; **Play** (1 click) runs Denoise Theater, then a BFS twin and — if diffusion fails — an autopsy. It does **not** hide that BFS is stronger.

Demo UI (GS-T48): theater auto-plays 10-step DDIM (noise → action sequence as a board path + arrows). The BFS twin reports solved/failed and node budget. Failures name the first illegal move, a stuck push, or exhausted iterations. After Play, a short GIF of the denoise run can be saved. No invented rates.

## Why

Image diffusion maps noise to pixels. This project asks whether the same idea can map noise to Sokoban moves, on a board small enough to train and serve on CPU.

## How

1. **Data.** `sokoban_data_gen.py` reverse-scrambles a solved 8×8 board, then BFS-solves it. Training keeps trajectories with `len(traj) >= 5`. As of GS-T47 it also drops `boxes_off_target < 2` for **future** datasets. Committed weights `sokoban_diffusion.pth` were trained with the length gate only.
2. **Train.** `sokoban_diffusion.py` learns to denoise length-20 action sequences conditioned on a 6-channel board encoding.
3. **Infer.** `sokoban_solve.diffusion_solve_fast` runs 10-step DDIM, 4 samples per iteration, up to 20 iterations, and executes only legal moves. The demo uses `diffusion_solve_report` so the UI can show denoise frames and an honest failure reason.
4. **Eval.** `eval/measure_solve_rate.py` supports three protocols. **Headline is scramble-hard.** Historical GS-T5 is kept labeled historical. Microban is a separate OOD table and is never mixed in.

## Quick Start

Python 3.10+ (3.12 used for the published numbers). GPU is optional and unused in CI/Docker.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

`requirements.txt` already uses the CPU torch extra index (`torch==2.5.1+cpu`). Editable installs of `pyproject.toml` need the same index:

```bash
pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cpu
```

Open http://localhost:5000. A puzzle is already on the board. **Play** starts Denoise Theater, then a BFS twin (Firstmate bar: ≤2 clicks from load; this demo ships 1). First request loads PyTorch + 3.7MB weights; cold start can take tens of seconds on a tiny dyno.

### Docker

```bash
docker build -t sokoflow .
docker run --rm -p 5000:5000 sokoflow
```

Or: `docker compose up --build`

`GET /health` should report `"model_loaded": true`.

### Gradio / Hugging Face Space

Source of truth: `gradio_app/`. Firstmate bar is ≤2 clicks from load; shipped path is 1: load shows a puzzle, **Play** runs denoise theater + BFS twin. Failure autopsy is shown when diffusion does not solve. A short GIF of the denoise run is offered as “Save denoise clip” after Play (no extra setup click).

```bash
pip install -r gradio_app/requirements.txt
PYTHONPATH=. python gradio_app/app.py
```

### Train / eval (optional)

Weights `sokoban_diffusion.pth` are already in git. You do **not** need to train to run the demo.

```bash
python sokoban_data_gen.py          # writes sokoban_dataset.npy (untracked)
python sokoban_diffusion.py         # overwrites sokoban_diffusion.pth
python eval/measure_solve_rate.py --protocol scramble-hard
python eval/measure_solve_rate.py --protocol microban
python eval/measure_solve_rate.py --protocol gs-t5          # historical protocol; dated JSON; does not overwrite 2026-08-24
python eval/measure_solve_rate.py --protocol gs-t5 --force  # replace eval/gs_t5_solve_rate.json
python eval/measure_solve_rate.py --smoke                   # 1 puzzle, does not overwrite tables
```

## Protocols

### Scramble-hard (headline, GS-T47)

**Definition.** Reverse-scramble 8×8 boards with the same `(boxes, scramble_steps)` configs as training / GS-T5. A board is eligible only if:

1. `boxes_off_target >= 2` (drops 1-box-off inflators)
2. BFS either fails at `max_nodes=30000` **or** returns a path of length `>= 5` (matches training `len(traj) >= 5`; drops short-path inflators)

Rejection-sample until N puzzles per config are eligible. Do not drop diffusion failures. BFS failures that pass (1) stay in the set. Microban is never mixed into this rate.

### GS-T5 (historical)

Same generation, **no** length gate, **no** 1-box-off gate. That mismatch inflated the published 20.8%: 111/240 boards were 1-box-off, and 43 of 50 diffusion wins were those boards.

### Train vs eval

| Protocol | `len(traj) >= 5` | `boxes_off_target >= 2` |
|---|---|---|
| Training (committed weights) | yes | no |
| Training (`generate_dataset` as of GS-T47) | yes | yes |
| GS-T5 eval (historical) | **no** | **no** |
| Scramble-hard eval (headline) | yes | yes |

## Results (scramble-hard, headline)

Date: **2026-09-21**. Model: `SokobanDiffusion` (`sokoban_diffusion.pth`). N=240 (20 per box-count / scramble config). Hardware: Intel Xeon, 4 cores, 16 GB RAM, **CPU**, Python 3.12.3, PyTorch 2.5.1+cpu. Protocol: scramble-hard as defined above. BFS `max_nodes=30000`; diffusion `max_iters=20`. Seed 42. Elapsed 71.3s.

| Difficulty | N | BFS | Diffusion | Diffusion given BFS |
|---|---:|---|---|---|
| easy (2 boxes) | 60 | 100.0% (60/60) | 8.3% (5/60) | 8.3% (5/60) |
| medium (3 boxes) | 80 | 98.8% (79/80) | 6.2% (5/80) | 6.3% (5/79) |
| hard (4 boxes) | 100 | 87.0% (87/100) | 5.0% (5/100) | 5.7% (5/87) |
| overall | 240 | 94.2% (226/240) | **6.2% (15/240)** | 6.6% (15/226) |

JSON: `eval/scramble_hard_solve_rate.json` (dated copy `eval/scramble_hard_solve_rate-2026-09-21.json`). All 15 diffusion wins had `boxes_off_target=2` and BFS length 5–10 (mean 6.9).

## Results (historical GS-T5)

**Do not treat 20.8% as a scramble-hard rate.** Kept so older citations stay checkable.

Date: **2026-08-24**. Same model weights. Protocol: no train-length filter, includes 1-box-off boards. Seed 42. Elapsed 57.5s. Hardware: Intel Xeon, 4 cores, 16 GB RAM, CPU, Python 3.12.3, PyTorch 2.13.0.

| Difficulty | N | BFS | Diffusion | Diffusion given BFS |
|---|---:|---|---|---|
| easy (2 boxes) | 60 | 100.0% (60/60) | 33.3% (20/60) | 33.3% (20/60) |
| medium (3 boxes) | 80 | 95.0% (76/80) | 18.8% (15/80) | 19.7% (15/76) |
| hard (4 boxes) | 100 | 93.0% (93/100) | 15.0% (15/100) | 16.1% (15/93) |
| overall | 240 | 95.4% (229/240) | **20.8% (50/240)** | 21.8% (50/229) |

Per-puzzle JSON: `eval/gs_t5_solve_rate.json`.

## Results (Microban OOD, separate)

**Not mixed into scramble-hard or GS-T5.** Expect low rates.

Source: David W. Skinner, Microban (April 2000), freely distributable with credit. Probe: the 38 of 155 published levels whose bounding box already fits 8×8. XSB void cells are sealed to walls; the level is center-padded with walls to 8×8. Train distribution is reverse-scramble with **perimeter walls only** — interior-wall Microban is true OOD. `seq_len=20`; mean BFS length on this subset is 34.4, so many optimal paths cannot be represented.

Date: **2026-09-21**. Same model. Elapsed 12.5s.

| Set | N | BFS | Diffusion | Diffusion given BFS |
|---|---:|---|---|---|
| Microban OOD (8×8-fitting subset) | 38 | 78.9% (30/38) | **2.6% (1/38)** | 3.3% (1/30) |

The single diffusion win is Microban 44 ("Duh!"), a 5×3 one-push puzzle. JSON: `eval/microban_ood_solve_rate.json`. Levels: `eval/microban_levels.py`.

## Limitations

- **Headline is 6.2%, not 20.8%.** GS-T5 included 1-box-off and short-path boards that training mostly never saw.
- BFS remains the honest baseline on scramble-hard: **94.2%** (226/240) with 14 BFS failures at `max_nodes=30000`. Diffusion failed on 225/240.
- 8×8 reverse scramble — not standard Sokoban, not warehouse robotics. Microban is OOD and reported separately.
- Demo solver uses in-memory sessions (`/api/solve_step`) plus a client-side path from `/api/new_game`. Gunicorn is **1 worker**. Stateless `/api/solve` is the production-shaped endpoint (rate-limited, 16KB body cap).
- CPU inference only in Docker/CI. Cold start is dominated by importing torch.

## API

| Method | Path | Notes |
|---|---|---|
| GET | `/health` | Liveness + whether weights loaded |
| GET | `/` | UI — puzzle loaded; Play runs the path |
| POST | `/api/new_game` | Scramble + solve. JSON body optional `{difficulty: int}` |
| POST | `/api/solve` | Stateless solve. Body `{grid, targets}` 8×8 |
| POST | `/api/solve_step` | Playback using `sokoflow_sid` cookie |

Guards: `MAX_CONTENT_LENGTH` default 16KiB; `RATE_LIMIT_PER_MINUTE` default 30 on `/api/solve`; `NEW_GAME_RATE_LIMIT_PER_MINUTE` default 120 on `/api/new_game` and `/api/solve_step` (separate per-IP buckets); `TRUST_PROXY` default unset (rate-limit key is `request.remote_addr`; `X-Forwarded-For` is ignored unless `TRUST_PROXY=1`); `CORS_ORIGINS` default `*`.

## Layout

| Path | Role |
|---|---|
| `app.py` | Flask app, health, CORS, rate limit |
| `sokoban_solve.py` | Shared CPU diffusion solve path, BFS twin report, autopsy |
| `sokoban_render.py` | Colorful board HTML/CSS + denoise/twin stage |
| `sokoban_clip.py` | Tiny GIF encoder for a denoise clip |
| `sokoban_diffusion.py` | Model + training (includes DDIM trace) |
| `sokoban_engine.py` | Game rules |
| `sokoban_data_gen.py` | Dataset + BFS |
| `sokoban_diffusion.pth` | Committed CPU weights (3.7MB) |
| `eval/measure_solve_rate.py` | GS-T5 / scramble-hard / Microban harness |
| `gradio_app/` | HF Space source of truth (≤2 clicks; ships 1-click Play → theater + twin) |
| `tests/` | Engine, actions, weight load, eval smoke, API guards |
| `Dockerfile` / `docker-compose.yml` | One-command Flask UI |

Install as a package with `pip install -e ".[dev]" --extra-index-url https://download.pytorch.org/whl/cpu` (`pyproject.toml` pins `torch==2.5.1+cpu`). `pip install -r requirements.txt` already includes that index.

## Deploy

See [DEPLOY.md](DEPLOY.md). Persistent Docker/Railway/Render/HF Spaces is the realistic host. Vercel Hobby cannot ship this app (measured CUDA torch **5306 MB**, CPU torch **731 MB**, both over the **500 MB** function cap). There is no `vercel.json`. Do not set the GitHub homepage to a Vercel URL.
