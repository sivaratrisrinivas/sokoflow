# SokoFlow

SokoFlow is a small app that tries to solve Sokoban puzzles by generating a list of moves, then cleaning that list up the way an image generator turns noise into a picture.

## Who it is for

Sokoban is a warehouse puzzle: you push boxes onto marked spots, and you cannot pull a box back. The usual computer approach is **search**: try legal moves until every box sits on a target.

This project is for people who want to watch a learned model play those puzzles in a browser, and for people who want to train or measure that model. The problem it takes on is whether a **diffusion model** (a program that starts from random noise and refines it into something structured) can output a full solution from the board, without using search as a backup while it plays.

Image generators such as DALL-E start from random pixels and refine them. SokoFlow uses the same idea, which this repo calls **flow**, but the output is a list of moves instead of a picture:

```
Random moves -> [step-by-step cleanup] -> a solution
```

## Try it

You need Python 3.8 or newer. Trained model weights are already in this repo (`sokoban_diffusion.pth`), so you do not have to train anything to try the page.

From the repo root:

```bash
pip install -r requirements.txt
python3 app.py
```

Then open http://localhost:5000 and click Start. An 8 by 8 board appears. The app makes a puzzle and the model tries to push every box onto a target. Click Stop to pause.

## How well it works

**Solve rate** means the share of puzzles that end with every box on a target.

On 24 August 2026 we ran the same 240 puzzles through two solvers:

1. **Search (BFS).** Breadth-first search is a classic method: it tries moves in order of length and aims for a shortest solution. If it looks at 30,000 board positions without finishing, it gives up.
2. **The model.** SokobanDiffusion, the trained network in `sokoban_diffusion.pth`, using the same play path as the web app.

**Solve rate versus BFS** means we compare those two shares on the same puzzles. It is a count of finished puzzles, not a speed contest, and not a claim that the model is better overall. On this test, search still finishes far more puzzles than the model.

Headline numbers from that GS-T5 measurement:

- The model solved **20.8% (50/240)** of the puzzles.
- Search solved **95.4% (229/240)**.

By difficulty (difficulty here is the number of boxes):

- easy (2 boxes): model 33.3% (20/60)
- medium (3 boxes): model 18.8% (15/80)
- hard (4 boxes): model 15.0% (15/100)

Search failed on 11 of 240 puzzles (`max_nodes=30000`). The model failed on 190 of 240.

Most of the model's wins are almost-finished boards. A **1-box-off** puzzle has only one box sitting off its target. 111 of the 240 puzzles are 1-box-off, and 43 of the model's 50 successes are those boards. Among the 50 puzzles the model solved, the search's shortest solutions averaged 3.7 moves. Training data generation in `generate_dataset` drops solutions shorter than 5 moves, so 20.8% is not a rate on long, heavily scrambled puzzles.

The published model solve rate is 20.8% (50/240). That figure is not re-filtered.

Full table (same numbers as `eval/gs_t5_solve_rate.json`). **Diffusion given BFS** is the model's solve rate counted only on puzzles that search also finished.

| Difficulty | N | BFS | Diffusion | Diffusion given BFS |
|---|---:|---|---|---|
| easy (2 boxes) | 60 | 100.0% (60/60) | 33.3% (20/60) | 33.3% (20/60) |
| medium (3 boxes) | 80 | 95.0% (76/80) | 18.8% (15/80) | 19.7% (15/76) |
| hard (4 boxes) | 100 | 93.0% (93/100) | 15.0% (15/100) | 16.1% (15/93) |
| overall | 240 | 95.4% (229/240) | 20.8% (50/240) | 21.8% (50/229) |

Measured on reverse-scrambled 8x8 boards. Model: SokobanDiffusion (`sokoban_diffusion.pth`). Dataset: 240 puzzles (20 per box-count and scramble config from `sokoban_data_gen.py`). Hardware: Intel Xeon, 4 cores, 16 GB RAM, CPU, Python 3.12.3, PyTorch 2.13.0. Per-puzzle outcomes: `eval/gs_t5_solve_rate.json`.

To run the same measurement:

```bash
python3 eval/measure_solve_rate.py
```

## Install, train, and other contributor details

The try-it commands above are enough to run the page. The rest of this file is for changing the model or the eval.

### Train or regenerate data

Training data is not committed (Git ignores `*.npy`). The trained weights file `sokoban_diffusion.pth` (3.6MB) is in the repo. If you want a fresh dataset or a freshly trained model:

```bash
# Generate training data (writes sokoban_dataset.npy, 2.2MB)
python3 sokoban_data_gen.py

# Train the model (writes sokoban_diffusion.pth)
python3 sokoban_diffusion.py

# Run the web server
python3 app.py
```

### Requirements

- Python 3.8+
- Flask >= 2.3.0
- Flask-CORS >= 4.0.0
- PyTorch >= 2.0.0
- NumPy >= 1.24.0

Flask is the small web server that serves the page. PyTorch is the library that runs the model.

### How play works

1. **Training:** the model learns to clean noisy move lists using shortest solutions from search.
2. **Play:** given a scrambled puzzle, the model starts from random moves and refines them (reverse diffusion).
3. **Execution:** valid moves run one at a time. If the model gets stuck, it generates again, up to 20 tries on hard puzzles.

At play time it uses a faster 10-step sampler (DDIM) instead of the 100-step training schedule (DDPM-style). It draws 4 candidate move lists and keeps the one that makes the most progress. There is no search fallback during play.

### Architecture

- **Board encoder:** a small convolutional network (CNN) that reads the 8 by 8 grid.
- **Action denoiser:** a transformer that cleans up the list of moves, given the board.
- **Training source:** shortest solutions from BFS.

Speed at play time is about 10-20 cleanup steps per puzzle.

### Files

- `app.py` - Flask web server with the model solver
- `sokoban_diffusion.py` - model architecture and training
- `sokoban_engine.py` - game logic (moves, rules, puzzle generation)
- `sokoban_data_gen.py` - generate training data from solved puzzles
- `templates/index.html` - web UI with auto-play
- `sokoban_diffusion.pth` - trained model weights (3.6MB)
- `sokoban_dataset.npy` - training dataset (2.2MB), created by `sokoban_data_gen.py`
- `eval/measure_solve_rate.py` - GS-T5 solve-rate measurement
- `eval/gs_t5_solve_rate.json` - recorded GS-T5 results
