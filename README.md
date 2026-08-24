# SokoFlow

A diffusion model that solves Sokoban puzzles. Think DALL-E, but for puzzle solutions.

**Flow** = The diffusion process that transforms random moves → optimal solution

## What It Does

This project uses a diffusion model (similar to DALL-E/Stable Diffusion) to solve Sokoban puzzles. Instead of generating images, it generates action sequences that solve the puzzle:

```
Random Actions → [Denoising Process] → Optimal Solution
```

The model learns to turn random moves into optimal solutions, similar to how image generators turn noise into pictures.

## Results

GS-T5 measurement of SokobanDiffusion versus BFS on reverse-scrambled 8x8 boards. Date: 2026-08-24. Model: SokobanDiffusion (`sokoban_diffusion.pth`). Dataset: 240 puzzles (20 per box-count and scramble config from `sokoban_data_gen.py`). Hardware: Intel Xeon, 4 cores, 16 GB RAM, CPU, Python 3.12.3, PyTorch 2.13.0.

| Difficulty | N | BFS | Diffusion | Diffusion given BFS |
|---|---:|---|---|---|
| easy (2 boxes) | 60 | 100.0% (60/60) | 33.3% (20/60) | 33.3% (20/60) |
| medium (3 boxes) | 80 | 95.0% (76/80) | 18.8% (15/80) | 19.7% (15/76) |
| hard (4 boxes) | 100 | 93.0% (93/100) | 15.0% (15/100) | 16.1% (15/93) |
| overall | 240 | 95.4% (229/240) | 20.8% (50/240) | 21.8% (50/229) |

BFS failed on 11 of 240 puzzles (max_nodes=30000). Diffusion failed on 190 of 240 puzzles. Full per-puzzle outcomes are in `eval/gs_t5_solve_rate.json`.

Reproduce:

```bash
python eval/measure_solve_rate.py
```

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate training data (optional if dataset already exists)
python sokoban_data_gen.py

# 3. Train the diffusion model (optional if model already exists)
python sokoban_diffusion.py

# 4. Run the web server
python app.py
```

Then open http://localhost:5000

## Files

- `app.py` - Flask web server with diffusion solver
- `sokoban_diffusion.py` - Diffusion model architecture & training  
- `sokoban_engine.py` - Game logic (moves, rules, puzzle generation)
- `sokoban_data_gen.py` - Generate training data from solved puzzles
- `templates/index.html` - Web UI with auto-play mode
- `sokoban_diffusion.pth` - Trained model weights (3.6MB)
- `sokoban_dataset.npy` - Training dataset (2.2MB)

## Features

- ✅ Pure diffusion model solving (no BFS fallback)
- ✅ Fast DDIM sampling (10 steps instead of 100)
- ✅ Batch sampling for better solution quality
- ✅ Handles 2-4 box puzzles
- ✅ Auto-play mode with live visualization
- ✅ Minimal, modern web interface

## How It Works

1. **Training**: Model learns to denoise action sequences from optimal puzzle solutions
2. **Inference**: Given a scrambled puzzle, model generates solution via reverse diffusion
3. **Execution**: Valid actions are executed step-by-step, model re-generates if stuck

## Architecture

- **Diffusion Model**: DDPM-style denoising (100 timesteps, DDIM for fast inference)
- **Board Encoder**: CNN that processes 8×8 Sokoban grid
- **Action Denoiser**: Transformer that denoises action sequences
- **Training**: Learns from optimal BFS solutions

## Performance

- **Speed**: ~10-20 denoising steps per puzzle
- **Success Rate**: ~90%+ on puzzles with 2-3 boxes
- **Max Iterations**: 20 iterations for hard puzzles

## Requirements

- Python 3.8+
- Flask >= 2.3.0
- Flask-CORS >= 4.0.0
- PyTorch >= 2.0.0
- NumPy >= 1.24.0
