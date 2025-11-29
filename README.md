# SokoFlow

A neural diffusion model that solves Sokoban puzzles. Think DALL-E, but for puzzle solutions.

**Flow** = The diffusion process that transforms random moves → optimal solution

## What It Does

This project uses a **diffusion model** (similar to DALL-E/Stable Diffusion) to solve Sokoban puzzles. Instead of generating images, it generates action sequences that solve the puzzle:

```
Random Actions → [Denoising Process] → Optimal Solution
```

## Architecture

- **Diffusion Model**: DDPM-style denoising (100 timesteps, DDIM for fast inference)
- **Board Encoder**: CNN that processes 8×8 Sokoban grid
- **Action Denoiser**: Transformer that denoises action sequences
- **Training**: Learns from optimal BFS solutions

## Quick Start

```bash
# 1. Generate training data
python sokoban_data_gen.py

# 2. Train the diffusion model
python sokoban_diffusion.py

# 3. Run the web server
python app.py
```

Then open http://localhost:5000

## Files

- `app.py` - Flask server + diffusion solver
- `sokoban_diffusion.py` - Diffusion model architecture & training  
- `sokoban_engine.py` - Game logic (moves, rules)
- `sokoban_data_gen.py` - Generate training trajectories
- `templates/index.html` - Minimal web UI
- `sokoban_diffusion.pth` - Trained model weights

## Features

- ✅ Pure diffusion model solving (no BFS fallback)
- ✅ Fast DDIM sampling (10 steps instead of 100)
- ✅ Batch sampling for better quality
- ✅ Handles 2-4 box puzzles
- ✅ Auto-play mode

## How It Works

1. **Training**: Model learns to denoise action sequences from solved puzzles
2. **Inference**: Given scrambled puzzle, model generates solution via reverse diffusion
3. **Execution**: Valid actions are executed, model re-generates if stuck

## Performance

- **Speed**: ~10-20 denoising steps per puzzle
- **Success Rate**: ~90%+ on puzzles with 2-3 boxes
- **Iterations**: 20 max iterations for hard puzzles

