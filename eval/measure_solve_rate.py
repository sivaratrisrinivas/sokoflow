#!/usr/bin/env python3
"""
GS-T5: SokobanDiffusion solve rate versus BFS, split by board difficulty.

Closest existing code is sokoban_data_gen.py (BFS solver and reverse-scramble
generation by box count). There is no bench/ or eval/ harness to extend, so
this script is new and reuses bfs_solve / SokobanGen plus the production
diffusion_solve_fast path in app.py.

From the repo root:

    python eval/measure_solve_rate.py

Writes eval/gs_t5_solve_rate.json and prints a markdown table to stdout.
"""

from __future__ import annotations

import json
import os
import platform
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
os.chdir(ROOT)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sokoban_data_gen import SokobanGen, bfs_solve  # noqa: E402
from app import diffusion_solve_fast  # noqa: E402

OUTPUT_PATH = ROOT / "eval" / "gs_t5_solve_rate.json"
MODEL_PATH = ROOT / "sokoban_diffusion.pth"
MODEL_NAME = "SokobanDiffusion"
SEED = 42
N_PER_CONFIG = 20
BFS_MAX_NODES = 30000
DIFFUSION_MAX_ITERS = 20
MAX_GEN_ATTEMPTS = 80

# Same difficulty labels and (boxes, scramble) pairs as sokoban_data_gen.generate_dataset.
CONFIGS = [
    {"difficulty": "easy", "num_boxes": 2, "scramble_steps": 20},
    {"difficulty": "easy", "num_boxes": 2, "scramble_steps": 30},
    {"difficulty": "easy", "num_boxes": 2, "scramble_steps": 40},
    {"difficulty": "medium", "num_boxes": 3, "scramble_steps": 20},
    {"difficulty": "medium", "num_boxes": 3, "scramble_steps": 30},
    {"difficulty": "medium", "num_boxes": 3, "scramble_steps": 40},
    {"difficulty": "medium", "num_boxes": 3, "scramble_steps": 50},
    {"difficulty": "hard", "num_boxes": 4, "scramble_steps": 20},
    {"difficulty": "hard", "num_boxes": 4, "scramble_steps": 30},
    {"difficulty": "hard", "num_boxes": 4, "scramble_steps": 40},
    {"difficulty": "hard", "num_boxes": 4, "scramble_steps": 50},
    {"difficulty": "hard", "num_boxes": 4, "scramble_steps": 60},
]


def cpu_model() -> str:
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
    except OSError:
        pass
    return platform.processor() or platform.machine() or "unknown"


def mem_total_mb() -> int | None:
    try:
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemTotal"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return None


def hardware_info() -> dict:
    import torch

    gpu = None
    if torch.cuda.is_available():
        gpu = torch.cuda.get_device_name(0)
    return {
        "cpu": cpu_model(),
        "cpu_cores": os.cpu_count(),
        "ram_mb": mem_total_mb(),
        "gpu": gpu,
        "device": "cuda" if gpu else "cpu",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "numpy": np.__version__,
    }


def boxes_off_target(grid: np.ndarray) -> int:
    return int(np.count_nonzero(grid == 3))


def generate_scrambled(num_boxes: int, scramble_steps: int) -> tuple[np.ndarray, np.ndarray, int, bool]:
    """Reverse-scramble from a solved board. Retry if every box stays on a target."""
    env = SokobanGen(num_boxes=num_boxes)
    already_solved = True
    grid = env.reset_solved()
    targets = env.targets.copy()
    attempts = 0
    for attempts in range(1, MAX_GEN_ATTEMPTS + 1):
        env.reset_solved()
        for _ in range(scramble_steps):
            env.step_reverse()
        if boxes_off_target(env.grid) > 0:
            already_solved = False
            grid = env.grid.copy()
            targets = env.targets.copy()
            break
        grid = env.grid.copy()
        targets = env.targets.copy()
    return grid, targets, attempts, already_solved


def rate(solved: int, n: int) -> float | None:
    if n == 0:
        return None
    return solved / n


def pct(solved: int, n: int) -> str:
    if n == 0:
        return "n/a"
    return f"{100.0 * solved / n:.1f}% ({solved}/{n})"


def summarize(rows: list[dict]) -> dict:
    n = len(rows)
    bfs_solved = sum(1 for r in rows if r["bfs_solved"])
    diffusion_solved = sum(1 for r in rows if r["diffusion_solved"])
    bfs_solvable = [r for r in rows if r["bfs_solved"]]
    diffusion_given_bfs = sum(1 for r in bfs_solvable if r["diffusion_solved"])
    bfs_times = [r["bfs_seconds"] for r in rows]
    diff_times = [r["diffusion_seconds"] for r in rows]
    bfs_lens = [r["bfs_length"] for r in rows if r["bfs_length"] is not None]
    diff_lens = [r["diffusion_length"] for r in rows if r["diffusion_length"] is not None]
    return {
        "n": n,
        "bfs_solved": bfs_solved,
        "bfs_failed": n - bfs_solved,
        "bfs_solve_rate": rate(bfs_solved, n),
        "diffusion_solved": diffusion_solved,
        "diffusion_failed": n - diffusion_solved,
        "diffusion_solve_rate": rate(diffusion_solved, n),
        "diffusion_solved_given_bfs": diffusion_given_bfs,
        "diffusion_solve_rate_given_bfs": rate(diffusion_given_bfs, len(bfs_solvable)),
        "already_solved_count": sum(1 for r in rows if r["already_solved"]),
        "mean_bfs_seconds": float(np.mean(bfs_times)) if bfs_times else None,
        "mean_diffusion_seconds": float(np.mean(diff_times)) if diff_times else None,
        "mean_bfs_length": float(np.mean(bfs_lens)) if bfs_lens else None,
        "mean_diffusion_length": float(np.mean(diff_lens)) if diff_lens else None,
    }


def markdown_table(results: dict) -> str:
    lines = [
        "| Difficulty | Boxes | Scramble | N | BFS | Diffusion | Diffusion given BFS |",
        "|---|---:|---:|---:|---|---|---|",
    ]
    for row in results["by_config"]:
        s = row["summary"]
        lines.append(
            "| {difficulty} | {boxes} | {scramble} | {n} | {bfs} | {diff} | {cond} |".format(
                difficulty=row["difficulty"],
                boxes=row["num_boxes"],
                scramble=row["scramble_steps"],
                n=s["n"],
                bfs=pct(s["bfs_solved"], s["n"]),
                diff=pct(s["diffusion_solved"], s["n"]),
                cond=pct(s["diffusion_solved_given_bfs"], s["bfs_solved"]),
            )
        )
    lines.append("")
    lines.append("| Difficulty | N | BFS | Diffusion | Diffusion given BFS |")
    lines.append("|---|---:|---|---|---|")
    for name in ("easy", "medium", "hard", "overall"):
        s = results["by_difficulty"][name]
        lines.append(
            "| {name} | {n} | {bfs} | {diff} | {cond} |".format(
                name=name,
                n=s["n"],
                bfs=pct(s["bfs_solved"], s["n"]),
                diff=pct(s["diffusion_solved"], s["n"]),
                cond=pct(s["diffusion_solved_given_bfs"], s["bfs_solved"]),
            )
        )
    failures = [p for p in results["puzzles"] if (not p["bfs_solved"]) or (not p["diffusion_solved"])]
    lines.append("")
    lines.append(f"Failures recorded: {len(failures)} of {results['dataset_size']} puzzles.")
    if failures:
        lines.append("")
        lines.append("| Puzzle | Difficulty | Boxes | Scramble | BFS | Diffusion |")
        lines.append("|---|---|---:|---:|---|---|")
        for p in failures:
            lines.append(
                "| {id} | {difficulty} | {boxes} | {scramble} | {bfs} | {diff} |".format(
                    id=p["id"],
                    difficulty=p["difficulty"],
                    boxes=p["num_boxes"],
                    scramble=p["scramble_steps"],
                    bfs="ok" if p["bfs_solved"] else "FAIL",
                    diff="ok" if p["diffusion_solved"] else "FAIL",
                )
            )
    return "\n".join(lines)


def run() -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model weights at {MODEL_PATH}")

    random.seed(SEED)
    np.random.seed(SEED)
    import torch

    torch.manual_seed(SEED)

    started = time.time()
    puzzles: list[dict] = []
    puzzle_id = 0

    for config in CONFIGS:
        for _ in range(N_PER_CONFIG):
            puzzle_id += 1
            grid, targets, gen_attempts, already_solved = generate_scrambled(
                config["num_boxes"], config["scramble_steps"]
            )

            t0 = time.perf_counter()
            bfs_path = bfs_solve(grid, targets, max_nodes=BFS_MAX_NODES)
            bfs_seconds = time.perf_counter() - t0
            bfs_solved = bfs_path is not None

            t0 = time.perf_counter()
            diff_path = diffusion_solve_fast(grid, targets, max_iters=DIFFUSION_MAX_ITERS)
            diff_seconds = time.perf_counter() - t0
            diffusion_solved = diff_path is not None

            puzzles.append(
                {
                    "id": puzzle_id,
                    "difficulty": config["difficulty"],
                    "num_boxes": config["num_boxes"],
                    "scramble_steps": config["scramble_steps"],
                    "generation_attempts": gen_attempts,
                    "already_solved": already_solved,
                    "boxes_off_target": boxes_off_target(grid),
                    "bfs_solved": bool(bfs_solved),
                    "bfs_length": None if bfs_path is None else len(bfs_path),
                    "bfs_seconds": bfs_seconds,
                    "diffusion_solved": bool(diffusion_solved),
                    "diffusion_length": None if diff_path is None else len(diff_path),
                    "diffusion_seconds": diff_seconds,
                }
            )
            status = "BFS {} DIFF {}".format(
                "ok" if bfs_solved else "FAIL",
                "ok" if diffusion_solved else "FAIL",
            )
            print(
                f"[{puzzle_id}/{len(CONFIGS) * N_PER_CONFIG}] "
                f"{config['difficulty']} boxes={config['num_boxes']} "
                f"scramble={config['scramble_steps']} {status}",
                flush=True,
            )

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for p in puzzles:
        grouped[(p["difficulty"], p["num_boxes"], p["scramble_steps"])].append(p)
    by_config = []
    for config in CONFIGS:
        key = (config["difficulty"], config["num_boxes"], config["scramble_steps"])
        by_config.append({**config, "summary": summarize(grouped[key])})

    by_difficulty = {
        name: summarize([p for p in puzzles if p["difficulty"] == name])
        for name in ("easy", "medium", "hard")
    }
    by_difficulty["overall"] = summarize(puzzles)

    results = {
        "task": "GS-T5",
        "metric": "solve_rate_vs_bfs_by_difficulty",
        "model_name": MODEL_NAME,
        "model_path": str(MODEL_PATH.name),
        "model_hparams": {
            "seq_len": 20,
            "timesteps": 100,
            "hidden_dim": 128,
            "ddim_steps": 10,
            "batch_samples": 4,
            "max_iters": DIFFUSION_MAX_ITERS,
        },
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "measured_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_size": len(puzzles),
        "n_per_config": N_PER_CONFIG,
        "seed": SEED,
        "bfs_max_nodes": BFS_MAX_NODES,
        "hardware": hardware_info(),
        "protocol": {
            "generation": (
                "Reverse-scramble from a solved 8x8 board using SokobanGen, "
                "matching sokoban_data_gen.generate_dataset difficulty labels. "
                "Retry scramble if no box is off a target. Do not drop BFS or "
                "diffusion failures."
            ),
            "bfs": "sokoban_data_gen.bfs_solve with max_nodes=30000",
            "diffusion": "app.diffusion_solve_fast with max_iters=20 (production path)",
        },
        "elapsed_seconds": time.time() - started,
        "by_difficulty": by_difficulty,
        "by_config": by_config,
        "puzzles": puzzles,
    }
    return results


def main() -> None:
    results = run()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    table = markdown_table(results)
    print()
    print(table)
    print()
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
