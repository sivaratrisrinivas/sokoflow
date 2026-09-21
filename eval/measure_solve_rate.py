#!/usr/bin/env python3
"""
Solve-rate harness.

Protocols:
  gs-t5          Historical (default for this filename). No train-length filter.
  scramble-hard  GS-T47 headline. Drops 1-box-off and short-path inflators;
                 length gate matches training len(traj) >= 5.
  microban       Separate OOD table. Never mix into scramble-hard or GS-T5.

From the repo root:

    python eval/measure_solve_rate.py --protocol scramble-hard
    python eval/measure_solve_rate.py --protocol microban
    python eval/measure_solve_rate.py --protocol gs-t5 --smoke
"""

from __future__ import annotations

import argparse
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

from eval.microban_levels import MICROBAN_LEVELS, MICROBAN_SOURCE  # noqa: E402
from eval.protocol import (  # noqa: E402
    MIN_BOXES_OFF_TARGET,
    MIN_TRAJECTORY_LEN,
    SCRAMBLE_HARD_DEFINITION,
    is_scramble_hard,
)
from eval.xsb import microban_to_8x8  # noqa: E402
from sokoban_data_gen import SokobanGen, bfs_solve, boxes_off_target  # noqa: E402
from sokoban_solve import diffusion_solve_fast  # noqa: E402

OUTPUT_PATH = ROOT / "eval" / "gs_t5_solve_rate.json"
SCRAMBLE_HARD_PATH = ROOT / "eval" / "scramble_hard_solve_rate.json"
MICROBAN_PATH = ROOT / "eval" / "microban_ood_solve_rate.json"
_model_env = os.environ.get("SOKOFLOW_MODEL_PATH", "sokoban_diffusion.pth")
MODEL_PATH = Path(_model_env)
if not MODEL_PATH.is_absolute():
    MODEL_PATH = ROOT / MODEL_PATH
MODEL_NAME = "SokobanDiffusion"
SEED = 42
N_PER_CONFIG = 20
BFS_MAX_NODES = 30000
DIFFUSION_MAX_ITERS = 20
MAX_GEN_ATTEMPTS = 80
MAX_HARD_ATTEMPTS = 250

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


def generate_scramble_hard(
    num_boxes: int, scramble_steps: int
) -> tuple[np.ndarray, np.ndarray, int, bool, list | None, float]:
    """Rejection-sample until the scramble-hard gates pass. Runs BFS as the length gate."""
    last = None
    bfs_seconds = 0.0
    for attempts in range(1, MAX_HARD_ATTEMPTS + 1):
        grid, targets, _gen_attempts, already_solved = generate_scrambled(num_boxes, scramble_steps)
        t0 = time.perf_counter()
        bfs_path = bfs_solve(grid, targets, max_nodes=BFS_MAX_NODES)
        bfs_seconds = time.perf_counter() - t0
        bfs_len = None if bfs_path is None else len(bfs_path)
        last = (grid, targets, attempts, already_solved, bfs_path, bfs_seconds)
        if is_scramble_hard(boxes_off_target=boxes_off_target(grid), bfs_length=bfs_len):
            return last
    raise RuntimeError(
        f"could not sample a scramble-hard puzzle in {MAX_HARD_ATTEMPTS} tries "
        f"(boxes={num_boxes}, scramble={scramble_steps})"
    )


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
        "already_solved_count": sum(1 for r in rows if r.get("already_solved")),
        "mean_bfs_seconds": float(np.mean(bfs_times)) if bfs_times else None,
        "mean_diffusion_seconds": float(np.mean(diff_times)) if diff_times else None,
        "mean_bfs_length": float(np.mean(bfs_lens)) if bfs_lens else None,
        "mean_diffusion_length": float(np.mean(diff_lens)) if diff_lens else None,
        "mean_boxes_off_target": float(np.mean([r["boxes_off_target"] for r in rows])) if rows else None,
    }


def gs_t5_protocol_text() -> dict:
    return {
        "generation": (
            "Reverse-scramble from a solved 8x8 board using SokobanGen, "
            "matching sokoban_data_gen.generate_dataset difficulty labels. "
            "Retry scramble if no box is off a target. Do not drop BFS or "
            "diffusion failures."
        ),
        "bfs": "sokoban_data_gen.bfs_solve with max_nodes=30000",
        "diffusion": "sokoban_solve.diffusion_solve_fast with max_iters=20 (production path)",
        "eval": (
            "GS-T5 does not drop short BFS trajectories. Training "
            "sokoban_data_gen.generate_dataset keeps only len(traj) >= 5. "
            "Those filters do not match. Historical — do not treat as scramble-hard."
        ),
        "train_filter": f"len(traj) >= {MIN_TRAJECTORY_LEN}",
        "eval_filter": "none (historical mismatch)",
    }


def scramble_hard_protocol_text() -> dict:
    return {
        "generation": SCRAMBLE_HARD_DEFINITION,
        "bfs": "sokoban_data_gen.bfs_solve with max_nodes=30000",
        "diffusion": "sokoban_solve.diffusion_solve_fast with max_iters=20 (production path)",
        "eval": (
            f"Eligible iff boxes_off_target >= {MIN_BOXES_OFF_TARGET} and "
            f"(BFS fail or bfs_length >= {MIN_TRAJECTORY_LEN}). "
            "Length gate matches training. 1-box-off boards are excluded."
        ),
        "train_filter": (
            f"len(traj) >= {MIN_TRAJECTORY_LEN}; generate_dataset also drops "
            f"boxes_off_target < {MIN_BOXES_OFF_TARGET} as of GS-T47 "
            "(committed weights were trained with the length gate only)"
        ),
        "eval_filter": (
            f"boxes_off_target >= {MIN_BOXES_OFF_TARGET} and "
            f"(bfs_length is None or bfs_length >= {MIN_TRAJECTORY_LEN})"
        ),
    }


def microban_protocol_text() -> dict:
    return {
        "generation": (
            "Load published Microban XSB levels whose bounding box is <= 8x8 "
            "(38 of 155). Seal XSB void to walls, center-pad with walls to 8x8. "
            "Not reverse-scramble. Separate OOD table only."
        ),
        "bfs": "sokoban_data_gen.bfs_solve with max_nodes=30000",
        "diffusion": "sokoban_solve.diffusion_solve_fast with max_iters=20 (production path)",
        "eval": (
            "No scramble-hard or GS-T5 filters. Every included Microban level is "
            "attempted. Do not mix this rate into headline tables."
        ),
        "source": MICROBAN_SOURCE,
        "limitations": [
            "Train distribution is reverse-scrambled 8x8 with perimeter walls only.",
            "Microban uses interior walls and classic Sokoban topology.",
            "Not the full 155-level set — only boards that fit 8x8.",
            "seq_len=20; longer BFS paths cannot be emitted by the model.",
        ],
    }


def _base_results(*, task: str, metric: str, protocol: dict, puzzles: list[dict], started: float, seed: int, extra: dict | None = None) -> dict:
    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for p in puzzles:
        grouped[(p.get("difficulty"), p.get("num_boxes"), p.get("scramble_steps"))].append(p)

    results = {
        "task": task,
        "metric": metric,
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
        "seed": seed,
        "bfs_max_nodes": BFS_MAX_NODES,
        "hardware": hardware_info(),
        "protocol": protocol,
        "elapsed_seconds": time.time() - started,
        "puzzles": puzzles,
    }
    if extra:
        results.update(extra)
    return results


def _measure_pair(grid, targets) -> tuple[list | None, float, list | None, float]:
    t0 = time.perf_counter()
    bfs_path = bfs_solve(grid, targets, max_nodes=BFS_MAX_NODES)
    bfs_seconds = time.perf_counter() - t0
    t0 = time.perf_counter()
    diff_path = diffusion_solve_fast(grid, targets, max_iters=DIFFUSION_MAX_ITERS)
    diff_seconds = time.perf_counter() - t0
    return bfs_path, bfs_seconds, diff_path, diff_seconds


def run(
    *,
    n_per_config: int = N_PER_CONFIG,
    configs: list[dict] | None = None,
    seed: int = SEED,
    protocol: str = "gs-t5",
    levels: list[dict] | None = None,
) -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing model weights at {MODEL_PATH}")

    random.seed(seed)
    np.random.seed(seed)
    import torch

    torch.manual_seed(seed)
    started = time.time()

    if protocol == "microban":
        return run_microban(seed=seed, started=started, levels=levels)

    selected = configs if configs is not None else CONFIGS
    puzzles: list[dict] = []
    puzzle_id = 0
    total = len(selected) * n_per_config
    hard = protocol == "scramble-hard"

    for config in selected:
        for _ in range(n_per_config):
            puzzle_id += 1
            if hard:
                grid, targets, gen_attempts, already_solved, bfs_path, bfs_seconds = generate_scramble_hard(
                    config["num_boxes"], config["scramble_steps"]
                )
                t0 = time.perf_counter()
                diff_path = diffusion_solve_fast(grid, targets, max_iters=DIFFUSION_MAX_ITERS)
                diff_seconds = time.perf_counter() - t0
            else:
                grid, targets, gen_attempts, already_solved = generate_scrambled(
                    config["num_boxes"], config["scramble_steps"]
                )
                bfs_path, bfs_seconds, diff_path, diff_seconds = _measure_pair(grid, targets)

            bfs_solved = bfs_path is not None
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
                    "scramble_hard_eligible": is_scramble_hard(
                        boxes_off_target=boxes_off_target(grid),
                        bfs_length=None if bfs_path is None else len(bfs_path),
                    ),
                }
            )
            status = "BFS {} DIFF {}".format(
                "ok" if bfs_solved else "FAIL",
                "ok" if diffusion_solved else "FAIL",
            )
            print(
                f"[{puzzle_id}/{total}] "
                f"{protocol} {config['difficulty']} boxes={config['num_boxes']} "
                f"scramble={config['scramble_steps']} {status}",
                flush=True,
            )

    grouped: dict[tuple, list[dict]] = defaultdict(list)
    for p in puzzles:
        grouped[(p["difficulty"], p["num_boxes"], p["scramble_steps"])].append(p)
    by_config = []
    for config in selected:
        key = (config["difficulty"], config["num_boxes"], config["scramble_steps"])
        by_config.append({**config, "summary": summarize(grouped[key])})

    by_difficulty = {
        name: summarize([p for p in puzzles if p["difficulty"] == name])
        for name in ("easy", "medium", "hard")
    }
    by_difficulty["overall"] = summarize(puzzles)

    task = "GS-T47-scramble-hard" if hard else "GS-T5"
    metric = "solve_rate_vs_bfs_scramble_hard" if hard else "solve_rate_vs_bfs_by_difficulty"
    proto = scramble_hard_protocol_text() if hard else gs_t5_protocol_text()
    results = _base_results(
        task=task,
        metric=metric,
        protocol=proto,
        puzzles=puzzles,
        started=started,
        seed=seed,
        extra={
            "n_per_config": n_per_config,
            "by_difficulty": by_difficulty,
            "by_config": by_config,
        },
    )
    return results


def run_microban(*, seed: int, started: float, levels: list[dict] | None = None) -> dict:
    selected = levels if levels is not None else MICROBAN_LEVELS
    puzzles: list[dict] = []
    total = len(selected)
    for i, level in enumerate(selected, start=1):
        grid, targets = microban_to_8x8(level["xsb"])
        bfs_path, bfs_seconds, diff_path, diff_seconds = _measure_pair(grid, targets)
        bfs_solved = bfs_path is not None
        diffusion_solved = diff_path is not None
        puzzles.append(
            {
                "id": i,
                "microban_id": level["id"],
                "name": level["name"],
                "native_width": level["width"],
                "native_height": level["height"],
                "already_solved": False,
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
        print(f"[{i}/{total}] microban {level['name']} {status}", flush=True)

    overall = summarize(puzzles)
    return _base_results(
        task="GS-T47-microban-ood",
        metric="solve_rate_vs_bfs_microban_ood",
        protocol=microban_protocol_text(),
        puzzles=puzzles,
        started=started,
        seed=seed,
        extra={"by_difficulty": {"overall": overall}, "source": MICROBAN_SOURCE},
    )


def markdown_table(results: dict) -> str:
    task = results.get("task", "")
    if task == "GS-T47-microban-ood":
        return markdown_microban(results)
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
    return "\n".join(lines)


def markdown_microban(results: dict) -> str:
    s = results["by_difficulty"]["overall"]
    lines = [
        "| Set | N | BFS | Diffusion | Diffusion given BFS |",
        "|---|---:|---|---|---|",
        "| Microban OOD (8×8-fitting subset) | {n} | {bfs} | {diff} | {cond} |".format(
            n=s["n"],
            bfs=pct(s["bfs_solved"], s["n"]),
            diff=pct(s["diffusion_solved"], s["n"]),
            cond=pct(s["diffusion_solved_given_bfs"], s["bfs_solved"]),
        ),
        "",
        "| Level | Size | Boxes off | BFS | Diffusion |",
        "|---|---|---:|---|---|",
    ]
    for p in results["puzzles"]:
        lines.append(
            "| {name} | {w}×{h} | {off} | {bfs} | {diff} |".format(
                name=p["name"],
                w=p["native_width"],
                h=p["native_height"],
                off=p["boxes_off_target"],
                bfs="ok" if p["bfs_solved"] else "FAIL",
                diff="ok" if p["diffusion_solved"] else "FAIL",
            )
        )
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GS-T5 / scramble-hard / Microban OOD measurement.")
    parser.add_argument(
        "--protocol",
        choices=("gs-t5", "scramble-hard", "microban"),
        default="gs-t5",
        help="gs-t5 is historical. scramble-hard is the GS-T47 headline. microban is OOD-only.",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny run; does not overwrite published tables.",
    )
    parser.add_argument("--n-per-config", type=int, default=None)
    parser.add_argument("--max-configs", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the historical eval/gs_t5_solve_rate.json (2026-08-24).",
    )
    return parser.parse_args(argv)


def dated_output_path(date: str, protocol: str = "gs-t5") -> Path:
    if protocol == "scramble-hard":
        return ROOT / "eval" / f"scramble_hard_solve_rate-{date}.json"
    if protocol == "microban":
        return ROOT / "eval" / f"microban_ood_solve_rate-{date}.json"
    return ROOT / "eval" / f"gs_t5_solve_rate-{date}.json"


def canonical_path(protocol: str) -> Path:
    if protocol == "scramble-hard":
        return SCRAMBLE_HARD_PATH
    if protocol == "microban":
        return MICROBAN_PATH
    return OUTPUT_PATH


def resolve_output_path(args: argparse.Namespace, results: dict) -> Path:
    """Never clobber the historical GS-T5 JSON unless --force is set."""
    protocol = getattr(args, "protocol", "gs-t5")
    if args.smoke:
        return args.output or (ROOT / "eval" / f"smoke_{protocol.replace('-', '_')}.json")
    requested = args.output
    if requested is None:
        if protocol == "gs-t5":
            requested = OUTPUT_PATH if args.force else dated_output_path(results["date"], protocol)
        else:
            # Headline / OOD tables: dated file. Also write canonical beside it in main().
            requested = dated_output_path(results["date"], protocol)
    requested = Path(requested)
    if requested.resolve() == OUTPUT_PATH.resolve() and not args.force:
        return dated_output_path(results["date"], "gs-t5")
    return requested


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configs = list(CONFIGS)
    if args.max_configs is not None:
        configs = configs[: max(1, args.max_configs)]
    n_per_config = N_PER_CONFIG if args.n_per_config is None else max(1, args.n_per_config)
    if args.smoke:
        if args.protocol == "microban":
            results = run(protocol="microban", seed=0, levels=MICROBAN_LEVELS[:1])
        else:
            configs = CONFIGS[:1]
            n_per_config = 1
            results = run(
                n_per_config=n_per_config,
                configs=configs,
                seed=0,
                protocol=args.protocol,
            )
    else:
        results = run(n_per_config=n_per_config, configs=configs, protocol=args.protocol)

    table = markdown_table(results)
    print()
    print(table)
    print()

    out = resolve_output_path(args, results)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(results, indent=2) + "\n"
    out.write_text(payload, encoding="utf-8")
    if args.smoke:
        print(f"Smoke write {out} (published tables left unchanged).")
        return
    if args.protocol in {"scramble-hard", "microban"} and args.output is None:
        canonical = canonical_path(args.protocol)
        canonical.write_text(payload, encoding="utf-8")
        print(f"Wrote {out} and {canonical}")
        return
    if out.resolve() != OUTPUT_PATH.resolve():
        print(f"Wrote {out} (historical {OUTPUT_PATH.name} left unchanged; pass --force to replace it).")
        return
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
