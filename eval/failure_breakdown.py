"""Error analysis for scramble-hard: why does diffusion fail?

Reruns the deterministic scramble-hard protocol (same seed, same production fast solver, so the
boards and outcomes match eval/scramble_hard_solve_rate.json), then replays each failed board
through the demo solver (sokoban_solve.diffusion_solve_report), which records why it stopped.
The demo solver samples separately, so its outcome can differ from the fast path; that count is
reported too.

python eval/failure_breakdown.py [--output eval/scramble_hard_failures.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "eval"))

import measure_solve_rate as msr  # noqa: E402
from sokoban_solve import diffusion_solve_report  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, default=ROOT / "eval" / "scramble_hard_failures.json")
    args = ap.parse_args()

    boards: list[tuple] = []
    fast = msr.diffusion_solve_fast

    def recording(grid, targets, max_iters=20):
        path = fast(grid, targets, max_iters=max_iters)
        boards.append((grid.copy(), targets.copy(), path is not None))
        return path

    msr.diffusion_solve_fast = recording
    results = msr.run(protocol="scramble-hard")
    committed = json.loads((ROOT / "eval" / "scramble_hard_solve_rate.json").read_text())["puzzles"]
    same = sum(
        a["diffusion_solved"] == b["diffusion_solved"] and a["bfs_length"] == b["bfs_length"]
        for a, b in zip(results["puzzles"], committed, strict=True)
    )

    reasons: Counter = Counter()
    first_illegal: Counter = Counter()
    by_off: Counter = Counter()
    demo_solved = 0
    rows = []
    for (grid, targets, solved), puzzle in zip(boards, results["puzzles"], strict=True):
        if solved:
            continue
        rep = diffusion_solve_report(grid, targets, max_iters=20, trace=False)
        if rep.get("solved"):
            demo_solved += 1
        reason = rep.get("reason") or "unknown"
        reasons[reason] += 1
        fi = rep.get("first_illegal") or {}
        first_illegal[fi.get("why", "none")] += 1
        off = rep.get("boxes_off")
        by_off[str(off)] += 1
        rows.append({"id": puzzle["id"], "num_boxes": puzzle["num_boxes"], "bfs_length": puzzle["bfs_length"],
                     "reason": reason, "first_illegal": fi.get("why"), "boxes_off_after": off,
                     "legal_prefix": len(rep.get("path") or []), "demo_solved": bool(rep.get("solved"))})

    out = {
        "puzzles": len(boards),
        "match_committed": same,
        "fast_failures": len(rows),
        "demo_solver_solved_fast_failures": demo_solved,
        "stop_reason": dict(reasons),
        "first_illegal_move": dict(first_illegal),
        "boxes_off_after": dict(by_off),
        "rows": rows,
    }
    args.output.write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
