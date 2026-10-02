#!/usr/bin/env python3
"""Regression gate: compare a fresh scramble-hard run to the committed results.

The scramble-hard run is deterministic (seed 42, iteration-capped diffusion), so every puzzle's
BFS and diffusion outcome must match eval/scramble_hard_solve_rate.json exactly. A rerun on
2026-10-02 on a heavily loaded machine matched all 240 puzzles, which is the evidence for using
an exact gate.

  python eval/measure_solve_rate.py --protocol scramble-hard --output /tmp/sh.json
  python eval/check_regression.py /tmp/sh.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

COMMITTED = Path(__file__).resolve().parent / "scramble_hard_solve_rate.json"
FIELDS = ("bfs_solved", "bfs_length", "diffusion_solved", "diffusion_length")


def compare(fresh: dict, committed: dict) -> list[str]:
    problems = []
    a = {p["id"]: p for p in fresh["puzzles"]}
    b = {p["id"]: p for p in committed["puzzles"]}
    if set(a) != set(b):
        problems.append(f"puzzle ids differ: fresh {len(a)}, committed {len(b)}")
    for pid in sorted(set(a) & set(b)):
        for f in FIELDS:
            if a[pid].get(f) != b[pid].get(f):
                problems.append(f"puzzle {pid} {f}: fresh {a[pid].get(f)} vs committed {b[pid].get(f)}")
    return problems


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    path = Path(argv[1])
    if not path.exists():
        print(f"no such file: {path}")
        return 2
    fresh = json.loads(path.read_text())
    committed = json.loads(COMMITTED.read_text())
    problems = compare(fresh, committed)
    o = fresh["by_difficulty"]["overall"]
    print(f"fresh run: BFS {o['bfs_solved']}/{o['n']}, diffusion {o['diffusion_solved']}/{o['n']}")
    if problems:
        print(f"REGRESSION: {len(problems)} differences")
        for p in problems[:20]:
            print("  " + p)
        return 1
    print("scramble-hard matches committed results on all puzzles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
