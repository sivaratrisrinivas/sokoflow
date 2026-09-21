"""Shared CPU diffusion + BFS solve path used by Flask and the Gradio Space."""

from __future__ import annotations

import os
import threading

import numpy as np
import torch

from sokoban_data_gen import bfs_solve_report as bfs_solve_report_indices
from sokoban_diffusion import SokobanDiffusion, state_to_tensor

DEFAULT_MODEL_PATH = os.environ.get("SOKOFLOW_MODEL_PATH", "sokoban_diffusion.pth")
ACTION_MAP = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
ACTION_DELTAS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
ILLEGAL_WHY = {
    "walk_into_wall": "walk into wall",
    "walk_out_of_bounds": "walk off the board",
    "stuck_push_wall": "stuck push into wall",
    "stuck_push_box": "stuck push into a box",
    "push_out_of_bounds": "stuck push off the board",
}

diffusion_model = SokobanDiffusion(seq_len=20, timesteps=100, hidden_dim=128)
_model_lock = threading.Lock()
_model_ready = False
_model_error: str | None = None


def model_path() -> str:
    return os.environ.get("SOKOFLOW_MODEL_PATH", DEFAULT_MODEL_PATH)


def model_error() -> str | None:
    return _model_error


def model_ready() -> bool:
    return _model_ready


def ensure_model_loaded() -> bool:
    """Load committed weights once. Safe to call from workers/threads."""
    global _model_ready, _model_error
    if _model_ready:
        return True
    with _model_lock:
        if _model_ready:
            return True
        path = model_path()
        if not os.path.exists(path):
            _model_error = f"missing weights at {path}"
            print(f"⚠️ Train model first: python sokoban_diffusion.py ({_model_error})")
            return False
        try:
            diffusion_model.load_state_dict(
                torch.load(path, map_location="cpu", weights_only=True)
            )
            diffusion_model.eval()
            torch.set_grad_enabled(False)
            _model_ready = True
            _model_error = None
            print("🎨 Diffusion model loaded!")
            return True
        except Exception as exc:  # pragma: no cover - reported via /health
            _model_error = str(exc)
            print(f"⚠️ Failed to load model: {exc}")
            return False


def player_pos(grid):
    found = np.argwhere(grid == 2)
    if len(found) == 0:
        return None
    return tuple(map(int, found[0]))


def is_valid_move(grid, targets, pos, action):
    dr, dc = ACTION_DELTAS[action]
    pr, pc = pos
    nr, nc = pr + dr, pc + dc

    if not (0 <= nr < 8 and 0 <= nc < 8):
        return None, None

    ng = grid.copy()

    if ng[nr, nc] == 1:
        return None, None

    if ng[nr, nc] in [0, 4]:
        ng[pr, pc] = 4 if targets[pr, pc] else 0
        ng[nr, nc] = 2
        return ng, (nr, nc)

    if ng[nr, nc] in [3, 5]:
        nnr, nnc = nr + dr, nc + dc
        if 0 <= nnr < 8 and 0 <= nnc < 8 and ng[nnr, nnc] in [0, 4]:
            ng[pr, pc] = 4 if targets[pr, pc] else 0
            ng[nr, nc] = 2
            ng[nnr, nnc] = 5 if targets[nnr, nnc] else 3
            return ng, (nr, nc)

    return None, None


def illegal_reason(grid, targets, pos, action) -> str | None:
    """Why a proposed action is illegal, or None if it is legal."""
    dr, dc = ACTION_DELTAS[action]
    pr, pc = pos
    nr, nc = pr + dr, pc + dc
    if not (0 <= nr < 8 and 0 <= nc < 8):
        return "walk_out_of_bounds"
    cell = int(grid[nr, nc])
    if cell == 1:
        return "walk_into_wall"
    if cell in (0, 4):
        return None
    if cell in (3, 5):
        nnr, nnc = nr + dr, nc + dc
        if not (0 <= nnr < 8 and 0 <= nnc < 8):
            return "push_out_of_bounds"
        ahead = int(grid[nnr, nnc])
        if ahead == 1:
            return "stuck_push_wall"
        if ahead in (3, 5):
            return "stuck_push_box"
        if ahead in (0, 4):
            return None
        return "stuck_push_wall"
    return "walk_into_wall"


def is_solved(grid):
    return bool(int(np.count_nonzero(grid == 3)) == 0)


def boxes_off_goal(grid) -> int:
    return int(np.count_nonzero(grid == 3))


def replay_until_illegal(grid, targets, actions: list[str]):
    """Walk a proposed sequence; stop at the first illegal. Always-legal prefix."""
    current = np.array(grid, dtype=int).copy()
    targets = np.array(targets, dtype=bool)
    pos = player_pos(current)
    legal: list[str] = []
    trail: list[tuple[int, int]] = []
    first_illegal = None
    if pos is not None:
        trail.append(pos)

    for index, action in enumerate(actions):
        if pos is None:
            first_illegal = {
                "index": index,
                "action": action,
                "why": "walk_out_of_bounds",
                "why_text": "no player on the board",
            }
            break
        why = illegal_reason(current, targets, pos, action)
        if why is not None:
            first_illegal = {
                "index": index,
                "action": action,
                "why": why,
                "why_text": ILLEGAL_WHY.get(why, why),
            }
            break
        nxt, new_pos = is_valid_move(current, targets, pos, action)
        if nxt is None:
            first_illegal = {
                "index": index,
                "action": action,
                "why": "walk_into_wall",
                "why_text": ILLEGAL_WHY["walk_into_wall"],
            }
            break
        current = nxt
        pos = new_pos
        legal.append(action)
        if pos is not None:
            trail.append(pos)
        if is_solved(current):
            break

    return {
        "grid": current,
        "legal": legal,
        "trail": trail,
        "first_illegal": first_illegal,
    }


def walk_path_frames(grid, targets, actions: list[str]) -> list[dict]:
    """Execute a known-legal path into board frames for twin playback."""
    current = np.array(grid, dtype=int).copy()
    targets = np.array(targets, dtype=bool)
    pos = player_pos(current)
    trail: list[tuple[int, int]] = [pos] if pos is not None else []
    frames = [{"grid": current.copy(), "trail": list(trail)}]
    for action in actions or []:
        if pos is None:
            break
        nxt, new_pos = is_valid_move(current, targets, pos, action)
        if nxt is None:
            break
        current, pos = nxt, new_pos
        if pos is not None:
            trail.append(pos)
        frames.append({"grid": current.copy(), "trail": list(trail)})
    return frames


def bfs_solve_report(grid, targets, max_nodes=30000) -> dict:
    """BFS twin payload: solved / failed / node budget. Path uses action names."""
    raw = bfs_solve_report_indices(grid, targets, max_nodes=max_nodes)
    path_idx = raw["path"]
    path = None if path_idx is None else [ACTION_MAP[int(i)] for i in path_idx]
    solved = bool(raw["solved"])
    nodes = int(raw["nodes"])
    budget = int(raw["max_nodes"])
    if solved:
        status = f"solved · {len(path)} moves · {nodes} nodes"
    elif raw["budget_exhausted"]:
        status = f"failed · {budget} node budget"
    else:
        status = f"failed · no path · {nodes} nodes"
    return {
        "solved": solved,
        "path": path or [],
        "nodes": nodes,
        "max_nodes": budget,
        "budget_exhausted": bool(raw["budget_exhausted"]),
        "status": status,
    }


def _empty_report(*, reason: str, autopsy: str, max_iters: int) -> dict:
    return {
        "solved": False,
        "path": [],
        "iters_used": 0,
        "max_iters": int(max_iters),
        "reason": reason,
        "first_illegal": None,
        "skipped_illegal": 0,
        "boxes_off": None,
        "denoise_frames": [],
        "autopsy": autopsy,
        "sample_index": 0,
    }


def _spread_indices(length: int, max_frames: int) -> list[int]:
    """Inclusive indices in [0, length-1], always including the ends."""
    if length <= 1:
        return [0]
    max_frames = max(2, int(max_frames))
    if length <= max_frames:
        return list(range(length))
    picks: list[int] = []
    for i in range(max_frames):
        idx = int(round(i * (length - 1) / (max_frames - 1)))
        if not picks or idx != picks[-1]:
            picks.append(idx)
    if picks[0] != 0:
        picks.insert(0, 0)
    if picks[-1] != length - 1:
        picks.append(length - 1)
    return picks


def theater_frames_from_path(grid, targets, path: list[str] | None, *, max_frames: int = 11) -> list[dict]:
    """Theater / GIF frames along the executed legal path (same path the twin uses)."""
    path = list(path or [])
    walked = walk_path_frames(grid, targets, path)
    picks = _spread_indices(len(walked), max_frames)
    n = len(picks)
    frames: list[dict] = []
    for i, wi in enumerate(picks):
        if i == 0:
            t = 100
        elif i == n - 1:
            t = 0
        else:
            t = int(round(100 * (1 - i / (n - 1))))
        frames.append(
            {
                "t": t,
                "actions": path[:wi],
                "legal_n": wi,
                "grid": walked[wi]["grid"],
                "trail": walked[wi]["trail"],
                "first_illegal": None,
            }
        )
    return frames


def _attach_theater(report: dict, start_grid, targets, *, trace: bool) -> dict:
    if trace:
        report["denoise_frames"] = theater_frames_from_path(
            start_grid, targets, report.get("path") or []
        )
    else:
        report["denoise_frames"] = []
    report["autopsy"] = format_autopsy(report)
    return report


def format_autopsy(report: dict) -> str:
    """Honest sentence. Never claims a solve when solved is false."""
    if report.get("solved"):
        moves = len(report.get("path") or [])
        iters = report.get("iters_used") or 0
        return f"Diffusion solved in {moves} moves after {iters} iteration(s)."

    parts: list[str] = []
    reason = report.get("reason")
    if reason == "model_missing":
        return report.get("autopsy") or f"Model not loaded: {model_error() or 'unknown'}"

    first = report.get("first_illegal")
    if first:
        step = int(first["index"]) + 1
        parts.append(
            f"First illegal move: {first['action']} at step {step} ({first['why_text']})."
        )
    skipped = int(report.get("skipped_illegal") or 0)
    if skipped:
        parts.append(f"Solver skipped {skipped} illegal action(s) and continued.")
    if reason == "no_progress":
        parts.append(
            f"Stuck after {report.get('iters_used') or 0} iteration(s): "
            "no new legal unvisited state."
        )
    elif reason == "exhausted_iters":
        parts.append(f"Exhausted {report.get('max_iters') or 0} iterations.")
    off = report.get("boxes_off")
    if off is None:
        pass
    elif off == 0:
        parts.append("No box is off a goal, but the solve still failed.")
    elif off == 1:
        parts.append("1 box still off goal.")
    else:
        parts.append(f"{off} boxes still off goal.")
    path = report.get("path") or []
    if path:
        parts.append(f"Legal prefix: {len(path)} move(s).")
    text = " ".join(parts).strip()
    return text or "Diffusion failed."


def diffusion_solve_report(grid, targets, max_iters=20, *, trace: bool = True, ddim_steps: int = 10) -> dict:
    """Run the demo solver and keep the evidence: denoise frames + autopsy."""
    max_iters = int(max_iters)
    if not ensure_model_loaded():
        return _empty_report(
            reason="model_missing",
            autopsy=f"Model not loaded: {model_error()}",
            max_iters=max_iters,
        )

    start_grid = np.array(grid, dtype=int).copy()
    targets = np.array(targets, dtype=bool)
    current_grid = start_grid.copy()
    current_pos = player_pos(current_grid)
    if current_pos is None:
        report = _empty_report(
            reason="no_progress",
            autopsy="No player on the board.",
            max_iters=max_iters,
        )
        report["boxes_off"] = boxes_off_goal(current_grid)
        report["autopsy"] = format_autopsy(report)
        return report

    solution: list[str] = []
    visited = {current_grid.tobytes()}
    first_illegal = None
    skipped_illegal = 0
    stalled_iters = 0
    iters_used = 0
    reason = "exhausted_iters"
    best_sample = 0

    for iteration in range(max_iters):
        iters_used = iteration + 1
        if is_solved(current_grid):
            reason = "solved"
            break

        state_tensor = state_to_tensor(current_grid).unsqueeze(0)
        batch_state = state_tensor.repeat(4, 1, 1, 1)

        with torch.inference_mode():
            all_actions = diffusion_model.sample_fast(batch_state, steps=ddim_steps)

        best_progress = 0
        best_result = None
        best_sample = 0
        sample_skipped: list[int] = []
        sample_first: list[dict | None] = []

        for seq_idx in range(4):
            test_grid = current_grid.copy()
            test_pos = current_pos
            test_solution: list[str] = []
            test_visited = visited.copy()
            actions = all_actions[seq_idx].tolist()
            skipped_here = 0
            first_here = None

            for step_i, action_idx in enumerate(actions):
                if is_solved(test_grid):
                    skipped_illegal += skipped_here
                    if first_illegal is None:
                        first_illegal = first_here
                    current_grid = test_grid
                    solution = solution + test_solution
                    report = {
                        "solved": True,
                        "path": solution,
                        "iters_used": iters_used,
                        "max_iters": max_iters,
                        "reason": "solved",
                        "first_illegal": first_illegal,
                        "skipped_illegal": skipped_illegal,
                        "boxes_off": 0,
                        "sample_index": seq_idx,
                    }
                    return _attach_theater(report, start_grid, targets, trace=trace)

                action = ACTION_MAP[int(action_idx)]
                why = illegal_reason(test_grid, targets, test_pos, action)
                new_grid, new_pos = is_valid_move(test_grid, targets, test_pos, action)

                if new_grid is None:
                    skipped_here += 1
                    if first_here is None:
                        first_here = {
                            "index": step_i,
                            "action": action,
                            "why": why or "walk_into_wall",
                            "why_text": ILLEGAL_WHY.get(why or "walk_into_wall", why or "illegal"),
                            "iteration": iteration,
                            "sample": seq_idx,
                        }
                    continue

                h = new_grid.tobytes()
                if h not in test_visited:
                    test_visited.add(h)
                    test_grid = new_grid
                    test_pos = new_pos
                    test_solution.append(action)

            sample_skipped.append(skipped_here)
            sample_first.append(first_here)
            boxes_done = np.count_nonzero(test_grid == 5)
            progress = len(test_solution) + boxes_done * 2
            if progress > best_progress:
                best_progress = progress
                best_result = (test_grid, test_pos, test_solution, test_visited)
                best_sample = seq_idx

        if best_result and best_result[2]:
            current_grid, current_pos, new_moves, visited = best_result
            solution.extend(new_moves)
            skipped_illegal += sample_skipped[best_sample]
            if first_illegal is None:
                first_illegal = sample_first[best_sample]
            stalled_iters = 0
        else:
            moved = False
            for action in ["UP", "DOWN", "LEFT", "RIGHT"]:
                new_grid, new_pos = is_valid_move(current_grid, targets, current_pos, action)
                if new_grid is not None and new_grid.tobytes() not in visited:
                    visited.add(new_grid.tobytes())
                    current_grid = new_grid
                    current_pos = new_pos
                    solution.append(action)
                    moved = True
                    break
            if first_illegal is None:
                first_illegal = next((item for item in sample_first if item), None)
            if not moved:
                stalled_iters += 1
                if stalled_iters >= 1 and iteration == max_iters - 1:
                    reason = "no_progress"
            else:
                stalled_iters = 0

    solved = is_solved(current_grid)
    if solved:
        reason = "solved"
    elif reason != "no_progress":
        reason = "exhausted_iters"

    report = {
        "solved": bool(solved),
        "path": solution,
        "iters_used": iters_used,
        "max_iters": max_iters,
        "reason": reason if not solved else "solved",
        "first_illegal": first_illegal,
        "skipped_illegal": skipped_illegal,
        "boxes_off": boxes_off_goal(current_grid),
        "sample_index": best_sample,
    }
    return _attach_theater(report, start_grid, targets, trace=trace)


def diffusion_solve_fast(grid, targets, max_iters=20):
    """Eval/API helper: action-name path if the board is solved, else None."""
    report = diffusion_solve_report(grid, targets, max_iters=max_iters, trace=False)
    if report["solved"]:
        return report["path"]
    return None
