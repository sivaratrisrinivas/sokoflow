"""Shared CPU diffusion solve path used by Flask and the Gradio Space."""

from __future__ import annotations

import os
import threading

import numpy as np
import torch

from sokoban_diffusion import SokobanDiffusion, state_to_tensor

DEFAULT_MODEL_PATH = os.environ.get("SOKOFLOW_MODEL_PATH", "sokoban_diffusion.pth")
ACTION_MAP = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
ACTION_DELTAS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}

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


def is_solved(grid):
    return bool(int(np.count_nonzero(grid == 3)) == 0)


def diffusion_solve_fast(grid, targets, max_iters=20):
    if not ensure_model_loaded():
        return None

    current_grid = grid.copy()
    current_pos = tuple(map(int, np.argwhere(current_grid == 2)[0]))
    solution = []
    visited = {current_grid.tobytes()}

    for _iteration in range(max_iters):
        if is_solved(current_grid):
            return solution

        state_tensor = state_to_tensor(current_grid).unsqueeze(0)
        batch_state = state_tensor.repeat(4, 1, 1, 1)

        with torch.inference_mode():
            all_actions = diffusion_model.sample_fast(batch_state, steps=10)

        best_progress = 0
        best_result = None

        for seq_idx in range(4):
            test_grid = current_grid.copy()
            test_pos = current_pos
            test_solution = []
            test_visited = visited.copy()

            actions = all_actions[seq_idx].tolist()

            for action_idx in actions:
                if is_solved(test_grid):
                    return solution + test_solution

                action = ACTION_MAP[action_idx]
                new_grid, new_pos = is_valid_move(test_grid, targets, test_pos, action)

                if new_grid is not None:
                    h = new_grid.tobytes()
                    if h not in test_visited:
                        test_visited.add(h)
                        test_grid = new_grid
                        test_pos = new_pos
                        test_solution.append(action)

            boxes_done = np.count_nonzero(test_grid == 5)
            progress = len(test_solution) + boxes_done * 2

            if progress > best_progress:
                best_progress = progress
                best_result = (test_grid, test_pos, test_solution, test_visited)

        if best_result and best_result[2]:
            current_grid, current_pos, new_moves, visited = best_result
            solution.extend(new_moves)
        else:
            for action in ["UP", "DOWN", "LEFT", "RIGHT"]:
                new_grid, new_pos = is_valid_move(current_grid, targets, current_pos, action)
                if new_grid is not None and new_grid.tobytes() not in visited:
                    visited.add(new_grid.tobytes())
                    current_grid = new_grid
                    current_pos = new_pos
                    solution.append(action)
                    break

    return solution if is_solved(current_grid) else None
