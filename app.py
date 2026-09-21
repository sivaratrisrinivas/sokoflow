"""
SokoFlow - Diffusion Sokoban Solver

A neural diffusion model that solves Sokoban puzzles.
Flow = The denoising process that transforms random moves → optimal solution.
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict, deque
from uuid import uuid4

import numpy as np
from flask import Flask, jsonify, make_response, render_template, request
from flask_cors import CORS

from sokoban_clip import denoise_gif_data_uri
from sokoban_engine import SokobanEnv
from sokoban_solve import (
    bfs_solve_report,
    diffusion_solve_fast,
    diffusion_solve_report,
    ensure_model_loaded,
    is_solved,
    is_valid_move,
    model_error,
    model_path,
)

APP_VERSION = "0.1.0"
DEFAULT_MODEL_PATH = os.environ.get("SOKOFLOW_MODEL_PATH", "sokoban_diffusion.pth")
MAX_CONTENT_BYTES = int(os.environ.get("MAX_CONTENT_LENGTH", str(16 * 1024)))
DEFAULT_RATE_LIMIT = int(os.environ.get("RATE_LIMIT_PER_MINUTE", "30"))
DEFAULT_NEW_GAME_RATE_LIMIT = int(os.environ.get("NEW_GAME_RATE_LIMIT_PER_MINUTE", "120"))

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_BYTES
app.config["JSONIFY_PRETTYPRINT_REGULAR"] = False

_cors_origins = os.environ.get("CORS_ORIGINS", "*").strip()
if _cors_origins == "*":
    CORS(app)
else:
    CORS(app, origins=[origin.strip() for origin in _cors_origins.split(",") if origin.strip()])

ACTION_MAP = {0: "UP", 1: "DOWN", 2: "LEFT", 3: "RIGHT"}
ACTION_DELTAS = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}

_sessions_lock = threading.Lock()
_sessions: dict[str, dict] = {}
MAX_SESSIONS = 256
SESSION_TTL_SECONDS = 30 * 60

_rate_lock = threading.Lock()
_rate_hits: dict[str, deque] = defaultdict(deque)


def to_python_types(obj):
    """
    Recursively convert numpy types to native Python types for JSON serialization.
    This bypasses any Flask/NumPy version incompatibilities.
    """
    if isinstance(obj, dict):
        return {k: to_python_types(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_python_types(x) for x in obj]
    if isinstance(obj, np.ndarray):
        return to_python_types(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj.item())
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def _trust_proxy() -> bool:
    return os.environ.get("TRUST_PROXY", "").strip().lower() in {"1", "true", "yes", "on"}


def _client_key() -> str:
    """Rate-limit identity. Ignore X-Forwarded-For unless TRUST_PROXY is set."""
    if _trust_proxy():
        forwarded = request.headers.get("X-Forwarded-For", "")
        if forwarded:
            return forwarded.split(",")[0].strip()
    return request.remote_addr or "unknown"


def rate_limit_ok(*, group: str, limit: int | None = None) -> bool:
    """Per-IP buckets are split by route group (solve vs demo) so caps do not share hits."""
    if limit is None:
        max_hits = int(os.environ.get("RATE_LIMIT_PER_MINUTE", str(DEFAULT_RATE_LIMIT)))
    else:
        max_hits = limit
    if max_hits <= 0:
        return True
    now = time.time()
    key = f"{group}:{_client_key()}"
    with _rate_lock:
        bucket = _rate_hits[key]
        while bucket and now - bucket[0] > 60:
            bucket.popleft()
        if len(bucket) >= max_hits:
            return False
        bucket.append(now)
        return True


def _prune_sessions(now: float) -> None:
    stale = [sid for sid, sess in _sessions.items() if now - sess["ts"] > SESSION_TTL_SECONDS]
    for sid in stale:
        _sessions.pop(sid, None)
    while len(_sessions) > MAX_SESSIONS:
        oldest = min(_sessions, key=lambda k: _sessions[k]["ts"])
        _sessions.pop(oldest, None)


def _store_session(env: SokobanEnv, path: list | None) -> str:
    sid = uuid4().hex
    now = time.time()
    with _sessions_lock:
        _prune_sessions(now)
        _sessions[sid] = {
            "env": env,
            "path": path or [],
            "index": 0,
            "ts": now,
        }
    return sid


def _get_session() -> tuple[str | None, dict | None]:
    sid = request.cookies.get("sokoflow_sid")
    if not sid:
        return None, None
    with _sessions_lock:
        sess = _sessions.get(sid)
        if sess:
            sess["ts"] = time.time()
        return sid, sess


def _json(payload, status=200, session_id=None):
    response = make_response(jsonify(to_python_types(payload)), status)
    if session_id:
        response.set_cookie(
            "sokoflow_sid",
            session_id,
            httponly=True,
            samesite="Lax",
            max_age=SESSION_TTL_SECONDS,
        )
    return response


@app.errorhandler(413)
def payload_too_large(_error):
    return _json({"error": "payload_too_large", "limit_bytes": MAX_CONTENT_BYTES}, 413)


@app.errorhandler(400)
def bad_request(_error):
    return _json({"error": "bad_request"}, 400)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/health")
def health():
    loaded = ensure_model_loaded()
    return _json(
        {
            "status": "ok" if loaded else "degraded",
            "model_loaded": loaded,
            "model_path": os.path.basename(model_path()),
            "model_error": model_error(),
            "version": APP_VERSION,
            "eval": {
                "task": "GS-T47-scramble-hard",
                "date": "2026-09-21",
                "diffusion_solve_rate": 0.0625,
                "bfs_solve_rate": 0.9416666666666667,
                "n": 240,
                "caveat": "scramble-hard: boxes_off>=2 and bfs_len>=5 or BFS fail; not GS-T5",
                "historical_gs_t5": {
                    "task": "GS-T5",
                    "date": "2026-08-24",
                    "diffusion_solve_rate": 0.208,
                    "bfs_solve_rate": 0.954,
                    "n": 240,
                    "caveat": "not scramble-hard; many wins are 1-box-off short trajectories",
                },
            },
        }
    )


def _parse_board(data):
    try:
        grid = np.array(data["grid"], dtype=int)
        targets = np.array(data["targets"], dtype=bool)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("grid and targets are required") from exc
    if grid.shape != (8, 8) or targets.shape != (8, 8):
        raise ValueError("grid and targets must be 8x8")
    return grid, targets


@app.route("/api/solve", methods=["POST"])
def api_solve():
    if not rate_limit_ok(group="solve"):
        return _json({"error": "rate_limited"}, 429)
    data = request.get_json(silent=True) or {}
    try:
        grid, targets = _parse_board(data)
    except ValueError as exc:
        return _json({"error": "invalid_board", "detail": str(exc)}, 400)
    max_iters = data.get("max_iters", 20)
    try:
        max_iters = int(max_iters)
    except (TypeError, ValueError):
        return _json({"error": "invalid_max_iters"}, 400)
    max_iters = max(1, min(max_iters, 20))
    path = diffusion_solve_fast(grid, targets, max_iters=max_iters)
    return _json(
        {
            "path": path,
            "solved": path is not None,
            "moves": 0 if path is None else len(path),
        }
    )


@app.route("/api/new_game", methods=["POST"])
def new_game():
    if not rate_limit_ok(
        group="demo",
        limit=int(os.environ.get("NEW_GAME_RATE_LIMIT_PER_MINUTE", str(DEFAULT_NEW_GAME_RATE_LIMIT))),
    ):
        return _json({"error": "rate_limited"}, 429)

    data = request.get_json(silent=True) or {}
    difficulty = data.get("difficulty", 20)
    try:
        difficulty = int(difficulty)
    except (TypeError, ValueError):
        return _json({"error": "invalid_difficulty"}, 400)
    difficulty = max(1, min(difficulty, 80))

    env = SokobanEnv(num_boxes=3)
    env.reset_solved()
    for _ in range(difficulty):
        env.step_reverse()

    report = diffusion_solve_report(env.grid, env.targets, max_iters=20, trace=True)
    bfs = bfs_solve_report(env.grid, env.targets, max_nodes=30000)
    solved = bool(report.get("solved"))
    executed_path = list(report.get("path") or [])
    # Public path is empty when unsolved so clients cannot treat a legal
    # prefix as success. Theater still ships executed prefixes in denoise.
    public_path = executed_path if solved else []
    frames = [
        {
            "t": frame["t"],
            "actions": frame["actions"],
            "legal_n": frame["legal_n"],
            "first_illegal": frame["first_illegal"],
        }
        for frame in report.get("denoise_frames") or []
    ]
    clip = None
    if report.get("denoise_frames"):
        clip = denoise_gif_data_uri(report["denoise_frames"], env.targets)

    sid = _store_session(env, public_path)
    return _json(
        {
            "grid": env.grid,
            "targets": env.targets,
            "solvable": solved,
            "moves": len(public_path),
            "path": public_path,
            "diffusion_solved": solved,
            "diffusion_reason": report.get("reason"),
            "autopsy": "" if solved else (report.get("autopsy") or ""),
            "denoise": frames,
            "clip_gif": clip,
            "bfs": {
                "solved": bfs["solved"],
                "path": bfs["path"],
                "nodes": bfs["nodes"],
                "max_nodes": bfs["max_nodes"],
                "budget_exhausted": bfs["budget_exhausted"],
                "status": bfs["status"],
            },
        },
        session_id=sid,
    )


@app.route("/api/solve_step", methods=["POST"])
def solve_step():
    if not rate_limit_ok(
        group="demo",
        limit=int(os.environ.get("NEW_GAME_RATE_LIMIT_PER_MINUTE", str(DEFAULT_NEW_GAME_RATE_LIMIT))),
    ):
        return _json({"error": "rate_limited"}, 429)

    sid, sess = _get_session()
    if not sess:
        return _json(
            {
                "action": "GIVE_UP",
                "grid": None,
                "solved": False,
                "gave_up": True,
                "steps": 0,
                "message": "No active game",
            }
        )

    env = sess["env"]
    solution_path = sess["path"]
    solution_index = sess["index"]

    if not solution_path:
        return _json(
            {
                "action": "GIVE_UP",
                "grid": env.grid,
                "solved": False,
                "gave_up": True,
                "steps": 0,
                "message": "Diffusion failed",
            },
            session_id=sid,
        )

    if solution_index >= len(solution_path):
        return _json(
            {
                "action": "DONE",
                "grid": env.grid,
                "solved": is_solved(env.grid),
                "steps": solution_index,
                "total": len(solution_path),
            },
            session_id=sid,
        )

    action = solution_path[solution_index]
    sess["index"] = solution_index + 1
    env.step(action)

    return _json(
        {
            "action": action,
            "grid": env.grid,
            "solved": is_solved(env.grid),
            "steps": sess["index"],
            "total": len(solution_path),
        },
        session_id=sid,
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("🚀 SokoFlow")
    print(f"   http://localhost:{port}")
    ensure_model_loaded()
    app.run(host="0.0.0.0", port=port, debug=False)
