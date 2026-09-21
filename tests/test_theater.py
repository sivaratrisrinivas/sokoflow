import numpy as np
import torch

from sokoban_clip import denoise_gif_bytes, raster_board
from sokoban_data_gen import SokobanGen, PLAYER, BOX, TARGET, FLOOR, WALL, bfs_solve
from sokoban_render import compose_stage, denoise_theater_html, twin_html
from sokoban_solve import (
    bfs_solve_report,
    format_autopsy,
    illegal_reason,
    replay_until_illegal,
)


def _one_push():
    env = SokobanGen(num_boxes=1)
    env.reset_solved()
    env.grid.fill(FLOOR)
    env.targets.fill(False)
    env.grid[0, :] = WALL
    env.grid[-1, :] = WALL
    env.grid[:, 0] = WALL
    env.grid[:, -1] = WALL
    env.player_pos = (3, 2)
    env.grid[3, 2] = PLAYER
    env.grid[3, 3] = BOX
    env.targets[3, 4] = True
    env.grid[3, 4] = TARGET
    return env


def test_illegal_reason_walk_and_stuck_push():
    env = _one_push()
    assert illegal_reason(env.grid, env.targets, (3, 2), "LEFT") is None
    assert illegal_reason(env.grid, env.targets, (3, 2), "UP") is None
    env.grid[1, 1] = PLAYER
    env.player_pos = (1, 1)
    assert illegal_reason(env.grid, env.targets, (1, 1), "UP") == "walk_into_wall"
    env.grid[1, 5] = PLAYER
    env.grid[1, 6] = BOX
    assert illegal_reason(env.grid, env.targets, (1, 5), "RIGHT") == "stuck_push_wall"


def test_replay_stops_at_first_illegal():
    env = _one_push()
    walked = replay_until_illegal(env.grid, env.targets, ["LEFT", "LEFT", "LEFT"])
    assert walked["first_illegal"]["action"] == "LEFT"
    assert walked["first_illegal"]["why"] == "walk_into_wall"
    assert walked["legal"] == ["LEFT"]


def test_autopsy_does_not_fake_success():
    text = format_autopsy(
        {
            "solved": False,
            "reason": "exhausted_iters",
            "max_iters": 20,
            "iters_used": 20,
            "path": ["UP", "RIGHT"],
            "boxes_off": 2,
            "skipped_illegal": 4,
            "first_illegal": {
                "index": 2,
                "action": "DOWN",
                "why_text": "stuck push into wall",
            },
        }
    )
    assert "Diffusion solved" not in text
    assert "stuck push into wall" in text
    assert "Exhausted 20 iterations" in text
    assert "2 boxes still off goal" in text


def test_bfs_report_one_push_and_budget():
    env = _one_push()
    report = bfs_solve_report(env.grid, env.targets, max_nodes=500)
    assert report["solved"] is True
    assert report["path"] == ["RIGHT"]
    assert report["nodes"] >= 1
    assert report["max_nodes"] == 500
    assert "solved" in report["status"]
    assert "nodes" in report["status"]
    assert bfs_solve(env.grid, env.targets, max_nodes=500) == [3]

    walled = _one_push()
    walled.grid[3, 4] = WALL
    walled.targets[3, 4] = False
    failed = bfs_solve_report(walled.grid, walled.targets, max_nodes=200)
    assert failed["solved"] is False
    assert failed["path"] == []
    assert "failed" in failed["status"]


def test_theater_and_twin_html():
    env = _one_push()
    frames = [
        {
            "t": 100,
            "actions": ["UP"] * 20,
            "grid": env.grid,
            "trail": [(3, 2)],
            "first_illegal": {"index": 0, "action": "UP", "why_text": "walk into wall"},
        },
        {
            "t": 0,
            "actions": ["RIGHT"] + ["UP"] * 19,
            "grid": env.grid,
            "trail": [(3, 2)],
            "first_illegal": {"index": 1, "action": "UP", "why_text": "walk into wall"},
        },
    ]
    theater = denoise_theater_html(frames, env.targets, interactive=True)
    assert "Denoise" in theater
    assert "name=\"denoise-t\"" in theater
    assert "noise" in theater
    assert "plan" in theater
    twin = twin_html(
        env.grid,
        env.grid,
        env.targets,
        diffusion_status="failed · 20 iters",
        bfs_status="solved · 1 moves · 8 nodes",
    )
    assert "Diffusion" in twin
    assert "BFS" in twin
    assert "failed · 20 iters" in twin
    assert "solved · 1 moves" in twin
    stage = compose_stage(
        mode="twin",
        grid=env.grid,
        targets=env.targets,
        theater_frames=frames,
        theater_interactive=True,
        diffusion_grid=env.grid,
        bfs_grid=env.grid,
        diffusion_status="failed",
        bfs_status="solved · 1 moves · 8 nodes",
        autopsy="First illegal move: UP at step 1 (walk into wall).",
    )
    assert "Denoise" in stage
    assert "autopsy" in stage
    assert "walk into wall" in stage


def test_denoise_gif_is_gif89a():
    env = _one_push()
    frames = [{"grid": env.grid, "trail": [(3, 2)]}, {"grid": env.grid, "trail": [(3, 2)]}]
    raw = denoise_gif_bytes(frames, env.targets)
    assert raw.startswith(b"GIF89a")
    assert raw.endswith(b";")
    img = raster_board(env.grid, env.targets)
    assert img.shape[0] == img.shape[1]
    assert img.max() <= 7


def test_sample_fast_trace_starts_from_noise():
    from sokoban_diffusion import SokobanDiffusion, state_to_tensor

    model = SokobanDiffusion(seq_len=20, timesteps=100, hidden_dim=128)
    model.eval()
    grid = np.zeros((8, 8), dtype=int)
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1
    grid[2, 2] = 2
    tensor = state_to_tensor(grid).unsqueeze(0)
    with torch.inference_mode():
        actions, traces = model.sample_fast_trace(tensor, steps=3)
    assert len(traces) == 4  # noise + 3 DDIM steps
    assert int(traces[0]["t"]) == 100
    assert tuple(actions.shape) == (1, 20)
    assert tuple(traces[-1]["actions"].shape) == (1, 20)
