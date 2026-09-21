import numpy as np
import pytest

from app import _rate_hits, _sessions, app, is_solved, is_valid_move


@pytest.fixture()
def client():
    app.config["TESTING"] = True
    _rate_hits.clear()
    _sessions.clear()
    with app.test_client() as test_client:
        yield test_client


def _board():
    board = {
        "grid": np.zeros((8, 8), dtype=int).tolist(),
        "targets": np.zeros((8, 8), dtype=bool).tolist(),
    }
    board["grid"][0] = [1] * 8
    board["grid"][7] = [1] * 8
    for row in board["grid"]:
        row[0] = 1
        row[7] = 1
    board["grid"][3][2] = 2
    board["grid"][3][3] = 3
    board["targets"][3][4] = True
    board["grid"][3][4] = 4
    return board


def test_health_reports_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["status"] in {"ok", "degraded"}
    assert payload["model_loaded"] is True
    assert payload["eval"]["task"] == "GS-T47-scramble-hard"
    assert payload["eval"]["diffusion_solve_rate"] == 0.0625
    assert payload["eval"]["historical_gs_t5"]["diffusion_solve_rate"] == 0.208


def test_index_renders(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"SokoFlow" in response.data
    assert b"Play" in response.data
    assert b"6.2%" in response.data
    assert b"20.8%" in response.data
    assert b"id=\"theater\"" in response.data
    assert b"id=\"twin\"" in response.data
    assert b"id=\"autopsy\"" in response.data
    assert b"--wall:" in response.data or b"var(--wall)" in response.data
    assert b"#C4A574" in response.data
    assert b"#E39B2D" in response.data
    assert b"#5BA8D4" in response.data
    assert b"#E25A45" in response.data
    assert b"#3DAA6D" in response.data


def test_new_game_rejects_payload_over_size_limit(client):
    too_big = b"x" * (20 * 1024)
    response = client.post(
        "/api/new_game",
        data=too_big,
        content_type="application/json",
    )
    assert response.status_code == 413
    assert response.get_json()["error"] == "payload_too_large"


def test_solve_requires_8x8_board(client):
    response = client.post(
        "/api/solve",
        json={"grid": [[0]], "targets": [[False]]},
    )
    assert response.status_code == 400
    assert response.get_json()["error"] == "invalid_board"


def test_rate_limit_on_solve(client, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "2")
    monkeypatch.setattr("app.diffusion_solve_fast", lambda *args, **kwargs: ["RIGHT"])
    board = _board()
    first = client.post("/api/solve", json=board)
    second = client.post("/api/solve", json=board)
    third = client.post("/api/solve", json=board)
    assert first.status_code == 200
    assert second.status_code == 200
    assert third.status_code == 429
    assert third.get_json()["error"] == "rate_limited"


def test_solve_and_new_game_use_separate_buckets(client, monkeypatch):
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1")
    monkeypatch.setenv("NEW_GAME_RATE_LIMIT_PER_MINUTE", "5")
    monkeypatch.setattr("app.diffusion_solve_fast", lambda *args, **kwargs: ["RIGHT"])
    monkeypatch.setattr(
        "app.diffusion_solve_report",
        lambda *args, **kwargs: {
            "solved": True,
            "path": ["RIGHT"],
            "denoise_frames": [],
            "autopsy": "",
            "reason": "solved",
        },
    )
    monkeypatch.setattr(
        "app.bfs_solve_report",
        lambda *args, **kwargs: {
            "solved": True,
            "path": ["RIGHT"],
            "nodes": 2,
            "max_nodes": 30000,
            "budget_exhausted": False,
            "status": "solved · 1 moves · 2 nodes",
        },
    )
    monkeypatch.setattr("app.denoise_gif_data_uri", lambda *args, **kwargs: None)
    board = _board()
    assert client.post("/api/solve", json=board).status_code == 200
    assert client.post("/api/solve", json=board).status_code == 429
    demo = client.post("/api/new_game", json={"difficulty": 8})
    assert demo.status_code == 200
    payload = demo.get_json()
    assert payload["solvable"] is True
    assert payload["diffusion_solved"] is True
    assert payload["path"] == ["RIGHT"]
    assert payload["moves"] == 1
    assert payload["bfs"]["solved"] is True
    assert payload["denoise"] == []


def test_rate_limit_ignores_x_forwarded_for_without_trust_proxy(client, monkeypatch):
    monkeypatch.delenv("TRUST_PROXY", raising=False)
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1")
    monkeypatch.setattr("app.diffusion_solve_fast", lambda *args, **kwargs: ["RIGHT"])
    board = _board()
    first = client.post("/api/solve", json=board, headers={"X-Forwarded-For": "1.1.1.1"})
    second = client.post("/api/solve", json=board, headers={"X-Forwarded-For": "8.8.8.8"})
    assert first.status_code == 200
    assert second.status_code == 429


def test_rate_limit_uses_x_forwarded_for_when_trust_proxy_set(client, monkeypatch):
    monkeypatch.setenv("TRUST_PROXY", "1")
    monkeypatch.setenv("RATE_LIMIT_PER_MINUTE", "1")
    monkeypatch.setattr("app.diffusion_solve_fast", lambda *args, **kwargs: ["RIGHT"])
    board = _board()
    first = client.post("/api/solve", json=board, headers={"X-Forwarded-For": "1.1.1.1"})
    second = client.post("/api/solve", json=board, headers={"X-Forwarded-For": "8.8.8.8"})
    assert first.status_code == 200
    assert second.status_code == 200


def _fake_bfs(*, solved=True, path=None):
    path = path if path is not None else (["RIGHT"] if solved else [])
    return {
        "solved": solved,
        "path": path,
        "nodes": 2,
        "max_nodes": 30000,
        "budget_exhausted": False,
        "status": "solved · 1 moves · 2 nodes" if solved else "failed · no path · 2 nodes",
    }


def test_new_game_unsolved_returns_empty_path_keeps_denoise_and_flag(client, monkeypatch):
    prefix = ["RIGHT", "UP"]
    monkeypatch.setattr(
        "app.diffusion_solve_report",
        lambda *args, **kwargs: {
            "solved": False,
            "path": prefix,
            "denoise_frames": [
                {"t": 100, "actions": [], "legal_n": 0, "first_illegal": None},
                {"t": 0, "actions": prefix, "legal_n": 2, "first_illegal": None},
            ],
            "autopsy": "Legal prefix: 2 move(s).",
            "reason": "exhausted_iters",
        },
    )
    monkeypatch.setattr("app.bfs_solve_report", lambda *args, **kwargs: _fake_bfs())
    monkeypatch.setattr("app.denoise_gif_data_uri", lambda *args, **kwargs: None)
    response = client.post("/api/new_game", json={"difficulty": 8})
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["solvable"] is False
    assert payload["diffusion_solved"] is False
    assert payload["path"] == []
    assert payload["moves"] == 0
    assert payload["denoise"][-1]["actions"] == prefix
    assert payload["denoise"][-1]["legal_n"] == 2
    assert "Legal prefix" in payload["autopsy"]

    step = client.post("/api/solve_step")
    assert step.status_code == 200
    body = step.get_json()
    assert body["gave_up"] is True
    assert body["solved"] is False
    assert body["action"] == "GIVE_UP"


def test_new_game_solved_still_returns_executed_path(client, monkeypatch):
    path = ["RIGHT"]
    monkeypatch.setattr(
        "app.diffusion_solve_report",
        lambda *args, **kwargs: {
            "solved": True,
            "path": path,
            "denoise_frames": [
                {"t": 100, "actions": [], "legal_n": 0, "first_illegal": None},
                {"t": 0, "actions": path, "legal_n": 1, "first_illegal": None},
            ],
            "autopsy": "",
            "reason": "solved",
        },
    )
    monkeypatch.setattr("app.bfs_solve_report", lambda *args, **kwargs: _fake_bfs(path=path))
    monkeypatch.setattr("app.denoise_gif_data_uri", lambda *args, **kwargs: None)
    response = client.post("/api/new_game", json={"difficulty": 8})
    payload = response.get_json()
    assert payload["solvable"] is True
    assert payload["diffusion_solved"] is True
    assert payload["path"] == path
    assert payload["moves"] == 1
    assert payload["denoise"][-1]["actions"] == path


def test_solve_unsolved_returns_null_path(client, monkeypatch):
    monkeypatch.setattr("app.diffusion_solve_fast", lambda *args, **kwargs: None)
    response = client.post("/api/solve", json=_board())
    assert response.status_code == 200
    payload = response.get_json()
    assert payload["path"] is None
    assert payload["solved"] is False
    assert payload["moves"] == 0


def test_is_solved_and_valid_move_helpers():
    grid = np.zeros((8, 8), dtype=int)
    targets = np.zeros((8, 8), dtype=bool)
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1
    grid[3, 2] = 2
    grid[3, 3] = 3
    targets[3, 4] = True
    grid[3, 4] = 4
    assert is_solved(grid) is False
    nxt, pos = is_valid_move(grid, targets, (3, 2), "RIGHT")
    assert pos == (3, 3)
    assert is_solved(nxt) is True
