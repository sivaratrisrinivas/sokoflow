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
    board = _board()
    assert client.post("/api/solve", json=board).status_code == 200
    assert client.post("/api/solve", json=board).status_code == 429
    demo = client.post("/api/new_game", json={"difficulty": 8})
    assert demo.status_code == 200
    assert demo.get_json()["solvable"] is True


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
