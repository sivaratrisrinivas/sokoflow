import json
from pathlib import Path

from eval.measure_solve_rate import CONFIGS, run

GS_T5 = Path(__file__).resolve().parent.parent / "eval" / "gs_t5_solve_rate.json"


def test_gs_t5_historical_table_is_the_published_measurement():
    data = json.loads(GS_T5.read_text(encoding="utf-8"))
    overall = data["by_difficulty"]["overall"]
    assert data["task"] == "GS-T5"
    assert data["date"] == "2026-08-24"
    assert data["dataset_size"] == 240
    assert overall["diffusion_solved"] == 50
    assert overall["bfs_solved"] == 229
    assert abs(overall["diffusion_solve_rate"] - 0.20833333333333334) < 1e-12
    assert abs(overall["bfs_solve_rate"] - 0.9541666666666667) < 1e-12


def test_eval_smoke_one_easy_puzzle():
    results = run(n_per_config=1, configs=CONFIGS[:1], seed=0)
    assert results["dataset_size"] == 1
    puzzle = results["puzzles"][0]
    assert puzzle["difficulty"] == "easy"
    assert "diffusion_solved" in puzzle
    assert "bfs_solved" in puzzle
    assert results["by_difficulty"]["overall"]["n"] == 1
    # Smoke must not clobber the published GS-T5 file.
    published = json.loads(GS_T5.read_text(encoding="utf-8"))
    assert published["date"] == "2026-08-24"
