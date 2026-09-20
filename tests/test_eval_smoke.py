import json
from pathlib import Path

from eval.measure_solve_rate import (
    CONFIGS,
    OUTPUT_PATH,
    dated_output_path,
    parse_args,
    resolve_output_path,
    run,
)

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
    published = json.loads(GS_T5.read_text(encoding="utf-8"))
    assert published["date"] == "2026-08-24"


def test_non_smoke_default_write_is_dated_not_historical():
    args = parse_args([])
    out = resolve_output_path(args, {"date": "2026-09-20"})
    assert out == dated_output_path("2026-09-20")
    assert out.resolve() != OUTPUT_PATH.resolve()
    assert out.name == "gs_t5_solve_rate-2026-09-20.json"


def test_force_writes_historical_path():
    args = parse_args(["--force"])
    out = resolve_output_path(args, {"date": "2026-09-20"})
    assert out.resolve() == OUTPUT_PATH.resolve()


def test_explicit_historical_output_still_requires_force():
    args = parse_args(["--output", str(OUTPUT_PATH)])
    out = resolve_output_path(args, {"date": "2026-09-20"})
    assert out == dated_output_path("2026-09-20")
