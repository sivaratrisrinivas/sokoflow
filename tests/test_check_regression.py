import copy
import json
from pathlib import Path

from eval.check_regression import COMMITTED, compare


def test_committed_matches_itself():
    data = json.loads(Path(COMMITTED).read_text())
    assert compare(data, data) == []


def test_flip_is_reported():
    data = json.loads(Path(COMMITTED).read_text())
    fresh = copy.deepcopy(data)
    fresh["puzzles"][0]["diffusion_solved"] = not fresh["puzzles"][0]["diffusion_solved"]
    problems = compare(fresh, data)
    assert len(problems) == 1 and "diffusion_solved" in problems[0]
