import numpy as np

from eval.microban_levels import MICROBAN_LEVELS, MICROBAN_SOURCE
from eval.xsb import microban_to_8x8
from sokoban_data_gen import PLAYER, WALL, bfs_solve


def test_microban_source_is_documented():
    assert MICROBAN_SOURCE["author"] == "David W. Skinner"
    assert MICROBAN_SOURCE["full_set_size"] == 155
    assert MICROBAN_SOURCE["included"] == len(MICROBAN_LEVELS) == 38


def test_microban_1_fits_and_has_a_player():
    level = next(item for item in MICROBAN_LEVELS if item["id"] == "1")
    grid, targets = microban_to_8x8(level["xsb"])
    assert grid.shape == (8, 8)
    assert targets.shape == (8, 8)
    assert int(np.count_nonzero(grid == PLAYER)) == 1
    assert int(np.count_nonzero(grid == WALL)) >= 28
    path = bfs_solve(grid, targets, max_nodes=30000)
    assert path is not None
    assert len(path) >= 1
