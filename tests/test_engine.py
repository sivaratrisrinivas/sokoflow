import numpy as np
import pytest

from sokoban_data_gen import ACTIONS, SokobanGen, bfs_solve, is_valid_move
from sokoban_engine import BOX, BOX_TARGET, FLOOR, PLAYER, TARGET, WALL, SokobanEnv


def _walled_board(num_boxes=1):
    env = SokobanEnv(width=8, height=8, num_boxes=num_boxes)
    env.grid.fill(FLOOR)
    env.targets.fill(False)
    env.grid[0, :] = WALL
    env.grid[-1, :] = WALL
    env.grid[:, 0] = WALL
    env.grid[:, -1] = WALL
    return env


def test_reset_solved_places_boxes_on_targets():
    env = SokobanEnv(num_boxes=3)
    env.reset_solved()
    assert int(np.count_nonzero(env.grid == WALL)) >= 28
    assert int(np.count_nonzero(env.grid == PLAYER)) == 1
    assert int(np.count_nonzero(env.grid == BOX_TARGET)) == 3
    assert int(np.count_nonzero(env.grid == BOX)) == 0
    assert int(np.count_nonzero(env.targets)) == 3


def test_walls_block_player():
    env = _walled_board()
    env.player_pos = (1, 1)
    env.grid[1, 1] = PLAYER
    before = env.grid.copy()
    env.step("UP")
    assert env.player_pos == (1, 1)
    assert np.array_equal(env.grid, before)


def test_player_moves_onto_floor():
    env = _walled_board()
    env.player_pos = (2, 2)
    env.grid[2, 2] = PLAYER
    env.step("RIGHT")
    assert env.player_pos == (2, 3)
    assert env.grid[2, 2] == FLOOR
    assert env.grid[2, 3] == PLAYER


def test_player_pushes_box_onto_target():
    env = _walled_board()
    env.player_pos = (3, 2)
    env.grid[3, 2] = PLAYER
    env.grid[3, 3] = BOX
    env.targets[3, 4] = True
    env.grid[3, 4] = TARGET
    env.step("RIGHT")
    assert env.player_pos == (3, 3)
    assert env.grid[3, 3] == PLAYER
    assert env.grid[3, 4] == BOX_TARGET
    assert int(np.count_nonzero(env.grid == BOX)) == 0


def test_cannot_push_box_into_wall():
    env = _walled_board()
    env.player_pos = (1, 5)
    env.grid[1, 5] = PLAYER
    env.grid[1, 6] = BOX
    env.step("RIGHT")
    assert env.player_pos == (1, 5)
    assert env.grid[1, 6] == BOX


def test_cannot_push_two_boxes():
    env = _walled_board()
    env.player_pos = (4, 2)
    env.grid[4, 2] = PLAYER
    env.grid[4, 3] = BOX
    env.grid[4, 4] = BOX
    env.step("RIGHT")
    assert env.player_pos == (4, 2)
    assert env.grid[4, 3] == BOX
    assert env.grid[4, 4] == BOX


def test_leaving_target_restores_target_tile():
    env = _walled_board()
    env.player_pos = (2, 2)
    env.targets[2, 2] = True
    env.grid[2, 2] = PLAYER
    env.step("DOWN")
    assert env.grid[2, 2] == TARGET
    assert env.grid[3, 2] == PLAYER


def test_is_valid_move_accepts_walk_and_rejects_wall():
    env = _walled_board()
    env.player_pos = (2, 2)
    env.grid[2, 2] = PLAYER
    walked, pos = is_valid_move(env.grid, env.targets, env.player_pos, 3)  # RIGHT
    assert walked is not None
    assert pos == (2, 3)
    blocked, blocked_pos = is_valid_move(env.grid, env.targets, (1, 1), 0)  # UP into wall
    assert blocked is None
    assert blocked_pos is None


def test_is_valid_move_push_matches_engine():
    env = _walled_board()
    env.player_pos = (5, 2)
    env.grid[5, 2] = PLAYER
    env.grid[5, 3] = BOX
    new_grid, new_pos = is_valid_move(env.grid, env.targets, env.player_pos, 3)
    env.step("RIGHT")
    assert new_pos == env.player_pos
    assert np.array_equal(new_grid, env.grid)


def test_unknown_action_raises():
    env = SokobanEnv(num_boxes=1)
    with pytest.raises(KeyError):
        env.step("TELEPORT")


def test_action_indices_are_four_cardinal_moves():
    assert set(ACTIONS) == {0, 1, 2, 3}
    assert ACTIONS[0] == (-1, 0)
    assert ACTIONS[1] == (1, 0)
    assert ACTIONS[2] == (0, -1)
    assert ACTIONS[3] == (0, 1)


def test_bfs_solves_one_push():
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
    path = bfs_solve(env.grid, env.targets, max_nodes=500)
    assert path == [3]
