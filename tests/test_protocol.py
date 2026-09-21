from eval.protocol import is_scramble_hard
from sokoban_data_gen import MIN_BOXES_OFF_TARGET, MIN_TRAJECTORY_LEN


def test_training_length_gate_is_five():
    assert MIN_TRAJECTORY_LEN == 5
    assert MIN_BOXES_OFF_TARGET == 2


def test_scramble_hard_drops_one_box_off():
    assert is_scramble_hard(boxes_off_target=1, bfs_length=12) is False
    assert is_scramble_hard(boxes_off_target=1, bfs_length=None) is False


def test_scramble_hard_drops_short_paths():
    assert is_scramble_hard(boxes_off_target=2, bfs_length=4) is False
    assert is_scramble_hard(boxes_off_target=3, bfs_length=0) is False


def test_scramble_hard_keeps_long_or_unsolved():
    assert is_scramble_hard(boxes_off_target=2, bfs_length=5) is True
    assert is_scramble_hard(boxes_off_target=3, bfs_length=20) is True
    assert is_scramble_hard(boxes_off_target=2, bfs_length=None) is True
