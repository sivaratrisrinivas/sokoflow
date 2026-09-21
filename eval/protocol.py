"""Train / eval protocol constants. GS-T47 scramble-hard is the headline protocol."""

from __future__ import annotations

from sokoban_data_gen import MIN_BOXES_OFF_TARGET, MIN_TRAJECTORY_LEN

# Re-export so eval code and tests can import a single protocol module.
__all__ = [
    "MIN_BOXES_OFF_TARGET",
    "MIN_TRAJECTORY_LEN",
    "SCRAMBLE_HARD_DEFINITION",
    "is_scramble_hard",
]

SCRAMBLE_HARD_DEFINITION = (
    "Reverse-scramble 8×8 boards using the same (boxes, scramble_steps) configs as "
    "training / GS-T5. A board is eligible only if (1) boxes_off_target >= 2 and "
    "(2) BFS either fails at max_nodes=30000 or returns a path of length >= 5. "
    "Those two gates drop the 1-box-off and short-path inflators that biased GS-T5. "
    "Gate (2) matches the training filter len(traj) >= 5. Do not drop diffusion "
    "failures. BFS failures that pass (1) stay in the set (they are hard, not trivial). "
    "Rejection-sample until N puzzles per config are eligible. Microban is a separate "
    "OOD table and is never mixed into this rate."
)


def is_scramble_hard(*, boxes_off_target: int, bfs_length: int | None) -> bool:
    """True when a generated board belongs on the scramble-hard table."""
    if boxes_off_target < MIN_BOXES_OFF_TARGET:
        return False
    if bfs_length is not None and bfs_length < MIN_TRAJECTORY_LEN:
        return False
    return True
