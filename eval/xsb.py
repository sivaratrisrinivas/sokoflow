"""XSB → 8×8 grid conversion for the Microban OOD probe."""

from __future__ import annotations

from collections import deque

import numpy as np

from sokoban_data_gen import BOX, BOX_TARGET, FLOOR, PLAYER, TARGET, WALL

XSB_TO_CELL = {
    "#": WALL,
    " ": FLOOR,
    ".": TARGET,
    "@": PLAYER,
    "+": PLAYER,  # player on goal
    "$": BOX,
    "*": BOX_TARGET,
}
GOAL_CHARS = {".", "+", "*"}


def parse_xsb(rows: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Parse XSB rows into a tight grid. Spaces are floor until void-sealing."""
    height = len(rows)
    width = max(len(row) for row in rows)
    padded = [row.ljust(width, " ") for row in rows]
    grid = np.zeros((height, width), dtype=int)
    targets = np.zeros((height, width), dtype=bool)
    for r, row in enumerate(padded):
        for c, ch in enumerate(row):
            if ch not in XSB_TO_CELL:
                raise ValueError(f"unsupported XSB char {ch!r} at {(r, c)}")
            grid[r, c] = XSB_TO_CELL[ch]
            if ch in GOAL_CHARS:
                targets[r, c] = True
    return grid, targets


def seal_xsb_void(grid: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Turn unreached 'floor' (XSB void outside the walls) into walls."""
    players = np.argwhere(grid == PLAYER)
    if len(players) != 1:
        raise ValueError(f"expected one player, found {len(players)}")
    start = tuple(map(int, players[0]))
    h, w = grid.shape
    seen = {start}
    queue = deque([start])
    walkable = {FLOOR, PLAYER, BOX, TARGET, BOX_TARGET}
    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if (nr, nc) in seen:
                continue
            if int(grid[nr, nc]) not in walkable:
                continue
            seen.add((nr, nc))
            queue.append((nr, nc))
    sealed = grid.copy()
    sealed_targets = targets.copy()
    for r in range(h):
        for c in range(w):
            if (r, c) not in seen and int(sealed[r, c]) != WALL:
                sealed[r, c] = WALL
                sealed_targets[r, c] = False
    return sealed, sealed_targets


def pad_to_8x8(grid: np.ndarray, targets: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Center the level in an 8×8 wall frame. Raises if it does not fit."""
    h, w = grid.shape
    if h > 8 or w > 8:
        raise ValueError(f"level {h}x{w} does not fit in 8x8")
    out = np.full((8, 8), WALL, dtype=int)
    out_t = np.zeros((8, 8), dtype=bool)
    r0 = (8 - h) // 2
    c0 = (8 - w) // 2
    out[r0 : r0 + h, c0 : c0 + w] = grid
    out_t[r0 : r0 + h, c0 : c0 + w] = targets
    return out, out_t


def microban_to_8x8(rows: list[str]) -> tuple[np.ndarray, np.ndarray]:
    grid, targets = parse_xsb(rows)
    grid, targets = seal_xsb_void(grid, targets)
    return pad_to_8x8(grid, targets)
