"""Colorful 8×8 Sokoban board HTML. Pieces are chromatic; chrome stays elsewhere."""

from __future__ import annotations

import numpy as np

# Distinct piece colors — not grayscale. Chrome (cream/sky/charcoal) lives in CSS.
PIECE_COLORS = {
    "wall": "#C4A574",
    "floor": "#F7F1E6",
    "box": "#E39B2D",
    "goal": "#5BA8D4",
    "player": "#E25A45",
    "box_on_goal": "#3DAA6D",
}

CELL_CLASS = {
    0: "floor",
    1: "wall",
    2: "player",
    3: "box",
    4: "goal",
    5: "box-on-goal",
}


def board_css() -> str:
    c = PIECE_COLORS
    return f"""
.soko-board {{
  display: grid;
  grid-template-columns: repeat(8, 44px);
  grid-template-rows: repeat(8, 44px);
  gap: 5px;
  padding: 10px;
  width: max-content;
  margin: 0 auto;
  background: {c["floor"]};
  border-radius: 18px;
}}
.soko-cell {{
  width: 44px;
  height: 44px;
  border-radius: 10px;
  position: relative;
  display: flex;
  align-items: center;
  justify-content: center;
}}
.soko-cell.floor {{ background: {c["floor"]}; box-shadow: inset 0 0 0 1px rgba(196, 165, 116, 0.25); }}
.soko-cell.wall {{ background: {c["wall"]}; border-radius: 8px; box-shadow: inset 0 -3px 0 rgba(90, 64, 32, 0.18); }}
.soko-cell.goal {{ background: {c["floor"]}; }}
.soko-cell.goal::after {{
  content: "";
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: {c["goal"]};
}}
.soko-cell.box::after, .soko-cell.box-on-goal::after {{
  content: "";
  width: 30px;
  height: 30px;
  border-radius: 7px;
}}
.soko-cell.box::after {{ background: {c["box"]}; box-shadow: 0 3px 0 rgba(160, 90, 10, 0.28); }}
.soko-cell.box-on-goal {{ background: {c["floor"]}; }}
.soko-cell.box-on-goal::after {{ background: {c["box_on_goal"]}; box-shadow: 0 3px 0 rgba(20, 90, 50, 0.28); }}
.soko-cell.player {{ background: {c["floor"]}; }}
.soko-cell.player::after {{
  content: "";
  width: 26px;
  height: 26px;
  border-radius: 50%;
  background: {c["player"]};
  box-shadow: 0 3px 0 rgba(140, 40, 28, 0.25);
}}
.soko-cell.player.goal::before {{
  content: "";
  position: absolute;
  width: 16px;
  height: 16px;
  border-radius: 50%;
  background: {c["goal"]};
}}
@media (max-width: 520px) {{
  .soko-board {{
    grid-template-columns: repeat(8, 34px);
    grid-template-rows: repeat(8, 34px);
    gap: 4px;
    padding: 8px;
  }}
  .soko-cell {{ width: 34px; height: 34px; border-radius: 8px; }}
  .soko-cell.box::after, .soko-cell.box-on-goal::after {{ width: 22px; height: 22px; }}
  .soko-cell.player::after {{ width: 20px; height: 20px; }}
}}
""".strip()


def _cell_classes(value: int, on_goal: bool) -> str:
    cls = CELL_CLASS.get(int(value), "floor")
    if int(value) == 2 and on_goal:
        return "soko-cell player goal"
    return f"soko-cell {cls}"


def board_html(grid: np.ndarray, targets: np.ndarray | None = None) -> str:
    """Render an 8×8 grid as colorful HTML. Used by the Gradio Space."""
    cells: list[str] = []
    for r in range(8):
        for c in range(8):
            v = int(grid[r][c])
            on_goal = bool(targets[r][c]) if targets is not None else False
            cells.append(f'<div class="{_cell_classes(v, on_goal)}"></div>')
    inner = "".join(cells)
    return (
        f'<style>{board_css()}</style>'
        f'<div class="soko-board" aria-label="Sokoban board">{inner}</div>'
    )


def empty_board_html() -> str:
    grid = np.ones((8, 8), dtype=int)
    grid[1:7, 1:7] = 0
    return board_html(grid)
