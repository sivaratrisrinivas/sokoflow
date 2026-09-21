"""Colorful 8×8 Sokoban board HTML. Pieces are chromatic; chrome stays elsewhere."""

from __future__ import annotations

import html as html_lib

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

ARROWS = {"UP": "↑", "DOWN": "↓", "LEFT": "←", "RIGHT": "→"}


def board_css() -> str:
    c = PIECE_COLORS
    return f"""
.soko-board {{
  --cell: 32px;
  --gap: 4px;
  display: grid;
  grid-template-columns: repeat(8, var(--cell));
  grid-template-rows: repeat(8, var(--cell));
  gap: var(--gap);
  padding: 8px;
  width: max-content;
  margin: 0 auto;
  background: {c["floor"]};
  border-radius: 16px;
}}
.soko-board.solo {{
  --cell: 44px;
  --gap: 5px;
  padding: 10px;
  border-radius: 18px;
}}
.soko-cell {{
  width: var(--cell);
  height: var(--cell);
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
  width: 36%;
  height: 36%;
  border-radius: 50%;
  background: {c["goal"]};
}}
.soko-cell.box::after, .soko-cell.box-on-goal::after {{
  content: "";
  width: 68%;
  height: 68%;
  border-radius: 7px;
}}
.soko-cell.box::after {{ background: {c["box"]}; box-shadow: 0 3px 0 rgba(160, 90, 10, 0.28); }}
.soko-cell.box-on-goal {{ background: {c["floor"]}; }}
.soko-cell.box-on-goal::after {{ background: {c["box_on_goal"]}; box-shadow: 0 3px 0 rgba(20, 90, 50, 0.28); }}
.soko-cell.player {{ background: {c["floor"]}; }}
.soko-cell.player::after {{
  content: "";
  width: 58%;
  height: 58%;
  border-radius: 50%;
  background: {c["player"]};
  box-shadow: 0 3px 0 rgba(140, 40, 28, 0.25);
}}
.soko-cell.player.goal::before {{
  content: "";
  position: absolute;
  width: 36%;
  height: 36%;
  border-radius: 50%;
  background: {c["goal"]};
}}
.soko-cell.trail::before {{
  content: "";
  position: absolute;
  width: 8px;
  height: 8px;
  border-radius: 50%;
  background: #D7E6F0;
}}
.stage-kicker {{
  text-align: center;
  font-size: 11px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: #8A847A;
  margin: 0 0 8px;
}}
.t-label {{
  text-align: center;
  font-size: 12px;
  color: #8A847A;
  margin: 0 0 8px;
}}
.action-strip {{
  display: flex;
  flex-wrap: wrap;
  gap: 2px;
  justify-content: center;
  margin: 8px auto 0;
  max-width: 340px;
}}
.act {{
  width: 16px;
  height: 16px;
  line-height: 16px;
  text-align: center;
  font-size: 12px;
  color: #2B2926;
}}
.act.dead {{ color: #8A847A; opacity: 0.35; }}
.act.bad {{ color: #E25A45; text-decoration: line-through; }}
.theater {{
  margin: 0 auto 22px;
  max-width: 420px;
}}
.theater .t-radio {{
  position: absolute;
  opacity: 0;
  pointer-events: none;
}}
.theater .frames .denoise-frame {{ display: none; }}
.scrub {{
  display: flex;
  align-items: center;
  justify-content: center;
  gap: 8px;
  margin-top: 12px;
  font-size: 11px;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #8A847A;
}}
.ticks {{
  display: flex;
  gap: 5px;
  align-items: center;
}}
.ticks label {{
  width: 10px;
  height: 10px;
  border-radius: 50%;
  background: #D7E6F0;
  cursor: pointer;
  display: inline-block;
}}
.save-clip {{
  display: block;
  text-align: center;
  margin-top: 10px;
  font-size: 12px;
  color: #8A847A;
  text-decoration: none;
}}
.twin {{
  display: flex;
  flex-wrap: wrap;
  justify-content: center;
  gap: 28px 32px;
  width: 100%;
}}
.stage {{ width: 100%; }}
.twin-pane {{ text-align: center; }}
.pane-status {{
  margin: 8px 0 0;
  font-size: 13px;
  color: #8A847A;
}}
.autopsy {{
  max-width: 440px;
  margin: 18px auto 0;
  text-align: center;
  font-size: 13px;
  line-height: 1.55;
  color: #2B2926;
}}
@media (max-width: 520px) {{
  .soko-board.solo {{ --cell: 34px; --gap: 4px; }}
  .soko-board {{ --cell: 28px; --gap: 3px; }}
}}
""".strip()


def _cell_classes(value: int, on_goal: bool, trailed: bool) -> str:
    cls = CELL_CLASS.get(int(value), "floor")
    classes = ["soko-cell", cls]
    if int(value) == 2 and on_goal:
        classes = ["soko-cell", "player", "goal"]
    if trailed and int(value) in (0, 4):
        classes.append("trail")
    return " ".join(classes)


def board_html(
    grid: np.ndarray,
    targets: np.ndarray | None = None,
    *,
    trail=None,
    solo: bool = False,
    include_style: bool = True,
) -> str:
    """Render an 8×8 grid as colorful HTML. Used by the Gradio Space."""
    trail_set = {tuple(p) for p in (trail or [])}
    cells: list[str] = []
    for r in range(8):
        for c in range(8):
            v = int(grid[r][c])
            on_goal = bool(targets[r][c]) if targets is not None else False
            cells.append(f'<div class="{_cell_classes(v, on_goal, (r, c) in trail_set)}"></div>')
    inner = "".join(cells)
    size_cls = "soko-board solo" if solo else "soko-board"
    board = f'<div class="{size_cls}" aria-label="Sokoban board">{inner}</div>'
    if include_style:
        return f"<style>{board_css()}</style>{board}"
    return board


def empty_board_html() -> str:
    grid = np.ones((8, 8), dtype=int)
    grid[1:7, 1:7] = 0
    return board_html(grid, solo=True)


def _action_strip(actions: list[str], first_illegal: dict | None) -> str:
    cut = int(first_illegal["index"]) if first_illegal else len(actions)
    bits: list[str] = []
    for i, action in enumerate(actions):
        glyph = ARROWS.get(action, "?")
        if first_illegal and i == cut:
            cls = "act bad"
        elif i > cut:
            cls = "act dead"
        else:
            cls = "act"
        bits.append(f'<span class="{cls}">{glyph}</span>')
    return f'<div class="action-strip" aria-label="Proposed actions">{"".join(bits)}</div>'


def _frame_caption(frame: dict, index: int, n: int) -> str:
    if index == 0:
        return "noise"
    if index == n - 1:
        return "plan"
    return f"t={int(frame['t'])}"


def denoise_theater_html(
    frames: list[dict],
    targets,
    *,
    active_index: int | None = None,
    interactive: bool = False,
    clip_href: str | None = None,
    include_style: bool = True,
) -> str:
    if not frames:
        return ""
    n = len(frames)
    if active_index is None:
        active_index = n - 1
    active_index = max(0, min(n - 1, int(active_index)))

    radio_css = []
    radios = []
    labels = []
    frame_divs = []
    for i, frame in enumerate(frames):
        checked = " checked" if i == active_index else ""
        radio_css.append(
            f'.theater #dt-{i}:checked ~ .frames .denoise-frame[data-i="{i}"]{{display:block}}'
        )
        radio_css.append(
            f'.theater #dt-{i}:checked ~ .scrub label[for="dt-{i}"]{{background:#2B2926}}'
        )
        radios.append(
            f'<input class="t-radio" type="radio" name="denoise-t" id="dt-{i}"{checked}>'
        )
        labels.append(
            f'<label for="dt-{i}" title="{html_lib.escape(_frame_caption(frame, i, n))}"></label>'
        )
        caption = html_lib.escape(_frame_caption(frame, i, n))
        board = board_html(frame["grid"], targets, trail=frame.get("trail"), include_style=False)
        strip = _action_strip(frame["actions"], frame.get("first_illegal"))
        hidden = "" if (not interactive and i == active_index) else ' style="display:none"'
        if interactive:
            hidden = ""
        frame_divs.append(
            f'<div class="denoise-frame" data-i="{i}"{hidden}>'
            f'<p class="t-label">{caption}</p>{board}{strip}</div>'
        )

    if interactive:
        body = (
            "".join(radios)
            + '<div class="frames">'
            + "".join(frame_divs)
            + "</div>"
            + '<div class="scrub"><span>noise</span><div class="ticks">'
            + "".join(labels)
            + "</div><span>plan</span></div>"
        )
        extra_css = "".join(radio_css)
    else:
        body = '<div class="frames">' + frame_divs[active_index] + "</div>"
        extra_css = ""

    clip = ""
    if clip_href:
        clip = f'<a class="save-clip" href="{clip_href}" download="sokoflow-denoise.gif">Save denoise clip</a>'

    theater = (
        f'<div class="theater">'
        f'<p class="stage-kicker">Denoise</p>'
        f"{body}{clip}</div>"
    )
    css = board_css() + extra_css
    if include_style:
        return f"<style>{css}</style>{theater}"
    return f"<style>{extra_css}</style>{theater}" if extra_css else theater


def twin_html(
    diffusion_grid,
    bfs_grid,
    targets,
    *,
    diffusion_status: str,
    bfs_status: str,
    diffusion_trail=None,
    bfs_trail=None,
    include_style: bool = False,
) -> str:
    left = board_html(diffusion_grid, targets, trail=diffusion_trail, include_style=False)
    right = board_html(bfs_grid, targets, trail=bfs_trail, include_style=False)
    d_stat = html_lib.escape(diffusion_status)
    b_stat = html_lib.escape(bfs_status)
    markup = (
        '<div class="twin" style="display:flex !important;flex-wrap:wrap !important;'
        'justify-content:center !important;gap:28px 32px !important;width:100% !important">'
        '<div class="twin-pane" style="flex:0 0 auto !important;width:auto !important;text-align:center">'
        '<p class="stage-kicker">Diffusion</p>'
        f'{left}<p class="pane-status">{d_stat}</p></div>'
        '<div class="twin-pane" style="flex:0 0 auto !important;width:auto !important;text-align:center">'
        '<p class="stage-kicker">BFS</p>'
        f'{right}<p class="pane-status">{b_stat}</p></div>'
        "</div>"
    )
    if include_style:
        return f"<style>{board_css()}</style>{markup}"
    return markup


def autopsy_html(text: str) -> str:
    if not text:
        return ""
    return f'<p class="autopsy">{html_lib.escape(text)}</p>'


def compose_stage(
    *,
    mode: str,
    grid=None,
    targets=None,
    theater_frames=None,
    theater_index: int | None = None,
    theater_interactive: bool = False,
    clip_href: str | None = None,
    diffusion_grid=None,
    bfs_grid=None,
    diffusion_status: str = "",
    bfs_status: str = "",
    diffusion_trail=None,
    bfs_trail=None,
    autopsy: str = "",
) -> str:
    """One HTML stage: solo puzzle, denoise auto-play, or theater + BFS twin."""
    chunks = [
        f'<style>{board_css()}</style>'
        '<div class="stage" style="width:100% !important;max-width:720px;margin:0 auto">'
    ]
    if mode == "single":
        chunks.append(board_html(grid, targets, solo=True, include_style=False))
        chunks.append("</div>")
        return "".join(chunks)

    if theater_frames:
        chunks.append(
            denoise_theater_html(
                theater_frames,
                targets,
                active_index=theater_index,
                interactive=theater_interactive,
                clip_href=clip_href,
                include_style=False,
            )
        )

    if mode == "twin":
        chunks.append(
            twin_html(
                diffusion_grid if diffusion_grid is not None else grid,
                bfs_grid if bfs_grid is not None else grid,
                targets,
                diffusion_status=diffusion_status,
                bfs_status=bfs_status,
                diffusion_trail=diffusion_trail,
                bfs_trail=bfs_trail,
            )
        )
        if autopsy:
            chunks.append(autopsy_html(autopsy))

    chunks.append("</div>")
    return "".join(chunks)
