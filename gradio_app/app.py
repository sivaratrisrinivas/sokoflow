"""
SokoFlow — Hugging Face Gradio Space (CPU).

Source of truth for https://huggingface.co/spaces/Srini410/sokoflow

Captain lock: a default puzzle is on the board at load. One Play click
runs diffusion on that board. No setup tabs, no second confirm.
"""

from __future__ import annotations

import os
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
for path in (ROOT, PARENT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

for candidate in (ROOT / "sokoban_diffusion.pth", PARENT / "sokoban_diffusion.pth"):
    if candidate.exists():
        os.environ.setdefault("SOKOFLOW_MODEL_PATH", str(candidate))
        break

import gradio as gr

from sokoban_engine import SokobanEnv
from sokoban_render import board_html, empty_board_html
from sokoban_solve import diffusion_solve_fast, ensure_model_loaded, is_solved, is_valid_move, model_error

try:
    import spaces
except ImportError:  # local Flask-style CPU run without ZeroGPU
    class spaces:  # type: ignore[no-redef]
        @staticmethod
        def GPU(*_args, **_kwargs):
            def deco(fn):
                return fn

            return deco


DEFAULT_SEED = 47
DEFAULT_BOXES = 3
DEFAULT_SCRAMBLE = 25

CSS = """
@import url("https://fonts.googleapis.com/css2?family=Instrument+Sans:wght@400;500;600&display=swap");

:root {
  --cream: #F3EEE4;
  --paper: #FAF7F1;
  --ink: #2B2926;
  --muted: #8A847A;
  --sky: #D7E6F0;
}

html, body, .gradio-container, .main, .contain {
  background: var(--cream) !important;
  font-family: "Instrument Sans", ui-sans-serif, system-ui, sans-serif !important;
  color: var(--ink) !important;
}

.gradio-container {
  max-width: 640px !important;
  margin: 0 auto !important;
  padding: 64px 20px 80px !important;
  min-height: 100vh;
}

footer, .footer, .built-with,
.icon-button, button.sm.secondary,
#settings-toolbar, .settings-toolbar,
.show-api, .api-docs, .gradio-info {
  display: none !important;
}

#wordmark {
  text-align: center;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  font-size: 13px;
  font-weight: 500;
  color: var(--muted);
  margin: 0 0 28px;
}

#board-wrap {
  background: transparent !important;
  border: none !important;
  box-shadow: none !important;
  padding: 0 !important;
}
#board-wrap .prose, #board-wrap .html-container {
  display: flex;
  justify-content: center;
}

#status p, #status {
  text-align: center;
  color: var(--muted) !important;
  font-size: 14px !important;
  min-height: 1.4em;
}

#play {
  display: block;
  width: 168px !important;
  margin: 8px auto 0 !important;
  background: var(--ink) !important;
  color: var(--paper) !important;
  border: none !important;
  border-radius: 999px !important;
  font-size: 15px !important;
  font-weight: 500 !important;
  letter-spacing: 0.02em;
  padding: 14px 0 !important;
  box-shadow: 0 8px 20px rgba(43, 41, 38, 0.12) !important;
}

#quiet-new {
  display: block;
  margin: 10px auto 0 !important;
  background: transparent !important;
  color: var(--muted) !important;
  border: none !important;
  box-shadow: none !important;
  font-size: 13px !important;
  width: auto !important;
}

#honest {
  text-align: center;
  color: var(--muted);
  font-size: 12px;
  line-height: 1.55;
  max-width: 340px;
  margin: 28px auto 0;
}

.block, .label-wrap, .empty { border: none !important; box-shadow: none !important; }
.prose { background: transparent !important; }
"""

THEME = gr.themes.Soft(
    font=[gr.themes.GoogleFont("Instrument Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
    primary_hue="slate",
    neutral_hue="stone",
).set(
    body_background_fill="#F3EEE4",
    body_background_fill_dark="#F3EEE4",
    background_fill_primary="#F3EEE4",
    background_fill_secondary="#F3EEE4",
    block_background_fill="#F3EEE4",
    block_border_width="0px",
    block_shadow="none",
    button_primary_background_fill="#2B2926",
    button_primary_text_color="#FAF7F1",
)


def empty_state() -> dict[str, Any]:
    return {"grid": None, "targets": None, "start_grid": None}


def scramble_puzzle(seed: int | None = None) -> dict[str, Any]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    env = SokobanEnv(num_boxes=DEFAULT_BOXES)
    env.reset_solved()
    for _ in range(DEFAULT_SCRAMBLE):
        env.step_reverse()
    grid = env.grid.copy()
    targets = env.targets.copy()
    return {"grid": grid, "targets": targets, "start_grid": grid.copy()}


def render_state(state: dict[str, Any]) -> str:
    if state.get("grid") is None:
        return empty_board_html()
    return board_html(state["grid"], state["targets"])


@spaces.GPU(duration=120)
def _solve(grid, targets):
    return diffusion_solve_fast(grid, targets, max_iters=20)


def load_default():
    ensure_model_loaded()
    state = scramble_puzzle(seed=DEFAULT_SEED)
    return render_state(state), "A puzzle is ready.", state


def new_puzzle(_state):
    state = scramble_puzzle(seed=None)
    return render_state(state), "A puzzle is ready.", state


def play(state: dict[str, Any]):
    if not state or state.get("grid") is None:
        state = scramble_puzzle(seed=DEFAULT_SEED)
        yield render_state(state), "A puzzle is ready.", state
        return

    grid = np.array(state["start_grid"] if state.get("start_grid") is not None else state["grid"], dtype=int)
    targets = np.array(state["targets"], dtype=bool)
    state["grid"] = grid.copy()
    yield render_state(state), "Running diffusion…", state

    if not ensure_model_loaded():
        yield render_state(state), f"Model not loaded: {model_error()}", state
        return

    path = _solve(grid, targets)
    if not path:
        yield render_state(state), "No path this time — expected on scramble-hard boards.", state
        return

    current = grid.copy()
    for i, action in enumerate(path, start=1):
        pos = tuple(map(int, np.argwhere(current == 2)[0]))
        nxt, _ = is_valid_move(current, targets, pos, action)
        if nxt is None:
            state["grid"] = current
            yield render_state(state), f"Stopped at {i - 1}/{len(path)}", state
            return
        current = nxt
        state["grid"] = current
        yield render_state(state), f"{i} / {len(path)}", state
        time.sleep(0.11)

    note = "Solved" if is_solved(current) else "Finished"
    yield render_state(state), note, state


with gr.Blocks(title="SokoFlow", theme=THEME, css=CSS, analytics_enabled=False) as demo:
    gr.HTML('<p id="wordmark">SokoFlow</p>')
    state = gr.State(empty_state())
    board = gr.HTML(empty_board_html(), elem_id="board-wrap")
    status = gr.Markdown("A puzzle is ready.", elem_id="status")
    btn_play = gr.Button("Play", variant="primary", elem_id="play")
    btn_new = gr.Button("New puzzle", elem_id="quiet-new")
    gr.HTML(
        '<p id="honest">One click runs diffusion on the board above. '
        "Scramble-hard: 6.2% (15/240) vs BFS 94.2%. GS-T5 20.8% is historical.</p>"
    )

    demo.load(load_default, inputs=None, outputs=[board, status, state])
    btn_play.click(play, inputs=[state], outputs=[board, status, state])
    btn_new.click(new_puzzle, inputs=[state], outputs=[board, status, state])


if __name__ == "__main__":
    demo.launch()
