"""
SokoFlow — Hugging Face Gradio Space (CPU).

Source of truth for https://huggingface.co/spaces/Srini410/sokoflow

UX bar (Firstmate, provisional): core objective in ≤2 clicks from load.
Exact N may update later. Shipped path is 1 click: a default puzzle is
already on the board; Play runs denoise theater + BFS twin. No setup tabs,
no second confirm.
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

from sokoban_clip import denoise_gif_data_uri
from sokoban_engine import SokobanEnv
from sokoban_render import compose_stage, empty_board_html
from sokoban_solve import (
    bfs_solve_report,
    diffusion_solve_report,
    ensure_model_loaded,
    model_error,
    walk_path_frames,
)

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
  max-width: 760px !important;
  margin: 0 auto !important;
  padding: 64px 20px 80px !important;
  min-height: 100vh;
}

.gradio-container .block {
  width: 100% !important;
  max-width: 100% !important;
}
.gradio-container .html-container {
  width: 100% !important;
  max-width: 100% !important;
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
  width: 100% !important;
  max-width: 100% !important;
}
#board-wrap .prose, #board-wrap .html-container, #board-wrap .prose * {
  max-width: none !important;
}
#board-wrap .prose, #board-wrap .html-container {
  display: block !important;
  width: 100% !important;
}
#board-wrap .stage, #board-wrap .theater, #board-wrap .twin {
  width: 100% !important;
}
#board-wrap .twin {
  display: flex !important;
  flex-wrap: wrap !important;
  justify-content: center !important;
  gap: 28px 32px !important;
}
#board-wrap .twin-pane {
  width: auto !important;
  flex: 0 0 auto !important;
  display: block !important;
}
#board-wrap .soko-board {
  width: max-content !important;
}

#status-line, #status p, #status {
  text-align: center;
  color: var(--muted) !important;
  font-size: 14px !important;
  min-height: 1.4em;
  border: none !important;
  box-shadow: none !important;
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
  max-width: 400px;
  margin: 28px auto 0;
}

.save-clip {
  color: #8A847A !important;
  text-decoration: none !important;
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


def render_solo(state: dict[str, Any]) -> str:
    if state.get("grid") is None:
        return empty_board_html()
    return compose_stage(mode="single", grid=state["grid"], targets=state["targets"])


@spaces.GPU(duration=120)
def _compute(grid, targets):
    report = diffusion_solve_report(grid, targets, max_iters=20, trace=True)
    bfs = bfs_solve_report(grid, targets, max_nodes=30000)
    return report, bfs


def status_html(text: str) -> str:
    return f'<p id="status-line">{text}</p>'


def headline(report: dict, bfs: dict) -> str:
    if report["solved"] and bfs["solved"]:
        return "Solved"
    if bfs["solved"] and not report["solved"]:
        return "BFS solved — diffusion did not"
    if report["solved"] and not bfs["solved"]:
        return "Diffusion solved — BFS did not"
    return "Neither solved"


def load_default():
    ensure_model_loaded()
    state = scramble_puzzle(seed=DEFAULT_SEED)
    return render_solo(state), status_html("A puzzle is ready."), state


def new_puzzle(_state):
    state = scramble_puzzle(seed=None)
    return render_solo(state), status_html("A puzzle is ready."), state


def play(state: dict[str, Any]):
    if not state or state.get("grid") is None:
        state = scramble_puzzle(seed=DEFAULT_SEED)
        yield render_solo(state), status_html("A puzzle is ready."), state
        return

    grid = np.array(state["start_grid"] if state.get("start_grid") is not None else state["grid"], dtype=int)
    targets = np.array(state["targets"], dtype=bool)
    state["grid"] = grid.copy()
    yield render_solo(state), status_html("Running diffusion…"), state

    if not ensure_model_loaded():
        yield render_solo(state), status_html(f"Model not loaded: {model_error()}"), state
        return

    report, bfs = _compute(grid, targets)
    frames = report.get("denoise_frames") or []
    clip = denoise_gif_data_uri(frames, targets) if frames else None

    for i in range(len(frames)):
        yield (
            compose_stage(
                mode="denoise",
                grid=grid,
                targets=targets,
                theater_frames=frames,
                theater_index=i,
                theater_interactive=False,
            ),
            status_html("Denoising…"),
            state,
        )
        time.sleep(0.11)

    d_play = walk_path_frames(grid, targets, report.get("path") or [])
    b_play = walk_path_frames(grid, targets, bfs.get("path") or [])
    n = max(len(d_play), len(b_play), 1)
    d_end = "solved · {} moves".format(len(report.get("path") or [])) if report["solved"] else "failed"
    if not report["solved"] and report.get("iters_used"):
        d_end = f"failed · {report['iters_used']} iters"

    for i in range(n):
        di = min(i, len(d_play) - 1)
        bi = min(i, len(b_play) - 1)
        d_status = d_end if di == len(d_play) - 1 else f"{di} / {max(len(d_play) - 1, 1)}"
        b_status = bfs["status"] if bi == len(b_play) - 1 else f"{bi} / {max(len(b_play) - 1, 1)}"
        yield (
            compose_stage(
                mode="twin",
                grid=grid,
                targets=targets,
                theater_frames=frames,
                theater_index=len(frames) - 1 if frames else None,
                theater_interactive=False,
                diffusion_grid=d_play[di]["grid"],
                bfs_grid=b_play[bi]["grid"],
                diffusion_status=d_status,
                bfs_status=b_status,
                diffusion_trail=d_play[di]["trail"],
                bfs_trail=b_play[bi]["trail"],
            ),
            status_html("Comparing…"),
            state,
        )
        time.sleep(0.09)

    state["grid"] = d_play[-1]["grid"]
    autopsy = "" if report["solved"] else (report.get("autopsy") or "")
    yield (
        compose_stage(
            mode="twin",
            grid=grid,
            targets=targets,
            theater_frames=frames,
            theater_index=len(frames) - 1 if frames else None,
            theater_interactive=bool(frames),
            clip_href=clip,
            diffusion_grid=d_play[-1]["grid"],
            bfs_grid=b_play[-1]["grid"],
            diffusion_status=d_end,
            bfs_status=bfs["status"],
            diffusion_trail=d_play[-1]["trail"],
            bfs_trail=b_play[-1]["trail"],
            autopsy=autopsy,
        ),
        status_html(headline(report, bfs)),
        state,
    )


with gr.Blocks(title="SokoFlow", theme=THEME, css=CSS, analytics_enabled=False) as demo:
    gr.HTML('<p id="wordmark">SokoFlow</p>')
    state = gr.State(empty_state())
    board = gr.HTML(empty_board_html(), elem_id="board-wrap")
    status = gr.HTML('<p id="status-line">A puzzle is ready.</p>', elem_id="status")
    btn_play = gr.Button("Play", variant="primary", elem_id="play")
    btn_new = gr.Button("New puzzle", elem_id="quiet-new")
    gr.HTML(
        '<p id="honest">One click runs denoise theater + a BFS twin on the board above. '
        "Scramble-hard: 6.2% (15/240) vs BFS 94.2%. GS-T5 20.8% is historical.</p>"
    )

    demo.load(load_default, inputs=None, outputs=[board, status, state])
    btn_play.click(play, inputs=[state], outputs=[board, status, state])
    btn_new.click(new_puzzle, inputs=[state], outputs=[board, status, state])


if __name__ == "__main__":
    demo.launch()
