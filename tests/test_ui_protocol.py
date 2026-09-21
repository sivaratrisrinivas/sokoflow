from pathlib import Path

from sokoban_render import PIECE_COLORS, board_html
import numpy as np

from eval.measure_solve_rate import OUTPUT_PATH, dated_output_path, parse_args, resolve_output_path

ROOT = Path(__file__).resolve().parent.parent


def test_piece_colors_are_not_grayscale():
    seen = set()
    chromatic = ("wall", "box", "goal", "player", "box_on_goal")
    for name, hex_color in PIECE_COLORS.items():
        assert hex_color not in seen
        seen.add(hex_color)
        r = int(hex_color[1:3], 16)
        g = int(hex_color[3:5], 16)
        b = int(hex_color[5:7], 16)
        if name in chromatic:
            assert max(r, g, b) - min(r, g, b) > 20, name
    assert PIECE_COLORS["floor"].lower() != "#ffffff"
    assert PIECE_COLORS["floor"].lower() != "#000000"


def test_board_html_emits_distinct_classes():
    grid = np.zeros((8, 8), dtype=int)
    grid[0, :] = 1
    grid[1, 1] = 2
    grid[1, 2] = 3
    grid[1, 3] = 4
    grid[1, 4] = 5
    html = board_html(grid)
    for token in ("wall", "floor", "player", "box", "goal", "box-on-goal"):
        assert token in html


def test_scramble_hard_default_write_is_dated():
    args = parse_args(["--protocol", "scramble-hard"])
    out = resolve_output_path(args, {"date": "2026-09-21"})
    assert out == dated_output_path("2026-09-21", "scramble-hard")
    assert out.resolve() != OUTPUT_PATH.resolve()


def test_gradio_core_objective_within_two_clicks():
    """Firstmate provisional bar: ≤2 clicks from load. Exact N may update later.

    Shipped path is 1: demo.load puts a puzzle on the board; Play runs
    denoise theater + BFS twin. No setup tabs. Scrub is HTML inside the
    stage after Play, not a Gradio setup slider.
    """
    text = (ROOT / "gradio_app" / "app.py").read_text(encoding="utf-8")
    assert 'gr.Button("Play"' in text
    assert "demo.load(load_default" in text
    assert "gr.Slider" not in text
    assert "gr.Tab" not in text
    assert text.count("gr.Button(") == 2  # Play + optional New; New is not required
    assert "denoise theater" in text.lower() or "Denoise" in text
    assert "bfs_solve_report" in text
    assert "diffusion_solve_report" in text
    flask = (ROOT / "templates" / "index.html").read_text(encoding="utf-8")
    assert 'id="btn">Play</button>' in flask
    assert "newGame();" in flask
    assert "playOnce()" in flask
    assert "onclick=\"playOnce()\"" in flask
    assert 'id="theater"' in flask
    assert 'id="twin"' in flask
    assert 'id="autopsy"' in flask
