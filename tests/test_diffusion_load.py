from pathlib import Path

import numpy as np
import torch

from sokoban_diffusion import SokobanDiffusion, state_to_tensor

WEIGHTS = Path(__file__).resolve().parent.parent / "sokoban_diffusion.pth"


def test_committed_weights_exist():
    assert WEIGHTS.is_file()
    assert WEIGHTS.stat().st_size > 100_000


def test_load_committed_weights_and_sample():
    model = SokobanDiffusion(seq_len=20, timesteps=100, hidden_dim=128)
    state = torch.load(WEIGHTS, map_location="cpu", weights_only=True)
    missing, unexpected = model.load_state_dict(state, strict=True)
    assert not missing
    assert not unexpected
    model.eval()

    grid = np.zeros((8, 8), dtype=int)
    grid[0, :] = 1
    grid[-1, :] = 1
    grid[:, 0] = 1
    grid[:, -1] = 1
    grid[2, 2] = 2
    grid[2, 3] = 3
    grid[2, 4] = 4
    tensor = state_to_tensor(grid).unsqueeze(0)

    with torch.inference_mode():
        actions = model.sample_fast(tensor, steps=2)

    assert tuple(actions.shape) == (1, 20)
    assert int(actions.min()) >= 0
    assert int(actions.max()) <= 3
