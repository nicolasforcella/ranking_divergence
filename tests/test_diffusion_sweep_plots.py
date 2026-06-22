import importlib.util
from pathlib import Path

import numpy as np


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "examples" / "plot_diffusion_sweeps.py"
SPEC = importlib.util.spec_from_file_location("plot_diffusion_sweeps", SCRIPT_PATH)
assert SPEC is not None
plot_diffusion_sweeps = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(plot_diffusion_sweeps)


def test_metric_grid_marks_missing_sweep_points():
    rows = [
        {"method": "candi", "nfe": 16, "temperature": 0.5, "gen_ppl": 10.0},
        {"method": "candi", "nfe": 16, "temperature": 1.0, "gen_ppl": 20.0},
        {"method": "candi", "nfe": 32, "temperature": 0.5, "gen_ppl": 30.0},
    ]
    nfes, temperatures, grid = plot_diffusion_sweeps.metric_grid(rows, "gen_ppl")
    assert nfes == [16, 32]
    assert temperatures == [0.5, 1.0]
    assert grid[0].tolist() == [10.0, 20.0]
    assert grid[1, 0] == 30.0
    assert np.isnan(grid[1, 1])
