import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "examples" / "plot_efficiency_pareto.py"
SPEC = importlib.util.spec_from_file_location("plot_efficiency_pareto", SCRIPT_PATH)
assert SPEC is not None
plot_efficiency_pareto = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(plot_efficiency_pareto)


def test_best_by_method_nfe_selects_temperature_with_lowest_distance():
    rows = [
        {"method": "duo", "nfe": 16, "temperature": 0.6, "rank_wasserstein": 0.2},
        {"method": "duo", "nfe": 16, "temperature": 0.7, "rank_wasserstein": 0.1},
    ]
    best = plot_efficiency_pareto.best_by_method_nfe(rows)
    assert len(best) == 1
    assert best[0]["temperature"] == 0.7


def test_pareto_frontier_minimizes_nfe_and_distance():
    points = [
        {"method": "a", "nfe": 8, "rank_wasserstein": 0.2},
        {"method": "b", "nfe": 8, "rank_wasserstein": 0.3},
        {"method": "a", "nfe": 16, "rank_wasserstein": 0.1},
        {"method": "ar", "nfe": 128, "rank_wasserstein": 0.15},
    ]
    frontier = plot_efficiency_pareto.pareto_frontier(points)
    assert [(point["nfe"], point["rank_wasserstein"]) for point in frontier] == [
        (8, 0.2),
        (16, 0.1),
    ]


def test_parameter_free_point_can_start_frontier_at_zero_nfe():
    points = [
        {"method": "periodic", "nfe": 0, "rank_wasserstein": 1.0},
        {"method": "duo", "nfe": 8, "rank_wasserstein": 0.1},
        {"method": "duo", "nfe": 16, "rank_wasserstein": 0.2},
    ]
    frontier = plot_efficiency_pareto.pareto_frontier(points)
    assert [(point["nfe"], point["rank_wasserstein"]) for point in frontier] == [
        (0, 1.0),
        (8, 0.1),
    ]
