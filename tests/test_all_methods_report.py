import importlib.util
import sys
from pathlib import Path


EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
sys.path.insert(0, str(EXAMPLES_DIR))
SCRIPT_PATH = EXAMPLES_DIR / "build_all_methods_report.py"
SPEC = importlib.util.spec_from_file_location("build_all_methods_report", SCRIPT_PATH)
assert SPEC is not None
build_all_methods_report = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(build_all_methods_report)


def test_select_best_rank_rows_keeps_all_metrics_from_winner():
    rows = [
        {"method": "duo", "nfe": 8, "rank_wasserstein": 0.2, "gen_ppl": 10},
        {"method": "duo", "nfe": 8, "rank_wasserstein": 0.1, "gen_ppl": 20},
    ]
    selected = build_all_methods_report.select_best_rank_rows(rows)
    assert selected == [rows[1]]
