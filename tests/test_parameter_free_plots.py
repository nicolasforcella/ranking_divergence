import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "examples" / "plot_parameter_free_baselines.py"
SPEC = importlib.util.spec_from_file_location("plot_parameter_free_baselines", SCRIPT_PATH)
assert SPEC is not None
plot_parameter_free_baselines = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(plot_parameter_free_baselines)


def test_display_names_cover_parameter_free_methods():
    assert set(plot_parameter_free_baselines.DISPLAY_NAMES) == {
        "top_k_iid_64",
        "mirror_5000",
        "periodic_400",
        "phrase_bank_5000",
    }
