import importlib.util
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "examples" / "openwebtext_analysis.py"
SPEC = importlib.util.spec_from_file_location("openwebtext_analysis", SCRIPT_PATH)
assert SPEC is not None
openwebtext_analysis = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(openwebtext_analysis)


def test_openwebtext_analysis_cli_defaults_to_plan_values():
    args = openwebtext_analysis.parse_args([])
    assert args.sample_length == 1024
    assert args.scorer_model == "gpt2-large"
    assert args.generator_model == "gpt2"
    assert args.top_k == 64
    assert args.mirror_k == 5000
    assert args.periodic_k == 400
    assert args.phrase_bank_m == 5000
    assert args.phrase_bank_n == 5


def test_openwebtext_analysis_output_path_construction():
    run_dir = openwebtext_analysis.timestamped_run_dir(Path("outputs/openwebtext_analysis"), "smoke")
    assert run_dir == Path("outputs/openwebtext_analysis/smoke")
