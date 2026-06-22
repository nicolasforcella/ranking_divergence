import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import torch


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "examples" / "evaluate_diffusion_sweeps.py"
SPEC = importlib.util.spec_from_file_location("evaluate_diffusion_sweeps", SCRIPT_PATH)
assert SPEC is not None
evaluate_diffusion_sweeps = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(evaluate_diffusion_sweeps)


def test_default_sweeps_have_method_labels():
    args = evaluate_diffusion_sweeps.parse_args([])
    sweeps = evaluate_diffusion_sweeps.parse_sweeps(args.sweep)
    assert set(sweeps) == {"mdlm", "candi", "duo"}
    assert args.scorer_model == "gpt2-large"
    assert args.max_length == 1024


def test_discovery_and_generation_loading(tmp_path):
    sweep_dir = tmp_path / "samples"
    sweep_dir.mkdir()
    path = sweep_dir / "samples_steps32_temp0.750.json"
    path.write_text(
        json.dumps(
            {
                "generated_seqs": ["one", "two"],
                "steps": 32,
                "temperature": 0.75,
                "generative_ppl": 10.0,
                "entropy": 3.0,
            }
        )
    )

    files = evaluate_diffusion_sweeps.discover_sweep_files({"duo": sweep_dir})
    assert len(files) == 1
    assert files[0].method == "duo"
    assert files[0].nfe == 32
    assert files[0].temperature_label == "0.750"

    texts, payload = evaluate_diffusion_sweeps.load_generation_file(files[0], limit_samples=1)
    assert texts == ["one"]
    assert payload["temperature"] == 0.75


def test_inventory_reports_missing_cartesian_grid_points(tmp_path):
    sweep_dir = tmp_path / "samples"
    sweep_dir.mkdir()
    for name in (
        "samples_steps16_temp0.500.json",
        "samples_steps16_temp1.000.json",
        "samples_steps32_temp0.500.json",
    ):
        (sweep_dir / name).write_text("{}")

    files = evaluate_diffusion_sweeps.discover_sweep_files({"candi": sweep_dir})
    result = evaluate_diffusion_sweeps.inventory(files)
    assert result["methods"]["candi"]["missing_grid_points"] == [
        {"nfe": 32, "temperature_label": "1.000"}
    ]


def test_combined_scorer_returns_finite_ppl_and_normalized_histogram():
    class TinyTokenizer:
        eos_token_id = 4
        pad_token_id = 4

        def pad(self, encoded, **_kwargs):
            rows = encoded["input_ids"]
            width = max(map(len, rows))
            input_ids = torch.tensor([row + [4] * (width - len(row)) for row in rows])
            attention_mask = torch.tensor(
                [[1] * len(row) + [0] * (width - len(row)) for row in rows]
            )
            return {"input_ids": input_ids, "attention_mask": attention_mask}

    class TinyModel:
        config = SimpleNamespace(vocab_size=5, n_positions=8)

        def __call__(self, input_ids, attention_mask):
            del attention_mask
            logits = torch.zeros((*input_ids.shape, self.config.vocab_size))
            logits[..., 0] = 1.0
            return SimpleNamespace(logits=logits)

    ppl, histogram = evaluate_diffusion_sweeps.score_token_ids(
        [[4, 0, 1, 2], [4, 0, 0]],
        TinyModel(),
        TinyTokenizer(),
        batch_size=2,
        max_length=8,
        rank_position_chunk=2,
        device="cpu",
        description="test",
    )
    assert ppl > 0
    assert torch.isfinite(torch.tensor(ppl))
    assert histogram.shape == (5,)
    assert torch.isclose(histogram.sum(), torch.tensor(1.0, dtype=torch.float64))
