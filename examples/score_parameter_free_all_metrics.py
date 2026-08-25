"""Score the parameter-free baselines (saved token IDs) on the full metric battery.

Produces one row per baseline (rank_wasserstein + all divergences + gen_ppl + entropy +
MAUVE + GM) against the SAME reference histogram / held-out OWT used by the s-flm sweeps,
so the baselines can be added at NFE=0 on the efficiency plots.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ranking_divergence import (
    compute_all,
    per_sample_unigram_entropy,
    rank_wasserstein_from_histograms,
    score_token_ids,
)
from ranking_divergence.data import OWT_HELDOUT_SPLIT, load_openwebtext_texts

_spec = importlib.util.spec_from_file_location(
    "edm", Path(__file__).resolve().parent / "evaluate_distributional_metrics.py"
)
edm = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(edm)

DISPLAY = {"mirror_5000": "Mirror", "periodic_400": "Periodic",
           "phrase_bank_5000": "Phrase bank", "top_k_iid_64": "Top-k IID"}


def main() -> None:
    import mauve

    p = argparse.ArgumentParser()
    p.add_argument("--token-dir", type=Path, default=Path("outputs/parameter_free_analysis/owt-128/tokens"))
    p.add_argument("--reference-histogram", type=Path,
                   default=Path("outputs/diffusion_sweep_analysis/duo-sflm-v1/reference_rank_histogram.pt"))
    p.add_argument("--scorer-model", default="gpt2-large")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--cache-dir", default="/users/staff/dmi-dmi/miele0000/.cache/discrete_diffusion/owt")
    p.add_argument("--num-reference", type=int, default=256)
    p.add_argument("--output", type=Path, default=Path("outputs/divergence_exploration/parameter_free_all_metrics.csv"))
    args = p.parse_args()

    reference = torch.load(args.reference_histogram, map_location="cpu", weights_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.scorer_model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.scorer_model).to(args.device).eval()

    print(f"Loading {args.num_reference} held-out OWT reference docs...")
    ref_texts = load_openwebtext_texts(split=OWT_HELDOUT_SPLIT, cache_dir=args.cache_dir, limit=args.num_reference)
    ref_feats = edm.featurize(ref_texts, model, tokenizer, args.device, 1024, 32)
    ref_grad = edm.mean_nll_gradient(ref_texts[:64], model, tokenizer, args.device, 512)

    rows = []
    for path in sorted(args.token_dir.glob("*.json")):
        name = path.stem
        token_ids = json.loads(path.read_text())
        gen_ppl, hist, model_entropy = score_token_ids(token_ids, model, tokenizer, batch_size=1, max_length=1024,
                                        rank_position_chunk=64, device=args.device, description=name)
        texts = tokenizer.batch_decode(token_ids, skip_special_tokens=False)
        gen_feats = edm.featurize(texts, model, tokenizer, args.device, 1024, 32)
        mauve_out = mauve.compute_mauve(p_features=gen_feats, q_features=ref_feats, device_id=0, verbose=False, batch_size=32)
        gen_grad = edm.mean_nll_gradient(texts[:64], model, tokenizer, args.device, 512)
        row = {
            "method": DISPLAY.get(name, name), "nfe": 0, "temperature_label": "",
            "unigram_entropy": per_sample_unigram_entropy(texts, token_ids=token_ids),
            "gen_ppl": gen_ppl,
            "model_entropy": model_entropy,
            "rank_wasserstein": rank_wasserstein_from_histograms(reference, hist, normalize=False),
            "mauve": float(mauve_out.mauve),
            "gm": float(((gen_grad - ref_grad) ** 2).sum().item()),
        }
        row.update(compute_all(reference, hist))
        rows.append(row)
        print(f"{row['method']:12s}: rankW={row['rank_wasserstein']:.3f} MAUVE={row['mauve']:.3f} "
              f"GM={row['gm']:.2f} genPPL={gen_ppl:.1f} H={row['unigram_entropy']:.2f}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0].keys())
    with args.output.open("w", newline="") as h:
        w = csv.DictWriter(h, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
