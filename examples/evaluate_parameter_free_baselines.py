from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ranking_divergence import (
    MirrorSampler,
    PeriodicSampler,
    PhraseBankSampler,
    RestrictedMarginalSampler,
    lexical_metrics,
    rank_wasserstein_from_histograms,
    score_token_ids,
)
from ranking_divergence.data import (
    DUO_OWT_CACHE_DIR,
    OWT_SAMPLER_SOURCE_SPLIT,
    load_openwebtext_texts,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate and evaluate parameter-free OpenWebText samplers."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/parameter_free_analysis/owt-128"),
    )
    parser.add_argument(
        "--reference-histogram",
        type=Path,
        default=Path(
            "outputs/diffusion_sweep_analysis/mdlm-candi-duo-50k/"
            "reference_rank_histogram.pt"
        ),
    )
    parser.add_argument("--cache-dir", default=DUO_OWT_CACHE_DIR)
    parser.add_argument("--sampler-source-split", default=OWT_SAMPLER_SOURCE_SPLIT)
    parser.add_argument("--num-sampler-source", type=int, default=4096)
    parser.add_argument("--num-samples", type=int, default=128)
    parser.add_argument("--sample-length", type=int, default=1024)
    parser.add_argument("--scorer-model", default="gpt2-large")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--rank-position-chunk", type=int, default=64)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--mirror-k", type=int, default=5000)
    parser.add_argument("--periodic-k", type=int, default=400)
    parser.add_argument("--phrase-bank-m", type=int, default=5000)
    parser.add_argument("--phrase-bank-n", type=int, default=5)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args(argv)


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.device = (
        "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    )
    if args.device == "auto":
        args.device = "cpu"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output_dir / "checkpoints"
    token_dir = args.output_dir / "tokens"
    checkpoint_dir.mkdir(exist_ok=True)
    token_dir.mkdir(exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.scorer_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Loading OpenWebText sampler source...")
    source_texts = load_openwebtext_texts(
        split=args.sampler_source_split,
        cache_dir=args.cache_dir,
        limit=args.num_sampler_source,
    )

    topk = RestrictedMarginalSampler.from_texts(source_texts, tokenizer, k=args.top_k)
    mirror_base = RestrictedMarginalSampler.from_texts(
        source_texts, tokenizer, k=args.mirror_k
    )
    samplers = {
        f"top_k_iid_{args.top_k}": topk,
        f"mirror_{args.mirror_k}": MirrorSampler(mirror_base),
        f"periodic_{args.periodic_k}": PeriodicSampler.from_texts(
            source_texts, tokenizer, k=args.periodic_k
        ),
        f"phrase_bank_{args.phrase_bank_m}": PhraseBankSampler.from_texts(
            source_texts,
            tokenizer,
            n=args.phrase_bank_n,
            m=args.phrase_bank_m,
        ),
    }

    print(f"Loading scorer {args.scorer_model}...")
    scorer = AutoModelForCausalLM.from_pretrained(args.scorer_model).to(args.device).eval()
    reference_histogram = torch.load(
        args.reference_histogram, map_location="cpu", weights_only=True
    )

    metadata = {
        "num_samples": args.num_samples,
        "sample_length": args.sample_length,
        "seed": args.seed,
        "scorer_model": args.scorer_model,
        "reference_histogram": str(args.reference_histogram),
        "num_sampler_source": args.num_sampler_source,
        "sampler_source_split": args.sampler_source_split,
        "cache_dir": args.cache_dir,
    }
    write_json(args.output_dir / "metadata.json", metadata)

    for name, sampler in samplers.items():
        checkpoint = checkpoint_dir / f"{name}.json"
        if checkpoint.exists() and not args.force:
            print(f"Skipping completed {name}")
            continue
        print(f"Generating {args.num_samples} samples from {name}...")
        token_ids = sampler.sample_token_ids(
            num_samples=args.num_samples,
            length=args.sample_length,
            seed=args.seed,
        )
        texts = tokenizer.batch_decode(token_ids, skip_special_tokens=False)
        write_json(token_dir / f"{name}.json", token_ids)
        gen_ppl, histogram = score_token_ids(
            token_ids,
            scorer,
            tokenizer,
            batch_size=args.batch_size,
            max_length=args.sample_length,
            rank_position_chunk=args.rank_position_chunk,
            device=args.device,
            description=name,
        )
        row = {
            "method": name,
            "display_name": name,
            "nfe": 0,
            "temperature": "",
            "temperature_label": "",
            "num_samples": args.num_samples,
            "gen_ppl": gen_ppl,
            "rank_wasserstein": rank_wasserstein_from_histograms(
                reference_histogram, histogram, normalize=False
            ),
        }
        row.update(lexical_metrics(texts, token_ids))
        write_json(checkpoint, row)

        rows = [
            json.loads(path.read_text(encoding="utf-8"))
            for path in sorted(checkpoint_dir.glob("*.json"))
        ]
        write_csv(args.output_dir / "metrics.csv", rows)
        write_json(args.output_dir / "metrics.json", {"metadata": metadata, "metrics": rows})

    rows = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(checkpoint_dir.glob("*.json"))
    ]
    write_csv(args.output_dir / "metrics.csv", rows)
    write_json(args.output_dir / "metrics.json", {"metadata": metadata, "metrics": rows})
    print(f"Wrote {len(rows)} parameter-free results to {args.output_dir}")


if __name__ == "__main__":
    main()
