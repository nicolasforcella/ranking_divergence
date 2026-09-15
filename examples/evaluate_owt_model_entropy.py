from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Sequence
from torch.utils.data import DataLoader

from tqdm.auto import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets
from torch.utils.data import DataLoader

from ranking_divergence.evaluation import (
    score_tokens_no_rank
)

from ranking_divergence.data import DUO_OWT_CACHE_DIR, OWT_HELDOUT_SPLIT, OWT_DATASET_ID

def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def build_data_loader(
    tokenizer,
    cache_dir: str,
    split: str = OWT_HELDOUT_SPLIT,
    batch_size: int = 1024,
    max_length: int = 1024,
):
    """Build a dataloader wrapper around OWT."""
    try:
        import datasets
    except ImportError as exc:
        raise ImportError("Install examples dependencies with `uv pip install -e '.[examples]'`.") from exc

    dataset = datasets.load_dataset(
        OWT_DATASET_ID,
        split=split,
        cache_dir=cache_dir,
        streaming=False,
        trust_remote_code=True,
    )

    def collect_fn(batch_rows):
        token_ids = [tokenizer.encode(row["text"], add_special_tokens=False)[:max_length] for row in batch_rows]

        return [ids for ids in token_ids if len(ids) >= 2]

    return DataLoader(dataset, batch_size=batch_size, collate_fn=collect_fn)


def build_samples_data_loader(
    tokenizer,
    samples_path: Path,
    batch_size: int = 1024,
    max_length: int = 1024,
):
    """Build a dataloader wrapper around a generated samples json."""
    texts = json.loads(samples_path.read_text(encoding="utf-8"))["generated_seqs"]

    def collect_fn(batch_rows):
        token_ids = [tokenizer.encode(text, add_special_tokens=False)[:max_length] for text in batch_rows]

        return [ids for ids in token_ids if len(ids) >= 2]

    return DataLoader(texts, batch_size=batch_size, collate_fn=collect_fn)

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate OWT Gen PPL and Entropy."
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/openwebtext_analysis/model_entropy"))
    parser.add_argument("--scorer-model", default="gpt2-large")
    parser.add_argument("--cache-dir", default=DUO_OWT_CACHE_DIR)
    parser.add_argument("--reference-split", default=OWT_HELDOUT_SPLIT)
    parser.add_argument("--samples-path", type=Path, default=None)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size-data-loader", type=int, default=1024)
    parser.add_argument("--batch-size-scoring", type=int, default=8)
    parser.add_argument("--rank-position-chunk", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["float16", "bfloat16", "float32", "auto"])
    args = parser.parse_args(argv)

    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)

    #Store location

    output_dir = args.output_dir
    scorer_model_tag = args.scorer_model.replace("/", "_").replace("-", "_")
    store_dir = output_dir / scorer_model_tag
    data_dir = store_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    #Store metadata for the run

    metadata_path = store_dir / "metadata.json"
    results = {
        "scorer_model": args.scorer_model,
        "device": args.device,
        "dtype": args.dtype,
        "max_length": args.max_length,
        "batch_size_data_loader": args.batch_size_data_loader,
        "batch_size_scoring": args.batch_size_scoring,
        "rank_position_chunk": args.rank_position_chunk,
        "reference_split": args.reference_split,
        "cache_dir": args.cache_dir,
        "samples_path": str(args.samples_path) if args.samples_path else None,
    }


    #Load scorer and tokenizer
    args.device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.scorer_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = args.dtype if args.dtype == "auto" else getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(args.scorer_model, dtype=dtype).to(args.device).eval()

    #Build data loader wrapper for not loading all samples at a time.
    if args.samples_path is not None:
        dataloader_wrapper = build_samples_data_loader(
            tokenizer,
            args.samples_path,
            batch_size=args.batch_size_data_loader,
            max_length=args.max_length,
        )
    else:
        dataloader_wrapper = build_data_loader(
            tokenizer,
            args.cache_dir,
            split=args.reference_split,
            batch_size=args.batch_size_data_loader,
            max_length=args.max_length,
        )

    progress_bar = tqdm(total=len(dataloader_wrapper), desc="Scoring OWT")
    batch_results = []
    for sample_batch in dataloader_wrapper:
        batch_results.append(
            score_tokens_no_rank(
                sample_batch,
                model,
                tokenizer,
                batch_size=args.batch_size_scoring,
                max_length=args.max_length,
                rank_position_chunk=args.rank_position_chunk,
                device=args.device,
                per_sample=True,
            )
        )
        
        progress_bar.update(1)

    progress_bar.close()

    result_per_sample = torch.cat(batch_results, dim=0)
    torch.save(result_per_sample, data_dir / "per_sample_metrics.pt")

    #Store the final results
    totals = result_per_sample.sum(dim=0)
    total_tokens = float(totals[2].item())

    gen_ppl = math.exp(float(totals[0].item()) /total_tokens)
    model_entropy = float(totals[1].item()) /total_tokens

    results["num_samples"] = int(result_per_sample.shape[0])
    results["num_tokens"] = total_tokens
    results["genPPL"] = gen_ppl
    results["model_entropy"] = model_entropy
    results["kl"] = math.log(gen_ppl) - model_entropy
    write_json(metadata_path, results)


if __name__ == "__main__":
    main()