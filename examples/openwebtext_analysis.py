from __future__ import annotations

import argparse
import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ranking_divergence import (
    MirrorSampler,
    PeriodicSampler,
    PhraseBankSampler,
    RestrictedMarginalSampler,
    duo_generative_perplexity,
    per_sample_unigram_entropy,
    rank_histogram,
    rank_wasserstein_from_histograms,
    rep_n,
    unique_ngram_ratios,
)
from ranking_divergence.data import (
    DUO_OWT_CACHE_DIR,
    OWT_HELDOUT_SPLIT,
    OWT_SAMPLER_SOURCE_SPLIT,
    load_openwebtext_texts,
    openwebtext_split_config,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the GPT-2/OpenWebText rank-divergence analysis.")
    parser.add_argument("--cache-dir", default=DUO_OWT_CACHE_DIR)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/openwebtext_analysis"))
    parser.add_argument("--sample-length", type=int, default=1024)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--num-reference", type=int, default=128)
    parser.add_argument("--num-sampler-source", type=int, default=4096)
    parser.add_argument("--scorer-model", default="gpt2-large")
    parser.add_argument("--generator-model", default="gpt2")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--mirror-k", type=int, default=5000)
    parser.add_argument("--periodic-k", type=int, default=400)
    parser.add_argument("--phrase-bank-m", type=int, default=5000)
    parser.add_argument("--phrase-bank-n", type=int, default=5)
    parser.add_argument("--reference-split", default=OWT_HELDOUT_SPLIT)
    parser.add_argument("--sampler-source-split", default=OWT_SAMPLER_SOURCE_SPLIT)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--plot-only", action="store_true")
    parser.add_argument("--metrics-csv", type=Path, default=None)
    parser.add_argument("--cache-scorer", action="store_true")
    return parser.parse_args(argv)


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


def timestamped_run_dir(output_dir: Path, run_name: str | None = None) -> Path:
    if run_name is None:
        run_name = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return output_dir / run_name


def slugify(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_")


@torch.inference_mode()
def generate_model_token_ids(
    model,
    tokenizer,
    *,
    num_samples: int,
    length: int,
    device: str,
    seed: int,
    temperature: float,
    top_p: float,
) -> list[list[int]]:
    torch.manual_seed(seed)
    model = model.to(device).eval()
    input_ids = torch.full((num_samples, 1), tokenizer.eos_token_id, dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    output = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=length,
        do_sample=True,
        top_p=top_p,
        top_k=0,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
    )
    samples = output[:, 1:].detach().cpu().tolist()
    eos_token_id = tokenizer.eos_token_id
    if eos_token_id is None:
        return samples

    trimmed_samples: list[list[int]] = []
    for sample in samples:
        if eos_token_id in sample:
            eos_index = sample.index(eos_token_id)
            sample = sample[: eos_index + 1]
        trimmed_samples.append(sample)
    return trimmed_samples


def decode_token_ids(tokenizer, token_ids: Sequence[Sequence[int]], *, skip_special_tokens: bool = False) -> list[str]:
    return tokenizer.batch_decode(token_ids, skip_special_tokens=skip_special_tokens)


def write_text_samples(path: Path, texts: Sequence[str]) -> None:
    path.write_text("\n\n".join(texts) + "\n", encoding="utf-8")


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def candidate_metrics(
    *,
    name: str,
    texts: list[str],
    token_ids: list[list[int]],
    reference_histogram: torch.Tensor,
    scorer,
    tokenizer,
    args: argparse.Namespace,
) -> dict[str, float | str]:
    comparison_histogram = rank_histogram(
        texts,
        scorer,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.sample_length,
        device=args.device,
        normalize=True,
        show_progress=True,
    )
    row: dict[str, float | str] = {
        "name": name,
        "unigram_entropy": per_sample_unigram_entropy(texts, tokenizer, token_ids=token_ids),
        "gen_ppl": duo_generative_perplexity(
            texts,
            scorer,
            tokenizer,
            batch_size=args.batch_size,
            max_length=args.sample_length,
            device=args.device,
            show_progress=True,
        ),
        "rank_wasserstein": rank_wasserstein_from_histograms(
            reference_histogram,
            comparison_histogram,
            normalize=False,
        ),
    }
    for n in range(1, 5):
        ratios = unique_ngram_ratios(texts, tokenizer, n=n, token_ids=token_ids)
        row[f"unique_{n}gram_sample"] = ratios["sample"]
        row[f"unique_{n}gram_corpus"] = ratios["corpus"]
    for n in range(1, 4):
        row[f"rep_{n}"] = rep_n(texts, tokenizer, n=n, token_ids=token_ids)
    return row


def write_metrics_csv(path: Path, rows: Sequence[dict[str, float | str]]) -> None:
    fieldnames = list(rows[0].keys()) if rows else ["name"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_metrics_csv(path: Path) -> list[dict[str, float | str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if key != "name" and value not in {"", None}:
                row[key] = float(value)
    return rows


def write_markdown_table(path: Path, rows: Sequence[dict[str, float | str]]) -> None:
    columns = ["name", "unigram_entropy", "gen_ppl", "rank_wasserstein"]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        values = [str(row["name"])]
        values.extend(f"{float(row[column]):.4g}" for column in columns[1:])
        lines.append("| " + " | ".join(values) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_latex_table(path: Path, rows: Sequence[dict[str, float | str]]) -> None:
    lines = [
        "\\begin{tabular}{lrrr}",
        "\\toprule",
        "Generator & Entropy & Gen-PPL & Rank-W \\\\",
        "\\midrule",
    ]
    for row in rows:
        name = str(row["name"]).replace("_", "\\_")
        lines.append(
            f"{name} & {float(row['unigram_entropy']):.4g} & "
            f"{float(row['gen_ppl']):.4g} & {float(row['rank_wasserstein']):.4g} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_metrics(rows: Sequence[dict[str, float | str]], plot_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    names = [str(row["name"]) for row in rows]

    def bar(metric: str, filename: str, ylabel: str) -> None:
        fig, ax = plt.subplots(figsize=(max(7, len(names) * 1.2), 4))
        ax.bar(names, [float(row[metric]) for row in rows])
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=30)
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=180)
        plt.close(fig)

    bar("unigram_entropy", "unigram_entropy.png", "Unigram entropy")
    bar("gen_ppl", "generative_perplexity.png", "Generative perplexity")
    bar("rank_wasserstein", "rank_wasserstein.png", "Rank-Wasserstein")
    for n in range(1, 5):
        bar(f"unique_{n}gram_sample", f"unique_{n}gram_sample.png", f"Unique {n}-grams")

    def scatter(x_metric: str, y_metric: str, filename: str) -> None:
        fig, ax = plt.subplots(figsize=(5, 4))
        xs = [float(row[x_metric]) for row in rows]
        ys = [float(row[y_metric]) for row in rows]
        ax.scatter(xs, ys)
        for name, x, y in zip(names, xs, ys):
            ax.annotate(name, (x, y), fontsize=8)
        ax.set_xlabel(x_metric)
        ax.set_ylabel(y_metric)
        if all(math.isfinite(x) and x > 0 for x in xs):
            ax.set_xscale("log")
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=180)
        plt.close(fig)

    scatter("gen_ppl", "rank_wasserstein", "gen_ppl_vs_rank_wasserstein.png")
    scatter("unigram_entropy", "rank_wasserstein", "entropy_vs_rank_wasserstein.png")


def run_plot_only(args: argparse.Namespace) -> None:
    if args.metrics_csv is None:
        raise SystemExit("--plot-only requires --metrics-csv")
    rows = load_metrics_csv(args.metrics_csv)
    output_dir = args.metrics_csv.parent
    (output_dir / "tables").mkdir(parents=True, exist_ok=True)
    plot_metrics(rows, output_dir / "plots")
    write_markdown_table(output_dir / "tables" / "metrics.md", rows)
    write_latex_table(output_dir / "tables" / "metrics.tex", rows)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    if args.plot_only:
        run_plot_only(args)
        return

    args.device = resolve_device(args.device)
    run_dir = timestamped_run_dir(args.output_dir, args.run_name)
    sample_dir = run_dir / "samples"
    token_dir = run_dir / "tokens"
    table_dir = run_dir / "tables"
    for path in (sample_dir, token_dir, table_dir):
        path.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.scorer_model)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading OpenWebText splits...")
    reference_texts = load_openwebtext_texts(
        split=args.reference_split,
        cache_dir=args.cache_dir,
        limit=args.num_reference,
    )
    sampler_source = load_openwebtext_texts(
        split=args.sampler_source_split,
        cache_dir=args.cache_dir,
        limit=args.num_sampler_source,
    )

    print(f"Loading scorer {args.scorer_model}...")
    scorer = AutoModelForCausalLM.from_pretrained(args.scorer_model)
    scorer = scorer.to(args.device).eval()

    safe_scorer_name = args.scorer_model.replace("/", "__")
    cache_path = Path("cache/reference_histograms") / f"{safe_scorer_name}_owt.pt"

    if cache_path.exists() and args.cache_scorer:
        print(f"Loading cached reference rank histogram from {cache_path}...")
        reference_histogram = torch.load(cache_path, map_location="cpu", weights_only=True)
    else:
        print("Computing held-out reference rank histogram...")
        reference_histogram = rank_histogram(
            reference_texts,
            scorer,
            tokenizer,
            batch_size=args.batch_size,
            max_length=args.sample_length,
            device=args.device,
            normalize=True,
            show_progress=True,
        )
        if args.cache_scorer:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(reference_histogram, cache_path)
            print(f"Cached reference rank histogram to {cache_path}")

    print(f"Generating from {args.generator_model}...")
    generator_tokenizer = AutoTokenizer.from_pretrained(args.generator_model)
    generator_tokenizer.pad_token = generator_tokenizer.eos_token
    generator = AutoModelForCausalLM.from_pretrained(args.generator_model)
    gpt2_ids = generate_model_token_ids(
        generator,
        generator_tokenizer,
        num_samples=args.num_samples,
        length=args.sample_length,
        device=args.device,
        seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    del generator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    topk = RestrictedMarginalSampler.from_texts(sampler_source, tokenizer, k=args.top_k)
    mirror_base = RestrictedMarginalSampler.from_texts(sampler_source, tokenizer, k=args.mirror_k)
    candidates = {
        args.generator_model: gpt2_ids,
        f"top_k_iid_{args.top_k}": topk.sample_token_ids(
            num_samples=args.num_samples,
            length=args.sample_length,
            seed=args.seed,
        ),
        f"mirror_{args.mirror_k}": MirrorSampler(mirror_base).sample_token_ids(
            num_samples=args.num_samples,
            length=args.sample_length,
            seed=args.seed,
        ),
        f"periodic_{args.periodic_k}": PeriodicSampler.from_texts(
            sampler_source,
            tokenizer,
            k=args.periodic_k,
        ).sample_token_ids(num_samples=args.num_samples, length=args.sample_length, seed=args.seed),
        f"phrase_bank_{args.phrase_bank_m}": PhraseBankSampler.from_texts(
            sampler_source,
            tokenizer,
            n=args.phrase_bank_n,
            m=args.phrase_bank_m,
        ).sample_token_ids(num_samples=args.num_samples, length=args.sample_length, seed=args.seed),
    }

    metadata = {
        "seed": args.seed,
        "sample_length": args.sample_length,
        "num_samples": args.num_samples,
        "num_reference": args.num_reference,
        "num_sampler_source": args.num_sampler_source,
        "scorer_model": args.scorer_model,
        "generator_model": args.generator_model,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "generator_top_k": 0,
        "generator_do_sample": True,
        "generator_eos_token_id": generator_tokenizer.eos_token_id,
        "generator_stops_at_eos": True,
        "sampler_parameters": {
            "top_k": args.top_k,
            "mirror_k": args.mirror_k,
            "periodic_k": args.periodic_k,
            "phrase_bank_m": args.phrase_bank_m,
            "phrase_bank_n": args.phrase_bank_n,
        },
        "splits": openwebtext_split_config()
        | {
            "reference_split": args.reference_split,
            "sampler_source_split": args.sampler_source_split,
            "cache_dir": args.cache_dir,
        },
    }
    write_json(run_dir / "metadata.json", metadata)

    rows = []
    full_metrics = {"metadata": metadata, "metrics": []}
    for name, ids in candidates.items():
        print(f"Evaluating {name}...")
        texts = decode_token_ids(tokenizer, ids, skip_special_tokens=False)
        display_texts = decode_token_ids(tokenizer, ids, skip_special_tokens=True)
        slug = slugify(name)
        write_text_samples(sample_dir / f"{slug}.txt", display_texts)
        write_json(token_dir / f"{slug}.json", ids)
        row = candidate_metrics(
            name=name,
            texts=texts,
            token_ids=ids,
            reference_histogram=reference_histogram,
            scorer=scorer,
            tokenizer=tokenizer,
            args=args,
        )
        rows.append(row)
        full_metrics["metrics"].append(row)

    write_metrics_csv(run_dir / "metrics.csv", rows)
    write_json(run_dir / "metrics.json", full_metrics)
    write_markdown_table(table_dir / "metrics.md", rows)
    write_latex_table(table_dir / "metrics.tex", rows)
    if not args.skip_plots:
        plot_metrics(rows, run_dir / "plots")
    print(f"Wrote OpenWebText analysis artifacts to {run_dir}")


if __name__ == "__main__":
    main()
