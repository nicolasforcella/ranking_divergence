from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence


DISPLAY_NAMES = {
    "top_k_iid_64": "Top-k IID",
    "mirror_5000": "Mirror",
    "periodic_400": "Periodic",
    "phrase_bank_5000": "Phrase bank",
}
COLORS = {
    "Top-k IID": "#bab0ac",
    "Mirror": "#b279a2",
    "Periodic": "#ff9da6",
    "Phrase bank": "#9d755d",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot parameter-free sampler metrics.")
    parser.add_argument(
        "--metrics-csv",
        type=Path,
        default=Path("outputs/parameter_free_analysis/owt-128/metrics.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/parameter_free_analysis/owt-128/plots"),
    )
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args(argv)


def load_rows(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    numeric = [
        "unigram_entropy",
        "gen_ppl",
        "rank_wasserstein",
        "rep_1",
        "rep_2",
        "rep_3",
        *[f"unique_{n}gram_sample" for n in range(1, 5)],
        *[f"unique_{n}gram_corpus" for n in range(1, 5)],
    ]
    for row in rows:
        row["display_name"] = DISPLAY_NAMES.get(row["method"], row["method"])
        for key in numeric:
            row[key] = float(row[key])
    return sorted(rows, key=lambda row: row["display_name"])


def plot_single_metric(rows: Sequence[dict], metric: str, ylabel: str, path: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    names = [row["display_name"] for row in rows]
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(
        names,
        [row[metric] for row in rows],
        color=[COLORS[name] for name in names],
    )
    if metric in {"gen_ppl", "rank_wasserstein"}:
        ax.set_yscale("log")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Parameter-free samplers: {ylabel}")
    ax.grid(axis="y", alpha=0.25, which="both")
    ax.bar_label(bars, fmt="%.4g", padding=3)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoffs(rows: Sequence[dict], path: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    pairs = [
        ("gen_ppl", "rank_wasserstein", "Generative perplexity", "Rank-Wasserstein"),
        ("unigram_entropy", "rank_wasserstein", "Unigram entropy", "Rank-Wasserstein"),
        ("unigram_entropy", "gen_ppl", "Unigram entropy", "Generative perplexity"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (x_metric, y_metric, xlabel, ylabel) in zip(axes, pairs):
        for row in rows:
            ax.scatter(
                row[x_metric],
                row[y_metric],
                s=90,
                color=COLORS[row["display_name"]],
                edgecolor="black",
                linewidth=0.5,
            )
            ax.annotate(
                row["display_name"],
                (row[x_metric], row[y_metric]),
                xytext=(5, 5),
                textcoords="offset points",
                fontsize=8,
            )
        if x_metric == "gen_ppl":
            ax.set_xscale("log")
        if y_metric in {"gen_ppl", "rank_wasserstein"}:
            ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, which="both")
    fig.suptitle("Parameter-free sampler metric tradeoffs")
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_grouped_metrics(
    rows: Sequence[dict],
    metrics: Sequence[str],
    labels: Sequence[str],
    title: str,
    path: Path,
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt
    import numpy as np

    names = [row["display_name"] for row in rows]
    x = np.arange(len(names))
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for index, (metric, label) in enumerate(zip(metrics, labels)):
        ax.bar(
            x + (index - (len(metrics) - 1) / 2) * width,
            [row[metric] for row in rows],
            width,
            label=label,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.05)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(ncols=2)
    fig.tight_layout()
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.metrics_csv)

    plot_single_metric(
        rows,
        "unigram_entropy",
        "Unigram entropy (nats)",
        args.output_dir / "unigram_entropy.png",
        args.dpi,
    )
    plot_single_metric(
        rows,
        "gen_ppl",
        "Generative perplexity",
        args.output_dir / "generative_perplexity.png",
        args.dpi,
    )
    plot_single_metric(
        rows,
        "rank_wasserstein",
        "Rank-Wasserstein distance",
        args.output_dir / "rank_wasserstein.png",
        args.dpi,
    )
    plot_tradeoffs(rows, args.output_dir / "metric_tradeoffs.png", args.dpi)
    plot_grouped_metrics(
        rows,
        [f"rep_{n}" for n in range(1, 4)],
        [f"Rep-{n}" for n in range(1, 4)],
        "Parameter-free sampler repetition",
        args.output_dir / "repetition.png",
        args.dpi,
    )
    plot_grouped_metrics(
        rows,
        [f"unique_{n}gram_sample" for n in range(1, 5)],
        [f"{n}-gram" for n in range(1, 5)],
        "Per-sample unique n-gram ratios",
        args.output_dir / "unique_ngrams_sample.png",
        args.dpi,
    )
    plot_grouped_metrics(
        rows,
        [f"unique_{n}gram_corpus" for n in range(1, 5)],
        [f"{n}-gram" for n in range(1, 5)],
        "Corpus unique n-gram ratios",
        args.output_dir / "unique_ngrams_corpus.png",
        args.dpi,
    )
    print(f"Wrote parameter-free plots to {args.output_dir}")


if __name__ == "__main__":
    main()
