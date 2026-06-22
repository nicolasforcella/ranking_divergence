from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path
from typing import Sequence

import numpy as np


DEFAULT_METRICS = Path("outputs/diffusion_sweep_analysis/mdlm-candi-duo-50k/metrics.csv")
METRIC_LABELS = {
    "unigram_entropy": "Unigram entropy (nats)",
    "gen_ppl": "Generative perplexity",
    "rank_wasserstein": "Rank-Wasserstein distance",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot temperature-by-NFE metrics for diffusion language model sweeps."
    )
    parser.add_argument("--metrics-csv", type=Path, default=DEFAULT_METRICS)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <metrics directory>/plots.",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--formats",
        nargs="+",
        choices=("png", "pdf", "svg"),
        default=["png"],
    )
    return parser.parse_args(argv)


def load_rows(path: Path) -> list[dict[str, str | int | float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows: list[dict[str, str | int | float]] = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No metric rows found in {path}.")

    required = {"method", "nfe", "temperature", *METRIC_LABELS}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing required columns: {sorted(missing)}")

    for row in rows:
        row["nfe"] = int(row["nfe"])
        row["temperature"] = float(row["temperature"])
        for metric in METRIC_LABELS:
            row[metric] = float(row[metric])
    return rows


def rows_by_method(
    rows: Sequence[dict[str, str | int | float]],
) -> dict[str, list[dict[str, str | int | float]]]:
    grouped: dict[str, list[dict[str, str | int | float]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["method"])].append(row)
    for method_rows in grouped.values():
        method_rows.sort(key=lambda row: (int(row["nfe"]), float(row["temperature"])))
    return dict(sorted(grouped.items()))


def metric_grid(
    rows: Sequence[dict[str, str | int | float]], metric: str
) -> tuple[list[int], list[float], np.ndarray]:
    nfes = sorted({int(row["nfe"]) for row in rows})
    temperatures = sorted({float(row["temperature"]) for row in rows})
    nfe_index = {value: index for index, value in enumerate(nfes)}
    temperature_index = {value: index for index, value in enumerate(temperatures)}
    grid = np.full((len(nfes), len(temperatures)), np.nan)
    for row in rows:
        grid[nfe_index[int(row["nfe"])], temperature_index[float(row["temperature"])]] = float(
            row[metric]
        )
    return nfes, temperatures, grid


def save_figure(fig, base_path: Path, formats: Sequence[str], dpi: int) -> None:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        fig.savefig(base_path.with_suffix(f".{extension}"), dpi=dpi, bbox_inches="tight")


def plot_metric_lines(
    method: str,
    rows: Sequence[dict[str, str | int | float]],
    metric: str,
    output_dir: Path,
    *,
    formats: Sequence[str],
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    nfes = sorted({int(row["nfe"]) for row in rows})
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(nfes)))
    for nfe, color in zip(nfes, colors):
        nfe_rows = [row for row in rows if int(row["nfe"]) == nfe]
        ax.plot(
            [float(row["temperature"]) for row in nfe_rows],
            [float(row[metric]) for row in nfe_rows],
            marker="o",
            markersize=3,
            linewidth=1.6,
            color=color,
            label=f"NFE {nfe}",
        )

    ax.set_title(f"{method.upper()}: {METRIC_LABELS[metric]} vs. temperature")
    ax.set_xlabel("Temperature")
    ax.set_ylabel(METRIC_LABELS[metric])
    if metric == "gen_ppl":
        ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(title="Sampling steps", ncols=2, fontsize=8)
    save_figure(fig, output_dir / "lines" / f"{method}_{metric}_vs_temperature", formats, dpi)
    plt.close(fig)


def plot_method_summary(
    method: str,
    rows: Sequence[dict[str, str | int | float]],
    output_dir: Path,
    *,
    formats: Sequence[str],
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt

    metrics = list(METRIC_LABELS)
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    nfes = sorted({int(row["nfe"]) for row in rows})
    colors = plt.cm.viridis(np.linspace(0.08, 0.92, len(nfes)))

    for ax, metric in zip(axes, metrics):
        for nfe, color in zip(nfes, colors):
            nfe_rows = [row for row in rows if int(row["nfe"]) == nfe]
            ax.plot(
                [float(row["temperature"]) for row in nfe_rows],
                [float(row[metric]) for row in nfe_rows],
                marker="o",
                markersize=2.5,
                linewidth=1.4,
                color=color,
                label=f"NFE {nfe}",
            )
        ax.set_xlabel("Temperature")
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.grid(alpha=0.25)
        if metric == "gen_ppl":
            ax.set_yscale("log")

    axes[-1].legend(title="Sampling steps", fontsize=8)
    fig.suptitle(f"{method.upper()} sweep", fontsize=14)
    fig.tight_layout()
    save_figure(fig, output_dir / "summaries" / f"{method}_metric_summary", formats, dpi)
    plt.close(fig)


def plot_metric_heatmap(
    method: str,
    rows: Sequence[dict[str, str | int | float]],
    metric: str,
    output_dir: Path,
    *,
    formats: Sequence[str],
    dpi: int,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    nfes, temperatures, grid = metric_grid(rows, metric)
    fig, ax = plt.subplots(figsize=(max(7, len(temperatures) * 0.27), 3.8))
    cmap = plt.get_cmap("magma").copy()
    cmap.set_bad(color="#d9d9d9")
    finite = grid[np.isfinite(grid)]
    norm = LogNorm(vmin=finite.min(), vmax=finite.max()) if metric == "gen_ppl" else None
    image = ax.imshow(grid, aspect="auto", origin="lower", cmap=cmap, norm=norm)

    tick_stride = max(1, math.ceil(len(temperatures) / 12))
    tick_positions = list(range(0, len(temperatures), tick_stride))
    if tick_positions[-1] != len(temperatures) - 1:
        tick_positions.append(len(temperatures) - 1)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([f"{temperatures[index]:.3g}" for index in tick_positions], rotation=45)
    ax.set_yticks(range(len(nfes)))
    ax.set_yticklabels(nfes)
    ax.set_xlabel("Temperature")
    ax.set_ylabel("NFE")
    ax.set_title(f"{method.upper()}: {METRIC_LABELS[metric]}")
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label(METRIC_LABELS[metric])
    save_figure(fig, output_dir / "heatmaps" / f"{method}_{metric}_heatmap", formats, dpi)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_dir = args.output_dir or args.metrics_csv.parent / "plots"
    grouped = rows_by_method(load_rows(args.metrics_csv))

    for method, rows in grouped.items():
        print(f"Plotting {method} ({len(rows)} configurations)...")
        plot_method_summary(method, rows, output_dir, formats=args.formats, dpi=args.dpi)
        for metric in METRIC_LABELS:
            plot_metric_lines(
                method,
                rows,
                metric,
                output_dir,
                formats=args.formats,
                dpi=args.dpi,
            )
            plot_metric_heatmap(
                method,
                rows,
                metric,
                output_dir,
                formats=args.formats,
                dpi=args.dpi,
            )

    print(f"Wrote sweep plots to {output_dir}")


if __name__ == "__main__":
    main()
