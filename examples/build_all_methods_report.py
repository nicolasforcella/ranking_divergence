from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import numpy as np

from plot_efficiency_pareto import (
    PARAMETER_FREE_NAMES,
    best_ar_point,
    best_by_method_nfe,
    load_parameter_free_points,
    load_rows,
    pareto_frontier,
    plot as plot_pareto,
    write_summary as write_pareto_summary,
)
from plot_diffusion_sweeps import (
    load_rows as load_sweep_rows,
    plot_method_summary,
    rows_by_method,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a consolidated all-method metrics report.")
    parser.add_argument(
        "--diffusion-metrics",
        type=Path,
        default=Path("outputs/diffusion_sweep_analysis/mdlm-candi-duo-50k/metrics.csv"),
    )
    parser.add_argument(
        "--ar-metrics",
        type=Path,
        default=Path(
            "outputs/diffusion_sweep_analysis/ar-openwebtext-20260622-004711/metrics.csv"
        ),
    )
    parser.add_argument(
        "--baseline-metrics",
        type=Path,
        default=Path("outputs/parameter_free_analysis/owt-128/metrics.csv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/all_methods_report"),
    )
    parser.add_argument("--ar-nfe", type=int, default=128)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args(argv)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def normalize_rows(
    diffusion_path: Path, ar_path: Path, baseline_path: Path, ar_nfe: int
) -> list[dict]:
    rows: list[dict] = []
    for source, path in (("diffusion", diffusion_path), ("ar", ar_path)):
        for row in read_csv(path):
            normalized = dict(row)
            normalized["source_group"] = source
            normalized["display_name"] = str(row["method"]).upper()
            if source == "ar":
                normalized["source_nfe"] = normalized["nfe"]
                normalized["nfe"] = ar_nfe
            rows.append(normalized)
    for row in read_csv(baseline_path):
        normalized = dict(row)
        name = str(row["method"])
        normalized["source_group"] = "parameter_free"
        normalized["display_name"] = PARAMETER_FREE_NAMES.get(name, name)
        normalized["nfe"] = 0
        normalized["temperature"] = ""
        normalized["temperature_label"] = ""
        rows.append(normalized)
    return rows


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


def select_best_rank_rows(rows: Sequence[dict]) -> list[dict]:
    selected: dict[tuple[str, int], dict] = {}
    for row in rows:
        key = (str(row["method"]), int(row["nfe"]))
        distance = float(row["rank_wasserstein"])
        if key not in selected or distance < float(selected[key]["rank_wasserstein"]):
            selected[key] = row
    return sorted(selected.values(), key=lambda row: (int(row["nfe"]), str(row["method"])))


def plot_efficiency_summary(rows: Sequence[dict], output: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    metrics = [
        ("unigram_entropy", "Unigram entropy (nats)", False),
        ("gen_ppl", "Generative perplexity", True),
        ("rank_wasserstein", "Rank-Wasserstein", True),
    ]
    styles = {
        "mdlm": ("#4c78a8", "o", "MDLM"),
        "candi": ("#f58518", "s", "CANDI"),
        "duo": ("#54a24b", "^", "DUO"),
        "ar": ("#e45756", "*", "AR"),
    }
    baseline_styles = ["X", "P", "v", "h"]
    baseline_colors = ["#b279a2", "#ff9da6", "#9d755d", "#bab0ac"]
    efficiencies = [0, 8, 16, 32, 64, 128]
    x_position = {nfe: index for index, nfe in enumerate(efficiencies)}
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.2))

    for ax, (metric, ylabel, log_scale) in zip(axes, metrics):
        for method, (color, marker, label) in styles.items():
            method_rows = sorted(
                [row for row in rows if row["method"] == method],
                key=lambda row: int(row["nfe"]),
            )
            if not method_rows:
                continue
            if method == "ar":
                ax.scatter(
                    [x_position[int(method_rows[0]["nfe"])]],
                    [float(method_rows[0][metric])],
                    color=color,
                    marker=marker,
                    s=190,
                    edgecolor="black",
                    linewidth=0.6,
                    label=label,
                    zorder=5,
                )
            else:
                ax.plot(
                    [x_position[int(row["nfe"])] for row in method_rows],
                    [float(row[metric]) for row in method_rows],
                    color=color,
                    marker=marker,
                    linewidth=1.7,
                    markersize=7,
                    label=label,
                )

        baseline_rows = [
            row for row in rows if row["source_group"] == "parameter_free"
        ]
        for row, marker, color in zip(baseline_rows, baseline_styles, baseline_colors):
            ax.scatter(
                [x_position[0]],
                [float(row[metric])],
                marker=marker,
                s=85,
                color=color,
                edgecolor="black",
                linewidth=0.4,
                label=row["display_name"],
            )
        if log_scale:
            ax.set_yscale("log")
        ax.set_xticks(range(len(efficiencies)))
        ax.set_xticklabels(["0\nparameter-free", "8", "16", "32", "64", "128"])
        ax.set_xlabel("NFE")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, which="both")

    axes[0].legend(fontsize=7, ncols=2)
    fig.suptitle("All methods: metrics at each method/NFE's best rank-distance temperature")
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_tradeoffs(all_rows: Sequence[dict], selected_rows: Sequence[dict], output: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    pairs = [
        ("gen_ppl", "rank_wasserstein", "Generative perplexity", "Rank-Wasserstein"),
        ("unigram_entropy", "rank_wasserstein", "Unigram entropy", "Rank-Wasserstein"),
        ("unigram_entropy", "gen_ppl", "Unigram entropy", "Generative perplexity"),
    ]
    colors = {"mdlm": "#4c78a8", "candi": "#f58518", "duo": "#54a24b", "ar": "#e45756"}
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ax, (x_metric, y_metric, xlabel, ylabel) in zip(axes, pairs):
        for method, color in colors.items():
            method_rows = [row for row in all_rows if row["method"] == method]
            ax.scatter(
                [float(row[x_metric]) for row in method_rows],
                [float(row[y_metric]) for row in method_rows],
                s=13,
                alpha=0.25,
                color=color,
                label=method.upper(),
            )
        for row in selected_rows:
            if row["source_group"] == "parameter_free":
                ax.scatter(
                    [float(row[x_metric])],
                    [float(row[y_metric])],
                    marker="X",
                    s=70,
                    edgecolor="black",
                    linewidth=0.4,
                    label=row["display_name"],
                )
            elif row["method"] in colors:
                ax.scatter(
                    [float(row[x_metric])],
                    [float(row[y_metric])],
                    s=35,
                    facecolor=colors[str(row["method"])],
                    edgecolor="black",
                    linewidth=0.45,
                )
        if x_metric == "gen_ppl":
            ax.set_xscale("log")
        if y_metric in {"gen_ppl", "rank_wasserstein"}:
            ax.set_yscale("log")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.25, which="both")
    axes[0].legend(fontsize=7, ncols=2)
    fig.suptitle("All configurations and selected best-rank points")
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_parameter_free_summary(rows: Sequence[dict], output: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    baseline_rows = [row for row in rows if row["source_group"] == "parameter_free"]
    baseline_rows.sort(key=lambda row: str(row["display_name"]))
    metrics = [
        ("unigram_entropy", "Unigram entropy"),
        ("gen_ppl", "Generative perplexity"),
        ("rank_wasserstein", "Rank-Wasserstein"),
    ]
    colors = ["#b279a2", "#ff9da6", "#9d755d", "#bab0ac"]
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.5))
    names = [str(row["display_name"]) for row in baseline_rows]
    for ax, (metric, ylabel) in zip(axes, metrics):
        ax.bar(names, [float(row[metric]) for row in baseline_rows], color=colors)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
        if metric in {"gen_ppl", "rank_wasserstein"}:
            ax.set_yscale("log")
    fig.suptitle("Parameter-free samplers: 128 generations each")
    fig.tight_layout()
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def write_markdown(path: Path, selected_rows: Sequence[dict]) -> None:
    lines = [
        "# All-method evaluation report",
        "",
        "The selected rows below minimize rank-Wasserstein within each method/NFE.",
        "",
        "| Method | NFE | Temperature | Entropy | Gen-PPL | Rank-W |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in selected_rows:
        temperature = row.get("temperature_label") or "—"
        lines.append(
            f"| {row['display_name']} | {row['nfe']} | {temperature} | "
            f"{float(row['unigram_entropy']):.4g} | {float(row['gen_ppl']):.4g} | "
            f"{float(row['rank_wasserstein']):.4g} |"
        )
    lines.extend(
        [
            "",
            "## Figures",
            "",
            "- `all_metrics_efficiency_summary.png`",
            "- `all_metric_tradeoffs.png`",
            "- `rank_wasserstein_efficiency_pareto.png`",
            "- `parameter_free_metrics_summary.png`",
            "- `summaries/`",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_rows = normalize_rows(
        args.diffusion_metrics, args.ar_metrics, args.baseline_metrics, args.ar_nfe
    )
    selected_rows = select_best_rank_rows(all_rows)
    write_csv(args.output_dir / "all_metrics.csv", all_rows)
    write_csv(args.output_dir / "best_rank_configurations.csv", selected_rows)

    plot_efficiency_summary(
        selected_rows,
        args.output_dir / "all_metrics_efficiency_summary.png",
        args.dpi,
    )
    plot_tradeoffs(
        all_rows,
        selected_rows,
        args.output_dir / "all_metric_tradeoffs.png",
        args.dpi,
    )
    plot_parameter_free_summary(
        selected_rows,
        args.output_dir / "parameter_free_metrics_summary.png",
        args.dpi,
    )

    diffusion_grouped = rows_by_method(load_sweep_rows(args.diffusion_metrics))
    ar_grouped = rows_by_method(load_sweep_rows(args.ar_metrics))
    for row in ar_grouped.get("ar", []):
        row["nfe"] = args.ar_nfe
    for method, method_rows in (diffusion_grouped | ar_grouped).items():
        plot_method_summary(
            method,
            method_rows,
            args.output_dir,
            formats=["png"],
            dpi=args.dpi,
        )

    diffusion_points = best_by_method_nfe(load_rows(args.diffusion_metrics))
    ar_point = best_ar_point(load_rows(args.ar_metrics), plotted_nfe=args.ar_nfe)
    baseline_points = load_parameter_free_points(args.baseline_metrics)
    pareto_points = diffusion_points + [ar_point] + baseline_points
    frontier = pareto_frontier(pareto_points)
    pareto_path = args.output_dir / "rank_wasserstein_efficiency_pareto.png"
    plot_pareto(pareto_points, frontier, pareto_path, args.dpi)
    write_pareto_summary(
        args.output_dir / "rank_wasserstein_efficiency_pareto_points.csv",
        pareto_points,
        frontier,
    )
    write_markdown(args.output_dir / "README.md", selected_rows)
    print(f"Wrote consolidated report to {args.output_dir}")


if __name__ == "__main__":
    main()
