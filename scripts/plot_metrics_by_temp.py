#!/usr/bin/env python3
"""Plot divergences against temperature for autoregressive models."""

from __future__ import annotations
import argparse
import math
from pathlib import Path
from typing import Sequence
import pandas as pd
import matplotlib.pyplot as plt

COL_NAME_TO_LABEL = {
    "model_entropy": "Model entropy (nats)",
    "unigram_entropy": "Unigram entropy (nats)",
    "gen_ppl": "Generative perplexity",
    "rank_wasserstein": "Rank-Wasserstein",
    "chi_square_bin20log5": "Chi-square (20+5 bins)",
    "kl_bin20log5": "KL (20+5 bins)",
    "js_bin20log5": "Jensen-Shannon (20+5 bins)",
    "mauve": "MAUVE",
    "gm": "Gradient Moment",
    "energy_distance": "Energy distance D²_E",
    "fmtyp_p": "FMTyp-p",
    "chi2": "Chi-square (25+10 bins)",
    "max_ratio": "Max-ratio (autoresearch, 15+15 bins)",
    "powmean": "Power-mean t=300 (autoresearch, 15+15 bins)",
    "trimmed_chi2": "Trimmed chi-square (drop top-2, 25+10 bins)",
    "topm_logratio_m3": "Top-m log-ratio (m=3, 15+15 bins)",
    "w1_log": "W1 (log ranks)",
    "w1_linear": "W1 (linear ranks)",
    "w1_sqrt": "W1 (sqrt ranks)",
}
#Metrics tu use log scale
LOG_SCALE = {
    "gen_ppl", "rank_wasserstein", "chi_square_bin20log5",
    "kl_bin20log5", "js_bin20log5", "chi2", "max_ratio", "powmean",
    "trimmed_chi2", "topm_logratio_m3", "w1_log", "w1_linear", "w1_sqrt"
}

DEFAULT_MODELS = [
    "openai-community_gpt2",
    "openai-community_gpt2-medium",
    "openai-community_gpt2-large",
    "openai-community_gpt2-xl",
]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("metrics_csv", type=Path, help="divergence_metrics.csv to plot")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("--output-dir", type=Path, default=None, help="Default: divergence_metrics.csv parent/metric_plots")
    parser.add_argument("--stability-csv", type=Path, default=None, help="stability.csv from evaluate_ranking_stability")
    parser.add_argument("--stability-sample-size", type=int, default=2048,help="Row of the stability CSV to read the noise floor from")
    parser.add_argument("--error-bars", choices=("none", "std", "sem"), default="none")
    parser.add_argument("--ncols", type=int, default=4)
    return parser.parse_args(argv)


def plot_grid(
    results_table: pd.DataFrame,
    model_names: Sequence[str],
    metric_columns: Sequence[str],
    figure_path: Path,
    max_columns: int,
    error_bar_mode: str = "none"
):

    columns = min(max_columns, len(metric_columns))
    rows = math.ceil(len(metric_columns) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows))
    axes = axes.flatten()
    #One legend for the whole figure.
    legend_handles = []
    legend_labels = []
    error_bars_drawn = False
    for ax, metric_name in zip(axes, metric_columns):
        error_column = f"{metric_name}_err" if f"{metric_name}_err" in results_table.columns else None
        needed_columns = ["temperature", metric_name]
        if error_column:
            needed_columns.append(error_column)

        for model_name in model_names:
            model_results = results_table[results_table["method"] == model_name][needed_columns]
            model_results = model_results.sort_values("temperature")

            model_label = model_name.split("_", 1)[-1]
            error_bars = None
            if error_column is not None:
                upper_error = model_results[error_column]
                lower_error = upper_error
                #Avoid negative lower error
                if metric_name in LOG_SCALE:
                    lower_error = lower_error.where(model_results[metric_name] - lower_error > 0)

                error_bars = [lower_error, upper_error]
                error_bars_drawn = True

            handle = ax.errorbar(
                model_results["temperature"], model_results[metric_name], yerr=error_bars,
                marker="o", markersize=4, linewidth=1.4, label=model_label,
            )
            #Add only first plot
            if model_label not in legend_labels:
                legend_handles.append(handle)
                legend_labels.append(model_label)

        ax.set_title(COL_NAME_TO_LABEL.get(metric_name, metric_name), fontsize=11)
        ax.set_xlabel("Temperature")
        ax.grid(alpha=0.25)

        if metric_name in LOG_SCALE:
            ax.set_yscale("log")

    for unused_axes in axes[len(metric_columns):]:
        unused_axes.axis("off")

    error_bar_desc = {"std": "1 std", "sem": "1 sem"}.get(error_bar_mode)
    figure.legend(
        legend_handles, legend_labels,
        title=f"Model (error bars: +-{error_bar_desc})" if error_bars_drawn else "Model",
        loc="upper center", ncol=max(len(legend_labels), 1), fontsize=9,
    )
    figure.tight_layout()
    figure.savefig(figure_path)
    plt.close(figure)
    
    return error_bars_drawn


def plot_entropy_vs_gen_ppl(results_table, models, output_path):

    fig, ax = plt.subplots(figsize=(8, 7))

    for model in models:
        model_data = results_table[results_table["method"] == model].sort_values("temperature")
        ax.plot(model_data["model_entropy"], model_data["gen_ppl"], marker="o", markersize=3, label=model.split("_", 1)[-1])

    ax.set_xlabel("Model entropy (nats)", fontsize=9)
    ax.set_ylabel("Generative perplexity", fontsize=9)
    ax.set_yscale("log")
    ax.tick_params(labelsize=8)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.savefig(output_path)
    plt.close(fig)


def best_score_table(
    results_table: pd.DataFrame,
    models: Sequence[str],
    metrics: Sequence[str],
    noise: pd.Series,
    sample_size: int,
    error_bars: str = "none",
):

    model_labels = [model.split("_", 1)[-1] for model in models]
    error_label = {"std": "sample std", "sem": "sem of the mean"}.get(error_bars)
    if error_label is not None and any(f"{metric}_err" in results_table.columns for metric in metrics):
        model_labels = [f"{label} (+- 1 {error_label})" for label in model_labels]
    header = [
        "| Measure | " + " | ".join(model_labels)
        + (f" | Noise (mean+2std, n={sample_size})" if not noise.empty else "")
        + f" | Ordering (best to worst) |",
        "|" + "---|" * (len(models) + 2 + (not noise.empty)),
    ]

    lines = header
    for metric in metrics:
        if metric in {"gen_ppl", "model_entropy", "unigram_entropy"}:
            continue
        metric_table =results_table[results_table[metric].notna()]
        best_index = {}
        for model in models:
            model_metric = metric_table[metric_table["method"] == model]
            best_index[model] = model_metric[metric].idxmin()

        ordered_results = sorted(best_index, key=lambda model: float(results_table.at[best_index[model], metric]))
        cells = []
        for model in models:
            index = best_index.get(model)
            cell = f"{results_table.at[index, metric]:.4g}"
            if f"{metric}_err" in results_table.columns:
                cell += f" +- {results_table.at[index, f'{metric}_err']:.3g}"     
            cells.append(f"**{cell}**" if model == ordered_results[0] else cell)

            
        noise_cell = f"{noise[metric]:.4g}" if metric in noise.index else "-"
        ordering = " > ".join(model.split("_", 1)[-1] for model in ordered_results)
        lines.append(f"| {metric} | " + " | ".join(cells) + (f" | {noise_cell}" if not noise.empty else "") + f" | {ordering} |")

    return "\n".join(lines) + "\n"


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    metrics_df = pd.read_csv(args.metrics_csv)
    metrics_df = metrics_df[metrics_df["method"].isin(args.models)]
    metrics_cols = [column for column in metrics_df.columns if column in COL_NAME_TO_LABEL]

    if args.error_bars != "none":
        for metric in metrics_cols:
            if f"{metric}_std" not in metrics_df.columns:
                continue
            spread = metrics_df[f"{metric}_std"]
            if args.error_bars == "sem":
                spread = spread / metrics_df["num_samples"].pow(0.5)
            metrics_df[f"{metric}_err"] = spread

    #Load errors from stability.csv
    noise = pd.Series(dtype=float)
    if args.stability_csv is not None:
        stability = pd.read_csv(args.stability_csv)
        rows = stability[stability["sample_size"] == args.stability_sample_size]
        noise = (rows["mean"] + 2 * rows["std"]).set_axis(rows["metric"]) #Use mean + 2 std

    out_dir = args.output_dir or args.metrics_csv.parent / "metric_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    figure = out_dir / "all_metrics.png"
    barred = plot_grid(
        metrics_df, args.models, metrics_cols, figure, args.ncols, args.error_bars
    )
    print(f"saved {figure}" + (f" (+-1 {args.error_bars} error bars)" if barred else ""))

    entropy_figure = out_dir / "model_entropy_vs_gen_ppl.png"
    plot_entropy_vs_gen_ppl(metrics_df, args.models, entropy_figure)
    print(f"saved {entropy_figure}")


    table = out_dir / "min_scores.md"
    table.write_text(
        best_score_table(
            metrics_df, args.models, metrics_cols, noise,
            args.stability_sample_size, args.error_bars,
        )
    )
    print(f"saved {table}")


if __name__ == "__main__":
    main()
