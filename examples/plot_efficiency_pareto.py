from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence


DEFAULT_DIFFUSION_METRICS = Path(
    "outputs/diffusion_sweep_analysis/mdlm-candi-duo-50k/metrics.csv"
)
DEFAULT_AR_METRICS = Path(
    "outputs/diffusion_sweep_analysis/ar-openwebtext-20260622-004711/metrics.csv"
)
DEFAULT_BASELINE_METRICS = Path(
    "outputs/parameter_free_analysis/owt-128/metrics.csv"
)
PARAMETER_FREE_NAMES = {
    "top_k_iid_64": "Top-k IID",
    "mirror_5000": "Mirror",
    "periodic_400": "Periodic",
    "phrase_bank_5000": "Phrase bank",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot best rank-Wasserstein quality against NFE and its Pareto frontier."
    )
    parser.add_argument("--diffusion-metrics", type=Path, default=DEFAULT_DIFFUSION_METRICS)
    parser.add_argument("--ar-metrics", type=Path, default=DEFAULT_AR_METRICS)
    parser.add_argument("--baseline-metrics", type=Path, default=DEFAULT_BASELINE_METRICS)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "outputs/diffusion_sweep_analysis/pareto/"
            "rank_wasserstein_efficiency_pareto.png"
        ),
    )
    parser.add_argument("--ar-nfe", type=int, default=128)
    parser.add_argument("--dpi", type=int, default=220)
    return parser.parse_args(argv)


def load_rows(path: Path) -> list[dict[str, str | int | float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows: list[dict[str, str | int | float]] = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}.")
    required = {"method", "nfe", "temperature", "temperature_label", "rank_wasserstein"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    for row in rows:
        row["nfe"] = int(row["nfe"])
        row["temperature"] = float(row["temperature"])
        row["rank_wasserstein"] = float(row["rank_wasserstein"])
    return rows


def best_by_method_nfe(
    rows: Sequence[dict[str, str | int | float]],
) -> list[dict[str, str | int | float]]:
    best: dict[tuple[str, int], dict[str, str | int | float]] = {}
    for row in rows:
        key = (str(row["method"]), int(row["nfe"]))
        if key not in best or float(row["rank_wasserstein"]) < float(
            best[key]["rank_wasserstein"]
        ):
            best[key] = dict(row)
    return sorted(best.values(), key=lambda row: (str(row["method"]), int(row["nfe"])))


def best_ar_point(
    rows: Sequence[dict[str, str | int | float]], *, plotted_nfe: int
) -> dict[str, str | int | float]:
    point = dict(min(rows, key=lambda row: float(row["rank_wasserstein"])))
    point["source_nfe"] = point["nfe"]
    point["nfe"] = plotted_nfe
    return point


def load_parameter_free_points(path: Path) -> list[dict[str, str | int | float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    points = []
    for row in rows:
        name = row.get("method", row.get("name", ""))
        if name not in PARAMETER_FREE_NAMES:
            continue
        points.append(
            {
                "method": name,
                "display_name": PARAMETER_FREE_NAMES[name],
                "nfe": 0,
                "temperature": "",
                "temperature_label": "",
                "rank_wasserstein": float(row["rank_wasserstein"]),
                "point_type": "parameter_free",
            }
        )
    if len(points) != len(PARAMETER_FREE_NAMES):
        found = {str(point["method"]) for point in points}
        raise ValueError(
            f"Missing parameter-free baselines in {path}: "
            f"{sorted(set(PARAMETER_FREE_NAMES) - found)}"
        )
    return points


def pareto_frontier(
    points: Sequence[dict[str, str | int | float]],
) -> list[dict[str, str | int | float]]:
    """Return points minimizing both NFE and rank-Wasserstein distance."""

    ordered = sorted(
        points,
        key=lambda point: (int(point["nfe"]), float(point["rank_wasserstein"])),
    )
    frontier: list[dict[str, str | int | float]] = []
    best_distance = float("inf")
    for point in ordered:
        distance = float(point["rank_wasserstein"])
        if distance < best_distance:
            frontier.append(point)
            best_distance = distance
    return frontier


def write_summary(
    path: Path, points: Sequence[dict[str, str | int | float]], frontier: Sequence[dict]
) -> None:
    frontier_keys = {
        (str(point["method"]), int(point["nfe"]), float(point["rank_wasserstein"]))
        for point in frontier
    }
    rows = []
    for point in sorted(points, key=lambda value: (int(value["nfe"]), str(value["method"]))):
        key = (
            str(point["method"]),
            int(point["nfe"]),
            float(point["rank_wasserstein"]),
        )
        rows.append(
            {
                "method": point["method"],
                "nfe": point["nfe"],
                "temperature": point["temperature"],
                "temperature_label": point["temperature_label"],
                "rank_wasserstein": point["rank_wasserstein"],
                "point_type": point.get("point_type", "model"),
                "pareto_optimal": key in frontier_keys,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def plot(points: Sequence[dict], frontier: Sequence[dict], output: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    styles = {
        "mdlm": {"color": "#4c78a8", "marker": "o", "label": "MDLM"},
        "candi": {"color": "#f58518", "marker": "s", "label": "CANDI"},
        "duo": {"color": "#54a24b", "marker": "^", "label": "DUO"},
    }
    efficiencies = [0, 8, 16, 32, 64, 128]
    x_position = {nfe: index for index, nfe in enumerate(efficiencies)}
    fig, (full_ax, zoom_ax) = plt.subplots(1, 2, figsize=(15.5, 5.7), sharex=True)

    methods = sorted({str(point["method"]) for point in points if point["method"] != "ar"})
    methods = [method for method in methods if method in styles]
    temperature_offsets = {"candi": 10, "duo": -14, "mdlm": -14}
    for method in methods:
        method_points = sorted(
            [point for point in points if point["method"] == method],
            key=lambda point: int(point["nfe"]),
        )
        style = styles.get(
            method,
            {"color": None, "marker": "o", "label": method.upper()},
        )
        for ax in (full_ax, zoom_ax):
            ax.plot(
                [x_position[int(point["nfe"])] for point in method_points],
                [float(point["rank_wasserstein"]) for point in method_points],
                color=style["color"],
                marker=style["marker"],
                markersize=7,
                linewidth=1.8,
                label=style["label"],
            )
        for point in method_points:
            zoom_ax.annotate(
                f"T={float(point['temperature']):.3g}",
                (
                    x_position[int(point["nfe"])],
                    float(point["rank_wasserstein"]),
                ),
                xytext=(0, temperature_offsets[method]),
                textcoords="offset points",
                ha="center",
                fontsize=7.5,
                color=style["color"],
            )

    ar_points = [point for point in points if point["method"] == "ar"]
    if ar_points:
        ar = ar_points[0]
        for ax in (full_ax, zoom_ax):
            ax.scatter(
                [x_position[int(ar["nfe"])]],
                [float(ar["rank_wasserstein"])],
                marker="*",
                s=260,
                color="#e45756",
                edgecolor="black",
                linewidth=0.8,
                zorder=5,
                label="AR",
            )
            ax.annotate(
                f"AR, T={float(ar['temperature']):.3g}",
                (x_position[int(ar["nfe"])], float(ar["rank_wasserstein"])),
                xytext=(-8, 12),
                textcoords="offset points",
                ha="right",
                fontsize=9,
            )

    baseline_markers = ["X", "P", "v", "h"]
    baseline_colors = ["#b279a2", "#ff9da6", "#9d755d", "#bab0ac"]
    baseline_points = sorted(
        [point for point in points if point.get("point_type") == "parameter_free"],
        key=lambda point: float(point["rank_wasserstein"]),
    )
    baseline_offsets = {
        "Periodic": -8,
        "Top-k IID": 10,
        "Phrase bank": 16,
        "Mirror": 0,
    }
    for point, marker, color in zip(baseline_points, baseline_markers, baseline_colors):
        full_ax.scatter(
            [x_position[0]],
            [float(point["rank_wasserstein"])],
            marker=marker,
            s=95,
            color=color,
            edgecolor="black",
            linewidth=0.5,
            label=str(point["display_name"]),
            zorder=4,
        )
        full_ax.annotate(
            str(point["display_name"]),
            (x_position[0], float(point["rank_wasserstein"])),
            xytext=(9, baseline_offsets[str(point["display_name"])]),
            textcoords="offset points",
            va="center",
            fontsize=8,
        )

    frontier = sorted(frontier, key=lambda point: int(point["nfe"]))
    full_ax.plot(
        [x_position[int(point["nfe"])] for point in frontier],
        [float(point["rank_wasserstein"]) for point in frontier],
        color="black",
        linestyle="--",
        linewidth=2.2,
        marker="D",
        markersize=5,
        label="Global Pareto frontier",
        zorder=4,
    )
    model_frontier = [point for point in frontier if int(point["nfe"]) > 0]
    zoom_ax.plot(
        [x_position[int(point["nfe"])] for point in model_frontier],
        [float(point["rank_wasserstein"]) for point in model_frontier],
        color="black",
        linestyle="--",
        linewidth=2.2,
        marker="D",
        markersize=5,
        label="Global Pareto frontier",
        zorder=4,
    )
    for ax in (full_ax, zoom_ax):
        ax.set_xticks(range(len(efficiencies)))
        ax.set_xticklabels(["0\nparameter-free", "8", "16", "32", "64", "128"])
        ax.set_xlabel("NFE (lower is more efficient)")
        ax.grid(alpha=0.28, which="both")

    full_ax.set_yscale("log")
    full_ax.set_ylabel("Minimum rank-Wasserstein over temperature")
    full_ax.set_title("Full comparison (log scale)")
    full_ax.legend(fontsize=8, ncols=2)
    zoom_ax.set_ylim(0, 0.27)
    zoom_ax.set_title("Model sweep detail (labels show best temperature)")
    zoom_ax.legend(fontsize=8)
    fig.suptitle("Generation efficiency vs. best rank-distribution match", fontsize=15)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    diffusion_points = best_by_method_nfe(load_rows(args.diffusion_metrics))
    ar_point = best_ar_point(load_rows(args.ar_metrics), plotted_nfe=args.ar_nfe)
    baseline_points = load_parameter_free_points(args.baseline_metrics)
    points = diffusion_points + [ar_point] + baseline_points
    frontier = pareto_frontier(points)

    plot(points, frontier, args.output, args.dpi)
    summary_path = args.output.with_name(f"{args.output.stem}_points.csv")
    write_summary(summary_path, points, frontier)
    print(f"Wrote Pareto plot to {args.output}")
    print(f"Wrote selected points to {summary_path}")


if __name__ == "__main__":
    main()
