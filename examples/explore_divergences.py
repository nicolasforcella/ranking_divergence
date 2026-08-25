from __future__ import annotations

import argparse
import csv
import math
import statistics
from pathlib import Path
from typing import Sequence

import torch

from ranking_divergence import DIVERGENCES, compute_all
from plot_efficiency_pareto import best_by_method_nfe, pareto_frontier

IDENTIFIER_COLUMNS = {
    "method",
    "display_name",
    "name",
    "nfe",
    "temperature",
    "temperature_label",
    "source_file",
    "source_run",
    "histogram_key",
    "point_type",
    "num_samples",
    "empty_samples",
    "min_tokens",
    "max_tokens",
}
TARGET_PREFIXES = ("unique_", "rep_")
TARGET_COLUMNS = {
    "rank_wasserstein",
    "gen_ppl",
    "source_gen_ppl",
    "source_entropy",
    "model_entropy",
    "unigram_entropy",
    "mean_tokens",
    "mauve",
    "gm",
    "energy_distance",
    "fmtyp_p",
    "rep_1",
    "rep_2",
    "rep_3",
}
STRING_COLUMNS = {
    "method",
    "display_name",
    "name",
    "temperature_label",
    "source_file",
    "source_run",
    "histogram_key",
    "point_type",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute alternative divergences from saved rank histograms and score them."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--baseline-run", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/divergence_exploration"))
    parser.add_argument("--output-name", default=None)
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--per-sample", help="Indicates that histograms where computed per sample.", action="store_true")
    return parser.parse_args(argv)


def numeric(value) -> float | None:
    if value in {None, ""}:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def read_csv(path: Path) -> list[dict[str, str | int | float]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows: list[dict[str, str | int | float]] = list(csv.DictReader(handle))
    for row in rows:
        for key, value in list(row.items()):
            if key in STRING_COLUMNS:
                continue
            number = numeric(value)
            if number is not None:
                if key == "nfe":
                    row[key] = int(number)
                else:
                    row[key] = number
    return rows


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def histogram_candidates(row: dict) -> list[str]:
    candidates: list[str] = []
    method = str(row.get("method", row.get("name", "")))
    source_file = row.get("source_file")
    if method and source_file:
        candidates.append(f"{method}__{Path(str(source_file)).stem}")
    if source_file:
        candidates.append(Path(str(source_file)).stem)
    if method:
        candidates.append(method)
    name = row.get("name")
    if name:
        candidates.append(str(name))
    if method and row.get("nfe", "") != "" and row.get("temperature_label", "") != "":
        candidates.append(f"{method}__samples_steps{int(row['nfe'])}_temp{row['temperature_label']}")
    return list(dict.fromkeys(candidates))


def load_histogram(path: Path) -> torch.Tensor:
    return torch.load(path, map_location="cpu", weights_only=True)


def densify(histogram: torch.Tensor) -> torch.Tensor:
    """Per-sample histograms are saved CSR-compressed; older runs are dense."""

    return histogram if histogram.layout == torch.strided else histogram.to_dense()


def merge_distributional_metrics(rows: list[dict], run_dir: Path) -> None:
    """Attach MAUVE/GM (from evaluate_distributional_metrics.py) onto rows by histogram key."""

    path = run_dir / "distributional_metrics.csv"
    if not path.exists():
        return
    lookup = {r["key"]: r for r in read_csv(path)}
    matched = 0
    for row in rows:
        extra = lookup.get(str(row.get("histogram_key")))
        if extra is None:
            continue
        matched += 1
        for column in ("mauve", "gm"):
            value = numeric(extra.get(column))
            if value is not None:
                row[column] = value
    print(f"Attached distributional metrics (MAUVE/GM) to {matched}/{len(rows)} rows from {path.name}")


def merge_de_fmtyp(rows: list[dict], run_dir: Path) -> None:
    """Attach DE / FMTyp-p (from evaluate_de_fmtyp.py) onto rows by histogram key."""

    path = run_dir / "de_fmtyp_metrics.csv"
    if not path.exists():
        return
    lookup = {r["key"]: r for r in read_csv(path)}
    matched = 0
    for row in rows:
        extra = lookup.get(str(row.get("histogram_key")))
        if extra is None:
            continue
        matched += 1
        for column in ("energy_distance", "fmtyp_p"):
            value = numeric(extra.get(column))
            if value is not None:
                row[column] = value
    print(f"Attached DE/FMTyp-p to {matched}/{len(rows)} rows from {path.name}")


def rows_with_divergences(run_dir: Path, reference: torch.Tensor, per_sample: bool=False) -> list[dict]:
    metrics_path = run_dir / "metrics.csv"
    histogram_dir = run_dir / "histograms"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing metrics CSV: {metrics_path}")
    if not histogram_dir.is_dir():
        raise FileNotFoundError(f"Missing histogram directory: {histogram_dir}")

    rows = []
    for row in read_csv(metrics_path):
        histogram_path = None
        histogram_key = None
        for candidate in histogram_candidates(row):
            candidate_path = histogram_dir / f"{candidate}.pt"
            if candidate_path.exists():
                histogram_path = candidate_path
                histogram_key = candidate
                print(f"Evaluating: {candidate_path}")
                break
        if histogram_path is None:
            raise FileNotFoundError(
                f"No histogram found for row in {metrics_path}: tried {histogram_candidates(row)}"
            )
        comparison = load_histogram(histogram_path)
        enriched = dict(row)
        enriched["source_run"] = run_dir.name
        enriched["histogram_key"] = histogram_key
        if per_sample:
            enriched["per_sample"] = True
            final_result_keys = None
            scored = []
            for sample_idx in range(comparison.shape[0]):
                sample = densify(comparison[sample_idx])
                if float(sample.sum()) <= 0:
                    # Empty generation: no rank histogram to score. Counted in empty_samples.
                    continue
                all_metrics = compute_all(reference, sample)
                if final_result_keys == None:
                    final_result_keys = all_metrics.keys()
                scored.append(list(all_metrics.values()))

            if not scored:
                raise ValueError(f"Every sample is empty in {histogram_path}")
            final_result_values = torch.tensor(scored, dtype=torch.float64)
            enriched["per_sample_scored"] = len(scored)
            means = final_result_values.mean(axis=0).tolist()
            std = final_result_values.std(axis=0).tolist()
            final_result_keys_std = [name + "_std" for name in final_result_keys]
            enriched.update(dict(zip(final_result_keys, means)))
            enriched.update(dict(zip(final_result_keys_std, std)))
        else:
            enriched.update(compute_all(reference, densify(comparison)))
        rows.append(enriched)
    return rows


def rank_values(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        stop = start + 1
        while stop < len(order) and values[order[stop]] == values[order[start]]:
            stop += 1
        average = 0.5 * (start + stop - 1) + 1.0
        for idx in order[start:stop]:
            ranks[idx] = average
        start = stop
    return ranks


def pearson(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    if len(xs) < 2:
        return None
    x_mean = sum(xs) / len(xs)
    y_mean = sum(ys) / len(ys)
    x_centered = [x - x_mean for x in xs]
    y_centered = [y - y_mean for y in ys]
    denominator = math.sqrt(sum(x * x for x in x_centered) * sum(y * y for y in y_centered))
    if denominator == 0:
        return None
    return sum(x * y for x, y in zip(x_centered, y_centered)) / denominator


def spearman(xs: Sequence[float], ys: Sequence[float]) -> float | None:
    return pearson(rank_values(xs), rank_values(ys))


def target_columns(rows: Sequence[dict]) -> list[str]:
    columns: list[str] = []
    for row in rows:
        for key, value in row.items():
            if key in DIVERGENCES or key in IDENTIFIER_COLUMNS:
                continue
            if key in TARGET_COLUMNS or key.startswith(TARGET_PREFIXES):
                if numeric(value) is not None and key not in columns:
                    columns.append(key)
    return columns


def correlation_rows(rows: Sequence[dict], divergence_names: Sequence[str]) -> list[dict]:
    targets = target_columns(rows)
    output = []
    for divergence in divergence_names:
        for target in targets:
            pairs = [
                (numeric(row.get(divergence)), numeric(row.get(target)))
                for row in rows
            ]
            pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
            if len(pairs) < 2:
                continue
            xs, ys = zip(*pairs)
            output.append(
                {
                    "divergence": divergence,
                    "target": target,
                    "pearson": pearson(xs, ys),
                    "spearman": spearman(xs, ys),
                    "n": len(pairs),
                }
            )
    return output


def eta_squared(rows: Sequence[dict], metric: str) -> tuple[float, float | None]:
    groups: dict[str, list[float]] = {}
    for row in rows:
        value = numeric(row.get(metric))
        method = row.get("method")
        if value is not None and method not in {None, ""}:
            groups.setdefault(str(method), []).append(value)
    values = [value for group in groups.values() for value in group]
    if len(groups) < 2 or len(values) <= len(groups):
        return 0.0, None
    overall = sum(values) / len(values)
    ss_between = sum(len(group) * (sum(group) / len(group) - overall) ** 2 for group in groups.values())
    ss_total = sum((value - overall) ** 2 for value in values)
    ss_within = ss_total - ss_between
    eta = ss_between / ss_total if ss_total > 0 else 0.0
    df_between = len(groups) - 1
    df_within = len(values) - len(groups)
    f_stat = (ss_between / df_between) / (ss_within / df_within) if ss_within > 0 and df_within > 0 else None
    return eta, f_stat


def v_curve_quality(rows: Sequence[dict], metric: str) -> float:
    groups: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        if row.get("temperature") in {None, ""} or row.get("nfe") in {None, ""}:
            continue
        if numeric(row.get(metric)) is None or numeric(row.get("temperature")) is None:
            continue
        groups.setdefault((str(row.get("method")), int(row.get("nfe"))), []).append(row)
    scores = []
    for group in groups.values():
        if len(group) < 3:
            continue
        ordered = sorted(group, key=lambda row: float(row["temperature"]))
        values = [float(row[metric]) for row in ordered]
        min_index = min(range(len(values)), key=lambda idx: values[idx])
        if min_index in {0, len(values) - 1}:
            scores.append(0.0)
            continue
        left_ok = all(values[idx] >= values[idx + 1] for idx in range(min_index))
        right_ok = all(values[idx] <= values[idx + 1] for idx in range(min_index, len(values) - 1))
        scores.append(1.0 if left_ok and right_ok else 0.0)
    return sum(scores) / len(scores) if scores else 0.0


def discrimination_rows(rows: Sequence[dict], divergence_names: Sequence[str]) -> list[dict]:
    output = []
    for divergence in divergence_names:
        eta, f_stat = eta_squared(rows, divergence)
        v_quality = v_curve_quality(rows, divergence)
        output.append(
            {
                "divergence": divergence,
                "eta_squared": eta,
                "anova_f": f_stat,
                "v_curve_quality": v_quality,
                "discrimination_score_raw": 0.5 * eta + 0.5 * v_quality,
            }
        )
    return output


def config_key(row: dict) -> tuple[str, int, str]:
    return (str(row.get("method", "")), int(row.get("nfe", 0)), str(row.get("temperature_label", "")))


def pareto_rows(rows: Sequence[dict], divergence_names: Sequence[str]) -> list[dict]:
    usable = [row for row in rows if row.get("nfe") not in {None, ""}]
    if not any("rank_wasserstein" in row for row in usable):
        return []
    reference_points = best_by_method_nfe(usable, metric_key="rank_wasserstein")
    reference_frontier = pareto_frontier(reference_points, metric_key="rank_wasserstein")
    reference_frontier_keys = {config_key(row) for row in reference_frontier}
    reference_by_group = {(str(row["method"]), int(row["nfe"])): float(row["rank_wasserstein"]) for row in reference_points}

    output = []
    for divergence in divergence_names:
        metric_rows = [row for row in usable if numeric(row.get(divergence)) is not None]
        if not metric_rows:
            continue
        best_points = best_by_method_nfe(metric_rows, metric_key=divergence)
        frontier = pareto_frontier(best_points, metric_key=divergence)
        frontier_keys = {config_key(row) for row in frontier}
        union = reference_frontier_keys | frontier_keys
        jaccard = len(reference_frontier_keys & frontier_keys) / len(union) if union else 0.0
        metric_by_group = {(str(row["method"]), int(row["nfe"])): float(row[divergence]) for row in best_points}
        common = sorted(set(reference_by_group) & set(metric_by_group))
        rank_corr = spearman(
            [reference_by_group[key] for key in common],
            [metric_by_group[key] for key in common],
        ) if len(common) >= 2 else None
        rank_score = 0.5 * (rank_corr + 1.0) if rank_corr is not None else 0.0
        output.append(
            {
                "divergence": divergence,
                "frontier_jaccard": jaccard,
                "best_group_spearman_vs_rank_wasserstein": rank_corr,
                "pareto_score_raw": 0.5 * jaccard + 0.5 * rank_score,
                "frontier_size": len(frontier),
            }
        )
    return output


def normalize_scores(values: dict[str, float]) -> dict[str, float]:
    finite = [value for value in values.values() if math.isfinite(value)]
    if not finite:
        return {key: 0.0 for key in values}
    low = min(finite)
    high = max(finite)
    if high == low:
        return {key: 1.0 for key in values}
    return {key: (value - low) / (high - low) if math.isfinite(value) else 0.0 for key, value in values.items()}


def scorecard_rows(
    divergence_names: Sequence[str],
    correlations: Sequence[dict],
    discriminations: Sequence[dict],
    paretos: Sequence[dict],
) -> list[dict]:
    corr_raw = {name: 0.0 for name in divergence_names}
    for name in divergence_names:
        vals = [abs(float(row["spearman"])) for row in correlations if row["divergence"] == name and row.get("spearman") is not None]
        corr_raw[name] = sum(vals) / len(vals) if vals else 0.0
    discr_raw = {str(row["divergence"]): float(row["discrimination_score_raw"]) for row in discriminations}
    pareto_raw = {str(row["divergence"]): float(row["pareto_score_raw"]) for row in paretos}
    corr = normalize_scores(corr_raw)
    discr = normalize_scores(discr_raw)
    pareto = normalize_scores(pareto_raw)
    rows = []
    for name in divergence_names:
        row = {
            "divergence": name,
            "combined_score": (corr.get(name, 0.0) + discr.get(name, 0.0) + pareto.get(name, 0.0)) / 3.0,
            "correlation_score": corr.get(name, 0.0),
            "discrimination_score": discr.get(name, 0.0),
            "pareto_score": pareto.get(name, 0.0),
            "correlation_score_raw": corr_raw.get(name, 0.0),
            "discrimination_score_raw": discr_raw.get(name, 0.0),
            "pareto_score_raw": pareto_raw.get(name, 0.0),
        }
        rows.append(row)
    return sorted(rows, key=lambda row: float(row["combined_score"]), reverse=True)


def write_markdown(path: Path, rows: Sequence[dict]) -> None:
    lines = ["# Divergence scorecard", "", "| Rank | Divergence | Combined | Corr. | Discr. | Pareto |", "| --- | --- | ---: | ---: | ---: | ---: |"]
    for index, row in enumerate(rows, start=1):
        lines.append(
            f"| {index} | {row['divergence']} | {float(row['combined_score']):.3f} | "
            f"{float(row['correlation_score']):.3f} | {float(row['discrimination_score']):.3f} | "
            f"{float(row['pareto_score']):.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_correlation_heatmap(path: Path, correlations: Sequence[dict]) -> None:
    import matplotlib.pyplot as plt

    divergences = list(DIVERGENCES)
    targets = sorted({str(row["target"]) for row in correlations})
    if not divergences or not targets:
        return
    values = [[0.0 for _ in targets] for _ in divergences]
    for row in correlations:
        if row.get("spearman") is None:
            continue
        values[divergences.index(str(row["divergence"]))][targets.index(str(row["target"]))] = float(row["spearman"])
    fig, ax = plt.subplots(figsize=(max(8, len(targets) * 0.65), max(7, len(divergences) * 0.32)))
    image = ax.imshow(values, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(targets)))
    ax.set_xticklabels(targets, rotation=45, ha="right")
    ax.set_yticks(range(len(divergences)))
    ax.set_yticklabels(divergences)
    fig.colorbar(image, ax=ax, label="Spearman correlation")
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    output_name = args.output_name or args.run_dir.name
    output_dir = args.output_dir / output_name
    reference_path = args.run_dir / "reference_rank_histogram.pt"
    if not reference_path.exists():
        raise FileNotFoundError(f"Missing reference histogram: {reference_path}")
    reference = load_histogram(reference_path)

    rows = rows_with_divergences(args.run_dir, reference, per_sample=args.per_sample)
    for baseline_run in args.baseline_run:
        rows.extend(rows_with_divergences(baseline_run, reference, per_sample=args.per_sample))
    merge_distributional_metrics(rows, args.run_dir)
    merge_de_fmtyp(rows, args.run_dir)

    divergence_names = list(DIVERGENCES)
    correlations = correlation_rows(rows, divergence_names)
    discriminations = discrimination_rows(rows, divergence_names)
    paretos = pareto_rows(rows, divergence_names)
    scorecard = scorecard_rows(divergence_names, correlations, discriminations, paretos)

    write_csv(output_dir / "divergence_metrics.csv", rows)
    write_csv(output_dir / "correlation_matrix.csv", correlations)
    write_csv(output_dir / "discrimination.csv", discriminations)
    write_csv(output_dir / "pareto_agreement.csv", paretos)
    write_csv(output_dir / "divergence_scorecard.csv", scorecard)
    write_markdown(output_dir / "divergence_scorecard.md", scorecard)
    if not args.skip_plots:
        plot_correlation_heatmap(output_dir / "correlation_heatmap.png", correlations)
    print(f"Wrote divergence exploration artifacts to {output_dir}")


if __name__ == "__main__":
    main()
