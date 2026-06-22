# Diffusion Sweep Analysis

`examples/evaluate_diffusion_sweeps.py` evaluates the saved MDLM, CANDI, and
DUO temperature-by-NFE sweeps. The three supplied directories are the script's
defaults, and their method labels are retained in every result row.

Run the full evaluation on a CUDA device:

```bash
uv run python examples/evaluate_diffusion_sweeps.py \
  --run-name mdlm-candi-duo-50k \
  --device cuda \
  --batch-size 1
```

The run is resumable: invoke the same command with the same `--run-name`, and
completed configuration checkpoints will be skipped. Use `--force` to recompute
them. A smaller end-to-end smoke run can be launched with:

```bash
uv run python examples/evaluate_diffusion_sweeps.py \
  --run-name sweep-smoke \
  --device cuda \
  --limit-configs 1 \
  --limit-samples 2 \
  --num-reference 2 \
  --max-length 64
```

To inspect available and missing sweep points without loading a model:

```bash
uv run python examples/evaluate_diffusion_sweeps.py \
  --run-name sweep-inventory \
  --inventory-only
```

Each run writes under `outputs/diffusion_sweep_analysis/<run-name>/`:

- `inventory.json`: discovered NFE/temperature grids and missing points.
- `metadata.json`: scorer, reference split, and evaluation settings.
- `reference_rank_histogram.pt`: cached held-out OpenWebText rank histogram.
- `checkpoints/*.json`: one resumable result per method/NFE/temperature.
- `metrics.csv` and `metrics.json`: consolidated results.

Every metric row includes `method`, `nfe`, the exact payload `temperature`, the
filename's rounded `temperature_label`, source path, sample count, source-file
PPL/entropy, recomputed generative PPL and entropy, rank-Wasserstein distance,
unique 1- through 4-gram ratios, Rep-1 through Rep-3, and token-count
diagnostics.

Additional or replacement sweeps can be supplied with repeatable
`--sweep METHOD=DIR` arguments.

## Plotting a completed sweep

Generate per-method temperature curves, three-panel summaries, and NFE by
temperature heatmaps for entropy, generative perplexity, and rank-Wasserstein:

```bash
uv run python examples/plot_diffusion_sweeps.py
```

By default this reads
`outputs/diffusion_sweep_analysis/mdlm-candi-duo-50k/metrics.csv` and writes
plots beside it under `plots/`. Generative-perplexity plots use a logarithmic
scale. Different input/output locations and multiple formats can be requested:

```bash
uv run python examples/plot_diffusion_sweeps.py \
  --metrics-csv outputs/diffusion_sweep_analysis/my-run/metrics.csv \
  --output-dir outputs/diffusion_sweep_analysis/my-run/plots \
  --formats png pdf
```

Compare the best temperature at every NFE across MDLM, CANDI, and DUO, with
the best AR result shown as a star at NFE 128 and the parameter-free samplers
shown at zero model NFE:

```bash
uv run python examples/plot_efficiency_pareto.py
```

Every model point is annotated with its selected temperature. The full panel
uses a logarithmic y-axis so the parameter-free baselines remain visible, while
a second panel gives a linear-scale view of the model frontier. A companion CSV
contains every selected point and a boolean indicating whether it lies on the
global efficiency/Rank-Wasserstein Pareto frontier.

## Matched parameter-free baselines and consolidated report

Generate 128 samples from each parameter-free sampler and score them against
the same cached reference histogram used by the diffusion sweep:

```bash
uv run python examples/evaluate_parameter_free_baselines.py \
  --device cuda \
  --num-samples 128
```

Generate dedicated entropy, gen-PPL, rank-distance, tradeoff, repetition, and
unique n-gram plots for those samplers:

```bash
uv run python examples/plot_parameter_free_baselines.py
```

Then build a single report containing normalized metrics, best-rank
configurations, an entropy/gen-PPL/rank efficiency summary, metric tradeoff
plots, and the Pareto frontier:

```bash
uv run python examples/build_all_methods_report.py
```
