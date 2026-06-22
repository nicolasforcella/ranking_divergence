# Important Results

## What rank-Wasserstein is measuring

The rank-Wasserstein metric compares generated text with held-out OpenWebText
using GPT-2-large as a fixed reference model. For every observed next token, it
records the token's rank under the reference model and compares the resulting
rank distribution with the human-text distribution. Lower values mean a closer
match.

The experiments show that this metric is not minimized simply by making text
more predictable or more diverse. It favors an intermediate regime whose token
rank statistics resemble real text.

## Autoregressive temperature sweep

The autoregressive sweep gives the clearest demonstration. Rank-Wasserstein is
strongly V-shaped as temperature changes:

| Temperature | Entropy | Gen-PPL | Rank-W | Rep-3 |
| ---: | ---: | ---: | ---: | ---: |
| 0.60 | 4.119 | 2.501 | 0.978 | 0.593 |
| **0.90** | **5.271** | **13.96** | **0.0261** | **0.116** |
| 1.00 | 5.591 | 34.20 | 0.610 | 0.0526 |

At low temperature, the model is highly predictable but excessively
repetitive: Rep-3 is about 0.59 and the rank distance is poor. Increasing the
temperature reduces repetition and moves the rank distribution much closer to
human text. The optimum occurs near temperature 0.90. Raising temperature
further continues to reduce repetition, but generative perplexity rises and
the rank distance worsens again.

This is useful evidence that rank-Wasserstein detects both sides of the
quality problem:

- overly concentrated, repetitive generation at low temperature;
- overly diffuse or unlikely token choices at high temperature.

The metric therefore does penalize repetitive collapse, but it does not equate
low repetition with quality. The best match lies between repetition and
unstructured diversity.

## MDLM, CANDI, and DUO sweeps

The diffusion models show the same broad temperature trend. Low temperatures
produce low-entropy, repetitive text; high temperatures sharply increase
generative perplexity and rank distance. Each method has an intermediate
temperature band with the closest rank-distribution match.

The preferred temperature is also fairly stable across NFE:

- DUO is best around temperature 0.625-0.650.
- CANDI is best around temperature 0.624-0.690.
- MDLM shifts from approximately 0.624 at NFE 8 to 0.690-0.755 at larger NFE.

The best rank-Wasserstein values at each NFE are:

| Method | NFE 8 | NFE 16 | NFE 32 | NFE 64 | NFE 128 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MDLM | 0.238 | 0.104 | 0.130 | 0.116 | 0.088 |
| CANDI | 0.241 | 0.165 | 0.152 | 0.124 | 0.139 |
| DUO | **0.098** | **0.0546** | **0.0239** | **0.0500** | **0.0246** |

DUO gives the strongest rank-distribution match throughout the NFE sweep.
Its NFE-32 result is the overall minimum, narrowly beating its NFE-128 result
and the best AR result. MDLM generally improves with additional NFE, although
the trend is not perfectly monotonic. CANDI improves through NFE 64 and then
slightly worsens at NFE 128.

These non-monotonic results matter: more sampling steps do not automatically
produce a better rank match. Temperature selection and sampling method can
matter more than raw NFE.

## Parameter-free samplers

The parameter-free samplers were reevaluated using 128 generations each and the
same reference rank histogram:

| Sampler | Entropy | Gen-PPL | Rank-W |
| --- | ---: | ---: | ---: |
| Periodic | 5.972 | 21.80 | 1.092 |
| Top-k IID | 3.561 | 146.0 | 1.424 |
| Phrase bank | 4.970 | 122.9 | 1.435 |
| Mirror | 5.153 | 1476 | 3.508 |

Periodic is the closest parameter-free baseline, but it remains roughly an
order of magnitude worse than the best learned-model configurations. Its high
Rep-n values also show that plausible entropy or perplexity alone does not
guarantee realistic sequential structure. Mirror is especially poor under both
generative perplexity and rank-Wasserstein.

## Final efficiency comparison

The final comparison chooses the lowest-rank-distance temperature separately
for every method and NFE. AR is plotted at the requested comparison cost of
NFE 128.

The global learned-model Pareto frontier is:

| Method | NFE | Temperature | Rank-W |
| --- | ---: | ---: | ---: |
| DUO | 8 | 0.625 | 0.0983 |
| DUO | 16 | 0.625 | 0.0546 |
| DUO | 32 | 0.625 | **0.0239** |

DUO at NFE 32 dominates the higher-cost points in this two-objective view. In
particular:

- AR at assigned NFE 128 reaches 0.0261.
- DUO at NFE 128 reaches 0.0246.
- DUO at NFE 32 reaches 0.0239.

Thus the best observed rank match is achieved by DUO at NFE 32, with slightly
better distance than AR at one quarter of the assigned NFE. AR nevertheless
provides an important validation result: its clean V-shaped temperature curve
shows that the metric identifies a narrow realistic operating regime rather
than mechanically rewarding low perplexity or maximal diversity.

## Where to find the figures

The consolidated tables and figures are generated under
`outputs/all_methods_report/`, including:

- `all_metrics_efficiency_summary.png`
- `all_metric_tradeoffs.png`
- `rank_wasserstein_efficiency_pareto.png`
- `parameter_free_metrics_summary.png`
- `summaries/`

Dedicated parameter-free plots are under
`outputs/parameter_free_analysis/owt-128/plots/`.

The numerical results should be interpreted as comparisons under the current
GPT-2-large/OpenWebText evaluation setup. Model sweeps use 256 generations per
configuration, while the parameter-free baselines use 128.
