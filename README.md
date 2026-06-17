# Ranking Divergence

Lightweight evaluation utilities for LLM rank-histogram divergences, plus an
OpenWebText analysis pipeline for comparing GPT-2 samples against simple
zero-parameter text baselines.

The core idea is described in [docs/ranking_summary_notes.md](docs/ranking_summary_notes.md):
use a fixed autoregressive reference LM as a ruler. For each real or generated
next-token transition, record the rank the reference model assigns to the
observed next token, build rank histograms for real and generated text, then
compare the histograms with a closed-form Wasserstein-1 distance on the log-rank
axis.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from ranking_divergence import rank_wasserstein

tokenizer = AutoTokenizer.from_pretrained("gpt2-large")
tokenizer.pad_token = tokenizer.eos_token
model = AutoModelForCausalLM.from_pretrained("gpt2-large")

result = rank_wasserstein(
    reference_texts=["Human-written reference text."],
    comparison_texts=["Generated text to evaluate."],
    model=model,
    tokenizer=tokenizer,
)
print(result.distance)
```

## Install

```bash
uv pip install -e .
```

For the OpenWebText analysis scripts:

```bash
uv pip install -e ".[examples]"
```

## What Is Included

- Rank histogram computation under a fixed causal LM reference model.
- Closed-form log-rank Wasserstein distance between normalized rank histograms.
- Rank-Wasserstein convenience wrapper for reference/generated text sets.
- Generative perplexity helpers, including a DUO-style variant that masks padding
  and handles EOS positions similarly to DUO.
- Unigram entropy, average per-sample unigram entropy, Rep-n, and unique n-gram
  ratios.
- Four zero-parameter samplers from [docs/baselines_to_use.pdf](docs/baselines_to_use.pdf):
  Top-k IID, Mirror-k, Periodic-k, and Phrase bank-m.
- OpenWebText split/cache helpers matching the DUO convention.
- An OpenWebText analysis CLI that generates samples, computes metrics, and
  writes CSV/JSON/Markdown/LaTeX/plot artifacts.
- Smoke/full bash scripts for CUDA machines.
- Unit tests for rank distance, samplers, split helpers, n-gram metrics, and CLI
  defaults. The full GPT/OpenWebText run remains a manual integration run.

## OpenWebText Analysis

The main analysis script is [examples/openwebtext_analysis.py](examples/openwebtext_analysis.py).
It follows the DUO OpenWebText conventions:

- sampler/training source: `openwebtext`, split `train[:-100000]`
- held-out reference/evaluation source: `openwebtext`, split `train[-100000:]`
- cache path: `/home/patrick/.cache/discrete_diffusion/owt`
- tokenizer: `gpt2`
- scorer/reference LM: `gpt2-large`
- default sample length/eval cap: `1024`

GPT-2 generation uses:

```text
do_sample=True
temperature=1.0
top_p=1.0
top_k=0
```

GPT-2 is allowed to stop naturally at EOS. The script trims generated token IDs
at the first EOS, so entropy and n-gram metrics do not count padding/EOS tails.
Baseline samplers generate exactly `sample_length` token IDs.

Run a small smoke test on CUDA device `0`:

```bash
./scripts/openwebtext_smoke.sh 0
```

Run a fuller analysis on CUDA device `0`:

```bash
./scripts/openwebtext_full.sh 0
```

Extra CLI args are forwarded after the CUDA device argument:

```bash
./scripts/openwebtext_full.sh 0 --run-name owt-gpt2-baselines --batch-size 1
```

Each analysis run writes artifacts under:

```text
outputs/openwebtext_analysis/<timestamp-or-run-name>/
```

Expected artifacts include generated sample text, generated token IDs, metadata,
`metrics.csv`, `metrics.json`, plots, and Markdown/LaTeX tables.

## Tracking The Original Goal

The original goal in [docs/ranking_summary_notes.md](docs/ranking_summary_notes.md)
is substantially implemented:

- A fixed reference LM is used as the common measuring stick.
- Real and generated text are converted into next-token rank histograms.
- Histograms are normalized before comparison.
- The log-rank Wasserstein distance is implemented with the closed-form
  cumulative-discrepancy formula.
- The implementation uses strict-rank semantics: rank is
  `count(logits > observed_logit) + 1`.
- The implementation batches model calls and vectorizes rank extraction within a
  batch.
- Diffusion/non-AR generators are supported conceptually because the metric only
  needs generated text or token sequences.

The current package is therefore a working prototype of the rank-divergence idea,
with an OpenWebText/GPT-2 analysis harness layered on top. It is not yet a fully
validated paper-grade reproduction.

## TODO / Known Risks

- Validate the OpenWebText pipeline against DUO outputs on a sufficiently large
  run, not just smoke tests.
- Investigate why `gpt2` scored by `gpt2-large` may underperform naive baselines
  in current OpenWebText analysis outputs, both on ranking and gen ppl / entropy. This should not be happening and is
  most likely a bug in generation, token handling, evaluation masking, metric
  aggregation, sample size, or baseline comparability.
- Cache and optionally reuse the held-out reference rank histogram across runs.
- Consider a token-ID-first evaluation mode for rank histograms and gen-PPL to
  avoid decode/re-tokenize drift when the generator and scorer tokenizers match.

## Development

Run tests:

```bash
uv run pytest
```

Check the analysis CLI:

```bash
uv run python examples/openwebtext_analysis.py --help
```
