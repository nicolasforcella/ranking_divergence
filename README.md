# Ranking Divergence

Lightweight evaluation utilities for LLM rank-histogram divergences.

The main metric follows `notes.pdf`: given a reference causal language model,
compute the histogram of the rank assigned to each observed next token, then
compare two histograms with the closed-form one-dimensional Wasserstein-1
distance on the log-rank axis.

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

For the OpenWebText example:

```bash
uv pip install -e ".[examples]"
```

## Included Metrics

- Rank-Wasserstein divergence over next-token rank histograms
- Generative perplexity under a fixed causal LM scorer
- Empirical unigram entropy
- Rep-n repetition score
- Four zero-parameter samplers from `baselines_to_use.pdf`: Top-k IID,
  Mirror-k, Periodic-k, and Phrase bank-m

## Example

The default example is intentionally small. Increase `--num-reference`,
`--num-samples`, and `--sample-length` for larger OpenWebText runs.

```bash
uv run python examples/evaluate_gpt2_openwebtext.py \
  --scorer-model gpt2-large \
  --generator-model gpt2 \
  --num-reference 32 \
  --num-sampler-source 2048 \
  --num-samples 16 \
  --sample-length 128
```
