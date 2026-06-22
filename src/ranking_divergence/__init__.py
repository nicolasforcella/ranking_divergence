"""Lightweight LLM rank-divergence metrics."""

from .baselines import (
    MirrorSampler,
    PhraseBankSampler,
    PeriodicSampler,
    RestrictedMarginalSampler,
    TopKSampler,
    build_phrase_bank,
    token_frequencies,
)
from .metrics import (
    duo_generative_perplexity,
    empirical_entropy,
    generative_perplexity,
    per_sample_unigram_entropy,
    rep_n,
    unique_ngram_ratios,
)
from .evaluation import lexical_metrics, score_token_ids
from .rank import (
    RankDivergenceResult,
    normalize_histogram,
    rank_histogram,
    rank_histogram_from_dataloader,
    rank_wasserstein,
    rank_wasserstein_from_histograms,
)

__all__ = [
    "PhraseBankSampler",
    "PeriodicSampler",
    "RankDivergenceResult",
    "MirrorSampler",
    "RestrictedMarginalSampler",
    "TopKSampler",
    "build_phrase_bank",
    "duo_generative_perplexity",
    "empirical_entropy",
    "generative_perplexity",
    "normalize_histogram",
    "lexical_metrics",
    "per_sample_unigram_entropy",
    "rank_histogram",
    "rank_histogram_from_dataloader",
    "rank_wasserstein",
    "rank_wasserstein_from_histograms",
    "rep_n",
    "score_token_ids",
    "token_frequencies",
    "unique_ngram_ratios",
]
