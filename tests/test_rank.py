import math

import torch
import pytest

from ranking_divergence import (
    MirrorSampler,
    RestrictedMarginalSampler,
    empirical_entropy,
    rank_wasserstein_from_histograms,
    rep_n,
)


def test_rank_wasserstein_identical_histograms_is_zero():
    hist = torch.tensor([3.0, 2.0, 1.0])
    assert rank_wasserstein_from_histograms(hist, hist) == 0.0


def test_rank_wasserstein_matches_closed_form():
    ref = torch.tensor([1.0, 0.0, 0.0])
    cmp = torch.tensor([0.0, 1.0, 0.0])
    assert rank_wasserstein_from_histograms(ref, cmp) == math.log(2.0)


def test_rep_n_repetition_score():
    assert rep_n("", token_ids=[[1, 2, 1, 2]], n=2) == pytest.approx(1.0 / 3.0)


def test_empirical_entropy_uses_nats():
    value = empirical_entropy("", token_ids=[[1, 1, 2, 2]])
    assert value == pytest.approx(math.log(2.0))


def test_mirror_sampler_returns_requested_odd_length():
    class Tokenizer:
        def batch_decode(self, samples, skip_special_tokens=True):
            return [" ".join(map(str, sample)) for sample in samples]

    base = RestrictedMarginalSampler(
        torch.tensor([1, 2]),
        torch.tensor([0.5, 0.5], dtype=torch.float64),
        Tokenizer(),
    )
    sample = MirrorSampler(base).sample_token_ids(num_samples=1, length=5, seed=0)[0]
    assert len(sample) == 5
