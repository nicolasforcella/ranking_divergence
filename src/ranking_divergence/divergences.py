"""Alternative divergences/measures between next-token rank histograms.

Every public divergence takes two histograms ``(reference, comparison)`` over ranks
(index 0 = rank 1) and returns a float. ``reference`` is the held-out/real-text
histogram (the "data" side) and ``comparison`` is the generated-text histogram.

Three families are provided:

* binned f-divergences -- coarsen ranks into a few linear head bins plus log-spaced
  tail bins, then apply a standard f-divergence (KL, JS, chi-square, ...);
* Wasserstein-1 with alternative ground costs on the rank axis;
* cheap rank-correlation / tail summaries on the histogram CDFs.

The registry :data:`DIVERGENCES` enumerates every measure so analysis code can iterate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from functools import partial
from typing import Callable

import torch

EPS = 1e-12


def _as_prob(histogram) -> torch.Tensor:
    """Return a normalized float64 CPU probability histogram."""

    tensor = torch.as_tensor(histogram, dtype=torch.float64).detach().cpu().flatten()
    total = tensor.sum()
    if total <= 0:
        raise ValueError("Cannot use an empty histogram.")
    return tensor / total


# ---------------------------------------------------------------------------
# Binning
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BinScheme:
    """Coarse binning over ranks: ``n_linear`` singleton head bins + ``n_log`` log bins.

    The first ``n_linear`` ranks each get their own bin; the remaining tail is split
    into ``n_log`` log-spaced bins spanning ranks ``n_linear+1 .. vocab_size``.
    """

    n_linear: int = 20
    n_log: int = 5

    def boundaries(self, vocab_size: int) -> list[int]:
        """0-based exclusive bin boundaries ``0 = b0 < b1 < ... < bm = vocab_size``."""

        n_linear = min(self.n_linear, vocab_size)
        bounds = list(range(0, n_linear + 1))  # singleton head bins
        if n_linear < vocab_size and self.n_log > 0:
            log_points = torch.logspace(
                math.log10(max(n_linear, 1)), math.log10(vocab_size), self.n_log + 1
            ).tolist()
            for point in log_points:
                index = max(n_linear, min(vocab_size, int(round(point))))
                if index > bounds[-1]:
                    bounds.append(index)
        if bounds[-1] != vocab_size:
            bounds.append(vocab_size)
        return bounds


DEFAULT_SCHEME = BinScheme(n_linear=20, n_log=5)


def coarsen_histogram(histogram, scheme: BinScheme = DEFAULT_SCHEME) -> torch.Tensor:
    """Aggregate a full-length rank histogram into the scheme's coarse bins (mass-conserving)."""

    probs = _as_prob(histogram)
    bounds = torch.tensor(scheme.boundaries(probs.numel()))
    cumulative = torch.cat([torch.zeros(1, dtype=probs.dtype), torch.cumsum(probs, 0)])
    return cumulative[bounds[1:]] - cumulative[bounds[:-1]]


def _smoothed(probs: torch.Tensor, eps: float) -> torch.Tensor:
    probs = probs + eps
    return probs / probs.sum()


# ---------------------------------------------------------------------------
# Binned f-divergences
# ---------------------------------------------------------------------------


def _binned_pair(reference, comparison, scheme: BinScheme, eps: float):
    p = _smoothed(coarsen_histogram(reference, scheme), eps)
    q = _smoothed(coarsen_histogram(comparison, scheme), eps)
    return p, q


def kl_divergence(reference, comparison, *, scheme: BinScheme = DEFAULT_SCHEME, eps: float = EPS) -> float:
    """KL(data || gen) on coarse bins."""

    p, q = _binned_pair(reference, comparison, scheme, eps)
    return float((p * (p / q).log()).sum())


def reverse_kl_divergence(reference, comparison, *, scheme: BinScheme = DEFAULT_SCHEME, eps: float = EPS) -> float:
    """KL(gen || data) on coarse bins."""

    return kl_divergence(comparison, reference, scheme=scheme, eps=eps)


def symmetric_kl_divergence(reference, comparison, *, scheme: BinScheme = DEFAULT_SCHEME, eps: float = EPS) -> float:
    p, q = _binned_pair(reference, comparison, scheme, eps)
    return float((p * (p / q).log()).sum() + (q * (q / p).log()).sum())


def jensen_shannon(reference, comparison, *, scheme: BinScheme = DEFAULT_SCHEME, eps: float = EPS) -> float:
    p, q = _binned_pair(reference, comparison, scheme, eps)
    m = 0.5 * (p + q)
    return float(0.5 * (p * (p / m).log()).sum() + 0.5 * (q * (q / m).log()).sum())


def chi_square(reference, comparison, *, scheme: BinScheme = DEFAULT_SCHEME, eps: float = EPS) -> float:
    """Pearson chi-square between coarse bins, with gen on the denominator."""

    p, q = _binned_pair(reference, comparison, scheme, eps)
    return float(((p - q) ** 2 / q).sum())


def hellinger(reference, comparison, *, scheme: BinScheme = DEFAULT_SCHEME, eps: float = EPS) -> float:
    p, q = _binned_pair(reference, comparison, scheme, eps)
    return float((1.0 / math.sqrt(2.0)) * torch.sqrt(((p.sqrt() - q.sqrt()) ** 2).sum()))


def total_variation(reference, comparison, *, scheme: BinScheme = DEFAULT_SCHEME, eps: float = EPS) -> float:
    p, q = _binned_pair(reference, comparison, scheme, eps)
    return float(0.5 * (p - q).abs().sum())


# ---------------------------------------------------------------------------
# Wasserstein-1 with alternative ground costs
# ---------------------------------------------------------------------------


def _ground_positions(vocab_size: int, cost: str, cap: float | None) -> torch.Tensor:
    pos = torch.arange(1, vocab_size + 1, dtype=torch.float64)
    if cost == "linear":
        return pos
    if cost == "sqrt":
        return torch.sqrt(pos)
    if cost == "log":
        return torch.log(pos)
    if cost == "capped_log":
        return torch.log(torch.clamp(pos, max=float(cap if cap is not None else 1000.0)))
    raise ValueError(f"Unknown ground cost {cost!r}.")


def wasserstein(reference, comparison, *, cost: str = "log", cap: float | None = None) -> float:
    """Closed-form Wasserstein-1 on the rank axis with a configurable ground cost.

    ``cost="log"`` reproduces :func:`ranking_divergence.rank.rank_wasserstein_from_histograms`.
    """

    p = _as_prob(reference)
    q = _as_prob(comparison)
    if p.shape != q.shape:
        raise ValueError("Histograms must have the same shape.")
    if p.numel() < 2:
        return 0.0
    cumulative = torch.cumsum(p - q, dim=0)[:-1].abs()
    positions = _ground_positions(p.numel(), cost, cap)
    increments = positions[1:] - positions[:-1]
    return float((cumulative * increments).sum())


# ---------------------------------------------------------------------------
# Rank-correlation / tail summaries
# ---------------------------------------------------------------------------


def ks_statistic(reference, comparison) -> float:
    """Kolmogorov-Smirnov statistic: max |CDF_data - CDF_gen|."""

    p = _as_prob(reference)
    q = _as_prob(comparison)
    return float((torch.cumsum(p, 0) - torch.cumsum(q, 0)).abs().max())


def cdf_l2(reference, comparison) -> float:
    """Cramer-von-Mises-style L2 distance between rank CDFs."""

    p = _as_prob(reference)
    q = _as_prob(comparison)
    return float(((torch.cumsum(p, 0) - torch.cumsum(q, 0)) ** 2).sum().sqrt())


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values)
    sorted_values = values[order]
    num_elem = values.numel()
    if num_elem == 0:
        return torch.empty_like(values)
    starts_mask = torch.ones(num_elem, dtype=torch.bool)
    starts_mask[1:] = sorted_values[1:] != sorted_values[:-1]
    group = torch.cumsum(starts_mask, 0) - 1
    starts = torch.nonzero(starts_mask, as_tuple=True)[0]
    stops = torch.cat([starts[1:], torch.tensor([num_elem])])
    group_ranks = (0.5 * (starts + stops - 1) + 1.0).to(values.dtype)
    ranks = torch.empty_like(values)
    ranks[order] = group_ranks[group]
    return ranks


def cdf_spearman(reference, comparison) -> float:
    """Spearman distance ``1 - rho`` between the two rank CDFs."""

    p = _as_prob(reference)
    q = _as_prob(comparison)
    p_ranks = _average_ranks(torch.cumsum(p, 0))
    q_ranks = _average_ranks(torch.cumsum(q, 0))
    p_centered = p_ranks - p_ranks.mean()
    q_centered = q_ranks - q_ranks.mean()
    denominator = torch.linalg.vector_norm(p_centered) * torch.linalg.vector_norm(q_centered)
    if denominator == 0:
        return 0.0
    rho = torch.clamp((p_centered * q_centered).sum() / denominator, -1.0, 1.0)
    return float(1.0 - rho)


def topk_mass_diff(reference, comparison, *, k: int = 10) -> float:
    """|fraction of mass in ranks 1..k| difference between data and gen."""

    p = _as_prob(reference)
    q = _as_prob(comparison)
    k = min(k, p.numel())
    return float((p[:k].sum() - q[:k].sum()).abs())


def mean_log_rank_gap(reference, comparison) -> float:
    """Absolute gap in expected log-rank under data vs gen."""

    p = _as_prob(reference)
    q = _as_prob(comparison)
    positions = torch.log(torch.arange(1, p.numel() + 1, dtype=torch.float64))
    return float(((p * positions).sum() - (q * positions).sum()).abs())


def _median_rank(probs: torch.Tensor) -> int:
    cumulative = torch.cumsum(probs, 0)
    return int((cumulative >= 0.5).nonzero()[0].item()) + 1


def median_rank_gap(reference, comparison) -> float:
    """Absolute gap between the median observed rank of data and gen."""

    p = _as_prob(reference)
    q = _as_prob(comparison)
    return float(abs(_median_rank(p) - _median_rank(q)))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

_BIN20LOG5 = BinScheme(20, 5)
_BIN50LOG10 = BinScheme(50, 10)

DIVERGENCES: dict[str, Callable[..., float]] = {
    # Wasserstein with alternative ground costs (log == existing rank_wasserstein).
    "w1_log": partial(wasserstein, cost="log"),
    "w1_linear": partial(wasserstein, cost="linear"),
    "w1_sqrt": partial(wasserstein, cost="sqrt"),
    "w1_capped_log_1000": partial(wasserstein, cost="capped_log", cap=1000.0),
    # Binned f-divergences (20 linear + 5 log).
    "kl_bin20log5": partial(kl_divergence, scheme=_BIN20LOG5),
    "reverse_kl_bin20log5": partial(reverse_kl_divergence, scheme=_BIN20LOG5),
    "symmetric_kl_bin20log5": partial(symmetric_kl_divergence, scheme=_BIN20LOG5),
    "js_bin20log5": partial(jensen_shannon, scheme=_BIN20LOG5),
    "chi_square_bin20log5": partial(chi_square, scheme=_BIN20LOG5),
    "hellinger_bin20log5": partial(hellinger, scheme=_BIN20LOG5),
    "total_variation_bin20log5": partial(total_variation, scheme=_BIN20LOG5),
    # Finer binning to probe sensitivity.
    "kl_bin50log10": partial(kl_divergence, scheme=_BIN50LOG10),
    "js_bin50log10": partial(jensen_shannon, scheme=_BIN50LOG10),
    # Rank-correlation / tail summaries.
    "ks_statistic": ks_statistic,
    "cdf_l2": cdf_l2,
    "cdf_spearman": cdf_spearman,
    "topk_mass_diff_1": partial(topk_mass_diff, k=1),
    "topk_mass_diff_10": partial(topk_mass_diff, k=10),
    "topk_mass_diff_100": partial(topk_mass_diff, k=100),
    "topk_mass_diff_1000": partial(topk_mass_diff, k=1000),
    "mean_log_rank_gap": mean_log_rank_gap,
    "median_rank_gap": median_rank_gap,
}


def compute_all(reference, comparison) -> dict[str, float]:
    """Evaluate every registered divergence for one (reference, comparison) pair."""

    return {name: fn(reference, comparison) for name, fn in DIVERGENCES.items()}
