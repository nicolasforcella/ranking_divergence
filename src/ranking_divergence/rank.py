from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import torch
from tqdm.auto import tqdm


@dataclass(frozen=True)
class RankDivergenceResult:
    """Rank-Wasserstein result with reusable normalized histograms."""

    distance: float
    reference_histogram: torch.Tensor
    comparison_histogram: torch.Tensor


def _as_text_list(texts: str | Sequence[str]) -> list[str]:
    if isinstance(texts, str):
        return [texts]
    return list(texts)


def _model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _ensure_pad_token(tokenizer) -> None:
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            raise ValueError("Tokenizer must define a pad token or eos token.")


def normalize_histogram(histogram: torch.Tensor) -> torch.Tensor:
    """Return a float64 probability histogram over ranks."""

    histogram = histogram.detach().to(dtype=torch.float64, device="cpu")
    total = histogram.sum()
    if total <= 0:
        raise ValueError("Cannot normalize an empty rank histogram.")
    return histogram / total


def _update_rank_histogram(
    histogram: torch.Tensor,
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> int:
    next_logits = logits[:, :-1, :]
    target_ids = input_ids[:, 1:]
    valid_targets = attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool()

    if not valid_targets.any():
        return 0

    observed_logits = next_logits.gather(-1, target_ids.unsqueeze(-1))
    ranks = (next_logits > observed_logits).sum(dim=-1) + 1
    ranks = ranks[valid_targets]
    histogram += torch.bincount(ranks - 1, minlength=histogram.numel()).to(histogram)
    return int(ranks.numel())


@torch.inference_mode()
def rank_histogram(
    texts: str | Sequence[str],
    model,
    tokenizer,
    *,
    batch_size: int = 8,
    max_length: int | None = None,
    device: str | torch.device | None = None,
    normalize: bool = True,
    show_progress: bool = False,
) -> torch.Tensor:
    """Compute the next-token rank histogram from ``notes.pdf``.

    Rank 1 is the highest-probability token under ``model``. The returned
    histogram has length equal to the reference model vocabulary size; index 0
    stores rank 1.
    """

    samples = _as_text_list(texts)
    if not samples:
        raise ValueError("At least one text sample is required.")

    _ensure_pad_token(tokenizer)
    if device is None:
        device = _model_device(model)
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    vocab_size = int(getattr(model.config, "vocab_size"))
    histogram = torch.zeros(vocab_size, dtype=torch.float64, device="cpu")
    iterator: Iterable[int] = range(0, len(samples), batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="rank histogram")

    for start in iterator:
        batch_texts = samples[start : start + batch_size]
        encoded = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=max_length is not None,
            max_length=max_length,
            return_attention_mask=True,
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        if input_ids.shape[1] < 2:
            continue
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        _update_rank_histogram(histogram, logits.cpu(), input_ids.cpu(), attention_mask.cpu())

    return normalize_histogram(histogram) if normalize else histogram


@torch.inference_mode()
def rank_histogram_from_dataloader(
    dataloader,
    model,
    *,
    device: str | torch.device | None = None,
    normalize: bool = True,
    show_progress: bool = False,
) -> torch.Tensor:
    """Compute a rank histogram from batches containing input IDs.

    Batches may be mappings with ``input_ids`` and optional ``attention_mask``,
    or tensors of input IDs. This is convenient for fixed-block OpenWebText
    dataloaders such as the DUO codebase uses.
    """

    if device is None:
        device = _model_device(model)
    device = torch.device(device)
    model = model.to(device)
    model.eval()

    vocab_size = int(getattr(model.config, "vocab_size"))
    histogram = torch.zeros(vocab_size, dtype=torch.float64, device="cpu")
    iterator = tqdm(dataloader, desc="rank histogram") if show_progress else dataloader

    for batch in iterator:
        if isinstance(batch, dict):
            input_ids = batch["input_ids"]
            attention_mask = batch.get("attention_mask")
        else:
            input_ids = batch
            attention_mask = None

        input_ids = input_ids.to(device)
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        else:
            attention_mask = attention_mask.to(device)
        if input_ids.shape[1] < 2:
            continue
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        _update_rank_histogram(histogram, logits.cpu(), input_ids.cpu(), attention_mask.cpu())

    return normalize_histogram(histogram) if normalize else histogram


def rank_wasserstein_from_histograms(
    reference_histogram: torch.Tensor,
    comparison_histogram: torch.Tensor,
    *,
    normalize: bool = True,
) -> float:
    """Closed-form log-rank Wasserstein distance between two histograms."""

    ref = normalize_histogram(reference_histogram) if normalize else reference_histogram.to(torch.float64).cpu()
    cmp = normalize_histogram(comparison_histogram) if normalize else comparison_histogram.to(torch.float64).cpu()
    if ref.shape != cmp.shape:
        raise ValueError("Histograms must have the same shape.")
    if ref.numel() < 2:
        return 0.0

    cumulative = torch.cumsum(ref - cmp, dim=0)[:-1].abs()
    ranks = torch.arange(1, ref.numel(), dtype=torch.float64)
    costs = torch.log(ranks + 1.0) - torch.log(ranks)
    return float((cumulative * costs).sum().item())


def rank_wasserstein(
    reference_texts: str | Sequence[str],
    comparison_texts: str | Sequence[str],
    model,
    tokenizer,
    *,
    batch_size: int = 8,
    max_length: int | None = None,
    device: str | torch.device | None = None,
    show_progress: bool = False,
) -> RankDivergenceResult:
    """Compute rank histograms for two text sets and their log-rank distance."""

    reference_histogram = rank_histogram(
        reference_texts,
        model,
        tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        normalize=True,
        show_progress=show_progress,
    )
    comparison_histogram = rank_histogram(
        comparison_texts,
        model,
        tokenizer,
        batch_size=batch_size,
        max_length=max_length,
        device=device,
        normalize=True,
        show_progress=show_progress,
    )
    distance = rank_wasserstein_from_histograms(
        reference_histogram,
        comparison_histogram,
        normalize=False,
    )
    return RankDivergenceResult(distance, reference_histogram, comparison_histogram)
