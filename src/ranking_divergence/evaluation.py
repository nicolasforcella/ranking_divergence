from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .metrics import per_sample_unigram_entropy, rep_n, unique_ngram_ratios


@torch.inference_mode()
def score_token_ids(
    token_ids: Sequence[Sequence[int]],
    model,
    tokenizer,
    *,
    batch_size: int,
    max_length: int,
    rank_position_chunk: int,
    device: str,
    description: str,
) -> tuple[float, torch.Tensor]:
    """Compute DUO-style gen-PPL and a normalized rank histogram in one model pass."""

    vocab_size = int(model.config.vocab_size)
    histogram = torch.zeros(vocab_size, dtype=torch.float64)
    total_loss = 0.0
    total_ppl_tokens = 0
    model_context = int(getattr(model.config, "n_positions", max_length))
    effective_length = min(max_length, model_context)

    iterator = tqdm(range(0, len(token_ids), batch_size), desc=description, leave=False)
    for start in iterator:
        batch_ids = [list(ids[:effective_length]) for ids in token_ids[start : start + batch_size]]
        encoded = tokenizer.pad(
            {"input_ids": batch_ids},
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        input_ids = encoded["input_ids"].to(device)
        attention_mask = encoded["attention_mask"].to(device)
        if input_ids.shape[1] < 2:
            continue

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
        labels = input_ids[:, 1:]
        rank_valid = attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool()
        first_eos = (input_ids == tokenizer.eos_token_id).cumsum(dim=-1) == 1
        non_eos = input_ids != tokenizer.eos_token_id
        ppl_valid = (first_eos[:, 1:] | non_eos[:, 1:]) & attention_mask[:, 1:].bool()

        for position in range(0, labels.shape[1], rank_position_chunk):
            stop = position + rank_position_chunk
            chunk_logits = logits[:, position:stop, :]
            chunk_labels = labels[:, position:stop]

            observed = chunk_logits.gather(-1, chunk_labels.unsqueeze(-1))
            ranks = (chunk_logits > observed).sum(dim=-1) + 1
            valid_ranks = ranks[rank_valid[:, position:stop]].detach().cpu()
            histogram += torch.bincount(valid_ranks - 1, minlength=vocab_size).to(torch.float64)

            losses = F.cross_entropy(
                chunk_logits.reshape(-1, vocab_size),
                chunk_labels.reshape(-1),
                reduction="none",
            ).reshape_as(chunk_labels)
            valid_ppl = ppl_valid[:, position:stop]
            total_loss += float(losses[valid_ppl].sum().item())
            total_ppl_tokens += int(valid_ppl.sum().item())

        del logits

    if histogram.sum() == 0:
        raise ValueError("No valid next-token positions found for rank scoring.")
    if total_ppl_tokens == 0:
        raise ValueError("No valid next-token positions found for perplexity.")
    gen_ppl = math.exp(total_loss / total_ppl_tokens)
    return gen_ppl, histogram / histogram.sum()


def lexical_metrics(texts: Sequence[str], token_ids: Sequence[Sequence[int]]) -> dict[str, float]:
    row = {
        "unigram_entropy": per_sample_unigram_entropy(texts, token_ids=token_ids),
        "mean_tokens": sum(map(len, token_ids)) / len(token_ids),
        "min_tokens": min(map(len, token_ids)),
        "max_tokens": max(map(len, token_ids)),
        "empty_samples": sum(not ids for ids in token_ids),
    }
    for n in range(1, 5):
        ratios = unique_ngram_ratios(texts, n=n, token_ids=token_ids)
        row[f"unique_{n}gram_sample"] = ratios["sample"]
        row[f"unique_{n}gram_corpus"] = ratios["corpus"]
    for n in range(1, 4):
        row[f"rep_{n}"] = rep_n(texts, n=n, token_ids=token_ids)
    return row
