from __future__ import annotations

import math
from typing import Sequence

from transformers import RepetitionPenaltyLogitsProcessor

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from .metrics import per_sample_unigram_entropy, rep_n, unique_ngram_ratios


def compute_causal_repetition_penalty(logits: torch.Tensor, input_ids: torch.Tensor, penalty: float = 1.2):
    """Compute repetition penalty for all tokens, not just the predicted one."""

    penalty_processor = RepetitionPenaltyLogitsProcessor(penalty)

    for token_pos in range(logits.shape[1]):

        logits[:, token_pos, :] = penalty_processor(
            input_ids=input_ids[:, :token_pos+1],
            scores=logits[:, token_pos, :],
        )

    return logits


def compute_model_entropy(logits: torch.Tensor):
    """Compute per token entropy based on logits."""
    log_probs = F.log_softmax(logits.float(), dim=-1)
    entropy = -(log_probs.exp() * log_probs).sum(-1)  #(B, T), nats

    return log_probs, entropy


@torch.inference_mode()
def score_token_ids(
    token_ids: Sequence[Sequence[int]],
    model,
    tokenizer,
    *,
    batch_size: int,
    start_token: int = 0,
    start_token_positions: torch.Tensor | Sequence[int] | None = None,
    max_length: int,
    rank_position_chunk: int,
    device: str,
    description: str,
    method: str = "count", #Options count, inv_prob, entropy_bins
    n_entropy_bins: int = 16,
    per_sample: bool = False,
    rep_penalty: float | None = None,
) -> tuple[float, torch.Tensor, float]:
    """Compute DUO-style gen-PPL and a normalized rank histogram in one model pass."""

    if method not in {"count", "inv_prob", "entropy_bins"}:
        raise ValueError(f"Unknown method: {method!r}")

    if start_token_positions is not None:
        if start_token != 0:
            raise ValueError("start_token and start_token_positions can not be used togeather.")
        start_token_positions = torch.as_tensor(start_token_positions, dtype=torch.int64)

    #!TODO: Implement for other methods if needed
    if per_sample:
        assert method == 'count', "Per sample is only implemented for count"

    
    vocab_size = int(model.config.vocab_size)
    n_bins = n_entropy_bins if method == "entropy_bins" else 1

    if per_sample:
        histogram = torch.zeros(len(token_ids), n_bins * vocab_size, dtype=torch.float64)
    else:
        histogram = torch.zeros(n_bins * vocab_size, dtype=torch.float64)
    entropy_edges = torch.linspace(0.0, math.log(vocab_size), n_bins + 1)[1:-1]

    total_loss = 0.0
    total_ppl_tokens = 0
    total_model_entropy = 0
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

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
        if rep_penalty is not None and rep_penalty != 1.0:
            logits = compute_causal_repetition_penalty(logits, input_ids, rep_penalty)
        logits = logits[:, :-1, :]

        labels = input_ids[:, 1:]
        rank_valid = attention_mask[:, :-1].bool() & attention_mask[:, 1:].bool()
        first_eos = (input_ids == tokenizer.eos_token_id).cumsum(dim=-1) == 1
        non_eos = input_ids != tokenizer.eos_token_id
        ppl_valid = (first_eos[:, 1:] | non_eos[:, 1:]) & attention_mask[:, 1:].bool()

        if start_token_positions is not None:
            #Mask the positions corresponding to the startof the generation.
            start_token_batch = start_token_positions[start : start + batch_size].to(device)
            scored_mask= torch.arange(labels.shape[1], device=device) >= (start_token_batch - 1).clamp(min=0).unsqueeze(1) 
            rank_valid = rank_valid & scored_mask
            ppl_valid = ppl_valid & scored_mask

        effective_start_token = start_token - 1 if start_token != 0 else 0 #Score exact amount of tokens except in the BOS token.
        for position in range(effective_start_token, labels.shape[1], rank_position_chunk):
            stop = position + rank_position_chunk
            chunk_logits = logits[:, position:stop, :]
            chunk_labels = labels[:, position:stop]

            losses = F.cross_entropy(
                chunk_logits.reshape(-1, vocab_size),
                chunk_labels.reshape(-1),
                reduction="none",
            ).reshape_as(chunk_labels)
            observed = chunk_logits.gather(-1, chunk_labels.unsqueeze(-1))
            ranks = (chunk_logits > observed).sum(dim=-1) + 1
            chunk_valid = rank_valid[:, position:stop]
            valid_ranks = ranks[chunk_valid].detach().cpu()
            weights = losses[chunk_valid].detach().double().exp().cpu() if method == "inv_prob" else None
            _, entropy = compute_model_entropy(chunk_logits)
            if not per_sample:
                if method == "entropy_bins":
                    entropy_bin = torch.bucketize(entropy[chunk_valid].detach().float().cpu(), entropy_edges)
                    flat_index = (valid_ranks - 1) * n_bins + entropy_bin
                else:
                    flat_index = valid_ranks - 1

                histogram += torch.bincount(flat_index, weights=weights, minlength=histogram.numel()).to(torch.float64)
            else:
                #Compute per sample histogram
                rows = chunk_labels.shape[0]
                row_index = torch.arange(rows, device=ranks.device).unsqueeze(1).expand_as(ranks)
                flat_index = (row_index * vocab_size + (ranks - 1))[chunk_valid].detach().cpu()
                counts = torch.bincount(flat_index, minlength=rows * vocab_size)
                histogram[start : start + rows] += counts.reshape(rows, vocab_size).to(torch.float64)

            valid_ppl = ppl_valid[:, position:stop]
            total_model_entropy += float(entropy[valid_ppl].sum().item())
            total_loss += float(losses[valid_ppl].sum().item())
            total_ppl_tokens += int(valid_ppl.sum().item())

        del logits

    if histogram.sum() == 0:
        raise ValueError("No valid next-token positions found for rank scoring.")
    if total_ppl_tokens == 0:
        raise ValueError("No valid next-token positions found for perplexity.")
    gen_ppl = math.exp(total_loss / total_ppl_tokens)
    model_entropy = total_model_entropy / total_ppl_tokens
    if per_sample:
        #Normalize each sample row independently.
        totals = histogram.sum(dim=-1, keepdim=True)
        return gen_ppl, histogram / totals.clamp(min=1.0), model_entropy

    return gen_ppl, histogram / histogram.sum(), model_entropy


@torch.inference_mode()
def score_tokens_no_rank(
    token_ids: Sequence[Sequence[int]],
    model,
    tokenizer,
    *,
    batch_size: int,
    max_length: int,
    rank_position_chunk: int,
    device: str,
    per_sample: bool = False,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute DUO-style gen-PPL and model entropy.
    If per sample --> Returns a tensor containing the per sample loss and entropy as well as the number of tokens.
    Else --> Returns a tensor containing aggregated genPPL and entropy and KL(q_gen, p_ref).
    KL assumes H(p_ref) is a good estimator of H(g_gen).
    """


    results_tensor = torch.zeros((len(token_ids), 3), dtype=torch.float64)
    #For each sample store cumulative loss, cumulative entropy and number of scored tokens

    model_context = int(getattr(model.config, "n_positions", max_length))
    effective_length = min(max_length, model_context)

    iterator = tqdm(total=len(token_ids), desc="Scoring tokens", leave=False, unit="sample")
    for start in range(0, len(token_ids), batch_size):
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
            iterator.update(len(batch_ids))
            continue

        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits[:, :-1, :]
        labels = input_ids[:, 1:]
        first_eos = (input_ids == tokenizer.eos_token_id).cumsum(dim=-1) == 1
        non_eos = input_ids != tokenizer.eos_token_id
        ppl_valid = (first_eos[:, 1:] | non_eos[:, 1:]) & attention_mask[:, 1:].bool()

        for position in range(0, labels.shape[1], rank_position_chunk):
            stop = position + rank_position_chunk
            chunk_logits = logits[:, position:stop, :]
            chunk_labels = labels[:, position:stop]
            if temperature != 1.0:
                chunk_logits = chunk_logits / temperature

            log_probs, entropy = compute_model_entropy(chunk_logits)

            loss = -log_probs.gather(-1, chunk_labels.unsqueeze(-1))


            valid_ppl = ppl_valid[:, position:stop]
            valid_mask = valid_ppl.to(loss.dtype)
            loss_valid = loss.squeeze(-1) * valid_mask
            entropy_valid = entropy * valid_mask

            rows = chunk_logits.shape[0]
            results_tensor[start : start + rows, 0] += loss_valid.sum(dim=1).double().cpu()
            results_tensor[start : start + rows, 1] += entropy_valid.sum(dim=1).double().cpu()
            results_tensor[start : start + rows, 2] += valid_ppl.sum(dim=1).double().cpu()

            del log_probs, entropy, loss, chunk_logits

        del logits
        iterator.update(len(batch_ids))

    iterator.close()

    if not per_sample:

        totals = results_tensor.sum(dim=0)
        total_tokens = totals[2].item()
        if total_tokens == 0:
            raise ValueError("No valid next-token positions found for perplexity.")
        gen_ppl = math.exp(totals[0].item() / total_tokens)
        model_entropy = totals[1].item() / total_tokens
        return torch.tensor(
            [gen_ppl, model_entropy, total_tokens, math.log(gen_ppl) - model_entropy],
            dtype=torch.float64,
        )
    else:
        return results_tensor

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
