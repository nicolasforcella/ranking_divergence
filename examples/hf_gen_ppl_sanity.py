from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import evaluate
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ranking_divergence import (
    duo_generative_perplexity,
    empirical_entropy,
    generative_perplexity,
    per_sample_unigram_entropy,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate with Hugging Face Transformers and independently check gen-PPL."
    )
    parser.add_argument("--generator-model", default="gpt2")
    parser.add_argument("--scorer-model", default="gpt2-large")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--sample-length", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--output", type=Path, default=Path("outputs/hf_gen_ppl_sanity.json"))
    return parser.parse_args()


@torch.inference_mode()
def generate_texts(args: argparse.Namespace, device: str) -> tuple[list[str], list[list[int]]]:
    tokenizer = AutoTokenizer.from_pretrained(args.generator_model)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(args.generator_model).to(device).eval()

    torch.manual_seed(args.seed)
    input_ids = torch.full(
        (args.num_samples, 1),
        tokenizer.eos_token_id,
        dtype=torch.long,
        device=device,
    )
    output_ids = model.generate(
        input_ids=input_ids,
        attention_mask=torch.ones_like(input_ids),
        max_new_tokens=args.sample_length,
        do_sample=True,
        temperature=1.0,
        top_p=1.0,
        top_k=0,
        pad_token_id=tokenizer.eos_token_id,
    )[:, 1:]

    samples: list[list[int]] = []
    for ids in output_ids.cpu().tolist():
        if tokenizer.eos_token_id in ids:
            ids = ids[: ids.index(tokenizer.eos_token_id) + 1]
        samples.append(ids)

    texts = tokenizer.batch_decode(samples, skip_special_tokens=False)
    del model
    torch.cuda.empty_cache()
    return texts, samples


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this sanity check.")
    device = "cuda:0"

    texts, generated_token_ids = generate_texts(args, device)
    lengths = [len(ids) for ids in generated_token_ids]
    per_sample_entropies = [
        empirical_entropy("", token_ids=[ids]) for ids in generated_token_ids
    ]
    mean_per_sample_entropy = per_sample_unigram_entropy(
        "",
        token_ids=generated_token_ids,
    )
    token_weighted_per_sample_entropy = sum(
        length * entropy
        for length, entropy in zip(lengths, per_sample_entropies, strict=True)
    ) / sum(lengths)
    pooled_entropy = empirical_entropy("", token_ids=generated_token_ids)

    scorer_tokenizer = AutoTokenizer.from_pretrained(args.scorer_model)
    scorer_tokenizer.pad_token = scorer_tokenizer.eos_token
    scorer = AutoModelForCausalLM.from_pretrained(args.scorer_model).to(device).eval()
    repo_ppl = generative_perplexity(
        texts,
        scorer,
        scorer_tokenizer,
        batch_size=args.batch_size,
        max_length=args.sample_length,
        device=device,
    )
    repo_duo_ppl = duo_generative_perplexity(
        texts,
        scorer,
        scorer_tokenizer,
        batch_size=args.batch_size,
        max_length=args.sample_length,
        device=device,
    )
    del scorer
    torch.cuda.empty_cache()

    hf_metric = evaluate.load("perplexity", module_type="metric")
    hf_result = hf_metric.compute(
        predictions=texts,
        model_id=args.scorer_model,
        batch_size=args.batch_size,
        add_start_token=False,
        device="cuda",
    )
    scored_token_counts = [
        max(
            0,
            len(
                scorer_tokenizer.encode(
                    text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=args.sample_length,
                )
            )
            - 1,
        )
        for text in texts
    ]
    hf_token_weighted_perplexity = math.exp(
        sum(
            count * math.log(perplexity)
            for count, perplexity in zip(
                scored_token_counts,
                hf_result["perplexities"],
                strict=True,
            )
        )
        / sum(scored_token_counts)
    )

    result = {
        "generator_model": args.generator_model,
        "scorer_model": args.scorer_model,
        "seed": args.seed,
        "num_samples": len(texts),
        "requested_sample_length": args.sample_length,
        "generated_token_lengths": lengths,
        "mean_per_sample_unigram_entropy_nats": mean_per_sample_entropy,
        "token_weighted_per_sample_unigram_entropy_nats": token_weighted_per_sample_entropy,
        "pooled_unigram_entropy_nats": pooled_entropy,
        "per_sample_unigram_entropies_nats": per_sample_entropies,
        "repo_gen_ppl": repo_ppl,
        "repo_duo_gen_ppl": repo_duo_ppl,
        "hf_mean_perplexity": hf_result["mean_perplexity"],
        "hf_token_weighted_perplexity": hf_token_weighted_perplexity,
        "hf_per_sample_perplexities": hf_result["perplexities"],
        "scored_token_counts": scored_token_counts,
        "sample_previews": [text[:200] for text in texts[:3]],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
