from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ranking_divergence import (
    MirrorSampler,
    PeriodicSampler,
    PhraseBankSampler,
    RestrictedMarginalSampler,
    empirical_entropy,
    generative_perplexity,
    rank_wasserstein,
    rep_n,
)
from ranking_divergence.data import load_openwebtext_texts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scorer-model", default="gpt2-large")
    parser.add_argument("--generator-model", default="gpt2")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--reference-split", default="train[-100000:]")
    parser.add_argument("--sampler-source-split", default="train[:-100000]")
    parser.add_argument("--num-reference", type=int, default=32)
    parser.add_argument("--num-sampler-source", type=int, default=2048)
    parser.add_argument("--num-samples", type=int, default=16)
    parser.add_argument("--sample-length", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=64)
    parser.add_argument("--mirror-k", type=int, default=5000)
    parser.add_argument("--periodic-k", type=int, default=400)
    parser.add_argument("--phrase-bank-m", type=int, default=5000)
    parser.add_argument("--phrase-bank-n", type=int, default=5)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.inference_mode()
def generate_model_samples(model, tokenizer, *, num_samples: int, length: int, device: str, seed: int) -> list[str]:
    torch.manual_seed(seed)
    model = model.to(device).eval()
    input_ids = torch.full((num_samples, 1), tokenizer.eos_token_id, dtype=torch.long, device=device)
    output = model.generate(
        input_ids=input_ids,
        max_new_tokens=length,
        do_sample=True,
        top_p=0.95,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.batch_decode(output[:, 1:], skip_special_tokens=True)


def evaluate_candidate(name: str, texts: list[str], reference_texts: list[str], scorer, tokenizer, args) -> dict:
    token_ids = [tokenizer.encode(text, add_special_tokens=False) for text in texts]
    divergence = rank_wasserstein(
        reference_texts,
        texts,
        scorer,
        tokenizer,
        batch_size=args.batch_size,
        max_length=args.sample_length,
        device=args.device,
        show_progress=True,
    )
    return {
        "name": name,
        "rank_wasserstein": divergence.distance,
        "gen_ppl": generative_perplexity(
            texts,
            scorer,
            tokenizer,
            batch_size=args.batch_size,
            max_length=args.sample_length,
            device=args.device,
            show_progress=True,
        ),
        "entropy": empirical_entropy(texts, tokenizer, token_ids=token_ids),
        "rep_1": rep_n(texts, tokenizer, n=1, token_ids=token_ids),
        "rep_2": rep_n(texts, tokenizer, n=2, token_ids=token_ids),
        "rep_3": rep_n(texts, tokenizer, n=3, token_ids=token_ids),
    }


def main() -> None:
    args = parse_args()
    args.device = resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.scorer_model)
    tokenizer.pad_token = tokenizer.eos_token

    print("Loading OpenWebText reference texts...")
    reference_texts = load_openwebtext_texts(
        split=args.reference_split,
        cache_dir=args.cache_dir,
        limit=args.num_reference,
    )
    sampler_source = load_openwebtext_texts(
        split=args.sampler_source_split,
        cache_dir=args.cache_dir,
        limit=args.num_sampler_source,
    )

    print(f"Loading scorer {args.scorer_model}...")
    scorer = AutoModelForCausalLM.from_pretrained(args.scorer_model)

    print(f"Generating from {args.generator_model}...")
    generator_tokenizer = AutoTokenizer.from_pretrained(args.generator_model)
    generator_tokenizer.pad_token = generator_tokenizer.eos_token
    generator = AutoModelForCausalLM.from_pretrained(args.generator_model)
    generated = generate_model_samples(
        generator,
        generator_tokenizer,
        num_samples=args.num_samples,
        length=args.sample_length,
        device=args.device,
        seed=args.seed,
    )
    del generator
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    topk = RestrictedMarginalSampler.from_texts(sampler_source, tokenizer, k=args.top_k)
    mirror_base = RestrictedMarginalSampler.from_texts(sampler_source, tokenizer, k=args.mirror_k)
    candidates = {
        args.generator_model: generated,
        f"top_k_iid_{args.top_k}": topk.sample(
            num_samples=args.num_samples,
            length=args.sample_length,
            seed=args.seed,
        ),
        f"mirror_{args.mirror_k}": MirrorSampler(mirror_base).sample(
            num_samples=args.num_samples,
            length=args.sample_length,
            seed=args.seed,
        ),
        f"periodic_{args.periodic_k}": PeriodicSampler.from_texts(
            sampler_source,
            tokenizer,
            k=args.periodic_k,
        ).sample(num_samples=args.num_samples, length=args.sample_length),
        f"phrase_bank_{args.phrase_bank_m}": PhraseBankSampler.from_texts(
            sampler_source,
            tokenizer,
            n=args.phrase_bank_n,
            m=args.phrase_bank_m,
        ).sample(num_samples=args.num_samples, length=args.sample_length, seed=args.seed),
    }

    results = [
        evaluate_candidate(name, texts, reference_texts, scorer, tokenizer, args)
        for name, texts in candidates.items()
    ]
    print(json.dumps(results, indent=2))
    if args.output is not None:
        args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
