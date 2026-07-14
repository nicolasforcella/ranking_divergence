from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import torch
import torch_xla.core.xla_model as xm
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generator-model", default="gpt2")
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--sample-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--steps-label", type=int, default=64)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--temperatures",
        type=float,
        nargs="+",
        default=[0.60, 0.625, 0.65, 0.675, 0.70, 0.725, 0.75, 0.775, 0.80, 0.825,
                  0.85, 0.875, 0.90, 0.925, 0.95, 0.975, 1.00],
    )
    return parser.parse_args(argv)


def generate_batch(model, tokenizer, *, num_samples, length, device, seed, temperature, top_p, batch_size):
    torch.manual_seed(seed)
    all_ids: list[list[int]] = []
    remaining = num_samples
    temperature_t = torch.tensor(temperature, device=device)
    while remaining > 0:
        n = min(batch_size, remaining)
        input_ids = torch.full((n, 1), tokenizer.eos_token_id, dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        output = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=length,
            min_new_tokens=length,  # static output shape avoids XLA recompilation
            do_sample=True,
            top_p=top_p,
            top_k=0,
            temperature=temperature_t,
            pad_token_id=tokenizer.eos_token_id,
            cache_implementation="static"
        )
        xm.mark_step()
        all_ids.extend(output[:, 1:].detach().cpu().tolist())
        remaining -= n
    return all_ids


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    device = xm.xla_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {args.generator_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.generator_model)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(args.generator_model).to(device).eval()

    for temperature in args.temperatures:
        out_path = args.output_dir / f"samples_steps{args.steps_label}_temp{temperature:.3f}.json"
        if out_path.exists() and not args.force:
            print(f"Skipping existing {out_path}")
            continue

        print(f"Generating temperature={temperature} -> {out_path}")
        token_ids = generate_batch(
            model,
            tokenizer,
            num_samples=args.num_samples,
            length=args.sample_length,
            device=device,
            seed=args.seed,
            temperature=temperature,
            top_p=args.top_p,
            batch_size=args.batch_size,
        )

        eos_token_id = tokenizer.eos_token_id
        trimmed: list[list[int]] = []
        for sample in token_ids:
            if eos_token_id is not None and eos_token_id in sample:
                sample = sample[: sample.index(eos_token_id) + 1]
            trimmed.append(sample)

        texts = tokenizer.batch_decode(trimmed, skip_special_tokens=True)
        payload = {
            "generated_seqs": texts,
            "steps": args.steps_label,
            "temperature": temperature,
            "generator_model": args.generator_model,
            "num_samples": args.num_samples,
            "sample_length": args.sample_length,
            "seed": args.seed,
            "top_p": args.top_p,
        }
        out_path.write_text(json.dumps(payload), encoding="utf-8")

    print("Done.")


if __name__ == "__main__":
    main()
