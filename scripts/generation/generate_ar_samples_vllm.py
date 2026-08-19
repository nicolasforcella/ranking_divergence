from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Sequence
import sys
from tqdm import tqdm

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--generator-model", default="openai-community/gpt2")
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--sample-length", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--devices", default="auto")
    parser.add_argument("--dtype", default="float32",
                        help="Model precision; 'float32' keeps the original weights, "
                             "'auto' lets vllm downcast fp32 checkpoints to fp16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8,
                        help="Fraction of GPU memory vllm may use for weights + KV cache; "
                             "keeps headroom for avoiding OOM errors.")
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

def generate_texts(model, num_samples, length, temperature, top_p):
    # Cached import, only reached after main() has already loaded vllm
    from vllm import SamplingParams
    from vllm.inputs import TokensPrompt

    # Start each sample with BOS or EOS depending on model
    tokenizer = model.get_tokenizer()
    start_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else tokenizer.eos_token_id
    prompts = [TokensPrompt(prompt_token_ids=[start_id]) for _ in range(num_samples)]
    # Cap new to fit in the model context window
    max_model_len = model.llm_engine.model_config.max_model_len
    max_tokens = min(length, max_model_len - 1)

    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=max_tokens)
    outputs = model.generate(prompts, sampling_params)

    texts = [output.outputs[0].text for output in outputs]

    return texts



def main(argv: Sequence[str] | None = None) -> None:
    #First parse the args
    args = parse_args(argv)
    if args.devices != "auto":
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices #Fix visible devices for vllm

    #Import after setting cuda visible devices to avoid errors
    from vllm import LLM

    model_tag = args.generator_model.replace("/", "_") #Convert HF name to valid directory name
    run_dir = args.output_dir / model_tag
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Loading {args.generator_model}...")
    model = LLM(model=args.generator_model, seed=args.seed, dtype=args.dtype, gpu_memory_utilization=args.gpu_memory_utilization)

    for temperature in tqdm(args.temperatures, desc="Temperatures"):
        out_path = run_dir / f"samples_steps{args.steps_label}_temp{temperature:.3f}.json"
        if out_path.exists() and not args.force:
            print(f"Skipping existing {out_path}")
            continue
        print(f"Generating temperature={temperature} -> {out_path}")

        texts = generate_texts(model, args.num_samples, args.sample_length, temperature, args.top_p)

        config_metadata = {
            "generated_seqs": texts,
            "steps": args.steps_label,
            "temperature": temperature,
            "generator_model": args.generator_model,
            "dtype": args.dtype,
            "num_samples": args.num_samples,
            "sample_length": args.sample_length,
            "seed": args.seed,
            "top_p": args.top_p,
        }
        with out_path.open("w") as f:
            json.dump(config_metadata, f)


        print(f"Wrote {len(texts)} samples to {out_path}")


if __name__ == "__main__":
    main()