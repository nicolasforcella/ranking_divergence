"""Copy from evaluate diffusion sweeps for the genPPL and Entropy computation."""
from __future__ import annotations
import argparse
import csv
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import NamedTuple, Sequence
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from ranking_divergence import lexical_metrics
from ranking_divergence.evaluation import score_tokens_no_rank


SAMPLE_RE = re.compile(r"^samples_steps(?P<nfe>\d+)_temp(?P<temperature>[0-9.]+)\.json$")


class SweepFile(NamedTuple):
    method: str
    path: Path
    nfe: int
    temperature_label: str

    @property
    def key(self) -> str:
        return f"{self.method}__{self.path.stem}"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate gen-PPL and model entropy over generation sweeps."
    )
    parser.add_argument(
        "--sweep",
        action="append",
        default=None,
        metavar="METHOD=DIR",
        help="Sweep directory. Repeatable. A bare DIR uses the directory name as METHOD.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gen_frontiers"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--scorer-model", default="gpt2-large")
    parser.add_argument("--scorer-temperature", default="1.0", help="Scorer temperature.")
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--rank-position-chunk", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="auto", choices=["float16", "bfloat16", "float32", "auto"], help="Scorer model's load dtype.")
    parser.add_argument("--limit-configs", type=int, default=None)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--per-sample", action="store_true", help="Store the per-sample metric tensor for each config.")
    parser.add_argument("--inventory-only", action="store_true")
    parser.add_argument("--force", action="store_true", help="Recompute existing checkpoints.")
    args = parser.parse_args(argv)

    return args


def parse_sweeps(values: Sequence[str] | None) -> dict[str, Path]:
    if not values:
        raise ValueError("Provide at least one --sweep METHOD=DIR or --sweep DIR.")
    sweeps: dict[str, Path] = {}
    for value in values:
        if "=" in value:
            method, directory = value.split("=", 1)
            method = method.strip()
        else:
            directory = value
            method = Path(directory).expanduser().resolve().name
        if not method:
            raise ValueError(f"Invalid --sweep {value!r}; METHOD is empty.")
        sweeps[method] = Path(directory).expanduser()
    return sweeps


def discover_sweep_files(sweeps: dict[str, Path]) -> list[SweepFile]:
    files: list[SweepFile] = []
    for method, directory in sweeps.items():
        if not directory.is_dir():
            raise FileNotFoundError(f"Sweep directory does not exist: {directory}")
        for path in directory.glob("samples_steps*_temp*.json"):
            match = SAMPLE_RE.fullmatch(path.name)
            if match is None:
                continue
            files.append(
                SweepFile(
                    method=method,
                    path=path.resolve(),
                    nfe=int(match.group("nfe")),
                    temperature_label=match.group("temperature"),
                )
            )
    return sorted(files, key=lambda item: (item.method, item.nfe, float(item.temperature_label)))


def inventory(files: Sequence[SweepFile]) -> dict:
    methods: dict[str, dict] = {}
    for method in sorted({item.method for item in files}):
        method_files = [item for item in files if item.method == method]
        nfes = sorted({item.nfe for item in method_files})
        temperatures = sorted({item.temperature_label for item in method_files}, key=float)
        observed = {(item.nfe, item.temperature_label) for item in method_files}
        missing = [
            {"nfe": nfe, "temperature_label": temperature}
            for nfe in nfes
            for temperature in temperatures
            if (nfe, temperature) not in observed
        ]
        methods[method] = {
            "directory": str(method_files[0].path.parent) if method_files else None,
            "num_files": len(method_files),
            "nfes": nfes,
            "temperature_labels": temperatures,
            "missing_grid_points": missing,
        }
    return {"num_files": len(files), "methods": methods}


def load_generation_file(item: SweepFile, limit_samples: int | None = None) -> tuple[list[str], dict]:
    payload = json.loads(item.path.read_text(encoding="utf-8"))
    texts = payload.get("generated_seqs")
    if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
        raise ValueError(f"{item.path} does not contain a string list at 'generated_seqs'.")
    payload_nfe = int(payload.get("steps", item.nfe))
    if payload_nfe != item.nfe:
        raise ValueError(f"NFE mismatch in {item.path}: filename={item.nfe}, payload={payload_nfe}.")
    if limit_samples is not None:
        texts = texts[:limit_samples]
    if not texts:
        raise ValueError(f"No generations found in {item.path}.")
    return texts, payload


def resolve_device(value: str) -> str:
    if value != "auto":
        return value
    return "cuda" if torch.cuda.is_available() else "cpu"


def run_dir_for(args: argparse.Namespace) -> Path:
    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return args.output_dir / run_name


def tokenize_texts(texts: Sequence[str], tokenizer) -> list[list[int]]:
    return [tokenizer.encode(text, add_special_tokens=False) for text in texts]


def write_json(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_checkpoints(checkpoint_dir: Path) -> list[dict]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(checkpoint_dir.glob("*.json"))
    ]


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    sweeps = parse_sweeps(args.sweep)
    files = discover_sweep_files(sweeps)
    if args.limit_configs is not None:
        files = files[: args.limit_configs]

    run_dir = run_dir_for(args)
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    data_dir = run_dir / "data"
    if args.per_sample:
        data_dir.mkdir(parents=True, exist_ok=True)

    inventory_payload = inventory(files)
    write_json(run_dir / "inventory.json", inventory_payload)
    print(json.dumps(inventory_payload, indent=2))
    if args.inventory_only:
        print(f"Wrote sweep inventory to {run_dir / 'inventory.json'}")
        return

    args.device = resolve_device(args.device)
    metadata_path = run_dir / "metadata.json"
    evaluation_config = {
        "scorer_model": args.scorer_model,
        "scorer_temperature": args.scorer_temperature,
        "device": args.device,
        "dtype": args.dtype,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "rank_position_chunk": args.rank_position_chunk,
        "limit_samples": args.limit_samples,
        "sweeps": {method: str(path) for method, path in sweeps.items()},
    }
    if metadata_path.exists() and not args.force:
        existing_metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        existing_config = {key: existing_metadata.get(key) for key in evaluation_config}
        if existing_config != evaluation_config:
            raise ValueError(
                f"{run_dir} contains checkpoints from different evaluation settings. "
                "Use a new --run-name or pass --force."
            )
        metadata = existing_metadata
    else:
        metadata = {"created_utc": datetime.now(timezone.utc).isoformat(), **evaluation_config}
        write_json(metadata_path, metadata)

    #Load scorer and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.scorer_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = args.dtype if args.dtype == "auto" else getattr(torch, args.dtype)
    model = AutoModelForCausalLM.from_pretrained(args.scorer_model, dtype=dtype).to(args.device).eval()

    for index, item in enumerate(files, start=1):
        checkpoint_path = checkpoint_dir / f"{item.key}.json"
        if checkpoint_path.exists() and not args.force:
            print(f"[{index}/{len(files)}] skipping completed {item.key}")
            continue

        print(f"[{index}/{len(files)}] evaluating {item.key}")

        texts, source = load_generation_file(item, args.limit_samples)
        token_ids = tokenize_texts(texts, tokenizer)

        #If matched temperature is set, retrieve it from the model (used for finding real temperature)
        temperature = float(source.get("temperature", item.temperature_label))
        scorer_temperature = temperature if args.scorer_temperature == "match" else float(args.scorer_temperature)

        result_per_sample = score_tokens_no_rank(
            token_ids,
            model,
            tokenizer,
            batch_size=args.batch_size,
            max_length=args.max_length,
            rank_position_chunk=args.rank_position_chunk,
            device=args.device,
            per_sample=True,
            temperature=scorer_temperature,
        )
        if args.per_sample:
            torch.save(result_per_sample, data_dir / f"{item.key}.pt")

        totals = result_per_sample.sum(dim=0)
        total_tokens = float(totals[2].item())
        gen_ppl = math.exp(float(totals[0].item()) /total_tokens)
        model_entropy = float(totals[1].item()) /total_tokens
        row = {
            "method": item.method,
            "nfe": item.nfe,
            "temperature": temperature,
            "temperature_label": item.temperature_label,
            "scorer_temperature": scorer_temperature,
            "source_file": str(item.path),
            "num_samples": len(texts),
            "num_tokens": total_tokens,
            "source_gen_ppl": source.get("generative_ppl"),
            "source_entropy": source.get("entropy"),
            "gen_ppl": gen_ppl,
            "model_entropy": model_entropy,
            "kl": math.log(gen_ppl) - model_entropy,
        }
        row.update(lexical_metrics(texts, token_ids))
        write_json(checkpoint_path, row)

        rows = sorted(
            load_checkpoints(checkpoint_dir),
            key=lambda value: (value["method"], value["nfe"], value["temperature"]),
        )
        write_csv(run_dir / "metrics.csv", rows)
        write_json(run_dir / "metrics.json", {"metadata": metadata, "metrics": rows})

    rows = sorted(
        load_checkpoints(checkpoint_dir),
        key=lambda value: (value["method"], value["nfe"], value["temperature"]),
    )
    write_csv(run_dir / "metrics.csv", rows)
    write_json(run_dir / "metrics.json", {"metadata": metadata, "metrics": rows})
    print(f"Wrote {len(rows)} evaluated configurations to {run_dir}")


if __name__ == "__main__":
    main()
