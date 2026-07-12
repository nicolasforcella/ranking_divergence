from __future__ import annotations

from itertools import islice
from typing import Iterable

OWT_SAMPLER_SOURCE_SPLIT = "train[:-100000]"
OWT_HELDOUT_SPLIT = "train[-100000:]"
DUO_SCRATCH_DIR = "/home/patrick/.cache/discrete_diffusion"
DUO_OWT_CACHE_DIR = f"{DUO_SCRATCH_DIR}/owt"


def openwebtext_split_config() -> dict[str, str]:
    """Return the OpenWebText split/cache convention used for this analysis."""

    return {
        "sampler_source_split": OWT_SAMPLER_SOURCE_SPLIT,
        "heldout_split": OWT_HELDOUT_SPLIT,
        "scratch_dir": DUO_SCRATCH_DIR,
        "cache_dir": DUO_OWT_CACHE_DIR,
    }


def load_openwebtext_texts(
    *,
    split: str = OWT_HELDOUT_SPLIT,
    cache_dir: str | None = None,
    limit: int | None = None,
    streaming: bool = False,
) -> list[str]:
    """Load raw OpenWebText strings using the DUO train/validation split style."""

    try:
        import datasets
    except ImportError as exc:
        raise ImportError("Install examples dependencies with `uv pip install -e '.[examples]'`.") from exc

    dataset = datasets.load_dataset(
        "Skylion007/openwebtext",
        split=split,
        cache_dir=cache_dir,
        streaming=streaming,
        trust_remote_code=True,
    )
    rows: Iterable[dict] = dataset
    if limit is not None:
        rows = islice(rows, limit)
    return [row["text"] for row in rows]
