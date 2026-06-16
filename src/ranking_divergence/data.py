from __future__ import annotations

from itertools import islice
from typing import Iterable


def load_openwebtext_texts(
    *,
    split: str = "train[-100000:]",
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
        "openwebtext",
        split=split,
        cache_dir=cache_dir,
        streaming=streaming,
        trust_remote_code=True,
    )
    rows: Iterable[dict] = dataset
    if limit is not None:
        rows = islice(rows, limit)
    return [row["text"] for row in rows]
