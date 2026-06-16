from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Sequence

import torch


def token_frequencies(texts: Sequence[str], tokenizer) -> Counter[int]:
    """Count tokenizer IDs in a reference corpus."""

    counts: Counter[int] = Counter()
    for text in texts:
        counts.update(tokenizer.encode(text, add_special_tokens=False))
    return counts


def build_phrase_bank(
    texts: Sequence[str],
    tokenizer,
    *,
    n: int = 5,
    m: int = 1000,
) -> list[tuple[int, ...]]:
    """Return the top-m token n-grams by frequency."""

    counts: Counter[tuple[int, ...]] = Counter()
    for text in texts:
        ids = tokenizer.encode(text, add_special_tokens=False)
        counts.update(tuple(ids[i : i + n]) for i in range(max(0, len(ids) - n + 1)))
    return [phrase for phrase, _ in counts.most_common(m)]


@dataclass
class RestrictedMarginalSampler:
    """IID sampler from the top-k empirical unigram distribution."""

    token_ids: torch.Tensor
    probabilities: torch.Tensor
    tokenizer: object

    @classmethod
    def from_texts(cls, texts: Sequence[str], tokenizer, *, k: int) -> "RestrictedMarginalSampler":
        counts = token_frequencies(texts, tokenizer)
        if not counts:
            raise ValueError("Cannot build a sampler from empty text.")
        common = counts.most_common(k)
        token_ids = torch.tensor([token_id for token_id, _ in common], dtype=torch.long)
        weights = torch.tensor([count for _, count in common], dtype=torch.float64)
        return cls(token_ids, weights / weights.sum(), tokenizer)

    def sample_token_ids(self, *, num_samples: int, length: int, seed: int | None = None) -> list[list[int]]:
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        draws = torch.multinomial(
            self.probabilities,
            num_samples * length,
            replacement=True,
            generator=generator,
        ).reshape(num_samples, length)
        ids = self.token_ids[draws]
        return ids.tolist()

    def sample(self, *, num_samples: int, length: int, seed: int | None = None) -> list[str]:
        return self.tokenizer.batch_decode(
            self.sample_token_ids(num_samples=num_samples, length=length, seed=seed),
            skip_special_tokens=True,
        )


TopKSampler = RestrictedMarginalSampler


@dataclass
class MirrorSampler:
    base_sampler: RestrictedMarginalSampler

    def sample_token_ids(self, *, num_samples: int, length: int, seed: int | None = None) -> list[list[int]]:
        half = max(1, length // 2)
        first_halves = self.base_sampler.sample_token_ids(
            num_samples=num_samples,
            length=half,
            seed=seed,
        )
        samples = []
        for ids in first_halves:
            repeated = []
            while len(repeated) < length:
                repeated.extend(ids)
            samples.append(repeated[:length])
        return samples

    def sample(self, *, num_samples: int, length: int, seed: int | None = None) -> list[str]:
        return self.base_sampler.tokenizer.batch_decode(
            self.sample_token_ids(num_samples=num_samples, length=length, seed=seed),
            skip_special_tokens=True,
        )


@dataclass
class PeriodicSampler:
    token_ids: list[int]
    tokenizer: object

    @classmethod
    def from_texts(cls, texts: Sequence[str], tokenizer, *, k: int) -> "PeriodicSampler":
        counts = token_frequencies(texts, tokenizer)
        token_ids = [token_id for token_id, _ in counts.most_common(k)]
        if not token_ids:
            raise ValueError("Cannot build a sampler from empty text.")
        return cls(token_ids, tokenizer)

    def sample_token_ids(self, *, num_samples: int, length: int, seed: int | None = None) -> list[list[int]]:
        del seed
        return [[self.token_ids[i % len(self.token_ids)] for i in range(length)] for _ in range(num_samples)]

    def sample(self, *, num_samples: int, length: int, seed: int | None = None) -> list[str]:
        return self.tokenizer.batch_decode(
            self.sample_token_ids(num_samples=num_samples, length=length, seed=seed),
            skip_special_tokens=True,
        )


@dataclass
class PhraseBankSampler:
    phrase_bank: list[tuple[int, ...]]
    tokenizer: object

    @classmethod
    def from_texts(
        cls,
        texts: Sequence[str],
        tokenizer,
        *,
        n: int = 5,
        m: int = 1000,
    ) -> "PhraseBankSampler":
        phrase_bank = build_phrase_bank(texts, tokenizer, n=n, m=m)
        if not phrase_bank:
            raise ValueError("Cannot build phrase bank from texts shorter than n.")
        return cls(phrase_bank, tokenizer)

    def sample_token_ids(self, *, num_samples: int, length: int, seed: int | None = None) -> list[list[int]]:
        generator = None
        if seed is not None:
            generator = torch.Generator().manual_seed(seed)
        samples = []
        for _ in range(num_samples):
            ids: list[int] = []
            while len(ids) < length:
                index = int(torch.randint(len(self.phrase_bank), (1,), generator=generator).item())
                ids.extend(self.phrase_bank[index])
            samples.append(ids[:length])
        return samples

    def sample(self, *, num_samples: int, length: int, seed: int | None = None) -> list[str]:
        return self.tokenizer.batch_decode(
            self.sample_token_ids(num_samples=num_samples, length=length, seed=seed),
            skip_special_tokens=True,
        )
