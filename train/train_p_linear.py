"""Training scaffold for the p-linear query-gating model.

This module defines utilities to construct a character-level HashingVectorizer
and linear classifiers for the various p-linear heads, and to export the
trained weights in a format that the Rust/WASM inference engine can consume.

Actual dataset loading and label construction are intentionally left to be
implemented once the concrete data sources are finalized.
"""

from __future__ import annotations
from typing import List, Mapping, Sequence, Dict
import numpy as np
from scipy.sparse import csr_matrix
from dataclasses import dataclass
from sklearn.feature_extraction.text import HashingVectorizer


P_LINEAR_HEADS: List[str] = [
    "simple",
    "complex",
    "needs_tools",
    "needs_memory",
    "high_risk",
    "code_like",
]


_FNV_OFFSET = 0xCBF29CE484222325
_FNV_PRIME = 0x100000001B3
_U64_MASK = 0xFFFFFFFFFFFFFFFF



@dataclass
class HashingConfig:
    """Configuration for the character-level hashing features."""

    n_features: int = 2**15
    ngram_min: int = 2
    ngram_max: int = 3
    analyzer: str = "char"


@dataclass
class TrainingExample:
    """Single training example for p-linear.

    Labels are expected to be 0/1 flags for the known heads (keys from
    P_LINEAR_HEADS). Heads that are missing from the mapping are treated as 0.
    """

    text: str
    labels: Mapping[str, int]


def _fnv1a_hash_ngram(chars: Sequence[str], n_features: int) -> int:
    h = _FNV_OFFSET

    for ch in chars:
        for b in ch.encode("utf-8"):
            h ^= b
            h = (h * _FNV_PRIME) & _U64_MASK

    if n_features > 0 and (n_features & (n_features - 1) == 0):
        return int(h & (n_features - 1))

    return int(h % n_features)


def build_feature_matrix(texts: Sequence[str], config: HashingConfig) -> csr_matrix:
    n_features = int(config.n_features)
    ngram_min = int(config.ngram_min)
    ngram_max = int(config.ngram_max)

    indptr: list[int] = [0]
    indices: list[int] = []
    data: list[float] = []

    for text in texts:
        chars = list(text)
        length = len(chars)
        counts: Dict[int, int] = {}

        for n in range(ngram_min, ngram_max + 1):
            if n <= 0 or n > length:
                continue

            for start in range(0, length - n + 1):
                idx = _fnv1a_hash_ngram(chars[start : start + n], n_features)
                counts[idx] = counts.get(idx, 0) + 1

        if counts:
            for idx, value in sorted(counts.items()):
                indices.append(idx)
                data.append(float(value))

        indptr.append(len(indices))

    return csr_matrix(
        (
            np.asarray(data, dtype=np.float32),
            np.asarray(indices, dtype=np.int32),
            np.asarray(indptr, dtype=np.int32),
        ),
        shape=(len(texts), n_features),
        dtype=np.float32,
    )


def build_vectorizer(config: HashingConfig | None = None) -> HashingVectorizer:
    """Construct a HashingVectorizer for character n-grams.

    The vectorizer is stateless (hashing trick), so we only need to persist the
    configuration (n_features and n-gram settings) alongside the trained
    linear models.
    """

    if config is None:
        config = HashingConfig()

    return HashingVectorizer(
        n_features=config.n_features,
        analyzer=config.analyzer,
        ngram_range=(config.ngram_min, config.ngram_max),
        alternate_sign=False,
        decode_error="ignore",
    )
