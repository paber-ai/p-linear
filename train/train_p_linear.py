"""Training scaffold for the p-linear query-gating model.

This module defines utilities to construct a character-level HashingVectorizer
and linear classifiers for the various p-linear heads, and to export the
trained weights in a format that the Rust/WASM inference engine can consume.

Actual dataset loading and label construction are intentionally left to be
implemented once the concrete data sources are finalized.
"""

from typing import List, Mapping
from dataclasses import dataclass


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
