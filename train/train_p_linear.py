"""Training scaffold for the p-linear query-gating model.

This module defines utilities to construct a character-level HashingVectorizer
and linear classifiers for the various p-linear heads, and to export the
trained weights in a format that the Rust/WASM inference engine can consume.

Actual dataset loading and label construction are intentionally left to be
implemented once the concrete data sources are finalized.
"""

from __future__ import annotations
import argparse
import json
import numpy as np
from pathlib import Path
from scipy.sparse import csr_matrix
from dataclasses import dataclass
from typing import List, Mapping, Sequence, Dict, Iterable
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier


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


def build_label_models(heads: Iterable[str] | None = None) -> Dict[str, SGDClassifier]:
    """Create an SGDClassifier per p-linear head.

    Each head is treated as an independent binary classifier over the same
    hashed feature space. We use logistic loss so that downstream consumers
    can interpret outputs as probabilities after applying a sigmoid.
    """

    if heads is None:
        heads = P_LINEAR_HEADS

    models: Dict[str, SGDClassifier] = {}

    for name in heads:
        models[name] = SGDClassifier(
            loss="log_loss",
            penalty="l2",
            alpha=1e-4,
            learning_rate="optimal",
            max_iter=1000,
            tol=1e-3,
            shuffle=True,
            class_weight="balanced",
            random_state=42,
        )

    return models


def export_weights(
    models: Mapping[str, SGDClassifier],
    config: HashingConfig,
    output_path: str | Path,
) -> None:
    """Export model weights and hashing config to JSON.

    The resulting file is intended to be ingested by a small generator that
    produces Rust constants used by the WASM inference engine.

    Parameters
    ----------
    models:
        Mapping from head name to trained SGDClassifier. Each classifier is
        expected to be binary (one set of coefficients).
    config:
        Hashing configuration used to construct the feature extractor.
    output_path:
        File path to write the JSON payload to.
    """

    payload: Dict[str, object] = {
        "hashing": {
            "n_features": config.n_features,
            "ngram_min": config.ngram_min,
            "ngram_max": config.ngram_max,
            "analyzer": config.analyzer,
        },
        "heads": {},
    }

    heads: Dict[str, object] = {}

    for name, model in models.items():
        if not hasattr(model, "coef_") or not hasattr(model, "intercept_"):
            raise ValueError(f"Model for head '{name}' is not trained (missing coef_).")

        coef = model.coef_
        intercept = model.intercept_

        if coef.shape[0] != 1:
            raise ValueError(
                f"Head '{name}' expected binary classifier (1 row), got {coef.shape[0]} rows."
            )

        heads[name] = {
            "weights": coef[0].tolist(),
            "bias": float(intercept[0]),
        }

    payload["heads"] = heads

    output_path = Path(output_path)
    output_path.write_text(json.dumps(payload), encoding="utf-8")


def train_on_examples(
    examples: Sequence[TrainingExample],
    config: HashingConfig | None = None,
) -> tuple[HashingConfig, Dict[str, SGDClassifier]]:
    """Train p-linear heads on an in-memory list of labeled examples.

    This helper is dataset-agnostic: it assumes that the caller has already
    constructed a list of TrainingExample instances from their preferred
    sources (Hugging Face datasets, logs, etc.). It fits one SGDClassifier per
    head over a shared hashed feature space.

    Heads that only see a single class in the provided labels are currently
    skipped to avoid fitting degenerate models; callers may choose to provide
    more diverse data for those heads.
    """

    if not examples:
        raise ValueError("No training examples provided to train_on_examples().")

    if config is None:
        config = HashingConfig()

    models = build_label_models()

    texts = [ex.text for ex in examples]
    X = build_feature_matrix(texts, config)

    for head, model in models.items():
        y = [int(ex.labels.get(head, 0)) for ex in examples]

        # Skip heads where all labels are identical; caller can decide how to
        # handle these cases (e.g., collect more data).
        if len(set(y)) < 2:
            continue

        model.fit(X, y)

    return config, models


def main() -> None:
    """Command-line entry point for p-linear training.

    This CLI expects a JSONL file where each line is a JSON object of the form:

    {"text": "...", "labels": {"simple": 1, "complex": 0, ...}}

    Only the heads defined in P_LINEAR_HEADS are used. Missing heads in the
    labels mapping default to 0.
    """

    parser = argparse.ArgumentParser(description="Train the p-linear model.")
    parser.add_argument(
        "--train-jsonl",
        type=str,
        required=True,
        help=(
            "Path to a JSONL file with one training example per line. "
            "Each line must contain at least a 'text' field and an optional "
            "'labels' mapping."
        ),
    )

    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Path to write the exported weights JSON for Rust/WASM.",
    )

    parser.add_argument(
        "--n-features",
        type=int,
        default=2**15,
        help="Number of hashing buckets (features) to use.",
    )

    parser.add_argument(
        "--ngram-min",
        type=int,
        default=2,
        help="Minimum character n-gram size.",
    )

    parser.add_argument(
        "--ngram-max",
        type=int,
        default=3,
        help="Maximum character n-gram size.",
    )

    args = parser.parse_args()

    config = HashingConfig(
        n_features=args.n_features,
        ngram_min=args.ngram_min,
        ngram_max=args.ngram_max,
    )

    train_path = Path(args.train_jsonl)

    if not train_path.is_file():
        raise SystemExit(f"Training JSONL file not found: {train_path}")

    examples: List[TrainingExample] = []

    with train_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line:
                continue

            obj = json.loads(line)

            text = obj.get("text")

            if not isinstance(text, str):
                raise ValueError("Each JSONL line must contain a 'text' string field.")

            raw_labels = obj.get("labels") or {}

            if not isinstance(raw_labels, Mapping):
                raise ValueError("'labels' must be a mapping from head name to 0/1.")

            # Normalize labels to int flags.
            labels: Dict[str, int] = {}

            for head in P_LINEAR_HEADS:
                value = raw_labels.get(head, 0)
                labels[head] = int(value)

            examples.append(TrainingExample(text=text, labels=labels))

    if not examples:
        raise SystemExit("No valid training examples were loaded from JSONL.")

    config, models = train_on_examples(examples, config)
    export_weights(models, config, args.output)

    print(f"Trained p-linear on {len(examples)} examples.")
    print(f"Exported weights to {args.output}.")


if __name__ == "__main__":
    main()
