import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
from .train_p_linear import (
    HashingConfig,
    P_LINEAR_HEADS,
)
from sklearn.metrics import precision_recall_curve


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _load_jsonl(
    path: Path, max_rows: int | None = None
) -> Tuple[List[str], Dict[str, List[int]]]:
    texts: List[str] = []
    labels: Dict[str, List[int]] = {head: [] for head in P_LINEAR_HEADS}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if max_rows is not None and len(texts) >= max_rows:
                break

            line = line.strip()

            if not line:
                continue

            obj = json.loads(line)
            text = obj.get("text")

            if not isinstance(text, str):
                continue

            raw_labels = obj.get("labels") or {}

            if not isinstance(raw_labels, dict):
                raw_labels = {}

            texts.append(text)

            for head in P_LINEAR_HEADS:
                labels[head].append(int(raw_labels.get(head, 0)) != 0)

    return texts, labels


def _load_weights(path: Path) -> tuple[HashingConfig, Dict[str, Dict[str, object]]]:
    data = json.loads(path.read_text(encoding="utf-8"))

    hashing = data.get("hashing")
    heads = data.get("heads")

    if not isinstance(hashing, dict) or not isinstance(heads, dict):
        raise ValueError("Weights JSON must contain 'hashing' and 'heads' objects.")

    config = HashingConfig(
        n_features=int(hashing.get("n_features")),
        ngram_min=int(hashing.get("ngram_min")),
        ngram_max=int(hashing.get("ngram_max")),
    )

    return config, heads

def _pick_threshold_max_f1(y_true: np.ndarray, y_score: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)

    if thresholds.size == 0:
        return 0.5

    denom = precision[:-1] + recall[:-1]

    f1 = np.divide(
        2.0 * precision[:-1] * recall[:-1],
        denom,
        out=np.zeros_like(denom),
        where=denom != 0,
    )

    best_idx = int(np.argmax(f1))

    return float(thresholds[best_idx])
