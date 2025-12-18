from __future__ import annotations
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
from .train_p_linear import (
    HashingConfig,
    P_LINEAR_HEADS,
    build_feature_matrix,
)
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


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


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate p-linear weights on a labeled JSONL dataset and suggest thresholds."
    )

    parser.add_argument(
        "--data-jsonl",
        type=str,
        default="data/data.jsonl",
        help="Path to labeled JSONL data (same schema as training).",
    )

    parser.add_argument(
        "--weights-json",
        type=str,
        default="artifacts/p_linear_weights.json",
        help="Path to exported weights JSON.",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/p_linear_thresholds.json",
        help="Where to write the thresholds + metrics JSON.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction of rows used for evaluation/threshold picking.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for the evaluation split.",
    )

    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional cap on number of rows loaded from JSONL (for quick runs).",
    )

    args = parser.parse_args()

    data_path = Path(args.data_jsonl)
    weights_path = Path(args.weights_json)
    output_path = Path(args.output)

    if not data_path.is_file():
        raise SystemExit(f"Data JSONL not found: {data_path}")

    if not weights_path.is_file():
        raise SystemExit(f"Weights JSON not found: {weights_path}")

    if not (0.0 < args.test_size < 1.0):
        raise SystemExit("test-size must be between 0 and 1")

    config, heads = _load_weights(weights_path)
    texts, labels = _load_jsonl(data_path, max_rows=args.max_rows)

    if not texts:
        raise SystemExit("No valid rows loaded from data JSONL.")

    n = len(texts)
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(n)
    eval_n = max(1, int(round(n * float(args.test_size))))
    eval_idx = perm[:eval_n]

    eval_texts = [texts[i] for i in eval_idx]

    y_eval: Dict[str, np.ndarray] = {
        head: np.asarray([labels[head][i] for i in eval_idx], dtype=np.int32)
        for head in P_LINEAR_HEADS
    }

    X_eval = build_feature_matrix(eval_texts, config)

    thresholds_out: Dict[str, float] = {}
    metrics_out: Dict[str, Dict[str, object]] = {}

    for head in P_LINEAR_HEADS:
        head_obj = heads.get(head)

        if not isinstance(head_obj, dict):
            raise ValueError(f"Missing head '{head}' in weights JSON.")

        weights = head_obj.get("weights")
        bias = head_obj.get("bias")

        if not isinstance(weights, list):
            raise ValueError(f"Head '{head}' has invalid weights.")

        w = np.asarray(weights, dtype=np.float32)
        b = float(bias)

        if w.shape[0] != int(config.n_features):
            raise ValueError(
                f"Head '{head}' weight length {w.shape[0]} != n_features {config.n_features}."
            )

        scores = np.asarray(X_eval.dot(w)).reshape(-1) + b
        probs = _sigmoid(scores)

        y_true = y_eval[head]

        threshold = 0.5

        if np.unique(y_true).size >= 2:
            threshold = _pick_threshold_max_f1(y_true, probs)

        y_pred = (probs >= threshold).astype(np.int32)

        tp = int(np.sum((y_pred == 1) & (y_true == 1)))
        fp = int(np.sum((y_pred == 1) & (y_true == 0)))
        tn = int(np.sum((y_pred == 0) & (y_true == 0)))
        fn = int(np.sum((y_pred == 0) & (y_true == 1)))

        head_metrics: Dict[str, object] = {
            "threshold": float(threshold),
            "support": int(y_true.shape[0]),
            "positives": int(np.sum(y_true)),
            "prevalence": float(np.mean(y_true)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }

        if np.unique(y_true).size >= 2:
            head_metrics["roc_auc"] = float(roc_auc_score(y_true, probs))
            head_metrics["average_precision"] = float(
                average_precision_score(y_true, probs)
            )

        thresholds_out[head] = float(threshold)
        metrics_out[head] = head_metrics

    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, object] = {
        "hashing": {
            "n_features": int(config.n_features),
            "ngram_min": int(config.ngram_min),
            "ngram_max": int(config.ngram_max),
        },
        "split": {
            "seed": int(args.seed),
            "test_size": float(args.test_size),
            "num_rows": int(n),
            "num_eval": int(eval_n),
        },
        "thresholds": thresholds_out,
        "metrics": metrics_out,
    }

    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(f"Evaluated {eval_n}/{n} rows.")
    print(f"Wrote thresholds + metrics to {output_path}.")


if __name__ == "__main__":
    main()
