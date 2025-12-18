import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple
from .train_p_linear import P_LINEAR_HEADS

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
