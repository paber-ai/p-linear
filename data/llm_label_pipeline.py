"""Interactive CLI for LLM-based auto-labeling into the p-linear JSONL format.

This script is intended to run from the p-linear project root, e.g.:

    python -m data.llm_label_pipeline

It will interactively ask for:
- A dataset identifier (for your own tracking, e.g. an HF dataset ID).
- A local data file path (CSV / JSON / Parquet) to load with pandas.
- A comma-separated list of columns whose contents should be concatenated
  into the input `text` for p-linear.

For each row, it calls a LLM to analyze the text and predict the
p-linear heads:

    simple, complex, needs_tools, needs_memory, high_risk, code_like

It writes one JSON object per line to a JSONL file (default: data/data.jsonl
relative to the project root) with the canonical schema expected by the
training code:

    {"text": "...", "labels": {"simple": 0/1, ...}}

Note: this script uses the official Ollama Python SDK with custom structured
outputs, You are responsible for
downloading any HF/Kaggle datasets locally (e.g., as CSV/JSON/Parquet) before
running the script.
"""
