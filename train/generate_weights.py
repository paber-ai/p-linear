"""Generate Rust weights.rs from p-linear exported weights JSON.

This script consumes the JSON produced by export_weights(...) in
train_p_linear.py and emits a Rust source file that defines the hashing
configuration and per-head weight arrays used by the WASM inference engine.

Usage (from the p-linear project root):

    python -m train.generate_weights_rs \
        --weights-json artifacts/p_linear_weights.json \
        --output model/src/weights.rs

The generated Rust module is intended to replace the placeholder
`model/src/weights.rs` that ships with the repository.
"""
