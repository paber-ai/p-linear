"""Training scaffold for the p-linear query-gating model.

This module defines utilities to construct a character-level HashingVectorizer
and linear classifiers for the various p-linear heads, and to export the
trained weights in a format that the Rust/WASM inference engine can consume.

Actual dataset loading and label construction are intentionally left to be
implemented once the concrete data sources are finalized.
"""
