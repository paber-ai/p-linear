use wasm_bindgen::prelude::*;

/// Output of the p-linear gating model.
///
/// This is a stub implementation that currently uses simple heuristics over
/// the input text. It is structured so that it can later be replaced by a
/// weight-based implementation generated from the Python training pipeline.
#[wasm_bindgen]
pub struct PLinearResult {
    p_simple: f32,
    p_complex: f32,
    p_needs_tools: f32,
    p_needs_memory: f32,
    p_high_risk: f32,
    p_code_like: f32,
}
