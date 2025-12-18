mod weights;
use wasm_bindgen::prelude::*;
use crate::weights::{HeadWeights, HEADS, NGRAM_MAX, NGRAM_MIN, N_FEATURES};

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

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

fn hash_ngram(chars: &[char]) -> usize {
    // Simple FNV-1a over UTF-8 bytes of each char.
    const FNV_OFFSET: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x100000001b3;

    let mut hash = FNV_OFFSET;

    for &ch in chars {
        let mut buf = [0u8; 4];

        for b in ch.encode_utf8(&mut buf).as_bytes() {
            hash ^= *b as u64;
            hash = hash.wrapping_mul(FNV_PRIME);
        }
    }

    if N_FEATURES.is_power_of_two() {
        (hash as usize) & (N_FEATURES - 1)
    } else {
        (hash as usize) % N_FEATURES
    }
}

fn linear_score_for_head(text: &str, head: &HeadWeights) -> Option<f32> {
    if head.weights.len() != N_FEATURES {
        return None;
    }

    let chars: Vec<char> = text.chars().collect();

    if chars.is_empty() {
        return Some(sigmoid(head.bias));
    }

    let len = chars.len();
    let mut sum = head.bias;

    for n in NGRAM_MIN..=NGRAM_MAX {
        if n == 0 || n > len {
            continue;
        }

        for start in 0..=(len - n) {
            let idx = hash_ngram(&chars[start..start + n]);

            if idx < head.weights.len() {
                sum += head.weights[idx];
            }
        }
    }

    Some(sigmoid(sum))
}

#[wasm_bindgen]
impl PLinearResult {
    #[wasm_bindgen(getter)]
    pub fn p_simple(&self) -> f32 {
        self.p_simple
    }

    #[wasm_bindgen(getter)]
    pub fn p_complex(&self) -> f32 {
        self.p_complex
    }

    #[wasm_bindgen(getter)]
    pub fn p_needs_tools(&self) -> f32 {
        self.p_needs_tools
    }

    #[wasm_bindgen(getter)]
    pub fn p_needs_memory(&self) -> f32 {
        self.p_needs_memory
    }

    #[wasm_bindgen(getter)]
    pub fn p_high_risk(&self) -> f32 {
        self.p_high_risk
    }

    #[wasm_bindgen(getter)]
    pub fn p_code_like(&self) -> f32 {
        self.p_code_like
    }
}

/// Analyze a query and return heuristic p-linear routing probabilities.
///
/// This function currently uses a very lightweight heuristic based on query
/// length and a few pattern checks (code-like markers). It is intended to be
/// swapped out with a real linear model whose weights are generated from the
/// Python training pipeline.
#[wasm_bindgen]
pub fn analyze_query(text: &str) -> PLinearResult {
    let len = text.chars().count() as f32;
    let normalized_len = (len / 200.0).min(1.0);

    // Naive code-like detection.
    let lower = text.to_lowercase();
    let has_code_markers = lower.contains("```")
        || lower.contains("fn ")
        || lower.contains("class ")
        || lower.contains("def ")
        || lower.contains("public static void")
        || lower.contains("#include");

    let mut p_code_like = if has_code_markers { 0.9 } else { 0.1 };

    // Simple vs complex: longer queries are treated as more complex.
    let mut p_complex = normalized_len;
    let mut p_simple = (1.0 - p_complex * 0.7).clamp(0.0, 1.0);

    // Placeholder estimates for tools, memory, and risk.
    let mut p_needs_tools = 0.2;
    let mut p_needs_memory = (normalized_len * 0.5).clamp(0.0, 1.0);
    let mut p_high_risk = 0.1;

    // If learned weights are available, override heuristic estimates per head.
    if let Some(p) = linear_score_for_head(text, HEADS.simple) {
        p_simple = p;
    }

    if let Some(p) = linear_score_for_head(text, HEADS.complex) {
        p_complex = p;
    }

    if let Some(p) = linear_score_for_head(text, HEADS.needs_tools) {
        p_needs_tools = p;
    }

    if let Some(p) = linear_score_for_head(text, HEADS.needs_memory) {
        p_needs_memory = p;
    }

    if let Some(p) = linear_score_for_head(text, HEADS.high_risk) {
        p_high_risk = p;
    }

    if let Some(p) = linear_score_for_head(text, HEADS.code_like) {
        p_code_like = p;
    }

    PLinearResult {
        p_simple,
        p_complex,
        p_needs_tools,
        p_needs_memory,
        p_high_risk,
        p_code_like,
    }
}
