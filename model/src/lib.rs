mod weights;
use wasm_bindgen::prelude::*;
use crate::weights::{HeadWeights, HEADS, NGRAM_MAX, NGRAM_MIN, N_FEATURES};

const THRESH_SIMPLE: f32 = 0.8516830801963806;
const THRESH_COMPLEX: f32 = 0.1393413543701172;
const THRESH_NEEDS_TOOLS: f32 = 0.6246927380561829;
const THRESH_NEEDS_MEMORY: f32 = 0.9285847544670105;
const THRESH_HIGH_RISK: f32 = 0.5;
const THRESH_CODE_LIKE: f32 = 0.609478771686554;

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
    p_needs_retrieval: f32,
    p_enable_reg: f32,
    p_high_risk: f32,
    p_code_like: f32,
    d_simple: bool,
    d_complex: bool,
    d_needs_tools: bool,
    d_needs_memory: bool,
    d_needs_retrieval: bool,
    d_enable_reg: bool,
    d_high_risk: bool,
    d_code_like: bool,
}

fn sigmoid(x: f32) -> f32 {
    let x = x.clamp(-16.0, 16.0);
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
    pub fn p_needs_retrieval(&self) -> f32 {
        self.p_needs_retrieval
    }

    #[wasm_bindgen(getter)]
    pub fn p_enable_reg(&self) -> f32 {
        self.p_enable_reg
    }

    #[wasm_bindgen(getter)]
    pub fn p_high_risk(&self) -> f32 {
        self.p_high_risk
    }

    #[wasm_bindgen(getter)]
    pub fn p_code_like(&self) -> f32 {
        self.p_code_like
    }

    #[wasm_bindgen(getter)]
    pub fn d_simple(&self) -> bool {
        self.d_simple
    }

    #[wasm_bindgen(getter)]
    pub fn d_complex(&self) -> bool {
        self.d_complex
    }

    #[wasm_bindgen(getter)]
    pub fn d_needs_tools(&self) -> bool {
        self.d_needs_tools
    }

    #[wasm_bindgen(getter)]
    pub fn d_needs_memory(&self) -> bool {
        self.d_needs_memory
    }

    #[wasm_bindgen(getter)]
    pub fn d_needs_retrieval(&self) -> bool {
        self.d_needs_retrieval
    }

    #[wasm_bindgen(getter)]
    pub fn d_enable_reg(&self) -> bool {
        self.d_enable_reg
    }

    #[wasm_bindgen(getter)]
    pub fn d_high_risk(&self) -> bool {
        self.d_high_risk
    }

    #[wasm_bindgen(getter)]
    pub fn d_code_like(&self) -> bool {
        self.d_code_like
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
    let needs_retrieval_markers = lower.contains("source")
        || lower.contains("sources")
        || lower.contains("citation")
        || lower.contains("citations")
        || lower.contains("cite")
        || lower.contains("reference")
        || lower.contains("references")
        || lower.contains("according to")
        || lower.contains("arxiv")
        || lower.contains("wikipedia")
        || lower.contains("pubmed")
        || lower.contains("doi")
        || lower.contains("where did you get")
        || lower.contains("is it true")
        || lower.contains("fact check")
        || lower.contains("verify");

    let high_risk_markers = lower.contains("medical")
        || lower.contains("diagnos")
        || lower.contains("treatment")
        || lower.contains("dosage")
        || lower.contains("prescription")
        || lower.contains("legal")
        || lower.contains("lawsuit")
        || lower.contains("contract")
        || lower.contains("tax")
        || lower.contains("investment")
        || lower.contains("financial advice")
        || lower.contains("suicide")
        || lower.contains("self-harm")
        || lower.contains("harm")
        || lower.contains("weapon");
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
    let mut p_high_risk = if high_risk_markers { 0.95_f32 } else { 0.05_f32 };

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
        p_high_risk = p_high_risk.max(p);
    }

    if let Some(p) = linear_score_for_head(text, HEADS.code_like) {
        p_code_like = p;
    }

    let d_simple = p_simple >= THRESH_SIMPLE;
    let d_complex = p_complex >= THRESH_COMPLEX;
    let d_needs_tools = p_needs_tools >= THRESH_NEEDS_TOOLS;
    let d_needs_memory = p_needs_memory >= THRESH_NEEDS_MEMORY;
    let d_high_risk = p_high_risk >= THRESH_HIGH_RISK;
    let d_code_like = p_code_like >= THRESH_CODE_LIKE;

    // Derived tasks (no additional heads required):
    // - needs_retrieval: retrieval tends to be valuable when the query is complex
    //   or explicitly needs memory/context.
    // - enable_reg: prefer enabling ReG when retrieval is needed *and* complexity
    //   is moderate/high (proxy for multi-hop).
    let p_needs_retrieval = p_needs_memory.max(p_complex).max(if needs_retrieval_markers { 0.8 } else { 0.0 });
    let d_needs_retrieval = d_needs_memory || d_complex || needs_retrieval_markers;

    let p_enable_reg = (0.65 * p_needs_retrieval + 0.35 * p_complex).clamp(0.0, 1.0);
    let d_enable_reg = d_needs_retrieval && p_complex >= 0.35;

    PLinearResult {
        p_simple,
        p_complex,
        p_needs_tools,
        p_needs_memory,
        p_needs_retrieval,
        p_enable_reg,
        p_high_risk,
        p_code_like,
        d_simple,
        d_complex,
        d_needs_tools,
        d_needs_memory,
        d_needs_retrieval,
        d_enable_reg,
        d_high_risk,
        d_code_like,
    }
}
