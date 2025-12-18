mod weights;
use wasm_bindgen::prelude::*;
use crate::weights::{HeadWeights, NGRAM_MIN, NGRAM_MAX, N_FEATURES};

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
