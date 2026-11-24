//! The `web-gpu` backend: on `wasm32`, the published gpu-lexer package
//! (`js/gpu-lexer.js`, vendored by `scripts/sync-upstream.mjs`) on the
//! browser's WebGPU, reached through wasm-bindgen. Without WebGPU (or when
//! it fails) the caller runs the CPU backend.

use js_sys::{Promise, Uint32Array};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

use crate::{Span, SyntaxClass};

#[wasm_bindgen(module = "/js/gpu-lexer.js")]
extern "C" {
    #[wasm_bindgen(js_name = nshGpuLexerAvailable)]
    fn available() -> bool;

    #[wasm_bindgen(js_name = nshGpuLexerParse, catch)]
    fn parse_js(code: &str) -> Result<Promise, JsValue>;
}

/// Whether the page has WebGPU.
pub fn webgpu_available() -> bool {
    available()
}

/// Spans from the browser's WebGPU, or `None` without it.
pub async fn parse(code: &str) -> Option<Vec<Span>> {
    if !available() {
        return None;
    }
    let value = JsFuture::from(parse_js(code).ok()?).await.ok()?;
    let flat = Uint32Array::new(&value).to_vec();
    Some(
        flat.as_chunks::<4>()
            .0
            .iter()
            .map(|span| Span {
                class: SyntaxClass::from_code(span[0] as u8),
                start: span[1] as usize,
                end: span[2] as usize,
                level: span[3].clamp(1, 16) as u8 - 1,
            })
            .collect(),
    )
}
