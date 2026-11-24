//! # nsh-gpu-lexer
//!
//! [gpu-lexer](https://github.com/vercel-labs/gpu-lexer) for
//! [Native Shell](https://nsh.dev): a tiny learned syntax highlighter
//! (about 50 thousand int6 weights) that colors source in any language
//! without a grammar or a language id.
//!
//! ```
//! let code = "const answer = 42; // the answer";
//! for span in nsh_gpu_lexer::parse(code) {
//!     println!("{:?} {}..{} {:.2}", span.class, span.start, span.end, span.confidence());
//! }
//! ```
//!
//! Offsets are UTF-16 code units, as upstream reports them; [`byte_spans`]
//! converts them to byte ranges of a `&str`.
//!
//! Backends (Cargo features):
//!
//! - `cpu` (always available): a Rust port of the tokenizer, feature packing
//!   and the full forward pass, parallel across parts (`parallel`, on by
//!   default off wasm) and vectorized across units. It reproduces the
//!   upstream JS reference.
//! - `gpu`: the upstream WGSL compute shader on wgpu (Vulkan, Metal, DX12),
//!   used for large inputs when an adapter is present.
//! - `web-gpu`: on `wasm32`, the published `gpu-lexer` package on the
//!   browser's WebGPU, falling back to the CPU without it.
//!
//! With the `nsh` feature (on by default) the crate is also a Native Shell
//! module: `import { GpuLexer } from "nsh-gpu-lexer"` in TSX, and the
//! [`nsh`] module's Editor and TextView highlighters in Rust, which an app
//! turns on explicitly.

#![warn(missing_docs)]

mod cpu;
mod math;
mod model;
pub mod prepare;
mod spans;

#[allow(missing_docs, dead_code)]
pub mod generated {
    //! Constants generated from the upstream checkpoint by
    //! `scripts/sync-upstream.mjs`.
    #[rustfmt::skip]
    pub mod model;
    #[cfg(any(feature = "gpu", test))]
    #[rustfmt::skip]
    pub mod runtime;
}

#[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
pub mod gpu;

#[cfg(all(feature = "web-gpu", target_arch = "wasm32"))]
pub mod web;

#[cfg(feature = "nsh")]
pub mod nsh;

/// The Native Shell module: what `import … from "nsh-gpu-lexer"` gives TSX.
#[cfg(feature = "nsh")]
#[nsh_core::prelude::native_module("nsh-gpu-lexer")]
fn native(registry: &mut nsh_core::prelude::Registry) {
    registry.module::<nsh::GpuLexer>();
}

pub use prepare::{Prepared, prepare};
pub use spans::{Span, spans_from_labels};

/// The nine classes the model predicts, in upstream's class order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[repr(u8)]
pub enum SyntaxClass {
    /// Text without a color: identifiers, punctuation, whitespace.
    Plain = 0,
    /// Comments.
    Comment = 1,
    /// Strings, characters, regular expressions.
    String = 2,
    /// Numbers.
    Number = 3,
    /// Keywords.
    Keyword = 4,
    /// Types.
    Type = 5,
    /// Functions and methods.
    Function = 6,
    /// Constants: `true`, `null`, enum members.
    Constant = 7,
    /// Operators.
    Operator = 8,
}

impl SyntaxClass {
    /// Every class, in code order.
    pub const ALL: [SyntaxClass; 9] = [
        SyntaxClass::Plain,
        SyntaxClass::Comment,
        SyntaxClass::String,
        SyntaxClass::Number,
        SyntaxClass::Keyword,
        SyntaxClass::Type,
        SyntaxClass::Function,
        SyntaxClass::Constant,
        SyntaxClass::Operator,
    ];

    /// The class of a label's low four bits; codes without a class are plain.
    pub fn from_code(code: u8) -> SyntaxClass {
        SyntaxClass::ALL
            .get(code as usize)
            .copied()
            .unwrap_or(SyntaxClass::Plain)
    }

    /// Upstream's name: `"plain"`, `"comment"`, ...
    pub fn name(self) -> &'static str {
        match self {
            SyntaxClass::Plain => "plain",
            SyntaxClass::Comment => "comment",
            SyntaxClass::String => "string",
            SyntaxClass::Number => "number",
            SyntaxClass::Keyword => "keyword",
            SyntaxClass::Type => "type",
            SyntaxClass::Function => "function",
            SyntaxClass::Constant => "constant",
            SyntaxClass::Operator => "operator",
        }
    }

    /// The class `name` names.
    pub fn from_name(name: &str) -> Option<SyntaxClass> {
        SyntaxClass::ALL.into_iter().find(|class| class.name() == name)
    }
}

/// Which backend runs the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// The GPU for inputs of at least [`Lexer::gpu_min_parts`] parts when
    /// one is available (feature `gpu`), the CPU otherwise.
    #[default]
    Auto,
    /// Always the CPU.
    Cpu,
    /// The GPU when available, else the CPU.
    Gpu,
}

/// Parts from which [`Backend::Auto`] moves to the GPU: upstream batches
/// requests up to this many parts, below which dispatch overhead dominates.
pub const DEFAULT_GPU_MIN_PARTS: usize = 16_384;

/// A configured lexer. The model's weights and any GPU device are shared
/// process-wide; a `Lexer` is only its settings and is cheap to copy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lexer {
    backend: Backend,
    gpu_min_parts: usize,
}

impl Default for Lexer {
    fn default() -> Self {
        Lexer::new()
    }
}

impl Lexer {
    /// The default lexer: [`Backend::Auto`].
    pub const fn new() -> Lexer {
        Lexer {
            backend: Backend::Auto,
            gpu_min_parts: DEFAULT_GPU_MIN_PARTS,
        }
    }

    /// Chooses the backend.
    pub const fn backend(mut self, backend: Backend) -> Lexer {
        self.backend = backend;
        self
    }

    /// The input size, in parts, from which [`Backend::Auto`] uses the GPU.
    pub const fn gpu_min_parts(mut self, parts: usize) -> Lexer {
        self.gpu_min_parts = parts;
        self
    }

    /// Highlights `code`; offsets are UTF-16 code units.
    pub fn parse(&self, code: &str) -> Vec<Span> {
        let units: Vec<u16> = code.encode_utf16().collect();
        self.parse_utf16(&units)
    }

    /// Highlights UTF-16 `code`.
    pub fn parse_utf16(&self, code: &[u16]) -> Vec<Span> {
        let prepared = prepare(code);
        let labels = self.labels(&prepared);
        spans_from_labels(&prepared.ranges, &labels)
    }

    /// One label byte per prepared part (class | level << 4).
    pub fn labels(&self, prepared: &Prepared) -> Vec<u8> {
        if prepared.is_empty() {
            return Vec::new();
        }
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        {
            let wants_gpu = match self.backend {
                Backend::Cpu => false,
                Backend::Gpu => true,
                Backend::Auto => prepared.len() >= self.gpu_min_parts,
            };
            if wants_gpu && let Some(labels) = gpu::labels(prepared) {
                return labels;
            }
        }
        cpu::labels(prepared)
    }

    /// Highlights `code`, on the browser's WebGPU when the crate runs on
    /// the web with feature `web-gpu` (the CPU without it); elsewhere the
    /// same as [`Lexer::parse`].
    pub async fn parse_async(&self, code: &str) -> Vec<Span> {
        #[cfg(all(feature = "web-gpu", target_arch = "wasm32"))]
        {
            if self.backend != Backend::Cpu
                && let Some(spans) = web::parse(code).await
            {
                return spans;
            }
        }
        self.parse(code)
    }

    /// The backend a call of `parts` parts would run on right now.
    pub fn backend_for(&self, parts: usize) -> &'static str {
        #[cfg(all(feature = "gpu", not(target_arch = "wasm32")))]
        {
            let wants_gpu = match self.backend {
                Backend::Cpu => false,
                Backend::Gpu => true,
                Backend::Auto => parts >= self.gpu_min_parts,
            };
            if wants_gpu && gpu::available() {
                return "gpu";
            }
        }
        let _ = parts;
        "cpu"
    }
}

/// Highlights `code` with the default lexer; offsets are UTF-16 code units.
pub fn parse(code: &str) -> Vec<Span> {
    Lexer::new().parse(code)
}

/// Highlights `code` on the CPU only.
pub fn parse_cpu(code: &str) -> Vec<Span> {
    Lexer::new().backend(Backend::Cpu).parse(code)
}

/// A span with byte offsets into the `&str` it came from.
#[derive(Debug, Clone, PartialEq)]
pub struct ByteSpan {
    /// What the model saw there.
    pub class: SyntaxClass,
    /// The span's bytes.
    pub range: std::ops::Range<usize>,
    /// The span's confidence level, 0 to 15 ([`Span::level`]).
    pub level: u8,
}

/// Converts UTF-16 `spans` of `code` to byte ranges. A span edge inside a
/// surrogate pair (the model splits only at code-unit granularity, but a
/// non-ASCII character is always inside one word part) moves to the end of
/// its character.
pub fn byte_spans(code: &str, spans: &[Span]) -> Vec<ByteSpan> {
    let mut out = Vec::with_capacity(spans.len());
    let mut chars = code.char_indices().peekable();
    let mut unit = 0usize;
    let mut byte = 0usize;
    let mut to_byte = |target: usize| -> usize {
        while unit < target {
            match chars.next() {
                Some((at, c)) => {
                    unit += c.len_utf16();
                    byte = at + c.len_utf8();
                }
                None => break,
            }
        }
        byte
    };
    for span in spans {
        let start = to_byte(span.start);
        let end = to_byte(span.end);
        if end > start {
            out.push(ByteSpan {
                class: span.class,
                range: start..end,
                level: span.level,
            });
        }
    }
    out
}

/// Provenance of the embedded model.
pub mod provenance {
    pub use crate::generated::model::{RUN_ID, UPSTREAM_COMMIT, UPSTREAM_VERSION, WEIGHTS_SHA256};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_small_snippets() {
        let code = "const answer = 42; // the answer\nfunction f(x) { return \"s\"; }\n";
        let spans = parse_cpu(code);
        assert_eq!(spans.first().unwrap().start, 0);
        assert_eq!(spans.last().unwrap().end, code.encode_utf16().count());
        for pair in spans.windows(2) {
            assert_eq!(pair[0].end, pair[1].start);
        }
        let comment = spans
            .iter()
            .find(|s| s.class == SyntaxClass::Comment)
            .expect("a comment");
        // Whitespace is never classified, so a comment splits at its spaces.
        assert_eq!(&code[comment.start..comment.end], "//");
        assert!(spans.iter().any(|s| s.class == SyntaxClass::Number));
        assert!(spans.iter().any(|s| s.class == SyntaxClass::Keyword));
    }

    #[test]
    fn empty_source_has_no_spans() {
        assert!(parse("").is_empty());
    }

    #[test]
    fn byte_spans_follow_utf8() {
        let code = "s = \"héllo 😀\" # ok";
        let spans = parse_cpu(code);
        let bytes = byte_spans(code, &spans);
        assert_eq!(bytes.last().unwrap().range.end, code.len());
        for span in &bytes {
            assert!(code.is_char_boundary(span.range.start) && code.is_char_boundary(span.range.end));
        }
    }
}
