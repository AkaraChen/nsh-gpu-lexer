//! The Native Shell module (feature `nsh`).
//!
//! - TSX: `import { GpuLexer } from "nsh-gpu-lexer"` ([`GpuLexer`]).
//! - Rust: [`editor::register_fallback`] and [`editor::GpuLexerHighlighter`]
//!   for `Editor`, [`text_view::code_highlighter`] for `TextView`.
//!
//! Nothing is registered by linking the crate: an app turns the Editor
//! highlighter on with [`editor::register_fallback`] (Rust) or
//! `GpuLexer.installFallback()` (TSX), and gives a `TextView` the code
//! highlighter itself.

pub mod editor;
pub mod text_view;

use nsh_core::prelude::*;

use crate::{Backend, Lexer, Span, SyntaxClass};

/// One of the nine classes gpu-lexer predicts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, PropEnum)]
pub enum GpuLexerTokenType {
    /// Not colored: identifiers, punctuation, whitespace.
    #[default]
    Plain,
    /// A comment.
    Comment,
    /// A string, character or regular expression.
    String,
    /// A number.
    Number,
    /// A keyword.
    Keyword,
    /// A type.
    Type,
    /// A function or method name.
    Function,
    /// A constant (`true`, `null`, enum members).
    Constant,
    /// An operator.
    Operator,
}

impl From<SyntaxClass> for GpuLexerTokenType {
    fn from(class: SyntaxClass) -> Self {
        match class {
            SyntaxClass::Plain => GpuLexerTokenType::Plain,
            SyntaxClass::Comment => GpuLexerTokenType::Comment,
            SyntaxClass::String => GpuLexerTokenType::String,
            SyntaxClass::Number => GpuLexerTokenType::Number,
            SyntaxClass::Keyword => GpuLexerTokenType::Keyword,
            SyntaxClass::Type => GpuLexerTokenType::Type,
            SyntaxClass::Function => GpuLexerTokenType::Function,
            SyntaxClass::Constant => GpuLexerTokenType::Constant,
            SyntaxClass::Operator => GpuLexerTokenType::Operator,
        }
    }
}

/// A highlighted range of the code, as upstream gpu-lexer's `parse` returns it.
#[derive(Debug, Clone, PartialEq, JsObject)]
pub struct GpuLexerSpan {
    /// What the model saw there.
    #[prop(rename = "type")]
    pub kind: GpuLexerTokenType,
    /// Where it starts, in UTF-16 code units (a JavaScript string index).
    pub start: u32,
    /// Where it ends (not included), in UTF-16 code units.
    pub end: u32,
    /// How sure the model is of `type`, in sixteenths from 1/16 to 1: the
    /// lowest among the span's parts, a ranking score rather than a
    /// calibrated probability.
    pub confidence: f64,
}

impl From<&Span> for GpuLexerSpan {
    fn from(span: &Span) -> Self {
        GpuLexerSpan {
            kind: span.class.into(),
            start: span.start as u32,
            end: span.end as u32,
            confidence: f64::from(span.confidence()),
        }
    }
}

/// Which backend runs the model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, PropEnum)]
pub enum GpuLexerBackend {
    /// The GPU for large inputs when the app was built with it, else the CPU.
    #[default]
    Auto,
    /// Always the CPU.
    Cpu,
    /// The GPU when available, else the CPU.
    Gpu,
}

impl From<GpuLexerBackend> for Backend {
    fn from(backend: GpuLexerBackend) -> Self {
        match backend {
            GpuLexerBackend::Auto => Backend::Auto,
            GpuLexerBackend::Cpu => Backend::Cpu,
            GpuLexerBackend::Gpu => Backend::Gpu,
        }
    }
}

/// Options of `GpuLexer.parse`.
#[derive(Debug, Clone, Default, PartialEq, JsObject)]
pub struct GpuLexerParseOptions {
    /// Which backend runs the model (default `"auto"`).
    pub backend: Option<GpuLexerBackend>,
}

/// Options of the Editor highlighter `GpuLexer.installFallback` installs.
#[derive(Debug, Clone, Default, PartialEq, JsObject)]
pub struct GpuLexerHighlightOptions {
    /// Spans whose confidence is below this (0 to 1) stay plain (default 0).
    pub min_confidence: Option<f64>,
    /// How long typing must pause before the whole text is highlighted
    /// again, in milliseconds (default 120). Lines keep their old tokens
    /// meanwhile.
    pub debounce_ms: Option<f64>,
}

impl GpuLexerHighlightOptions {
    pub(crate) fn settings(&self) -> editor::HighlightSettings {
        let mut settings = editor::HighlightSettings::default();
        if let Some(min) = self.min_confidence {
            settings.min_confidence = min.clamp(0.0, 1.0) as f32;
        }
        if let Some(ms) = self.debounce_ms {
            settings.debounce = std::time::Duration::from_millis(ms.max(0.0) as u64);
        }
        settings
    }
}

/// What the module runs: versions and the model's provenance.
#[derive(Debug, Clone, PartialEq, JsObject)]
pub struct GpuLexerInfo {
    /// This crate's version.
    pub version: String,
    /// The upstream gpu-lexer version the weights come from.
    pub upstream_version: String,
    /// The promoted checkpoint's run id.
    pub run_id: String,
    /// Whether the crate was built with the wgpu backend.
    pub gpu_backend: bool,
}

/// gpu-lexer: a tiny learned syntax highlighter that colors code in any
/// language, with no grammar and no language id.
#[module]
pub struct GpuLexer;

#[module]
impl GpuLexer {
    /// Highlights `code` off the UI thread: non-overlapping spans that cover
    /// it, offsets in UTF-16 code units, exactly as upstream `parse(code)`.
    pub async fn parse(code: String, options: Option<GpuLexerParseOptions>) -> Vec<GpuLexerSpan> {
        let backend = options.and_then(|o| o.backend).unwrap_or_default();
        let lexer = Lexer::new().backend(backend.into());
        #[cfg(not(target_arch = "wasm32"))]
        {
            blocking(move || to_js(&lexer.parse(&code))).await.unwrap_or_default()
        }
        #[cfg(target_arch = "wasm32")]
        {
            to_js(&lexer.parse_async(&code).await)
        }
    }

    /// Highlights `code` at once, inside the call, on the CPU: for short
    /// snippets (a code block) where a Promise is in the way.
    pub fn parse_sync(code: String) -> Vec<GpuLexerSpan> {
        to_js(&Lexer::new().backend(Backend::Cpu).parse(&code))
    }

    /// Makes gpu-lexer the highlighter of this engine's editors for text
    /// whose language is unknown or not set (Plain Text), and of the
    /// `"gpu-lexer"` language. Explicit: nothing is highlighted by it until
    /// this is called. Calling it again replaces the options.
    pub fn install_fallback(options: Option<GpuLexerHighlightOptions>) {
        editor::register_fallback_with(options.unwrap_or_default().settings());
    }

    /// Makes gpu-lexer the highlighter of the languages `languages` names
    /// (names, aliases or extensions: `"rust"`, `"ts"`), in place of the
    /// built-in highlighting.
    pub fn install_for(languages: Vec<String>, options: Option<GpuLexerHighlightOptions>) {
        let settings = options.unwrap_or_default().settings();
        for language in languages {
            editor::register_for_with(&language, settings);
        }
    }

    /// The versions and the model's provenance.
    pub fn info() -> GpuLexerInfo {
        GpuLexerInfo {
            version: env!("CARGO_PKG_VERSION").to_owned(),
            upstream_version: crate::provenance::UPSTREAM_VERSION.to_owned(),
            run_id: crate::provenance::RUN_ID.to_owned(),
            gpu_backend: cfg!(all(feature = "gpu", not(target_arch = "wasm32"))),
        }
    }
}

fn to_js(spans: &[Span]) -> Vec<GpuLexerSpan> {
    spans.iter().map(GpuLexerSpan::from).collect()
}
