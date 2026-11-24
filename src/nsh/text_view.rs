//! gpu-lexer for `TextView`'s code blocks.
//!
//! A Rust app gives the text views below it the highlighter with
//! `provide_context`:
//!
//! ```ignore
//! use nsh_core::prelude::*;
//! provide_context(nsh_gpu_lexer::nsh::text_view::code_highlighter(Default::default()));
//! ```
//!
//! A TSX app passes `renderCodeBlock={(code) => <LexedCode code={code} />}`
//! (`nsh-gpu-lexer/text-view`).

use std::ops::Range;

use nsh_base::{editor::EditorToken, text_view::{CodeBlock, CodeHighlighter}};
use nsh_core::{element::StyleRefinement, prelude::*};

use super::editor::editor_token;
use crate::{Backend, Lexer, byte_spans};

/// How code blocks are highlighted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CodeHighlightSettings {
    /// Spans below this confidence (0 to 1) stay plain.
    pub min_confidence: f32,
    /// Only blocks without a language (`language: None`) are highlighted by
    /// gpu-lexer when set; the others show plain.
    pub unnamed_only: bool,
}

impl Default for CodeHighlightSettings {
    fn default() -> Self {
        CodeHighlightSettings { min_confidence: 0.0, unnamed_only: false }
    }
}

/// The style of an editor token, in the reading palette the built-in
/// highlighting of `Editor` and `TextView` uses (light, dark).
pub fn token_style(token: EditorToken) -> StyleRefinement {
    let s = StyleRefinement::default();
    let color = AdaptiveColor::rgb;
    match token {
        EditorToken::Comment => s.text_color(color(0x6e7781, 0x7d7a72)).italic(),
        EditorToken::Keyword => s.text_color(color(0x8250df, 0xc4a7ff)),
        EditorToken::String => s.text_color(color(0x116329, 0x9fd68a)),
        EditorToken::Number => s.text_color(color(0x953800, 0xf5b27a)),
        EditorToken::Title => s.text_color(color(0x0550ae, 0x82b8ff)),
        EditorToken::Attr => s.text_color(color(0x7d4e00, 0xf0c674)),
        EditorToken::Built => s.text_color(color(0x0a7285, 0x6fd3d8)),
        EditorToken::Deletion => s.text_color(color(0xcf222e, 0xf87171)),
    }
}

/// The styled byte ranges of `code`.
pub fn highlight_code(code: &str, settings: &CodeHighlightSettings) -> Vec<(Range<usize>, StyleRefinement)> {
    let spans = Lexer::new().backend(Backend::Cpu).parse(code);
    byte_spans(code, &spans)
        .into_iter()
        .filter(|span| f32::from(span.level + 1) / 16.0 >= settings.min_confidence)
        .filter_map(|span| editor_token(span.class).map(|token| (span.range, token_style(token))))
        .collect()
}

/// A `CodeHighlighter` for `TextView` that colors code blocks with
/// gpu-lexer: give it to the views below with `provide_context`.
pub fn code_highlighter(settings: CodeHighlightSettings) -> CodeHighlighter {
    CodeHighlighter::new(move |block: &CodeBlock| {
        if settings.unnamed_only && block.language.as_deref().is_some_and(|l| !l.is_empty()) {
            return Vec::new();
        }
        highlight_code(&block.code, &settings)
    })
}
