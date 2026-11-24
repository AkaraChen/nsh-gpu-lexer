//! gpu-lexer as an `Editor` highlighter.
//!
//! The model reads the whole text at once (a token's class can depend on a
//! quote a thousand lines up), while the editor asks for tokens line by
//! line as the text changes. [`GpuLexerHighlighter`] keeps the result of
//! the last whole-text run: after an edit it waits for typing to pause
//! ([`HighlightSettings::debounce`]), runs the model on a snapshot of the
//! whole text off the editor's thread, and then gives the editor the lines
//! whose tokens changed. Meanwhile every line keeps the tokens it had,
//! moved with the text by the editor.
//!
//! Nothing is registered by linking the crate. A Rust app calls
//! [`register_fallback`] (or [`register_for`]) in its root before it mounts
//! its editors, or in its crate's `Registry::setup`; TSX calls
//! `GpuLexer.installFallback()`.

use std::{
    ops::Range,
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use nsh_base::editor::{
    EditorHighlighter, EditorLanguage, EditorText, EditorTextEdit, EditorToken, EditorTokens, register_highlighter,
    register_language,
};
use nsh_core::time::Instant;

use crate::{Lexer, SyntaxClass, byte_spans};

/// The editor token a class is colored as: comment, keyword and string as
/// themselves, number and constant as numbers, functions as titles, types
/// as built-ins; plain and operator keep the text's color.
pub fn editor_token(class: SyntaxClass) -> Option<EditorToken> {
    Some(match class {
        SyntaxClass::Comment => EditorToken::Comment,
        SyntaxClass::Keyword => EditorToken::Keyword,
        SyntaxClass::String => EditorToken::String,
        SyntaxClass::Number | SyntaxClass::Constant => EditorToken::Number,
        SyntaxClass::Function => EditorToken::Title,
        SyntaxClass::Type => EditorToken::Built,
        SyntaxClass::Plain | SyntaxClass::Operator => return None,
    })
}

/// How the highlighter runs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HighlightSettings {
    /// Spans below this confidence (0 to 1) stay plain.
    pub min_confidence: f32,
    /// How long typing must pause before the whole text runs again.
    pub debounce: Duration,
    /// The lexer (backend) to run.
    pub lexer: Lexer,
}

impl Default for HighlightSettings {
    fn default() -> Self {
        HighlightSettings {
            min_confidence: 0.0,
            debounce: Duration::from_millis(120),
            lexer: Lexer::new(),
        }
    }
}

/// A line's tokens: byte ranges within the line and their kinds.
pub type LineTokens = Vec<(Range<usize>, EditorToken)>;

/// Runs the model over `text` and cuts its tokens into lines (split at
/// `\n`, which the editor's text uses alone).
pub fn line_tokens(text: &str, settings: &HighlightSettings) -> Vec<LineTokens> {
    let spans = settings.lexer.parse(text);
    let mut lines: Vec<LineTokens> = vec![Vec::new()];
    let mut line_starts = vec![0usize];
    for (at, _) in text.match_indices('\n') {
        line_starts.push(at + 1);
        lines.push(Vec::new());
    }
    let mut line = 0;
    for span in byte_spans(text, &spans) {
        if f32::from(span.level + 1) / 16.0 < settings.min_confidence {
            continue;
        }
        let Some(token) = editor_token(span.class) else {
            continue;
        };
        let mut start = span.range.start;
        while start < span.range.end {
            while line + 1 < line_starts.len() && line_starts[line + 1] <= start {
                line += 1;
            }
            let line_start = line_starts[line];
            let line_end = line_starts.get(line + 1).map_or(text.len(), |next| next - 1);
            let end = span.range.end.min(line_end);
            if end > start {
                lines[line].push((start - line_start..end - line_start, token));
            }
            start = match line_starts.get(line + 1) {
                Some(&next) => next.max(end),
                None => break,
            };
        }
    }
    lines
}

/// The result of one whole-text run.
struct Run {
    version: u64,
    lines: Vec<LineTokens>,
}

#[derive(Default)]
struct Shared {
    finished: Option<Run>,
    running: bool,
}

/// The `EditorHighlighter` that colors a text with gpu-lexer: see the
/// module's documentation.
pub struct GpuLexerHighlighter {
    settings: HighlightSettings,
    /// Counts edits; a run applies only to the version it read.
    version: Arc<AtomicU64>,
    last_edit: Option<Instant>,
    /// The first line edited since the last run.
    dirty_from: usize,
    /// What the editor has of each line, as we gave it (or `None` when
    /// it must be given again).
    given: Vec<Option<LineTokens>>,
    /// The tokens of the current text, once a run of it finished.
    current: Option<Vec<LineTokens>>,
    shared: Arc<(Mutex<Shared>, Condvar)>,
}

impl GpuLexerHighlighter {
    /// A highlighter with `settings`.
    pub fn new(settings: HighlightSettings) -> Self {
        GpuLexerHighlighter {
            settings,
            version: Arc::new(AtomicU64::new(0)),
            last_edit: None,
            dirty_from: 0,
            given: Vec::new(),
            current: None,
            shared: Arc::new((Mutex::new(Shared::default()), Condvar::new())),
        }
    }

    fn start_run(&mut self, text: &EditorText) {
        let version = self.version.load(Ordering::SeqCst);
        let source = text.rope().to_string();
        let settings = self.settings;
        let shared = self.shared.clone();
        shared.0.lock().unwrap().running = true;
        let work = move || {
            let lines = line_tokens(&source, &settings);
            let (lock, ready) = &*shared;
            let mut state = lock.lock().unwrap();
            state.finished = Some(Run { version, lines });
            state.running = false;
            ready.notify_all();
        };
        #[cfg(not(target_arch = "wasm32"))]
        std::thread::spawn(work);
        // No threads on the web: the run takes this call.
        #[cfg(target_arch = "wasm32")]
        work();
    }

    /// Takes a finished run of the current text, waiting up to `budget`.
    fn collect(&mut self, budget: Duration) {
        let version = self.version.load(Ordering::SeqCst);
        let (lock, ready) = &*self.shared;
        let mut state = lock.lock().unwrap();
        if state.finished.is_none() && state.running {
            state = ready.wait_timeout(state, budget).unwrap().0;
        }
        let Some(run) = state.finished.take() else { return };
        if run.version != version {
            return;
        }
        drop(state);
        // Lines whose tokens are new are not current until given.
        self.given.resize(run.lines.len(), None);
        for (line, tokens) in run.lines.iter().enumerate() {
            if self.given[line].as_ref() != Some(tokens) {
                self.given[line] = None;
            }
        }
        self.current = Some(run.lines);
        self.dirty_from = usize::MAX;
    }
}

impl Default for GpuLexerHighlighter {
    fn default() -> Self {
        GpuLexerHighlighter::new(HighlightSettings::default())
    }
}

impl EditorHighlighter for GpuLexerHighlighter {
    fn edited(&mut self, edit: &EditorTextEdit) {
        self.version.fetch_add(1, Ordering::SeqCst);
        self.last_edit = Some(Instant::now());
        self.current = None;
        self.dirty_from = self.dirty_from.min(edit.first_line);
        let first = edit.first_line.min(self.given.len());
        let old_end = (edit.first_line + edit.old_line_count).min(self.given.len());
        self.given
            .splice(first..old_end, std::iter::repeat_n(None, edit.new_line_count));
    }

    fn highlight(&mut self, text: &EditorText, last_line: usize, budget: Duration, tokens: &mut EditorTokens) -> bool {
        if self.current.is_none() {
            let running = self.shared.0.lock().unwrap().running;
            let quiet = self.last_edit.is_none_or(|at| at.elapsed() >= self.settings.debounce);
            if !running && quiet {
                self.start_run(text);
            } else if !running {
                // Typing: wait out the pause, within the budget (the
                // editor calls again; the web has no thread to sleep on).
                #[cfg(not(target_arch = "wasm32"))]
                {
                    let left = self.settings.debounce.saturating_sub(self.last_edit.unwrap().elapsed());
                    std::thread::sleep(left.min(budget));
                }
                return false;
            }
            self.collect(budget);
            if self.current.is_none() {
                return false;
            }
        }
        let current = self.current.as_ref().unwrap();
        let last = last_line.min(current.len().saturating_sub(1));
        for (line, tokens_of_line) in current.iter().enumerate().take(last + 1) {
            if self.given[line].is_none() {
                tokens.line(line, tokens_of_line.iter().cloned());
                self.given[line] = Some(tokens_of_line.clone());
            }
        }
        true
    }

    fn frontier(&self) -> usize {
        match &self.current {
            Some(_) => self.given.iter().position(Option::is_none).unwrap_or(self.given.len()),
            None => self.dirty_from.min(self.given.len()),
        }
    }
}

/// The language editors name to be highlighted by gpu-lexer whatever the
/// code is: `language="gpu-lexer"` (or `"auto"`).
pub static GPU_LEXER_LANGUAGE: EditorLanguage = EditorLanguage::new("GPU Lexer").aliases(&["gpu-lexer", "auto"]);

/// Makes gpu-lexer the highlighter of this engine's editors whose language
/// is unknown or not set (Plain Text), and of `language="gpu-lexer"`, with
/// the default settings. Call it in the engine: a Rust app's root before it
/// mounts its editors, or an app crate's `Registry::setup`.
///
/// # Panics
///
/// Outside an engine.
pub fn register_fallback() {
    register_fallback_with(HighlightSettings::default());
}

/// [`register_fallback`] with `settings`.
pub fn register_fallback_with(settings: HighlightSettings) {
    register_language(&GPU_LEXER_LANGUAGE);
    register_for_with("gpu-lexer", settings);
    register_for_with("Plain Text", settings);
}

/// Makes gpu-lexer the highlighter of `language` (a name, an alias or an
/// extension), in place of the built-in highlighting.
pub fn register_for(language: &str) {
    register_for_with(language, HighlightSettings::default());
}

/// [`register_for`] with `settings`.
pub fn register_for_with(language: &str, settings: HighlightSettings) {
    register_highlighter(language, move || Box::new(GpuLexerHighlighter::new(settings)));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tokens_stay_within_lines() {
        let text = "/* a\nmulti-line comment */\nlet s = \"x\";\n";
        let lines = line_tokens(text, &HighlightSettings::default());
        assert_eq!(lines.len(), text.split('\n').count());
        for (line, tokens) in text.split('\n').zip(&lines) {
            for (range, _) in tokens {
                assert!(range.end <= line.len(), "{range:?} in {line:?}");
            }
        }
        assert!(lines[1].iter().any(|(_, t)| *t == EditorToken::Comment));
    }

    #[test]
    fn confidence_threshold_drops_tokens() {
        let text = "fn main() { let x = 1; }";
        let all = line_tokens(text, &HighlightSettings::default());
        let none = line_tokens(
            text,
            &HighlightSettings {
                min_confidence: 1.01,
                ..Default::default()
            },
        );
        assert!(!all[0].is_empty());
        assert!(none[0].is_empty());
    }

    #[test]
    fn highlighter_gives_every_line_then_only_changes() {
        let text = EditorText::default();
        let mut highlighter = GpuLexerHighlighter::new(HighlightSettings {
            debounce: Duration::ZERO,
            ..Default::default()
        });
        highlighter.edited(&EditorTextEdit::lines(0, 1, 1));
        let mut tokens = EditorTokens::default();
        let mut done = false;
        for _ in 0..100 {
            if highlighter.highlight(&text, usize::MAX, Duration::from_millis(50), &mut tokens) {
                done = true;
                break;
            }
        }
        assert!(done);
        assert_eq!(highlighter.frontier(), 1);
    }
}
