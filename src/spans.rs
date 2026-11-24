//! Spans from per-part labels: upstream `spans.js`.

use crate::SyntaxClass;

/// A highlighted range of the source.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Span {
    /// What the model saw there.
    pub class: SyntaxClass,
    /// Start, in UTF-16 code units (as JavaScript strings index).
    pub start: usize,
    /// End (exclusive), in UTF-16 code units.
    pub end: usize,
    /// The lowest confidence level among the span's parts, 0 to 15:
    /// the selected class's softmax probability in sixteenths, rounded down.
    pub level: u8,
}

impl Span {
    /// How sure the model is of `class`, from 1/16 to 1, as upstream reports
    /// it: a ranking score, not a calibrated probability.
    pub fn confidence(&self) -> f32 {
        f32::from(self.level + 1) / 16.0
    }
}

/// Merges adjacent parts of one class into spans, keeping the lowest level.
pub fn spans_from_labels(ranges: &[u32], labels: &[u8]) -> Vec<Span> {
    let mut spans: Vec<Span> = Vec::new();
    for (index, &label) in labels.iter().enumerate() {
        let start = ranges[index * 2] as usize;
        let end = ranges[index * 2 + 1] as usize;
        let class = SyntaxClass::from_code(label & 15);
        let level = label >> 4;
        match spans.last_mut() {
            Some(last) if last.class == class && last.end == start => {
                last.end = end;
                last.level = last.level.min(level);
            }
            _ => spans.push(Span { class, start, end, level }),
        }
    }
    spans
}
