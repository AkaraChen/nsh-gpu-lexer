//! CPU throughput at several input sizes: `cargo bench --bench throughput`.
//! Prints a Markdown table (median of several runs per size).

use std::time::Instant;

use nsh_gpu_lexer::{Backend, Lexer};

fn main() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
    let three = std::fs::read_to_string(root.join("fixtures/src/three.min.js")).unwrap();
    let lodash = std::fs::read_to_string(root.join("fixtures/src/lodash.js")).unwrap();
    let corpus = format!("{lodash}\n{three}");
    let lexer = Lexer::new().backend(Backend::Cpu);
    let threads = std::thread::available_parallelism().map_or(1, |n| n.get());
    println!("CPU backend, {threads} threads ({} {})", std::env::consts::OS, std::env::consts::ARCH);
    println!();
    println!("| Input | UTF-16 units | Parts | Spans | Median | Throughput |");
    println!("|---|---:|---:|---:|---:|---:|");
    let mut sizes: Vec<(String, String)> = Vec::new();
    for size in [1_000usize, 10_000, 100_000] {
        let mut end = size.min(corpus.len());
        while !corpus.is_char_boundary(end) {
            end -= 1;
        }
        sizes.push((format!("lodash.js, first {} KB", size / 1000), corpus[..end].to_owned()));
    }
    sizes.push(("three.min.js".to_owned(), three.clone()));
    sizes.push(("three.min.js × 10".to_owned(), three.repeat(10)));
    let _ = lexer.parse("warm up");
    for (label, text) in sizes {
        let units: Vec<u16> = text.encode_utf16().collect();
        let runs = if units.len() > 1_000_000 { 3 } else { 9 };
        let mut times = Vec::new();
        let mut spans = 0;
        let mut parts = 0;
        for _ in 0..runs {
            let started = Instant::now();
            let prepared = nsh_gpu_lexer::prepare(&units);
            let labels = lexer.labels(&prepared);
            spans = nsh_gpu_lexer::spans_from_labels(&prepared.ranges, &labels).len();
            times.push(started.elapsed().as_secs_f64());
            parts = prepared.len();
        }
        times.sort_by(f64::total_cmp);
        let median = times[times.len() / 2];
        println!(
            "| {label} | {} | {parts} | {spans} | {:.1} ms | {:.2} MB/s |",
            units.len(),
            median * 1e3,
            units.len() as f64 / median / 1e6
        );
    }
}
