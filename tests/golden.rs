//! Parity with the upstream JS reference (`scripts/golden.mjs`): the CPU
//! backend's label for every prepared part of every fixture, and the spans
//! rebuilt from them, against the reference's.
//!
//! The only difference allowed is a near-tie: a part whose reference
//! probability for its own class is below one half (level < 8), where
//! the last bits of a transcendental function can swap the top two classes.
//! The test prints the counts, and `PARITY_REPORT=path` writes them as JSON.

use std::{fs, io::Read, path::Path};

use nsh_gpu_lexer::{Backend, Lexer, prepare, spans_from_labels};

fn golden_labels(name: &str) -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("tests/golden")
        .join(format!("{name}.labels.gz"));
    let mut out = Vec::new();
    flate2::read::GzDecoder::new(fs::File::open(path).unwrap())
        .read_to_end(&mut out)
        .unwrap();
    out
}

fn source(name: &str, synthetic: bool) -> String {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    if synthetic {
        let cases: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(root.join("tests/golden/synthetic.json")).unwrap()).unwrap();
        cases[name].as_str().unwrap().to_owned()
    } else {
        fs::read_to_string(root.join("fixtures/src").join(name)).unwrap()
    }
}

#[test]
fn cpu_matches_js_reference() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let manifest: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(root.join("tests/golden/manifest.json")).unwrap()).unwrap();
    let lexer = Lexer::new().backend(Backend::Cpu);
    let mut report = Vec::new();
    let (mut total_parts, mut total_spans, mut total_class, mut total_level, mut total_span_diff) = (0, 0, 0, 0, 0);
    let mut failures = Vec::new();
    for file in manifest["files"].as_array().unwrap() {
        let name = file["name"].as_str().unwrap();
        let text = source(name, file["synthetic"].as_bool().unwrap());
        let units: Vec<u16> = text.encode_utf16().collect();
        assert_eq!(
            units.len() as u64,
            file["utf16Length"].as_u64().unwrap(),
            "{name}: source length"
        );
        let prepared = prepare(&units);
        let expected = golden_labels(name);
        assert_eq!(prepared.len(), expected.len(), "{name}: part count");
        let started = std::time::Instant::now();
        let actual = lexer.labels(&prepared);
        let elapsed = started.elapsed();
        let mut class_diff = 0;
        let mut level_diff = 0;
        for (index, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
            if a & 15 != e & 15 {
                class_diff += 1;
                if e >> 4 >= 8 {
                    failures.push(format!(
                        "{name} part {index}: class {} != {} at level {}",
                        a & 15,
                        e & 15,
                        e >> 4
                    ));
                }
            } else if a != e {
                level_diff += 1;
            }
        }
        let ours = spans_from_labels(&prepared.ranges, &actual);
        let theirs = spans_from_labels(&prepared.ranges, &expected);
        assert_eq!(
            theirs.len() as u64,
            file["spans"].as_u64().unwrap(),
            "{name}: golden span count"
        );
        // Spans whose (type, start, end) the reference does not have.
        let mut theirs_set: std::collections::HashSet<(u8, usize, usize)> =
            theirs.iter().map(|s| (s.class as u8, s.start, s.end)).collect();
        let mut span_diff = 0;
        for span in &ours {
            if !theirs_set.remove(&(span.class as u8, span.start, span.end)) {
                span_diff += 1;
            }
        }
        span_diff = span_diff.max(theirs_set.len());
        println!(
            "{name}: {} parts, {} spans, class diff {class_diff}, level diff {level_diff}, span diff {span_diff}, {:.1} ms",
            expected.len(),
            theirs.len(),
            elapsed.as_secs_f64() * 1e3
        );
        report.push(serde_json::json!({
            "name": name, "parts": expected.len(), "spans": theirs.len(),
            "classDiff": class_diff, "levelDiff": level_diff, "spanDiff": span_diff,
        }));
        total_parts += expected.len();
        total_spans += theirs.len();
        total_class += class_diff;
        total_level += level_diff;
        total_span_diff += span_diff;
    }
    println!(
        "total: {total_parts} parts, {total_spans} spans; class diff {total_class}, level diff {total_level}, span diff {total_span_diff}"
    );
    if let Ok(path) = std::env::var("PARITY_REPORT") {
        fs::write(
            path,
            serde_json::to_string_pretty(&serde_json::json!({
                "parts": total_parts, "spans": total_spans, "classDiff": total_class,
                "levelDiff": total_level, "spanDiff": total_span_diff, "files": report,
            }))
            .unwrap(),
        )
        .unwrap();
    }
    assert!(
        failures.is_empty(),
        "class differences outside near-ties:\n{}",
        failures.join("\n")
    );
    // Near-ties are allowed but must stay rare.
    assert!(
        total_class * 10_000 <= total_parts,
        "{total_class} of {total_parts} parts differ"
    );
}
