//! The GPU backend against the JS reference (feature `gpu`).
//!
//! Upstream reports that its WebGPU runtime agrees with the JS reference on
//! types and ranges except near-ties (4 of 168,813 spans of three.min.js);
//! f16 storage moves confidence levels a little. This test checks the same:
//! every class difference is at a reference level of at most 8 (p < 9/16) and
//! they stay under 0.1% of parts. Without an adapter it is skipped, unless
//! `NSH_GPU_LEXER_REQUIRE_GPU=1`.
#![cfg(feature = "gpu")]

use std::{fs, io::Read, path::Path};

use nsh_gpu_lexer::{gpu, prepare, spans_from_labels, Backend, Lexer};

fn golden_labels(name: &str) -> Vec<u8> {
    let path = Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden").join(format!("{name}.labels.gz"));
    let mut out = Vec::new();
    flate2::read::GzDecoder::new(fs::File::open(path).unwrap()).read_to_end(&mut out).unwrap();
    out
}

#[test]
fn gpu_matches_js_reference() {
    if !gpu::available() {
        assert!(std::env::var_os("NSH_GPU_LEXER_REQUIRE_GPU").is_none(), "no GPU adapter");
        eprintln!("skipped: no GPU adapter");
        return;
    }
    let (adapter, f16) = gpu::adapter_info().unwrap();
    println!("adapter: {adapter}, f16: {f16}");
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let manifest: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(root.join("tests/golden/manifest.json")).unwrap()).unwrap();
    let lexer = Lexer::new().backend(Backend::Gpu);
    let (mut parts, mut class_diff, mut level_diff, mut spans, mut span_diff) = (0usize, 0usize, 0usize, 0usize, 0usize);
    let mut far = Vec::new();
    for file in manifest["files"].as_array().unwrap() {
        if file["synthetic"].as_bool().unwrap() {
            continue;
        }
        let name = file["name"].as_str().unwrap();
        let text = fs::read_to_string(root.join("fixtures/src").join(name)).unwrap();
        let units: Vec<u16> = text.encode_utf16().collect();
        let prepared = prepare(&units);
        let expected = golden_labels(name);
        let started = std::time::Instant::now();
        let actual = gpu::labels(&prepared).expect("GPU labels");
        let elapsed = started.elapsed();
        assert_eq!(actual.len(), expected.len());
        let mut file_class = 0;
        for (index, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
            if a & 15 != e & 15 {
                file_class += 1;
                if e >> 4 > 8 {
                    far.push(format!("{name} part {index}: {} != {} (level {})", a & 15, e & 15, e >> 4));
                }
            } else if a != e {
                level_diff += 1;
            }
        }
        let ours: std::collections::HashSet<_> =
            spans_from_labels(&prepared.ranges, &actual).iter().map(|s| (s.class, s.start, s.end)).collect();
        let theirs = spans_from_labels(&prepared.ranges, &expected);
        let differ = theirs.iter().filter(|s| !ours.contains(&(s.class, s.start, s.end))).count();
        println!("{name}: class diff {file_class}, span diff {differ} of {}, {:.1} ms", theirs.len(), elapsed.as_secs_f64() * 1e3);
        parts += expected.len();
        class_diff += file_class;
        spans += theirs.len();
        span_diff += differ;
        // The lexer routes Backend::Gpu here too.
        assert_eq!(lexer.backend_for(prepared.len()), "gpu");
    }
    println!("total: {parts} parts, class diff {class_diff}, level diff {level_diff}; {spans} spans, span diff {span_diff}");
    assert!(far.is_empty(), "class differences outside near-ties:\n{}", far.join("\n"));
    assert!(class_diff * 1000 <= parts, "{class_diff} of {parts} parts differ");
}
