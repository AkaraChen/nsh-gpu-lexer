# nsh-gpu-lexer

[gpu-lexer](https://github.com/vercel-labs/gpu-lexer) for [Native Shell](https://nsh.dev): a tiny learned syntax
highlighter (about 50 thousand int6 weights) that colors code in any language with no grammar and no language id,
as a Rust nshm usable from **TSX and Rust on every NSH platform**, with explicit hooks into NSH's `Editor` and
`TextView`.

- **CPU backend** (default, every platform including wasm): a Rust port of upstream's tokenizer, feature packing
  and full forward pass that reproduces upstream's JS reference **bit for bit** on all test fixtures (below).
- **GPU backend** (`gpu` feature): upstream's WGSL compute shader on wgpu (Vulkan, Metal, DX12), used for large
  inputs.
- **Web GPU backend** (`web-gpu` feature, wasm32): the published `gpu-lexer` npm package on the browser's WebGPU,
  falling back to the CPU without it.

The model, its weights and its shader are gpu-lexer **0.0.6** (commit `6760268`, checkpoint
`20261117T071411.182Z`), generated into this crate by a script, never copied by hand.

## Install

```sh
nsh add github:AkaraChen/nsh-gpu-lexer          # in an NSH app (TSX or Rust)
```

or, in a crate's `Cargo.toml`:

```toml
nsh-gpu-lexer = { git = "https://github.com/AkaraChen/nsh-gpu-lexer" }
```

It is not on crates.io yet. It needs `nsh-core` / `nsh-base` 0.6, and an NSH that installs nshm (`nsh add`, nsh.json `"dependencies"`): the nightly builds after 0.6.1 (`nsh update nightly`). A Rust app can depend on the crate in its `Cargo.toml` with any NSH 0.6.

## TSX

```tsx
import { GpuLexer } from "nsh-gpu-lexer";

const spans = await GpuLexer.parse(code);
// [{ type: "keyword", start: 0, end: 5, confidence: 0.9375 }, ...]
```

`GpuLexer.parse(code)` returns exactly what upstream `parse(code)` returns: non-overlapping spans that cover the
code, `type` one of `plain comment string number keyword type function constant operator`, `start`/`end` in UTF-16
code units (JavaScript string indices), and `confidence` in sixteenths (1/16 to 1, the lowest of the span's parts).
It runs off the UI thread (`blocking`) and resolves with the spans. `GpuLexer.parseSync(code)` does the same inside
the call, for short snippets. `GpuLexer.info()` gives the versions and the checkpoint.

### Editor: explicit opt-in

Nothing is highlighted by gpu-lexer just because the crate is linked. Turn it on:

```tsx
// Editors whose language is unset or unknown (Plain Text), and language="gpu-lexer":
await GpuLexer.installFallback({ minConfidence: 0.5 });

// Or replace the built-in highlighting of named languages:
await GpuLexer.installFor(["vala", "nim"]);
```

Both run gpu-lexer as an NSH `EditorHighlighter` in Rust, off the UI thread (see
[how the editor highlighter works](#how-the-editor-highlighter-works)). They are `async` because NSH 0.6 runs a
module's synchronous calls outside the engine's reactive runtime, where editor registrations live.

Or color one editor yourself with decorations, from `nsh-gpu-lexer/editor` (this crate's `nsh.json` exports):

```tsx
import { Editor } from "nsh/base";
import { useGpuLexerDecorations } from "nsh-gpu-lexer/editor";

const code = signal(source);
const decorations = useGpuLexerDecorations(code, { minConfidence: 0.5 });
<Editor value={code()} onChange={({ value }) => code.set(value)} highlighting={false} decorations={decorations()} />;
```

`spansToDecorations(code, spans, minConfidence)` converts spans yourself.

### TextView

```tsx
import { TextView } from "nsh/base";
import { LexedCode, renderLexedCodeBlock } from "nsh-gpu-lexer/text-view";

<TextView markdown={doc} renderCodeBlock={renderLexedCodeBlock} />
<TextView markdown={doc} renderCodeBlock={(code) => <LexedCode code={code} minConfidence={0.5} />} />
```

## Rust

```rust
use nsh_gpu_lexer::{parse, byte_spans, Lexer, Backend, SyntaxClass};

let spans = parse(code);                          // UTF-16 offsets, as upstream
let bytes = byte_spans(code, &spans);             // byte ranges of the &str
let spans = Lexer::new().backend(Backend::Cpu).parse(code);
```

`Lexer::labels(&prepare(&utf16))` gives the raw per-part label bytes; `Span::confidence()` the upstream fraction.

Editor and TextView, explicitly (feature `nsh`, on by default):

```rust
use nsh_core::prelude::*;
use nsh_gpu_lexer::nsh::{editor, text_view};

// In a Rust app's root before it mounts its editors, or in an app crate's Registry::setup:
editor::register_fallback();                      // Plain Text + language="gpu-lexer"
editor::register_for("vala");                     // a named language
editor::register_fallback_with(editor::HighlightSettings { min_confidence: 0.5, ..Default::default() });

// TextView code blocks below this component:
provide_context(text_view::code_highlighter(Default::default()));
```

`editor::GpuLexerHighlighter` is the `EditorHighlighter` itself, for `nsh_base::editor::register_highlighter`
with settings of your own; `editor::line_tokens` and `text_view::highlight_code` are the pure functions under them.

### Token mapping

| gpu-lexer | Editor token / color |
|---|---|
| comment | `comment` |
| keyword | `keyword` |
| string | `string` |
| number, constant | `number` |
| function | `title` |
| type | `built` |
| operator, plain | not colored |

Every API takes a confidence threshold: spans below it stay plain.

### How the editor highlighter works

The model reads the whole text at once (a token's class can depend on a quote a thousand lines up), while NSH's
`EditorHighlighter` is asked for tokens line by line as the text changes. `GpuLexerHighlighter` keeps the result of
the last whole-text run: after an edit it waits for typing to pause (`debounce`, 120 ms by default), runs the model
on a snapshot of the whole text on a thread of its own, then gives the editor only the lines whose tokens changed.
Meanwhile every line keeps the tokens it had, which the editor moves with the text. On the web (no threads) the run
takes the highlight call.

## Backends

| Feature | Default | What |
|---|---|---|
| `cpu` | yes | Always compiled (it is every other backend's fallback); the name of the CPU backend. |
| `parallel` | yes | The CPU passes on rayon (ignored on wasm). |
| `nsh` | yes | The Native Shell module: `GpuLexer` for TSX, the Editor and TextView highlighters. |
| `gpu` | no | wgpu compute. `Backend::Auto` uses it from `DEFAULT_GPU_MIN_PARTS` (16,384 parts, upstream's batch size) when an adapter exists; `Backend::Gpu` always tries it. Storage is f16 when the adapter has `SHADER_F16`, else f32 (`NSH_GPU_LEXER_NO_F16=1` forces f32, `NSH_GPU_LEXER_NO_GPU=1` disables the GPU). |
| `web-gpu` | no | On wasm32, `parse_async` / `GpuLexer.parse` run the vendored npm bundle (`js/gpu-lexer.js`, through wasm-bindgen) on the browser's WebGPU, and the CPU without it. |

### A fix to upstream's shader

Running upstream's shader on Mesa's lavapipe disagreed with the JS reference on 0.8% of three.min.js's parts.
The cause is a race in `tree_down_classify`: the in-block top-down pass writes each child's returned context over
its upward state in workgroup memory while the threads of partner channels (`h ^ 1 << depth`) may still read that
upward state. Where a block's 32 channels run as one lockstep SIMD group (Apple GPUs, where upstream measured) the
race is benign; on 8- or 16-wide hardware it is not. `scripts/sync-upstream.mjs` patches the shader (and the
vendored npm bundle) to compute a level's children before writing them after a barrier; the patch fails loudly if
upstream's shader changes. With it, lavapipe matches the reference (table below).

## Parity with upstream

`scripts/golden.mjs` runs upstream's JS reference (`packages/training/src/tree-model.js`, `treeProbabilities`, on
the deployed int6 checkpoint) over the fixtures and writes one label byte per prepared part (class, and
`min(15, floor(16 p))` of its probability, as the GPU runtime packs labels) to `tests/golden/`. The fixtures are 30
files of upstream's verification set (its website examples and language probes, which upstream keeps out of
training: three.min.js, lodash.js, React, Python, Go, Rust, Ruby, shell, PowerShell, CSS/SCSS, HTML, Vue, Svelte,
TOML, Markdown, WGSL, GraphQL, Nim, CoffeeScript, WAT, Haxe, CMake) plus six synthetic edge cases (CRLF, non-ASCII
and surrogate pairs, long words, symbols, deep nesting), 647,517 parts and 424,788 spans in all.

| Backend | Parts with another class | Parts with another confidence level | Spans that differ (type or range) |
|---|---:|---:|---:|
| CPU (`tests/golden.rs`, Linux, macOS, Windows) | **0** | **0** | **0** of 424,788 |
| GPU, lavapipe, f32 storage (`tests/gpu.rs`) | 1 | 14 | 3 of 424,643 (0.0007%) |
| GPU, lavapipe, f16 storage | 14 | 201 | 25 of 424,643 (0.006%) |
| GPU, macOS runner's Metal (Apple Paravirtual device), f16 | 17 | 321 | 31 of 424,643 (0.007%) |

The CPU backend computes as the reference does: every activation rounded to `f32` where the reference stores it in
a `Float32Array`, every sum accumulated in `f64` in the reference's order, and `exp` / `tanh` from fdlibm as V8
computes them (`libm`'s msun `exp` / `expm1`, and msun's `tanh` ported in `src/math.rs`), so it matches on every
platform. Vectorization runs across output units and parallelism across parts, never across a sum. The tests allow
a class difference only at a near-tie (reference level ≤ 7 for the CPU, ≤ 8 for the GPU, i.e. the winning class's
probability under 1/2 or 9/16) and at most 0.01% (CPU) / 0.1% (GPU) of parts; the CPU has none at all. The GPU
rows exclude the synthetic cases; its differences are all such near-ties, as upstream reports for its own WebGPU
runtime (4 of 168,813 three.min.js spans).

## Performance

CPU backend, `cargo bench --bench throughput` on the GitHub Actions `ubuntu-24.04` runner (median, prepare +
model + spans):

Measured in [CI run 45338839369](https://github.com/AkaraChen/nsh-gpu-lexer/actions/runs/45338839369) (4 vCPUs, AMD EPYC 9V74):

| Input | UTF-16 units | Parts | Spans | Median | Throughput |
|---|---:|---:|---:|---:|---:|
| lodash.js, first 1 KB | 1,000 | 397 | 242 | 2.3 ms | 0.43 MB/s |
| lodash.js, first 10 KB | 10,000 | 4,289 | 2,114 | 18.0 ms | 0.56 MB/s |
| lodash.js, first 100 KB | 100,000 | 40,967 | 25,428 | 166.2 ms | 0.60 MB/s |
| three.min.js | 555,650 | 248,387 | 168,813 | 1,212 ms | 0.46 MB/s |
| three.min.js × 10 | 5,556,500 | 2,483,870 | 1,559,126 | 12,304 ms | 0.45 MB/s |

Every CI run prints the current numbers in the `CPU throughput` job's summary. On an 8-thread desktop
(local, same build) three.min.js takes about 490 ms; single-threaded, about 2.9 s.

The CPU backend is exact rather than fast: it spends its time in about 450 `f64` `exp`/`tanh` per part, which the
parity above needs. Highlighting a typical source file (10 to 100 KB) takes 20 to 170 ms on CI's 4 vCPUs, off the UI thread (the editor keeps its old tokens meanwhile).
For multi-megabyte inputs use the `gpu` feature (or `web-gpu` in browsers); upstream measured 190 ms for ten copies
of three.min.js on an Apple M4 Pro.

## Platforms and CI

[GitHub Actions](.github/workflows/ci.yml) checks:

- `cargo fmt --check`, `cargo clippy -D warnings` (default, `gpu`, CPU only, wasm32 + `web-gpu`);
- `cargo test --release` (unit tests and the CPU parity above) on Linux, macOS and Windows, and
  `cargo build --features gpu` there;
- `cargo build --features gpu` for `aarch64-linux-android`, `aarch64-apple-ios` and `aarch64-apple-ios-sim`;
- wasm32 builds: CPU only, the NSH module, and `web-gpu`;
- GPU parity on Mesa lavapipe (software Vulkan) with f16 and with f32 storage, and on the macOS runner's Metal
  device (Apple Paravirtual device; skipped, not failed, if a VM exposes none);
- `nsh check` and `nsh test` of `examples/playground` with NSH's nightly build (nshm `"dependencies"` are newer than the 0.6.1 release): `GpuLexer.parse` / `parseSync`, the editor
  highlighted only after `installFallback`, an editor left alone staying plain, `useGpuLexerDecorations`, and
  `renderLexedCodeBlock`, checked through the colors the headless runner resolves (`app.spanStyle`);
- that `scripts/sync-upstream.mjs` and `scripts/golden.mjs` reproduce the committed model, shader, bundle and
  goldens from upstream.

Not verified here: the GPU backend's speed and results on real GPUs (CI has only lavapipe and, possibly, a
virtualized Metal device), the `web-gpu` path in a browser (it is compiled, not run), and running on iOS and Android
devices (the crate is compiled for them, not run).

## Example

`examples/playground` is an NSH TSX app: paste code into the editor and it is highlighted as you type, with a
slider for the confidence threshold, span counts by class, and a TextView code block. Run it with
`nsh run examples/playground`; its tests with `nsh test` in that directory.

## Updating the model

```sh
bash scripts/fetch-upstream.sh .upstream                       # the pinned commit and npm tarball (integrity-checked)
node scripts/sync-upstream.mjs .upstream/gpu-lexer .upstream/npm   # model/, src/generated/, js/
node scripts/fetch-fixtures.mjs                                 # fixtures/src against their SHA-256
node scripts/golden.mjs .upstream/gpu-lexer                     # tests/golden/
```

The pinned version, commit and npm integrity are at the top of `scripts/sync-upstream.mjs` and
`scripts/fetch-upstream.sh`. The sync script refuses a checkpoint whose format, feature version, sizes or class
order this port does not implement.

## License and provenance

This crate is MIT licensed ([LICENSE](LICENSE)). It ports and embeds gpu-lexer by Shu Ding, also MIT
(Copyright (c) 2026 Shu Ding): the int6 weights and metadata of the promoted checkpoint
(`packages/training/active/`), the runtime weights and WGSL shader of `packages/core`, and the published
`gpu-lexer@0.0.6` bundle, which upstream distributes under the MIT License in its repository and npm package.
[NOTICE](NOTICE) lists every file taken from upstream and the one change made to it (the shader fix above).

Upstream's [MODEL_CARD.md](https://github.com/vercel-labs/gpu-lexer/blob/main/MODEL_CARD.md) describes the model:
trained on permissively licensed, pinned repositories and npm packages with labels from Shiki 4.4.3; intended for
experimental syntax coloring, not for parsing, linting or any decision where a wrong label matters; accuracy varies
by language (89.89% per-part agreement with Shiki on its verification set). No training data is redistributed here.
The fixtures in `fixtures/src` are third-party files under their own licenses, listed in
[fixtures/sources.json](fixtures/sources.json), used only as test data.
