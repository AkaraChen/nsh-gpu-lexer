#!/usr/bin/env node
// Generates the golden labels the CPU backend is tested against, with the
// upstream JS reference forward pass (packages/training/src/tree-model.js,
// `treeProbabilities`) on the promoted deployed int6 checkpoint.
//
//   node scripts/golden.mjs <gpu-lexer checkout>
//
// For each fixture (fixtures/src/* and the synthetic cases below) it writes
// tests/golden/<name>.labels.gz: one byte per prepared part, the argmax class
// in the low four bits and min(15, floor(16 * p)) of its probability in the
// high four, as the GPU runtime packs labels. tests/golden/manifest.json
// records each source's hash and its part and span counts.
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath, pathToFileURL } from "node:url";
import { gzipSync } from "node:zlib";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const upstream = resolve(process.argv[2] ?? join(root, ".upstream/gpu-lexer"));
const load = (path) => import(pathToFileURL(join(upstream, path)));
const { prepareTreeSource, releaseTreePrepared } = await load("packages/core/src/prepare-tree.js");
const { treeFeatureIndices } = await load("packages/core/src/tree-features.js");
const { spansFromRanges } = await load("packages/core/src/spans.js");
const { treeProbabilities } = await load("packages/training/src/tree-model.js");
const { dequantizeTensors } = await load("packages/training/src/quantization.js");

const active = join(upstream, "packages/training/active");
const metadata = JSON.parse(readFileSync(join(active, "model-active.json"), "utf8"));
const model = dequantizeTensors(readFileSync(join(active, "weights-int6-active.bin")), metadata.quantization);
const shape = {
  hiddenSize: metadata.hiddenSize,
  classifierSize: metadata.architecture.classifierDimensions,
  hashBuckets: metadata.architecture.lexemeHashBuckets,
};

export const synthetic = {
  "synthetic-crlf.txt": "fn main() {\r\n\tlet x = \"a\\\"b\";\r\n\t// done\r\n}\r\n\r",
  "synthetic-unicode.txt": "const 名前 = \"héllo 😀\"; // ünïcödé\nlet ß = 'x'; /* 🦀 */\n",
  "synthetic-long-word.txt": `${"a".repeat(300)} = ${"9".repeat(200)};\n${" ".repeat(150)}x\n`,
  "synthetic-one-char.txt": "x",
  "synthetic-symbols.txt": "<>?.??->=>::</{{${}}///**/--\\\n#!@%^&*()[]{}|~`",
  "synthetic-nesting.txt": "f(" + "[{(".repeat(40) + "1" + ")}]".repeat(40) + ");\n",
};

function goldenLabels(source) {
  const prepared = prepareTreeSource(source);
  try {
    const [data, ranges, , count] = prepared;
    const features = Array.from({ length: count }, (_, index) =>
      Uint16Array.from(treeFeatureIndices(data, index, shape.hashBuckets, metadata.featureVersion)));
    const probabilities = treeProbabilities(model, { features }, shape);
    const labels = new Uint8Array(count);
    for (let index = 0; index < count; index++) {
      const p = probabilities[index];
      let predicted = 0;
      for (let value = 1; value < p.length; value++) if (p[value] > p[predicted]) predicted = value;
      labels[index] = predicted | (Math.min(15, Math.floor(16 * p[predicted])) << 4);
    }
    // Spans exactly as the npm runtime rebuilds them from packed label words.
    const words = new Uint32Array((count + 3) >> 2);
    for (let index = 0; index < count; index++) words[index >> 2] |= labels[index] << ((index & 3) * 8);
    const spans = spansFromRanges(ranges, words);
    return { labels, spans, ranges: Uint32Array.from(ranges) };
  } finally {
    releaseTreePrepared(prepared);
  }
}

const sources = JSON.parse(readFileSync(join(root, "fixtures/sources.json"), "utf8")).files
  .map(({ name }) => [name, readFileSync(join(root, "fixtures/src", name), "utf8")]);
sources.push(...Object.entries(synthetic));
mkdirSync(join(root, "tests/golden"), { recursive: true });
writeFileSync(join(root, "tests/golden/synthetic.json"), JSON.stringify(synthetic, null, 2) + "\n");
const manifest = { upstreamCommit: "676026856660a22567345014835867a9dfe3fbdb", runId: metadata.runId, files: [] };
for (const [name, source] of sources) {
  const started = performance.now();
  const { labels, spans } = goldenLabels(source);
  const spanDigest = createHash("sha256");
  for (const span of spans) spanDigest.update(`${span.type},${span.start},${span.end},${span.confidence * 16}\n`);
  writeFileSync(join(root, "tests/golden", `${name}.labels.gz`), gzipSync(labels, { level: 9 }));
  manifest.files.push({
    name,
    synthetic: name in synthetic,
    sourceSha256: createHash("sha256").update(source).digest("hex"),
    utf16Length: source.length,
    parts: labels.length,
    spans: spans.length,
    spansSha256: spanDigest.digest("hex"),
  });
  console.log(`${name}: ${labels.length} parts, ${spans.length} spans, ${(performance.now() - started).toFixed(0)} ms`);
}
writeFileSync(join(root, "tests/golden/manifest.json"), JSON.stringify(manifest, null, 2) + "\n");
