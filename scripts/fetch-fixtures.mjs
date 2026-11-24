#!/usr/bin/env node
// Downloads the fixture sources listed in fixtures/sources.json into
// fixtures/src/ and checks them against their recorded SHA-256.
// `--record` writes the hashes of freshly downloaded files instead.
import { createHash } from "node:crypto";
import { mkdirSync, readFileSync, writeFileSync, existsSync } from "node:fs";
import { dirname, join, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const manifestPath = join(root, "fixtures/sources.json");
const manifest = JSON.parse(readFileSync(manifestPath, "utf8"));
const record = process.argv.includes("--record");
mkdirSync(join(root, "fixtures/src"), { recursive: true });
for (const file of manifest.files) {
  const path = join(root, "fixtures/src", file.name);
  let bytes = existsSync(path) ? readFileSync(path) : null;
  if (!bytes || record) {
    const response = await fetch(file.url);
    if (!response.ok) throw new Error(`${file.url}: HTTP ${response.status}`);
    bytes = Buffer.from(await response.arrayBuffer());
    writeFileSync(path, bytes);
  }
  const sha256 = createHash("sha256").update(bytes).digest("hex");
  if (record) file.sha256 = sha256;
  else if (file.sha256 !== sha256) throw new Error(`${file.name}: sha256 ${sha256} != ${file.sha256}`);
  console.log(`${file.name}: ${bytes.length} bytes`);
}
if (record) writeFileSync(manifestPath, JSON.stringify(manifest, null, 2) + "\n");
