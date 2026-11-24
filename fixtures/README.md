# Fixtures

`src/` holds the sources the parity tests run on: files of upstream gpu-lexer's
verification set (its website examples and language probes, which upstream
reserves from training), pinned to a version or a commit, permissive licenses
only. `sources.json` lists where each file comes from, its license and its
SHA-256. Each file remains under its own license and copyright; they are here
only as test data.

`node scripts/fetch-fixtures.mjs` checks the files against their hashes (and
downloads missing ones); `--record` downloads them again and records new hashes.
