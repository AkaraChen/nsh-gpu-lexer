#!/usr/bin/env bash
# Fetches upstream gpu-lexer at the pinned commit and its published npm
# package into $1 (default: .upstream), verifying the npm tarball's integrity.
set -euo pipefail

COMMIT=676026856660a22567345014835867a9dfe3fbdb
VERSION=0.0.6
INTEGRITY=sha512-hL0WzAtwijM3sdBsGqdEZuknwXCIup9c74YAQ/QTrQv/MQQZXUEcwr47u1fNmF6EysWPsD41jEFhMrQs1gI2PQ==

dest=${1:-.upstream}
mkdir -p "$dest"
if [ ! -d "$dest/gpu-lexer/.git" ]; then
  git init -q "$dest/gpu-lexer"
  git -C "$dest/gpu-lexer" remote add origin https://github.com/vercel-labs/gpu-lexer.git
fi
git -C "$dest/gpu-lexer" fetch -q --depth 1 origin "$COMMIT"
git -C "$dest/gpu-lexer" checkout -q --detach FETCH_HEAD
test "$(git -C "$dest/gpu-lexer" rev-parse HEAD)" = "$COMMIT"

tarball="$dest/gpu-lexer-$VERSION.tgz"
curl -sSfL "https://registry.npmjs.org/gpu-lexer/-/gpu-lexer-$VERSION.tgz" -o "$tarball"
actual="sha512-$(openssl dgst -sha512 -binary "$tarball" | base64 | tr -d '\n')"
if [ "$actual" != "$INTEGRITY" ]; then
  echo "npm tarball integrity mismatch: $actual" >&2
  exit 1
fi
rm -rf "$dest/npm" && mkdir -p "$dest/npm"
tar -xzf "$tarball" -C "$dest/npm" --strip-components=1
echo "upstream: $dest/gpu-lexer ($COMMIT), npm: $dest/npm (gpu-lexer@$VERSION)"
