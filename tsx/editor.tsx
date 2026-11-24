// Editor helpers of nsh-gpu-lexer: gpu-lexer's spans as Editor decorations.
//
// Two ways to color an Editor with gpu-lexer, both explicit:
//
// 1. `GpuLexer.installFallback()` (from "nsh-gpu-lexer") makes gpu-lexer the
//    highlighter of editors whose language is unknown or unset, and of
//    `language="gpu-lexer"`: it runs off the UI thread in Rust.
// 2. `useGpuLexerDecorations(code)` here gives decorations for any one
//    editor, with `highlighting={false}` so they color the text alone.
import { computed, effect, signal, type Accessor } from "nsh";
import type { EditorDecoration, EditorPosition } from "nsh/base";
import { GpuLexer, type GpuLexerSpan } from "nsh-gpu-lexer";
import { tokenOf } from "./tokens";

export { EDITOR_TOKENS, TOKEN_CLASSES, tokenOf } from "./tokens";

/** Where each line of `code` starts, in UTF-16 code units (an Editor's lines end at `\n`). */
export function lineStarts(code: string): number[] {
  const starts = [0];
  for (let index = code.indexOf("\n"); index >= 0; index = code.indexOf("\n", index + 1)) starts.push(index + 1);
  return starts;
}

/** Decorations that color `spans` of `code` as the Editor colors its tokens. */
export function spansToDecorations(
  code: string,
  spans: readonly GpuLexerSpan[],
  minConfidence = 0,
): EditorDecoration[] {
  const starts = lineStarts(code);
  let line = 0;
  const position = (offset: number): EditorPosition => {
    while (line + 1 < starts.length && starts[line + 1] <= offset) line++;
    return { line, column: offset - starts[line] };
  };
  const decorations: EditorDecoration[] = [];
  for (const span of spans) {
    const token = tokenOf(span, minConfidence);
    if (!token) continue;
    const start = position(span.start);
    const end = position(span.end);
    decorations.push({ range: { start, end }, token });
  }
  return decorations;
}

export interface GpuLexerDecorationsOptions {
  /** Spans whose confidence is below this (0 to 1) stay plain. */
  minConfidence?: number | Accessor<number>;
  /** How long `code` must stay unchanged before it is highlighted again, in ms (default 120). */
  debounceMs?: number;
}

/**
 * Decorations coloring `code()` with gpu-lexer, for an Editor with
 * `highlighting={false}`. The whole text is highlighted again once it stops
 * changing for `debounceMs`; meanwhile the editor keeps the decorations it
 * has, moved with the text.
 */
export function useGpuLexerDecorations(
  code: Accessor<string>,
  options: GpuLexerDecorationsOptions = {},
): Accessor<EditorDecoration[]> {
  const result = signal<{ code: string; spans: GpuLexerSpan[] }>({ code: "", spans: [] });
  const debounceMs = options.debounceMs ?? 120;
  let version = 0;
  effect(() => {
    const text = code();
    const id = ++version;
    const timer = setTimeout(() => {
      GpuLexer.parse(text).then((spans) => {
        if (id === version) result.set({ code: text, spans });
      });
    }, debounceMs);
    return () => clearTimeout(timer);
  });
  return computed(() => {
    const { code: text, spans } = result();
    const min = typeof options.minConfidence === "function" ? options.minConfidence() : options.minConfidence ?? 0;
    return spansToDecorations(text, spans, min);
  });
}
