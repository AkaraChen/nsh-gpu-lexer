// What gpu-lexer's nine classes are colored as: the Editor's token kinds,
// and Tailwind classes in the reading palette the built-in highlighting of
// Editor and TextView uses (light, then dark).
import type { EditorToken } from "nsh/base";
import type { GpuLexerSpan, GpuLexerTokenType } from "nsh-gpu-lexer";

/** The Editor token each class is colored as; `null` keeps the text's color. */
export const EDITOR_TOKENS: Record<GpuLexerTokenType, EditorToken | null> = {
  plain: null,
  comment: "comment",
  keyword: "keyword",
  string: "string",
  number: "number",
  constant: "number",
  function: "title",
  type: "built",
  operator: null,
};

/** Tailwind classes of each Editor token, as the built-in highlighting colors it. */
export const TOKEN_CLASSES: Record<EditorToken, string> = {
  comment: "text-[#6e7781] dark:text-[#7d7a72] italic",
  keyword: "text-[#8250df] dark:text-[#c4a7ff]",
  string: "text-[#116329] dark:text-[#9fd68a]",
  number: "text-[#953800] dark:text-[#f5b27a]",
  title: "text-[#0550ae] dark:text-[#82b8ff]",
  attr: "text-[#7d4e00] dark:text-[#f0c674]",
  built: "text-[#0a7285] dark:text-[#6fd3d8]",
  deletion: "text-[#cf222e] dark:text-[#f87171]",
};

/** The Editor token of a span, or `null` when it stays plain (its class, or a confidence under `minConfidence`). */
export function tokenOf(span: GpuLexerSpan, minConfidence = 0): EditorToken | null {
  return span.confidence >= minConfidence ? EDITOR_TOKENS[span.type] : null;
}
