// TextView helpers of nsh-gpu-lexer: code blocks colored by gpu-lexer.
//
//   <TextView markdown={doc} renderCodeBlock={renderLexedCodeBlock} />
//
// or, to choose the threshold or the look, render <LexedCode> yourself.
import { effect, signal, Scroll, Text, View } from "nsh";
import type { TextViewCodeBlock } from "nsh/base";
import { GpuLexer, type GpuLexerSpan } from "nsh-gpu-lexer";
import { TOKEN_CLASSES, tokenOf } from "./tokens";

export { EDITOR_TOKENS, TOKEN_CLASSES, tokenOf } from "./tokens";

/** Code at most this long is highlighted inside the render; longer code is highlighted off the UI thread and shows plain until then. */
export const SYNC_LIMIT = 20_000;

export interface LexedCodeProps {
  /** The code, or a TextView code block. */
  code: string | TextViewCodeBlock;
  /** Spans whose confidence is below this (0 to 1) stay plain. */
  minConfidence?: number;
  /** Tailwind classes of the code's text. */
  class?: string;
  /** Tailwind classes of the box around it. */
  boxClass?: string;
}

/** Code colored by gpu-lexer: one Text, a nested Text per colored span. */
export function LexedCode(props: LexedCodeProps) {
  const source = typeof props.code === "string" ? props.code : props.code.code;
  const wrap = typeof props.code === "string" ? false : props.code.wrap ?? false;
  const spans = signal<GpuLexerSpan[]>(source.length <= SYNC_LIMIT ? GpuLexer.parseSync(source) : []);
  if (source.length > SYNC_LIMIT) {
    effect(() => {
      GpuLexer.parse(source).then((result) => spans.set(result));
    });
  }
  const pieces = () => {
    const min = props.minConfidence ?? 0;
    const out: { text: string; class?: string }[] = [];
    let at = 0;
    for (const span of spans()) {
      const token = tokenOf(span, min);
      if (!token) continue;
      if (span.start > at) out.push({ text: source.slice(at, span.start) });
      out.push({ text: source.slice(span.start, span.end), class: TOKEN_CLASSES[token] });
      at = span.end;
    }
    if (at < source.length) out.push({ text: source.slice(at) });
    return out;
  };
  const body = (
    <Text class={`font-mono text-sm text-foreground ${props.class ?? ""}`}>
      {pieces().map((piece) => (piece.class ? <Text class={piece.class}>{piece.text}</Text> : piece.text))}
    </Text>
  );
  return (
    <View class={props.boxClass ?? "rounded-md bg-muted p-3"}>
      {wrap ? body : <Scroll horizontal>{body}</Scroll>}
    </View>
  );
}

/** A TextView `renderCodeBlock` that colors every code block with gpu-lexer. */
export function renderLexedCodeBlock(code: TextViewCodeBlock) {
  return <LexedCode code={code} />;
}
