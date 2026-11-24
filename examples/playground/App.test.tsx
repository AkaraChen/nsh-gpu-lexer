// `nsh test` builds a headless runner with nsh-gpu-lexer and runs these.
import { expect, open, test } from "nsh/test";
import { GpuLexer } from "nsh-gpu-lexer";

test("GpuLexer.parse returns upstream's spans", async () => {
  const code = "const answer = 42; // the answer";
  const spans = await GpuLexer.parse(code);
  expect(spans[0]).toEqual({ type: "keyword", start: 0, end: 5, confidence: spans[0].confidence });
  expect(spans[spans.length - 1].end).toBe(code.length);
  for (let i = 1; i < spans.length; i++) expect(spans[i].start).toBe(spans[i - 1].end);
  expect(spans.some((s) => s.type === "number" && code.slice(s.start, s.end) === "42")).toBe(true);
  expect(spans.some((s) => s.type === "comment")).toBe(true);
  for (const span of spans) {
    expect(span.confidence > 0 && span.confidence <= 1).toBe(true);
    expect(Number.isInteger(span.confidence * 16)).toBe(true);
  }
  // UTF-16 offsets, as JavaScript strings index.
  const emoji = "let s = \"😀\"; // é";
  const last = (await GpuLexer.parse(emoji)).at(-1)!;
  expect(last.end).toBe(emoji.length);
  expect(GpuLexer.parseSync(code)).toEqual(spans);
  expect(await GpuLexer.parse("")).toEqual([]);
});

test("the playground counts the spans of its code", async () => {
  const app = await open("./App.tsx", { platform: "linux", width: 1000, height: 900 });
  await app.waitFor("text:gpu-lexer playground");
  await app.waitUntil(() => app.texts().some((t) => t.startsWith("Spans: ") && t !== "Spans: 0"));
  expect(app.texts().some((t) => /keyword [1-9]/.test(t))).toBe(true);
});

test("the editor is highlighted by gpu-lexer once installed as the fallback", async () => {
  const app = await open("./App.tsx", { platform: "linux", width: 1000, height: 900 });
  await app.waitFor("text:gpu-lexer playground");
  // "def" is a keyword: the fallback highlighter colors it as the editor
  // colors keywords, which plain text is not.
  await app.waitUntil(() => app.spanStyle("text:def").color !== app.spanStyle("text:fib(n - 1)").color, {
    timeout: 10_000,
  });
  const keyword = app.spanStyle("text:def");
  expect(keyword.color).toMatch(/^#8250df/i);
});

test("nothing is highlighted by gpu-lexer until it is asked to", async () => {
  const app = await open("./Explicit.tsx", { platform: "linux", width: 800, height: 600 });
  // useGpuLexerDecorations colors its editor (built-in highlighting off) ...
  await app.waitUntil(() => app.spanStyle("text:def answer").color.toLowerCase().startsWith("#8250df"), {
    timeout: 10_000,
  });
  // ... while the editor without a language stays plain: no fallback was installed.
  await app.wait(500);
  expect(app.spanStyle("text:while ready").color.toLowerCase().startsWith("#8250df")).toBe(false);
});

test("TextView code blocks are colored by renderLexedCodeBlock", async () => {
  const app = await open("./App.tsx", { platform: "linux", width: 1000, height: 900 });
  await app.waitFor("text:gpu-lexer playground");
  await app.waitUntil(() => app.spanStyle("text:const answer").color.toLowerCase().startsWith("#8250df"));
  expect(app.spanStyle("text:the answer").italic).toBe(true);
});
