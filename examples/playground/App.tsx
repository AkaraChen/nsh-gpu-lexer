// gpu-lexer playground: paste code into the editor, it is highlighted as you
// type by gpu-lexer (no language id), and the slider hides the spans the
// model is less sure of.
import { computed, effect, mount, signal, Text, View } from "nsh";
import { Editor, Slider, TextView } from "nsh/base";
import { GpuLexer, type GpuLexerSpan } from "nsh-gpu-lexer";
import { renderLexedCodeBlock } from "nsh-gpu-lexer/text-view";
import { SAMPLE } from "./sample";

const CLASSES = ["keyword", "type", "function", "string", "number", "constant", "comment", "operator"] as const;

export function Playground() {
  const code = signal(SAMPLE);
  const minConfidence = signal(0);
  const spans = signal<GpuLexerSpan[]>([]);

  // Explicit: the editor below has no language, so gpu-lexer highlights it
  // only once it is installed as the fallback highlighter.
  effect(() => {
    void GpuLexer.installFallback({ minConfidence: minConfidence() });
  });

  // The same spans through the TSX API, for the counts under the editor.
  let version = 0;
  effect(() => {
    const text = code();
    const id = ++version;
    GpuLexer.parse(text).then((result) => {
      if (id === version) spans.set(result);
    });
  });

  const shown = computed(() => spans().filter((s) => s.type !== "plain" && s.confidence >= minConfidence()));
  const counts = computed(() =>
    CLASSES.map((type) => `${type} ${shown().filter((s) => s.type === type).length}`).join(" · "),
  );
  const info = GpuLexer.info();

  return (
    <View class="flex-1 gap-3 bg-background p-4">
      <Text class="text-lg font-semibold text-foreground">gpu-lexer playground</Text>
      <Text class="text-xs text-muted-foreground">
        {`nsh-gpu-lexer ${info.version} · gpu-lexer ${info.upstreamVersion} · checkpoint ${info.runId}`}
      </Text>
      <View class="flex-row items-center gap-3">
        <Text class="text-sm text-foreground">{`Min confidence ${minConfidence().toFixed(2)}`}</Text>
        <Slider
          accessibilityLabel="Min confidence"
          value={minConfidence()}
          min={0}
          max={1}
          step={1 / 16}
          onChange={({ value }) => minConfidence.set(typeof value === "number" ? value : value[0])}
          class="h-6 w-48 justify-center"
          trackClass="h-1 rounded-full bg-muted"
          rangeClass="h-1 rounded-full bg-primary"
          thumbClass="h-4 w-4 rounded-full bg-primary"
        />
      </View>
      <Editor
        testID="editor"
        defaultValue={SAMPLE}
        onChange={({ value }) => code.set(value)}
        class="min-h-64 flex-1 rounded-md border border-border"
      />
      <Text testID="span-count" class="text-sm text-foreground">{`Spans: ${shown().length}`}</Text>
      <Text testID="class-counts" class="text-xs text-muted-foreground">{counts()}</Text>
      <TextView
        markdown={"A code block, colored with `renderCodeBlock`:\n\n```\nconst answer = 42; // the answer\n```\n"}
        renderCodeBlock={renderLexedCodeBlock}
      />
    </View>
  );
}

mount(() => <Playground />);
