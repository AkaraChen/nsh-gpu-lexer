// Test fixtures: an editor with gpu-lexer's decorations (`useGpuLexerDecorations`),
// and one left alone, which gpu-lexer must not color (nothing is installed).
import { mount, signal, View } from "nsh";
import { Editor } from "nsh/base";
import { useGpuLexerDecorations } from "nsh-gpu-lexer/editor";

const DECORATED = "def answer():\n    return 42  # the answer\n";
const UNTOUCHED = "while ready:\n    pass\n";

function Explicit() {
  const code = signal(DECORATED);
  const decorations = useGpuLexerDecorations(code, { debounceMs: 0 });
  return (
    <View class="flex-1 gap-2 p-2">
      <Editor
        testID="decorated"
        defaultValue={DECORATED}
        onChange={({ value }) => code.set(value)}
        highlighting={false}
        decorations={decorations()}
        class="h-24"
      />
      <Editor testID="untouched" defaultValue={UNTOUCHED} class="h-24" />
    </View>
  );
}

mount(() => <Explicit />);
