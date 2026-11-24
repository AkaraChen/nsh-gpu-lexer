// The code the playground starts with: a few languages in one text, as the
// model needs no language id.
export const SAMPLE = `// TypeScript
export async function greet(name: string): Promise<string> {
  const count = 42;
  return \`Hello, \${name}! (\${count})\`;
}

# Python
def fib(n: int) -> int:
    """Return the n-th Fibonacci number."""
    return n if n < 2 else fib(n - 1) + fib(n - 2)

/* Rust */
fn main() {
    let v: Vec<u32> = (1..=3).collect();
    println!("{:?} {}", v, true);
}
`;
