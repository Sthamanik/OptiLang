# OptiLang User Guide

## Overview

OptiLang is a Python-inspired interpreted language designed for learning, execution tracing, and code-quality analysis. The runtime accepts source code as text, validates it, executes it, profiles it, and can attach optimization suggestions and a score report.

## Execution Pipeline

The standard flow is:

1. Tokenize source text
2. Parse tokens into an AST
3. Run semantic validation
4. Execute the program
5. Collect profiling data
6. Run optimization detectors
7. Calculate a final score

In code, the most common end-to-end path is:

```python
from optilang import analyze, calculate_score, execute
from optilang.lexer import tokenize
from optilang.parser import parse

source = """
def square(x):
    return x * x

print(square(8))
"""

result = execute(source)
ast = parse(tokenize(source))
report = analyze(ast, result.profiling, result.symbol_table)
score = calculate_score(
    profiling_data=result.profiling.to_dict() if result.profiling else None,
    optimizer_report=report,
    source_lines=source.count("\n") + 1,
    errors=result.errors,
)
```

## Supported Syntax

### Literals and Expressions

- Integers and floating-point numbers
- Strings with single or double quotes
- `True`, `False`, and `None`
- Arithmetic operators: `+`, `-`, `*`, `/`, `//`, `%`, `**`
- Comparison operators: `==`, `!=`, `<`, `<=`, `>`, `>=`
- Logical operators: `and`, `or`, `not`
- Parenthesized expressions

### Statements

- Variable assignment and augmented assignment
- `if`, `elif`, `else`
- `while`
- `for ... in ...`
- `break`, `continue`, `pass`
- Function definitions and `return`
- `try`, `except`, `finally`

### Data Structures

- Lists
- Dictionaries
- Nested indexing like `items[0]` or `mapping["key"]`

## Built-Ins

OptiLang currently provides:

- `print`
- `range`
- `len`
- `str`
- `int`
- `float`
- `bool`
- `list`
- `dict`

## Execution Results

`execute(source)` returns an `ExecutionResult` object with:

- `output`: captured stdout-like program output
- `errors`: lexer, parser, semantic, or runtime errors
- `execution_time`: total execution time in seconds
- `profiling`: `ProfilingData` when profiling is enabled
- `symbol_table`: final global values after execution

If an error occurs during lexing, parsing, semantic analysis, or runtime execution, OptiLang stops the pipeline and records the error message in `result.errors`.

## Profiling Data

When profiling is enabled, `result.profiling` includes:

- Per-line execution counts and timing
- Per-function call counts and timing
- Recursion depth tracking
- Peak memory estimate
- Complexity estimate and confidence

Use `result.profiling.to_dict()` when you need JSON-friendly data for reporting or scoring.

## Optimization Suggestions

The optimizer emits ranked suggestions for:

1. `unused_vars`
2. `dead_code`
3. `constant_folding`
4. `early_return`
5. `loop_invariant`
6. `string_concat_loop`
7. `nested_loops`
8. `hot_loop`
9. `repeated_computation`
10. `expensive_calls`

Each suggestion includes a line number, severity, description, recommended fix, and impact score.

## Scoring

`calculate_score(...)` returns a `ScoreReport` built from:

- Correctness
- Efficiency and complexity
- Quality
- Maintainability

The score report contains the final numeric score, a grade, the detected complexity class, per-dimension breakdowns, and a short narrative explanation.

## Current Scope

OptiLang is Python-inspired, not full Python. The 1.0.0 release is intentionally focused on the core educational language and analysis pipeline. If you extend the runtime, update the README and `docs/` at the same time.
