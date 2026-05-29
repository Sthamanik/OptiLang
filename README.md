# OptiLang

A Python-inspired interpreter with built-in profiling, static complexity analysis, optimization detection, and scoring.

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI version](https://img.shields.io/pypi/v/optilang.svg)](https://pypi.org/project/optilang/)
[![PyPI downloads](https://img.shields.io/pypi/dm/optilang.svg)](https://pypi.org/project/optilang/)

OptiLang `1.0.2` ships the full source-to-insight pipeline:

`source -> tokens -> AST -> semantic checks -> execution -> profiling -> static complexity -> optimization suggestions -> score`

## Release Highlights

- Python-like language core with variables, control flow, functions, recursion, lists, dictionaries, tuple unpacking, method calls, slicing, and exception handling
- Runtime execution with line-level and function-level profiling
- Static Big-O complexity analysis using AST structure — no runtime guessing, confidence scores included
- Extended complexity class support: `O(1)` through `O(n!)`, `O(2^n)`, `O(∞)`, `O(n+m)`, `O(n*m)`, and more
- Ten optimization detectors for performance and maintainability issues
- Five-dimension scoring system (correctness, efficiency, complexity, quality, maintainability) with a final `0–100` score, grade, and narrative explanation
- Reorganized package layout (`core`, `analysis`, `runtime`, `types`, `utils`) with full backward compatibility

## Quick Start

### Install from PyPI

```bash
pip install optilang
```

### Install from Source

```bash
git clone https://github.com/Sthamanik/optilang.git
cd optilang
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

### End-to-End Example

```python
from optilang import analyze, calculate_score, execute
from optilang.lexer import tokenize
from optilang.parser import parse

source = """
total = 0
for i in range(10):
    total += i
print(total)
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

print(result.output)                # 45
print(score.grade, score.score)     # e.g. Excellent 95.0
print(score.complexity_class)       # e.g. O(n)

for suggestion in report.suggestions:
    print(f"{suggestion.pattern}: {suggestion.description}")
```

For quick optimization suggestions from source text only:

```python
from optilang import analyze_source

report = analyze_source(source)
```

### Static Complexity Analysis

```python
from optilang import analyze_complexity, analyze_function_complexity
from optilang.lexer import tokenize
from optilang.parser import parse

source = """
def bubble_sort(arr):
    for i in range(len(arr)):
        for j in range(len(arr) - i - 1):
            if arr[j] > arr[j + 1]:
                arr[j], arr[j + 1] = arr[j + 1], arr[j]
"""

ast = parse(tokenize(source))
result = analyze_complexity(ast)
print(result.complexity)     # O(n²)
print(result.confidence)     # 1.0
print(result.explanation)

# Per-function breakdown
fn_results = analyze_function_complexity(ast)
for name, r in fn_results.items():
    print(f"{name}: {r.complexity} (confidence={r.confidence})")
```

## Architecture

```
┌──────────┐    ┌──────────┐    ┌─────────────┐    ┌──────────┐    ┌──────────┐
│  Lexer   │ -> │  Parser  │ -> │  Semantic   │ -> │ Executor │ -> │ Profiler │
│ (Tokens) │    │  (AST)   │    │ (Annot AST) │    │ (Runtime)│    │ (Metrics)│
└──────────┘    └──────────┘    └─────────────┘    └──────────┘    └──────────┘
                     │                                    │              │
                     v                                    v              v
               ┌──────────┐                        ┌──────────┐    ┌──────────┐
               │Complexity│                        │Optimizer │    │  Scorer  │
               │ (Big-O)  │                        │(Patterns)│    │ (0-100)  │
               └──────────┘                        └──────────┘    └──────────┘
```

## What OptiLang Supports

### Language Features

- Numbers, strings, booleans, and `None`
- Arithmetic, comparison, logical, unary, and augmented assignment operators (`+=`, `-=`, `*=`, etc.)
- Variables and lexical scoping
- `if` / `elif` / `else`
- `while` and `for ... in ...`
- `break`, `continue`, and `pass`
- Function definitions, calls, default parameters, returns, and recursion
- Lists, dictionaries, and index access
- Tuple unpacking: `a, b = 1, 2` and swap syntax `arr[i], arr[j] = arr[j], arr[i]`
- Method calls: `obj.method(args)`
- List and string slicing: `arr[1:3]`, `s[::-1]`
- Indexed augmented assignment: `arr[i] += 1`
- `try` / `except` / `finally`

### Built-In Functions and Types

| Name | Description |
|------|-------------|
| `print` | Print values to output |
| `range` | Integer range iterator |
| `len` | Length of a list or string |
| `str` | Convert to string |
| `int` | Convert to integer |
| `float` | Convert to float |
| `bool` | Convert to boolean |
| `list` | Convert to list |
| `dict` | Create/convert a dictionary |

## Analysis Features

### Profiling

Execution returns `ExecutionResult`, which can include:

- Captured program output
- Execution time
- Line execution counts and timings
- Function call counts and recursion depth
- Peak memory estimate
- Heuristic runtime complexity estimate
- Final symbol table

### Static Complexity Analysis

`analyze_complexity(ast)` returns a `ComplexityResult` with:

- A `Complexity` enum value representing the Big-O class
- Confidence score (`1.0` = provably derived from AST structure, lower = heuristic)
- Human-readable explanation of the derivation
- Uses symbolic `ComplexityExpr` algebra for exact reasoning about loop nesting, parameters, and recursion

Supported complexity classes:

| Class | Label |
|-------|-------|
| Constant | `O(1)` |
| Logarithmic | `O(log n)` |
| Linear | `O(n)` |
| Linearithmic | `O(n log n)` |
| Quadratic | `O(n²)` |
| Quadratic-log | `O(n² log n)` |
| Cubic | `O(n³)` |
| Quartic | `O(n⁴)` |
| Polynomial | `O(n^k)` |
| Exponential | `O(2^n)` |
| Factorial | `O(n!)` |
| Two-variable | `O(n+m)`, `O(n*m)` |
| Unbounded | `O(∞)` |
| Unknown | `O(?)` |

### Optimization Detectors

OptiLang ships with ten detectors across three categories:

**Static (AST only)**
1. `unused_vars` — variables defined but never read
2. `dead_code` — unreachable statements after `return`/`break`/`continue`
3. `constant_folding` — expressions that can be evaluated at parse time
4. `early_return` — functions that can exit earlier to reduce nesting

**Hybrid (AST + profiling)**

5. `loop_invariant` — computations inside loops that don't change per iteration
6. `string_concat_loop` — string `+=` inside loops (use list + join instead)
7. `nested_loops` — deeply nested loop structures with high complexity

**Dynamic (profiling-driven)**

8. `hot_loop` — loops that account for a disproportionate share of execution time
9. `repeated_computation` — identical sub-expressions evaluated multiple times
10. `expensive_calls` — function calls with high per-call cost

### Scoring

`calculate_score(...)` returns a `ScoreReport` with:

- Final score from `0` to `100`
- Grade label: `Excellent`, `Good`, `Fair`, or `Poor`
- Complexity class string
- Five-dimension breakdown:

| Dimension | Max Points | What it measures |
|-----------|-----------|-----------------|
| Correctness | 35 | Error density |
| Efficiency | 15 | Optimizer pattern findings |
| Complexity | 15 | Big-O class |
| Quality | 20 | Runtime anti-patterns |
| Maintainability | 15 | Style and structure |

- Beginner-friendly narrative summary

## Project Layout

```text
optilang/                        # Package root
  __init__.py                    # Public API and backward-compat lazy loading
  core/                          # Language frontend
    lexer.py                     # Tokenizer
    parser.py                    # AST builder
    ast_nodes.py                 # AST node definitions
    token.py                     # Token types
  analysis/                      # Static analysis
    complexity.py                # Big-O complexity analyzer
    optimizer.py                 # Optimization pattern detectors
    scoring.py                   # Five-dimension scorer
    semantic_analyzer.py         # Semantic checks and annotation
  runtime/                       # Execution layer
    executor.py                  # AST tree-walking interpreter
    profiler.py                  # Line/function profiling
  types/                         # Shared data types
    models.py                    # ExecutionResult, ScoreReport, etc.
    constants.py                 # Complexity labels and scoring points
  utils/                         # Utilities
    errors.py                    # Error types
  patterns/                      # Pattern classification constants
tests/                           # Test suite (pytest)
  test_complexity.py
  test_errors.py
  test_executor.py
  test_integration.py
  test_lexer.py
  test_models.py
  test_optimizer.py
  test_parser.py
  test_profiler.py
  test_scoring.py
  test_semantic_analyzer.py
docs/                            # Documentation
  README.md
  USER_GUIDE.md
  API_REFERENCE.md
```

> **Backward compatibility:** The old flat import paths (`from optilang.lexer import tokenize`, `from optilang.optimizer import ...`, etc.) continue to work via lazy module aliasing in `__init__.py`.

## Documentation

- [Documentation Index](docs/README.md)
- [User Guide](docs/USER_GUIDE.md)
- [API Reference](docs/API_REFERENCE.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

## Development

```bash
pip install -e ".[dev]"
python3 -m pytest
black optilang tests
mypy optilang
flake8 optilang
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for workflow, quality checks, and documentation expectations.

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Team

- Manik Kumar Shrestha - [GitHub](https://github.com/Sthamanik)
- Om Shree Mahat - [GitHub](https://github.com/itsomshree)
- Aashish Rimal - [GitHub](https://github.com/aashishrimal22)

## Contact

- Email: shresthamanik1820@gmail.com
- Issues: [GitHub Issues](https://github.com/Sthamanik/optilang/issues)
