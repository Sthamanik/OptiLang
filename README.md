# OptiLang

A Python-inspired interpreter with built-in profiling, optimization analysis, and scoring.

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![PyPI version](https://img.shields.io/pypi/v/optilang.svg)](https://pypi.org/project/optilang/)
[![PyPI downloads](https://img.shields.io/pypi/dm/optilang.svg)](https://pypi.org/project/optilang/)

OptiLang `1.0.0` is the first stable release of the project. It ships the full source-to-insight pipeline:

`source -> tokens -> AST -> semantic checks -> execution -> profiling -> optimization suggestions -> score`

## Release Highlights

- Python-like language core with variables, control flow, functions, recursion, lists, dictionaries, and exception handling
- Runtime execution with line-level and function-level profiling
- Ten optimization detectors for performance and maintainability issues
- Four-dimension scoring system with a final `0-100` score, grade, and narrative explanation
- End-to-end Python API for execution, analysis, and scoring

## Quick Start

### Install From Source

```bash
git clone https://github.com/Sthamanik/optilang.git
cd optilang
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e .
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

If you only need optimization suggestions from source text, use `analyze_source(source)` from `optilang` or `optilang.optimizer`.

## 🏗️ Architecture

```
┌──────────┐    ┌──────────┐    ┌─────────────┐    ┌──────────┐    ┌──────────┐
│  Lexer   │ -> │  Parser  │ -> │  Semantic   │ -> │ Executor │ -> │ Profiler │
│ (Tokens) │    │  (AST)   │    │ (Annot AST) │    │ (Runtime)│    │ (Metrics)│
└──────────┘    └──────────┘    └─────────────┘    └──────────┘    └──────────┘
                                                         │              │
                                                         v              v
                                                   ┌──────────┐    ┌──────────┐
                                                   │Optimizer │    │  Scorer  │
                                                   │(Patterns)│    │ (0-100)  │
                                                   └──────────┘    └──────────┘
```

## What OptiLang Supports

### Language Features

- Numbers, strings, booleans, and `None`
- Arithmetic, comparison, logical, unary, and augmented assignment operators
- Variables and lexical scoping
- `if` / `elif` / `else`
- `while` and `for ... in ...`
- `break`, `continue`, and `pass`
- Function definitions, calls, parameters, returns, and recursion
- Lists, dictionaries, and index access
- `try` / `except` / `finally`

### Built-In Functions and Types

- `print`
- `range`
- `len`
- `str`
- `int`
- `float`
- `bool`
- `list`
- `dict`

## Analysis Features

### Profiling

Execution returns `ExecutionResult`, which can include:

- Captured program output
- Execution time
- Line execution counts and timings
- Function call counts and recursion depth
- Peak memory estimate
- Heuristic complexity estimate
- Final symbol table

### Optimization Detectors

OptiLang `1.0.0` ships with ten detectors:

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

### Scoring

`calculate_score(...)` returns a `ScoreReport` with:

- Final score from `0` to `100`
- Grade label such as `Excellent`, `Good`, or `Fair`
- Complexity class
- Dimension breakdown for correctness, efficiency/complexity, quality, and maintainability
- Beginner-friendly narrative summary

## Project Layout

```text
optilang/
  lexer.py
  parser.py
  semantic_analyzer.py
  executor.py
  profiler.py
  optimizer.py
  scoring.py
  models.py
tests/
docs/
```

## Documentation

- [Documentation Index](docs/README.md)
- [User Guide](docs/USER_GUIDE.md)
- [API Reference](docs/API_REFERENCE.md)
- [Contributing Guide](CONTRIBUTING.md)
- [Changelog](CHANGELOG.md)

## Development

```bash
python3 -m pip install -e ".[dev]"
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
