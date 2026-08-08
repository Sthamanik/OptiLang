# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [1.0.3] - 2026-08-08

### Fixed
- Crash in recursive-function complexity analysis (`UnboundLocalError`) when
  a recursive function's body calls another function before/around its own
  recursive call
- `print()`/`str()` fabricating phantom complexity from scalar identifier
  arguments, inflating simple constant-time programs to `O(n)`
- Empirical complexity fallback discarding the AST, causing recursion depth
  to be conflated with loop nesting depth and producing incorrect classes
  such as `O(n⁴)` for permutation-style recursion
- Generic identifier reads (assignment/return/arithmetic) being treated as
  size-bound even for scalar parameter values, and missing self-recursion
  detection when analyzing a function in isolation via
  `analyze_function_complexity()`
- `range()` bound resolution for `len(x)` calls and arithmetic expressions
  (e.g. `range(len(arr) - i - 1)`), including an enclosing loop's own
  iterator being miscounted as an independent size variable

## [1.0.2] - 2026-05-29

### Added
- Indexed assignment and indexed augmented assignment support
- Enhanced indexing and slicing support in parser and executor
- `MethodCallNode` support with parsing and execution for method calls
- Tuple unpacking with indices and O(n!) complexity detection
- Exponential and unbounded complexity classes
- Loop depth tracking in executor for improved execution control

### Changed
- Major module restructuring: moved code into `core/`, `runtime/`, `analysis/`, `types/` packages with backward-compatible lazy loading
- Enhanced complexity detection with nested structure checks, loop depth analysis, and hot line ratio integration
- Simplified scorer complexity classification and improved efficiency suggestion output
- Removed ML-based analysis pipeline; simplified to static pattern detection
- Improved code formatting and organization across multiple files
- Cleaned up tests and reorganized imports

### Fixed
- Improved profiler complexity detection logic for more accurate hot line analysis

## [1.0.1] - 2026-05-20

### Changed
- Updated README for clarity and accuracy in features and installation instructions

## [1.0.0] - 2026-04-10

### Added
- Stable OptiLang interpreter pipeline: lexer, parser, semantic analyzer, executor, profiler, optimizer, and scorer
- Python-like language support for expressions, control flow, functions, recursion, lists, dictionaries, indexing, and try/except/finally
- Runtime profiling with line stats, function stats, memory estimation, and heuristic complexity detection
- Ten optimization detectors covering performance and maintainability anti-patterns
- Four-dimension scoring system with final score, grade, dimension breakdown, and narrative feedback
- User and API documentation under `README.md` and `docs/`

### Changed
- Promoted project metadata and package versioning to the `1.0.0` stable release
- Aligned README, contributing guide, and documentation links with the current API surface

## [0.1.0] - 2026-02-19

### Added
- Project initialization
- Repository setup
- Development environment configuration

[Unreleased]: https://github.com/Sthamanik/optilang/compare/v1.0.3...HEAD
[1.0.3]: https://github.com/Sthamanik/optilang/releases/tag/v1.0.3
[1.0.2]: https://github.com/Sthamanik/optilang/releases/tag/v1.0.2
[1.0.1]: https://github.com/Sthamanik/optilang/releases/tag/v1.0.1
[1.0.0]: https://github.com/Sthamanik/optilang/releases/tag/v1.0.0
[0.1.0]: https://github.com/Sthamanik/optilang/releases/tag/v0.1.0
