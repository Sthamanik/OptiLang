# OptiLang API Reference

## Public Entry Points

### `execute(source, timeout_seconds=5.0, enable_profiling=True) -> ExecutionResult`

Runs the full execution pipeline:

`tokenize -> parse -> semantic analyze -> execute`

Arguments:

- `source`: OptiLang source code as a string
- `timeout_seconds`: maximum execution time before a timeout error
- `enable_profiling`: whether profiling data should be collected

Returns:

- `ExecutionResult`

### `analyze(ast, profiling=None, symbol_table=None) -> OptimizationReport`

Runs all optimizer detectors against an already-parsed AST.

Use this when you already have:

- `ast` from `parse(tokenize(source))`
- `profiling` from `execute(source).profiling`
- `symbol_table` from `execute(source).symbol_table`

### `analyze_source(source) -> OptimizationReport`

Convenience helper that runs the full pipeline from source text and returns only the optimization report.

### `calculate_score(profiling_data, optimizer_report=None, source_lines=1, errors=None) -> ScoreReport`

Creates the final score and narrative report.

Arguments:

- `profiling_data`: `result.profiling.to_dict()` or `None`
- `optimizer_report`: `OptimizationReport` or `None`
- `source_lines`: number of lines in the original source
- `errors`: `ExecutionResult.errors`

## Low-Level Helpers

### `tokenize(source) -> list[Token]`

Converts source text into tokens.

### `parse(tokens) -> ProgramNode`

Builds the AST from tokenized input.

## Data Models

### `ExecutionResult`

Fields:

- `output: str`
- `errors: list[str]`
- `execution_time: float`
- `profiling: ProfilingData | None`
- `symbol_table: dict[str, Any]`

### `Suggestion`

Fields:

- `line: int`
- `pattern: str`
- `severity: str`
- `description: str`
- `suggestion: str`
- `impact_score: float`

### `OptimizationReport`

Fields:

- `suggestions: list[Suggestion]`

Note: scoring data is not stored on `OptimizationReport`. Use `ScoreReport` for final scores.

### `DimensionScores`

Fields:

- `correctness`
- `efficiency_complexity`
- `quality`
- `maintainability`
- `complexity_subscore`
- `efficiency_subscore`
- `profiling_partial`
- `optimizer_partial`

### `ScoreReport`

Fields:

- `score`
- `grade`
- `complexity_class`
- `dimensions`
- `narrative`
- `error_count`
- `lines_profiled`
- `cv`

## Profiling Structures

Important profiling types:

- `ProfilingData`
- `LineStats`
- `FunctionStats`
- `ProfilerConfig`

`ProfilingData.to_dict()` is the standard bridge from runtime output into the scoring system.
