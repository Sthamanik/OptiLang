"""
End-to-end demonstration of the OptiLang analysis pipeline.

Pipeline for each program:
    1. execute(source)          → run the code, collect profiling data
    2. Optimizer(...).run()     → analyse AST for optimization suggestions
    3. calculate_score(...)     → four-dimension score + narrative

Four programs are demonstrated, each designed to highlight a different
outcome:

    Program 1 — Perfect          Simple O(1) code, no issues
    Program 2 — Linear loop      O(n) efficiency, minor suggestions
    Program 3 — Nested loops     O(n²) complexity, quality issues
    Program 4 — Broken code      Runtime errors, structural problems
"""

import textwrap
from dataclasses import dataclass
from typing import Any, Dict, Optional

from optilang import execute
from optilang.lexer import tokenize
from optilang.models import ExecutionResult, OptimizationReport
from optilang.optimizer import Optimizer
from optilang.parser import parse
from optilang.scoring import ScoreReport, calculate_score

SEPARATOR = "=" * 64


@dataclass
class DemoReport:
    """Combined report for one demo program."""

    execution: ExecutionResult
    optimizer: Optional[OptimizationReport]
    score: ScoreReport
    optimizer_note: Optional[str] = None


def _bar(score: float, max_score: float, width: int = 30) -> str:
    """Render a simple ASCII progress bar for a dimension score."""
    filled = int(round((score / max_score) * width))
    empty = width - filled
    return f"[{'█' * filled}{'░' * empty}] {score:.1f}/{max_score:.0f}"


def _source_lines(source: str) -> list[str]:
    """Return source lines without disturbing line-number alignment."""
    return source.splitlines()


def _source_lookup(source: str) -> Dict[int, str]:
    """Map 1-based line numbers to source text."""
    return {index: line for index, line in enumerate(_source_lines(source), start=1)}


def _safe_repr(value: Any, limit: int = 48) -> str:
    """Short repr for demo output."""
    if value.__class__.__name__ == "UserFunction" and hasattr(value, "name"):
        return f"<function {value.name}>"

    rendered = repr(value)
    if len(rendered) > limit:
        return rendered[: limit - 3] + "..."
    return rendered


def _display_output_block(label: str, lines: list[str]) -> None:
    """Print a label followed by an indented multi-line block."""
    print(f"  {label:<10}:")
    for line in lines:
        print(f"    {line}")


def _display_execution(result: ExecutionResult) -> None:
    """Print the execution result section."""
    print("\n── Result ──")
    print(f"  Status    : {'ok' if not result.errors else 'error'}")
    print(f"  Exec time : {result.execution_time * 1000:.3f} ms")

    output_lines = result.output.splitlines() or ["<no output>"]
    _display_output_block("Output", output_lines)

    error_lines = result.errors or ["none"]
    _display_output_block("Errors", error_lines)

    if result.symbol_table:
        symbols = [
            f"{name} = {_safe_repr(value)}"
            for name, value in sorted(result.symbol_table.items())
        ]
    else:
        symbols = ["<none>"]
    _display_output_block("Symbols", symbols)


def _display_profiling(source: str, result: ExecutionResult) -> None:
    """Print profiling highlights for the executed program."""
    print("\n── Profiling ──")

    profiling = result.profiling
    if profiling is None:
        print("  Unavailable: execution stopped before profiling data was produced.")
        return

    source_lookup = _source_lookup(source)
    top_lines = sorted(
        profiling.line_stats.values(),
        key=lambda stats: (stats.execution_count, stats.total_time_ms),
        reverse=True,
    )[:3]
    top_functions = sorted(
        profiling.function_stats.values(),
        key=lambda stats: (stats.call_count, stats.total_time_ms),
        reverse=True,
    )[:3]

    print(f"  Total time : {profiling.total_execution_time_ms:.3f} ms")
    print(f"  Lines exec.: {profiling.total_lines_executed}")
    print(f"  Lines prof.: {len(profiling.line_stats)}")
    print(f"  Peak memory: {profiling.peak_memory_bytes} bytes")
    print(
        "  Complexity : "
        f"{profiling.complexity_estimate} "
        f"({profiling.complexity_method}, conf={profiling.complexity_confidence:.2f})"
    )

    if top_lines:
        print("  Hot lines  :")
        for stats in top_lines:
            snippet = source_lookup.get(stats.line_number, "").strip() or "<blank>"
            print(
                f"    L{stats.line_number:>2} | count={stats.execution_count:<5} "
                f"time={stats.total_time_ms:.3f} ms | {snippet}"
            )

    if top_functions:
        print("  Functions  :")
        for stats in top_functions:
            print(
                f"    {stats.name:<12} calls={stats.call_count:<3} "
                f"avg={stats.avg_time_ms:.3f} ms total={stats.total_time_ms:.3f} ms"
            )


def _display_suggestions(
    report: Optional[OptimizationReport], note: Optional[str]
) -> None:
    """Print optimizer suggestions."""
    print("\n── Suggestions ──")

    if report is None:
        print("  Unavailable.")
        if note:
            wrapped = textwrap.fill(
                note,
                width=60,
                initial_indent="  Note      : ",
                subsequent_indent="              ",
            )
            print(wrapped)
        return

    if not report.suggestions:
        print("  No optimization suggestions.")
        return

    print(f"  Suggestions: {len(report.suggestions)}")
    for suggestion in report.suggestions:
        print(
            f"  Line {suggestion.line:>2} "
            f"[{suggestion.severity.upper():<6}] "
            f"{suggestion.pattern} "
            f"(impact {suggestion.impact_score:.1f})"
        )
        print(
            textwrap.fill(
                suggestion.description,
                width=60,
                initial_indent="    What : ",
                subsequent_indent="           ",
            )
        )
        print(
            textwrap.fill(
                suggestion.suggestion,
                width=60,
                initial_indent="    Fix  : ",
                subsequent_indent="           ",
            )
        )


def _display_score(score_report: ScoreReport) -> None:
    """Print the existing scoring report."""
    dims = score_report.dimensions

    print("\n── Overall Score ──")
    print(f"  Score      : {score_report.score:.1f} / 100")
    print(f"  Grade      : {score_report.grade}")
    print(f"  Complexity : {score_report.complexity_class}")
    print(f"  CV         : {score_report.cv:.3f}")
    print(f"  Errors     : {score_report.error_count}")
    print(f"  Lines prof.: {score_report.lines_profiled}")

    print("\n── Dimension Breakdown ──")
    print(f"  Correctness          {_bar(dims.correctness, 35)}")
    print(f"  Efficiency+Complexity{_bar(dims.efficiency_complexity, 30)}")
    print(f"    └ Complexity sub   {_bar(dims.complexity_subscore, 15)}")
    print(f"    └ Efficiency sub   {_bar(dims.efficiency_subscore, 15)}")
    print(f"  Quality              {_bar(dims.quality, 20)}")
    print(f"  Maintainability      {_bar(dims.maintainability, 15)}")

    if dims.profiling_partial or dims.optimizer_partial:
        flags = []
        if dims.profiling_partial:
            flags.append("profiling (partial credit awarded)")
        if dims.optimizer_partial:
            flags.append("optimizer (partial credit awarded)")
        print(f"\n  ⚠  Data unavailable: {', '.join(flags)}")

    print("\n── Narrative ──")
    wrapped = textwrap.fill(
        score_report.narrative,
        width=60,
        initial_indent="  ",
        subsequent_indent="  ",
    )
    print(wrapped)


def _display(title: str, source: str, demo_report: DemoReport) -> None:
    """Print a formatted end-to-end report for one program."""
    print(SEPARATOR)
    print(f"  {title}")
    print(SEPARATOR)

    print("\n── Source Code ──")
    for line_number, line in enumerate(_source_lines(source), start=1):
        print(f"  {line_number:>2}│ {line}")

    _display_execution(demo_report.execution)
    _display_profiling(source, demo_report.execution)
    _display_suggestions(demo_report.optimizer, demo_report.optimizer_note)
    _display_score(demo_report.score)
    print()


def _run_pipeline(source: str) -> DemoReport:
    """Full pipeline: execute → optimize → score."""
    result = execute(source)

    optimizer_report: Optional[OptimizationReport] = None
    optimizer_note: Optional[str] = None
    try:
        ast = parse(tokenize(source))
        optimizer_report = Optimizer(
            ast,
            result.profiling,
            result.symbol_table or None,
        ).run()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        optimizer_note = f"Optimizer skipped: {exc}"

    score_report = calculate_score(
        profiling_data=result.profiling.to_dict() if result.profiling else None,
        optimizer_report=optimizer_report,
        source_lines=len(_source_lines(source)),
        errors=result.errors,
    )
    return DemoReport(
        execution=result,
        optimizer=optimizer_report,
        score=score_report,
        optimizer_note=optimizer_note,
    )


# Program 1 — Perfect: simple O(1) code, no issues

PROGRAM_1 = """\
def greet(name):
    message = "Hello, " + name
    return message

result = greet("OptiLang")
print(result)
"""

# Program 2 — Linear loop: O(n) efficiency, minor style suggestions

PROGRAM_2 = """\
def sum_range(n):
    total = 0
    for i in range(n):
        total += i
    return total

unused_var = 999
result = sum_range(100)
print(result)
"""

# Program 3 — Nested loops: O(n²) complexity, quality and maintainability issues

PROGRAM_3 = """\
result = 0
for i in range(50):
    for j in range(50):
        scale = 50 * 3
        result += i + j + scale
print(result)
"""

# Program 4 — Broken code: runtime error, dead code, unused variables

PROGRAM_4 = """\
def divide(a, b):
    return a / b
    leftover = 99

unused_var = 42
answer = divide(10, 0)
print(answer)
"""

# Program 5 — Complex code: nested loops with multiple operations, moderate complexity o(n2)
PROGRAM_5 = """
numbers = [1, 2, 3]

for i in numbers:
    for j in numbers:
        print(i, j)
"""

# Program 6 — Simple loop: O(n) efficiency, straightforward code, no issues
PROGRAM_6 = """
numbers = [1, 2, 3, 4, 5]

total = 0

for num in numbers:
    total += num

print(total)
"""

# Main

if __name__ == "__main__":
    programs = [
        ("Program 1 — Perfect", PROGRAM_1),
        ("Program 2 — Linear Loop", PROGRAM_2),
        ("Program 3 — Nested Loops", PROGRAM_3),
        ("Program 4 — Broken Code", PROGRAM_4),
        ("Program 5 — Complex Code", PROGRAM_5),
        ("Program 6 — Simple Loop", PROGRAM_6),
    ]

    print()
    print("  OptiLang End-to-End Demo")
    print("  Result, profiling, suggestions, and scoring")
    print()

    for title, source in programs:
        demo_report = _run_pipeline(source)
        _display(title, source, demo_report)

    print(SEPARATOR)
    print("  Demo complete.")
    print(SEPARATOR)
    print()
