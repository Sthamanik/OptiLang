"""
Integration tests for OptiLang — verifies all three stated objectives:
  1. Execute PyLite programs correctly with execution metrics
  2. Profiling data is present and meaningful
  3. Non-functional: profiling overhead is acceptable

Note: result.profiling is a ProfilingData object — access via attributes
      (.line_stats, .function_stats, etc.) or call .to_dict() for dict access.
"""

from __future__ import annotations
import textwrap
import time
from optilang import execute
from optilang.models import ExecutionResult

# ── Objective 1: Execute PyLite programs correctly ──


def test_simple_arithmetic_program() -> None:
    result = execute("print(2 + 3)")
    assert result.output == "5"
    assert result.errors == []


def test_function_and_recursion() -> None:
    src = textwrap.dedent("""\
        def factorial(n):
            if n <= 1:
                return 1
            return n * factorial(n - 1)
        print(factorial(5))""")
    result = execute(src)
    assert result.output == "120"
    assert result.errors == []


def test_loop_program() -> None:
    src = "total = 0\nfor i in range(10):\n    total += i\nprint(total)"
    result = execute(src)
    assert result.output == "45"
    assert result.errors == []


def test_nested_loops() -> None:
    src = textwrap.dedent("""\
        total = 0
        for i in range(3):
            for j in range(3):
                total += 1
        print(total)""")
    result = execute(src)
    assert result.output == "9"
    assert result.errors == []


def test_try_except_program() -> None:
    src = textwrap.dedent("""\
        try:
            x = 1 / 0
        except:
            print('handled')
        finally:
            print('done')""")
    result = execute(src)
    assert result.output == "handled\ndone"
    assert result.errors == []


def test_returns_execution_result_instance() -> None:
    result = execute("x = 1")
    assert isinstance(result, ExecutionResult)


def test_syntax_error_captured() -> None:
    result = execute("def f(\n    pass")
    assert result.errors != []
    assert result.output == ""


def test_runtime_error_captured() -> None:
    result = execute("print(undefined_var)")
    assert result.errors != []
    assert result.output == ""


def test_handled_exception_no_errors() -> None:
    result = execute("try:\n    x = 1 / 0\nexcept:\n    pass")
    assert result.errors == []


# ── Objective 2: Profiling data is present and meaningful ──


def test_profiling_not_none() -> None:
    result = execute("x = 1")
    assert result.profiling is not None


def test_profiling_total_time_positive() -> None:
    result = execute("x = 1")
    assert result.profiling is not None
    assert result.profiling.total_execution_time_ms > 0


def test_profiling_line_stats_populated() -> None:
    result = execute("x = 1\ny = 2")
    assert result.profiling is not None
    assert len(result.profiling.line_stats) >= 1


def test_loop_line_execution_count() -> None:
    src = "for i in range(5):\n    x = i"
    result = execute(src)
    assert result.profiling is not None
    all_counts = [s.execution_count for s in result.profiling.line_stats.values()]
    assert 5 in all_counts


def test_nested_loop_inner_execution_count() -> None:
    src = "for i in range(4):\n    for j in range(4):\n        x = i + j"
    result = execute(src)
    assert result.profiling is not None
    all_counts = [s.execution_count for s in result.profiling.line_stats.values()]
    assert 16 in all_counts


def test_function_stats_populated() -> None:
    src = "def f():\n    pass\nf()\nf()\nf()"
    result = execute(src)
    assert result.profiling is not None
    assert "f" in result.profiling.function_stats
    assert result.profiling.function_stats["f"].call_count == 3


def test_recursive_function_max_depth() -> None:
    src = textwrap.dedent("""\
        def f(n):
            if n <= 0:
                return 0
            return f(n - 1)
        f(5)""")
    result = execute(src)
    assert result.profiling is not None
    assert result.profiling.function_stats["f"].max_recursion_depth > 0


def test_get_hottest_lines_returns_results() -> None:
    src = "for i in range(100):\n    x = i * 2"
    result = execute(src)
    assert result.profiling is not None
    hottest = sorted(
        result.profiling.line_stats.values(),
        key=lambda s: s.total_time_ms,
        reverse=True,
    )[:3]
    assert len(hottest) <= 3


def test_get_most_executed_lines_correct() -> None:
    src = "for i in range(10):\n    x = i"
    result = execute(src)
    assert result.profiling is not None
    most = sorted(
        result.profiling.line_stats.values(),
        key=lambda s: s.execution_count,
        reverse=True,
    )
    assert most[0].execution_count == 10


# ── Objective 3: Profiling overhead is acceptable ──


def test_profiling_same_output_with_and_without() -> None:
    src = "for i in range(50):\n    x = i * 2\nprint(x)"
    r_with = execute(src, enable_profiling=True)
    r_without = execute(src, enable_profiling=False)
    assert r_with.output == r_without.output


def test_profiling_overhead_acceptable() -> None:
    src = "for i in range(100):\n    x = i * 2"
    t0 = time.perf_counter()
    execute(src, enable_profiling=False)
    t_without = time.perf_counter() - t0

    t0 = time.perf_counter()
    execute(src, enable_profiling=True)
    t_with = time.perf_counter() - t0

    assert t_with < t_without * 10 + 0.5


# ── Supervisor Benchmark Programs ──


def test_benchmark_loop() -> None:
    """Supervisor: test programs with loops."""
    src = "for i in range(100):\n    x = i * 2\nprint(x)"
    result = execute(src)
    assert result.output == "198"
    assert result.errors == []
    assert result.profiling is not None
    all_counts = [s.execution_count for s in result.profiling.line_stats.values()]
    assert 100 in all_counts


def test_benchmark_recursion() -> None:
    """Supervisor: test programs with recursion."""
    src = textwrap.dedent("""\
        def fib(n):
            if n <= 1:
                return n
            return fib(n - 1) + fib(n - 2)
        print(fib(8))""")
    result = execute(src)
    assert result.output == "21"
    assert result.errors == []
    assert result.profiling is not None
    assert "fib" in result.profiling.function_stats
    assert result.profiling.function_stats["fib"].call_count > 1


def test_benchmark_nested_loops() -> None:
    """Supervisor: nested loops execution count validation."""
    src = textwrap.dedent("""\
        total = 0
        for i in range(10):
            for j in range(10):
                total += 1
        print(total)""")
    result = execute(src)
    assert result.output == "100"
    assert result.profiling is not None
    all_counts = [s.execution_count for s in result.profiling.line_stats.values()]
    assert 100 in all_counts


def test_benchmark_mixed_program() -> None:
    """Supervisor: combining functions, loops, conditionals."""
    src = textwrap.dedent("""\
        def is_even(n):
            return n % 2 == 0
        total = 0
        for i in range(10):
            if is_even(i):
                total += 1
        print(total)""")
    result = execute(src)
    assert result.output == "5"
    assert result.errors == []


# ── execute() convenience function ──


def test_execute_timeout_respected() -> None:
    result = execute("while True:\n    pass", timeout_seconds=0.1)
    assert len(result.errors) == 1


def test_execute_profiling_disabled() -> None:
    result = execute("x = 1", enable_profiling=False)
    assert result.profiling is None


def test_execute_profiling_enabled_by_default() -> None:
    result = execute("x = 1")
    assert result.profiling is not None
    assert len(result.profiling.line_stats) >= 1
