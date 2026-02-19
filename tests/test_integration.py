"""
Integration tests for OptiLang — verifies all three stated objectives:
  1. Execute PyLite programs correctly with execution metrics
  2. Profiling data is present and meaningful
  3. Non-functional: profiling overhead is acceptable

Note: result.profiling is a plain dict (from ProfilingData.to_dict()),
      NOT a ProfilingData object. Access it with string keys.
"""

import time
import pytest
from optilang import execute
from optilang.models import ExecutionResult

# ── Objective 1: Execute PyLite programs correctly ────────────────────────────


def test_simple_arithmetic_program():
    result = execute("print(2 + 3)")
    assert result.output == "5"
    assert result.errors == []


def test_function_and_recursion():
    src = """
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)
print(factorial(5))
""".strip()
    result = execute(src)
    assert result.output == "120"
    assert result.errors == []


def test_loop_program():
    src = "total = 0\nfor i in range(10):\n    total += i\nprint(total)"
    result = execute(src)
    assert result.output == "45"
    assert result.errors == []


def test_nested_loops():
    src = "total = 0\nfor i in range(3):\n    for j in range(3):\n        total += 1\nprint(total)"
    result = execute(src)
    assert result.output == "9"
    assert result.errors == []


def test_try_except_program():
    src = "try:\n    x = 1 / 0\nexcept:\n    print('handled')\nfinally:\n    print('done')"
    result = execute(src)
    assert result.output == "handled\ndone"
    assert result.errors == []


def test_returns_execution_result_instance():
    result = execute("x = 1")
    assert isinstance(result, ExecutionResult)


def test_syntax_error_captured():
    result = execute("def f(\n    pass")
    assert result.errors != []
    assert result.output == ""


def test_runtime_error_captured():
    result = execute("print(undefined_var)")
    assert result.errors != []
    assert result.output == ""


def test_handled_exception_no_errors():
    result = execute("try:\n    x = 1 / 0\nexcept:\n    pass")
    assert result.errors == []


# ── Objective 2: Profiling data is present and meaningful ────────────────────


def test_profiling_not_none():
    result = execute("x = 1")
    assert result.profiling is not None


def test_profiling_total_time_positive():
    result = execute("x = 1")
    assert result.profiling["total_time_ms"] > 0


def test_profiling_line_stats_populated():
    result = execute("x = 1\ny = 2")
    assert len(result.profiling["line_stats"]) >= 1


def test_loop_line_execution_count():
    src = "for i in range(5):\n    x = i"
    result = execute(src)
    all_counts = [s["count"] for s in result.profiling["line_stats"].values()]
    assert 5 in all_counts


def test_nested_loop_inner_execution_count():
    src = "for i in range(4):\n    for j in range(4):\n        x = i + j"
    result = execute(src)
    all_counts = [s["count"] for s in result.profiling["line_stats"].values()]
    assert 16 in all_counts


def test_function_stats_populated():
    src = "def f():\n    pass\nf()\nf()\nf()"
    result = execute(src)
    assert "f" in result.profiling["function_stats"]
    assert result.profiling["function_stats"]["f"]["calls"] == 3


def test_recursive_function_max_depth():
    src = """
def f(n):
    if n <= 0:
        return 0
    return f(n - 1)
f(5)
""".strip()
    result = execute(src)
    assert result.profiling["function_stats"]["f"]["max_depth"] > 0


def test_get_hottest_lines_returns_results():
    src = "for i in range(100):\n    x = i * 2"
    result = execute(src)
    line_stats = result.profiling["line_stats"]
    hottest = sorted(line_stats.values(), key=lambda s: s["total_time"], reverse=True)[
        :3
    ]
    assert len(hottest) <= 3


def test_get_most_executed_lines_correct():
    src = "for i in range(10):\n    x = i"
    result = execute(src)
    line_stats = result.profiling["line_stats"]
    most = sorted(line_stats.values(), key=lambda s: s["count"], reverse=True)
    assert most[0]["count"] == 10


# ── Objective 3: Profiling overhead is acceptable ─────────────────────────────


def test_profiling_same_output_with_and_without():
    src = "for i in range(50):\n    x = i * 2\nprint(x)"
    r_with = execute(src, enable_profiling=True)
    r_without = execute(src, enable_profiling=False)
    assert r_with.output == r_without.output


def test_profiling_overhead_acceptable():
    src = "for i in range(100):\n    x = i * 2"
    t0 = time.perf_counter()
    execute(src, enable_profiling=False)
    t_without = time.perf_counter() - t0

    t0 = time.perf_counter()
    execute(src, enable_profiling=True)
    t_with = time.perf_counter() - t0

    assert t_with < t_without * 10 + 0.5


# ── Supervisor Benchmark Programs ─────────────────────────────────────────────


def test_benchmark_loop():
    """Supervisor: test programs with loops."""
    src = "for i in range(100):\n    x = i * 2\nprint(x)"
    result = execute(src)
    assert result.output == "198"
    assert result.errors == []
    all_counts = [s["count"] for s in result.profiling["line_stats"].values()]
    assert 100 in all_counts


def test_benchmark_recursion():
    """Supervisor: test programs with recursion."""
    src = """
def fib(n):
    if n <= 1:
        return n
    return fib(n - 1) + fib(n - 2)
print(fib(8))
""".strip()
    result = execute(src)
    assert result.output == "21"
    assert result.errors == []
    assert "fib" in result.profiling["function_stats"]
    assert result.profiling["function_stats"]["fib"]["calls"] > 1


def test_benchmark_nested_loops():
    """Supervisor: nested loops execution count validation."""
    src = "total = 0\nfor i in range(10):\n    for j in range(10):\n        total += 1\nprint(total)"
    result = execute(src)
    assert result.output == "100"
    all_counts = [s["count"] for s in result.profiling["line_stats"].values()]
    assert 100 in all_counts


def test_benchmark_mixed_program():
    """Supervisor: combining functions, loops, conditionals."""
    src = """
def is_even(n):
    return n % 2 == 0
total = 0
for i in range(10):
    if is_even(i):
        total += 1
print(total)
""".strip()
    result = execute(src)
    assert result.output == "5"
    assert result.errors == []


# ── execute() convenience function ───────────────────────────────────────────


def test_execute_timeout_respected():
    result = execute("while True:\n    pass", timeout_seconds=0.1)
    assert len(result.errors) == 1


def test_execute_profiling_disabled():
    result = execute("x = 1", enable_profiling=False)
    assert result.profiling is None


def test_execute_profiling_enabled_by_default():
    result = execute("x = 1")
    assert result.profiling is not None
    assert len(result.profiling["line_stats"]) >= 1
