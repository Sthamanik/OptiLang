"""
OptiLang Profiler - Tracks execution metrics for optimization analysis.

This module provides line-by-line profiling during code execution, collecting:
- Execution count per line
- Time spent on each line (total, average, min, max)
- Function call statistics (with caller tracking and recursion depth)
- Memory estimation (variable count + byte-level size estimation)
- Complexity detection (O(1), O(n), O(n^2), etc.)
- High-level summary for web API consumption
"""

import json
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class LineStats:
    """Statistics for a single line of code."""

    line_number: int
    execution_count: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    min_time_ms: float = float("inf")  # fastest single execution
    max_time_ms: float = 0.0  # slowest single execution
    memory_vars: int = 0  # number of variables in scope
    memory_bytes: int = 0  # estimated memory usage in bytes

    def update_time(self, elapsed_ms: float) -> None:
        """Update timing statistics after a line executes."""
        self.total_time_ms += elapsed_ms
        self.execution_count += 1
        self.avg_time_ms = self.total_time_ms / self.execution_count

        if elapsed_ms < self.min_time_ms:
            self.min_time_ms = elapsed_ms
        if elapsed_ms > self.max_time_ms:
            self.max_time_ms = elapsed_ms

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        safe_min = (
            round(self.min_time_ms, 3) if self.min_time_ms != float("inf") else 0.0
        )
        return {
            "line": self.line_number,
            "count": self.execution_count,
            "total_time_ms": round(self.total_time_ms, 3),
            "avg_time_ms": round(self.avg_time_ms, 3),
            "min_time_ms": safe_min,
            "max_time_ms": round(self.max_time_ms, 3),
            "memory_vars": self.memory_vars,
            "memory_bytes": self.memory_bytes,
        }


@dataclass
class FunctionStats:
    """Statistics for a single user-defined function."""

    name: str
    call_count: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    min_time_ms: float = float("inf")  # fastest single call
    max_time_ms: float = 0.0  # slowest single call
    max_recursion_depth: int = 0
    callers: Dict[str, int] = field(default_factory=dict)  # who called this

    def record_call(
        self,
        elapsed_ms: float,
        depth: int = 0,
        caller: Optional[str] = None,
    ) -> None:
        """Record a completed function call."""
        self.call_count += 1
        self.total_time_ms += elapsed_ms
        self.avg_time_ms = self.total_time_ms / self.call_count
        self.max_recursion_depth = max(self.max_recursion_depth, depth)

        if elapsed_ms < self.min_time_ms:
            self.min_time_ms = elapsed_ms
        if elapsed_ms > self.max_time_ms:
            self.max_time_ms = elapsed_ms

        if caller:
            self.callers[caller] = self.callers.get(caller, 0) + 1

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        safe_min = (
            round(self.min_time_ms, 3) if self.min_time_ms != float("inf") else 0.0
        )
        return {
            "name": self.name,
            "calls": self.call_count,
            "total_time_ms": round(self.total_time_ms, 3),
            "avg_time_ms": round(self.avg_time_ms, 3),
            "min_time_ms": safe_min,
            "max_time_ms": round(self.max_time_ms, 3),
            "max_recursion_depth": self.max_recursion_depth,
            "callers": self.callers,
        }


@dataclass
class ProfilingData:
    """Complete profiling data for a single code execution session."""

    line_stats: Dict[int, LineStats] = field(default_factory=dict)
    function_stats: Dict[str, FunctionStats] = field(default_factory=dict)
    total_execution_time_ms: float = 0.0
    total_lines_executed: int = 0
    peak_memory_bytes: int = 0  # highest memory observed at any point
    complexity_estimate: str = "O(1)"  # detected time complexity class
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "line_stats": {
                line: stats.to_dict() for line, stats in sorted(self.line_stats.items())
            },
            "function_stats": {
                fname: fstats.to_dict() for fname, fstats in self.function_stats.items()
            },
            "total_time_ms": round(self.total_execution_time_ms, 3),
            "total_lines_executed": self.total_lines_executed,
            "lines_profiled": len(self.line_stats),
            "peak_memory_bytes": self.peak_memory_bytes,
            "complexity_estimate": self.complexity_estimate,
        }


def estimate_memory_bytes(env_values: Dict[str, Any]) -> int:
    """
    Estimate the total memory used by variables currently in scope.

    Uses sys.getsizeof for each variable. For containers (list, dict),
    it also accounts for the size of contained elements up to one level
    deep to avoid slow deep recursion on large nested structures.

    Args:
        env_values: Dictionary of variable name to value from the environment

    Returns:
        Estimated total memory in bytes
    """
    total = 0
    for value in env_values.values():
        try:
            total += sys.getsizeof(value)

            if isinstance(value, list):
                for item in value:
                    try:
                        total += sys.getsizeof(item)
                    except (TypeError, ValueError):
                        total += 28  # fallback for unknown types

            elif isinstance(value, dict):
                for k, v in value.items():
                    try:
                        total += sys.getsizeof(k) + sys.getsizeof(v)
                    except (TypeError, ValueError):
                        total += 56  # fallback

        except (TypeError, ValueError):
            total += 28  # fallback: approximate size of a small Python object

    return total


def detect_complexity(line_stats: Dict[int, LineStats]) -> str:
    """
    Estimate the time complexity class of the executed program.

    This works by examining the maximum line execution count relative to
    the number of unique lines profiled. It is a heuristic approach, not
    a formal proof, but provides a useful approximation for the
    optimization scorer and web interface.

    Complexity classes returned:
        O(1)         - Every line ran at most once
        O(log n)     - Max count 2-15, suggests binary-search style
        O(n)         - Max count up to 1,000
        O(n log n)   - Max count suggests a sorting-style pattern
        O(n^2)       - Max count suggests nested loops
        O(n^3)+      - Very high execution counts

    Args:
        line_stats: Dictionary of line number to LineStats

    Returns:
        A string representing the estimated complexity class
    """
    if not line_stats:
        return "O(1)"

    counts = [s.execution_count for s in line_stats.values()]
    max_count = max(counts)
    unique_lines = len(line_stats)

    if max_count <= 1:
        return "O(1)"

    if max_count <= 15:
        return "O(log n)"

    if max_count <= 1_000:
        return "O(n)"

    if max_count <= 10_000:
        ratio = max_count / max(unique_lines, 1)
        if ratio < 200:
            return "O(n log n)"
        return "O(n^2)"

    if max_count <= 1_000_000:
        return "O(n^2)"

    return "O(n^3) or worse"


class Profiler:
    """
    Profiler that tracks execution metrics during code interpretation.

    The profiler is designed to be lightweight and non-intrusive. It hooks
    into the executor via start_line/end_line and start_function_call/
    end_function_call calls placed around every statement and function body.

    Usage::

        profiler = Profiler()
        profiler.start()

        profiler.start_line(line_number, env_values)
        # ... execute statement ...
        profiler.end_line(line_number)

        profiler.start_function_call("my_func", caller="parent_func")
        # ... execute function body ...
        profiler.end_function_call("my_func")

        profiler.stop()
        data = profiler.get_data()
        summary = profiler.get_summary()
    """

    def __init__(self) -> None:
        self.data = ProfilingData()
        self._current_line_start: Optional[float] = None
        # Each entry: (func_name, start_time, depth, caller)
        self._function_call_stack: List[Tuple[str, float, int, Optional[str]]] = []
        self._enabled = True

    # ── Session Control ──

    def start(self) -> None:
        """Begin a profiling session. Call this before any code executes."""
        self.data.start_time = time.perf_counter()

    def stop(self) -> None:
        """
        End the profiling session and compute final aggregates.

        Calculates total execution time, total lines executed, and the
        time complexity estimate.
        """
        if self.data.start_time is not None:
            self.data.end_time = time.perf_counter()
            self.data.total_execution_time_ms = (
                self.data.end_time - self.data.start_time
            ) * 1000

        self.data.total_lines_executed = sum(
            s.execution_count for s in self.data.line_stats.values()
        )
        self.data.complexity_estimate = detect_complexity(self.data.line_stats)

    # ── Line Profiling ──

    def start_line(
        self,
        line_number: int,
        env_values: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Called immediately before a statement on a given line executes.

        Args:
            line_number: The source line number about to execute
            env_values:  The current environment's variable dictionary
                         used for memory estimation. Pass None to skip.
        """
        if not self._enabled:
            return

        if line_number not in self.data.line_stats:
            self.data.line_stats[line_number] = LineStats(line_number)

        if env_values is not None:
            var_count = len(env_values)
            mem_bytes = estimate_memory_bytes(env_values)
        else:
            var_count = 0
            mem_bytes = 0

        line_stat = self.data.line_stats[line_number]
        line_stat.memory_vars = var_count
        line_stat.memory_bytes = mem_bytes

        if mem_bytes > self.data.peak_memory_bytes:
            self.data.peak_memory_bytes = mem_bytes

        self._current_line_start = time.perf_counter()

    def end_line(self, line_number: int) -> None:
        """
        Called immediately after a statement on a given line finishes.

        Args:
            line_number: The source line number that just finished executing
        """
        if not self._enabled or self._current_line_start is None:
            return

        elapsed_ms = (time.perf_counter() - self._current_line_start) * 1000

        if line_number in self.data.line_stats:
            self.data.line_stats[line_number].update_time(elapsed_ms)

        self._current_line_start = None

    # ── Function Profiling ──

    def start_function_call(
        self,
        function_name: str,
        caller: Optional[str] = None,
    ) -> None:
        """
        Called when execution enters a user-defined function.

        Args:
            function_name: Name of the function being entered
            caller:        Name of the calling function, if any
        """
        if not self._enabled:
            return

        depth = len(self._function_call_stack)
        start_time = time.perf_counter()
        self._function_call_stack.append((function_name, start_time, depth, caller))

        if function_name not in self.data.function_stats:
            self.data.function_stats[function_name] = FunctionStats(function_name)

    def end_function_call(self, function_name: str) -> None:
        """
        Called when execution exits a user-defined function.

        Args:
            function_name: Name of the function being exited. Used as a
                           fallback key when the call stack is empty.
        """
        if not self._enabled or not self._function_call_stack:
            return

        fname, start_time, depth, caller = self._function_call_stack.pop()
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Prefer the name recorded on entry; fall back to the argument
        # so the parameter is always referenced (avoids unused-argument).
        resolved = fname if fname else function_name
        if resolved in self.data.function_stats:
            self.data.function_stats[resolved].record_call(
                elapsed_ms, depth=depth, caller=caller
            )

    # ── Query Methods ──

    def get_data(self) -> ProfilingData:
        """Return the full profiling data object."""
        return self.data

    def get_call_stack(self) -> List[str]:
        """
        Return the names of functions currently on the call stack.

        The list is ordered outermost to innermost, so the last element
        is the currently-executing function. This is the public alternative
        to accessing ``_function_call_stack`` directly.

        Returns:
            List of function name strings, empty when no function is active.
        """
        return [entry[0] for entry in self._function_call_stack]

    def get_hottest_lines(self, top_n: int = 10) -> List[LineStats]:
        """
        Return the lines that consumed the most total execution time.

        Args:
            top_n: How many lines to return

        Returns:
            List of LineStats sorted by total_time_ms descending
        """
        return sorted(
            self.data.line_stats.values(),
            key=lambda x: x.total_time_ms,
            reverse=True,
        )[:top_n]

    def get_most_executed_lines(self, top_n: int = 10) -> List[LineStats]:
        """
        Return the lines that were executed most often.

        Args:
            top_n: How many lines to return

        Returns:
            List of LineStats sorted by execution_count descending
        """
        return sorted(
            self.data.line_stats.values(),
            key=lambda x: x.execution_count,
            reverse=True,
        )[:top_n]

    def get_hottest_functions(self, top_n: int = 5) -> List[FunctionStats]:
        """
        Return the functions that consumed the most total execution time.

        Args:
            top_n: How many functions to return

        Returns:
            List of FunctionStats sorted by total_time_ms descending
        """
        return sorted(
            self.data.function_stats.values(),
            key=lambda x: x.total_time_ms,
            reverse=True,
        )[:top_n]

    def get_most_called_functions(self, top_n: int = 5) -> List[FunctionStats]:
        """
        Return the functions that were called most often.

        Args:
            top_n: How many functions to return

        Returns:
            List of FunctionStats sorted by call_count descending
        """
        return sorted(
            self.data.function_stats.values(),
            key=lambda x: x.call_count,
            reverse=True,
        )[:top_n]

    def get_summary(self) -> Dict[str, Any]:
        """
        Return a concise high-level summary of the profiling session.

        This is the primary data structure consumed by the FastAPI
        interpreter service and the web front-end dashboard.

        Returns:
            Dictionary with key metrics ready for JSON serialization
        """
        hottest_lines = self.get_hottest_lines(3)
        most_executed = self.get_most_executed_lines(3)
        hottest_funcs = self.get_hottest_functions(3)

        return {
            "total_time_ms": round(self.data.total_execution_time_ms, 3),
            "total_lines_executed": self.data.total_lines_executed,
            "unique_lines_profiled": len(self.data.line_stats),
            "peak_memory_bytes": self.data.peak_memory_bytes,
            "peak_memory_kb": round(self.data.peak_memory_bytes / 1024, 2),
            "complexity_estimate": self.data.complexity_estimate,
            "functions_called": len(self.data.function_stats),
            "total_function_calls": sum(
                f.call_count for f in self.data.function_stats.values()
            ),
            "hottest_lines": [s.to_dict() for s in hottest_lines],
            "most_executed_lines": [s.to_dict() for s in most_executed],
            "hottest_functions": [f.to_dict() for f in hottest_funcs],
        }

    # ── Control ──

    def reset(self) -> None:
        """Reset all profiling data for a fresh session."""
        self.data = ProfilingData()
        self._current_line_start = None
        self._function_call_stack = []

    def enable(self) -> None:
        """Enable profiling (on by default)."""
        self._enabled = True

    def disable(self) -> None:
        """
        Disable profiling entirely.

        Use this when you want to run code without any measurement overhead,
        for example during warm-up runs before benchmarking.
        """
        self._enabled = False


def profile_execution(
    executor_func: Callable[..., Any],
    code: str,
    *args: Any,
    **kwargs: Any,
) -> Tuple[Any, ProfilingData]:
    """
    Convenience wrapper to profile a code execution function.

    Args:
        executor_func: The executor function to call
        code:          PyLite source code string
        *args:         Additional positional arguments for executor_func
        **kwargs:      Additional keyword arguments for executor_func

    Returns:
        Tuple of (execution_result, ProfilingData)
    """
    _profiler = Profiler()
    _profiler.start()
    result = executor_func(code, profiler=_profiler, *args, **kwargs)
    _profiler.stop()
    return result, _profiler.get_data()


if __name__ == "__main__":
    _demo_profiler = Profiler()
    _demo_profiler.start()

    _fake_env: Dict[str, Any] = {
        "i": 0,
        "total": 0,
        "label": "optilang",
        "items": [1, 2, 3],
    }

    for _i in range(10):
        _demo_profiler.start_line(1, _fake_env)
        time.sleep(0.001)
        _demo_profiler.end_line(1)

        _demo_profiler.start_line(2, _fake_env)
        time.sleep(0.0005)
        _demo_profiler.end_line(2)

    _demo_profiler.start_function_call("compute", caller=None)
    _demo_profiler.start_line(5, {"result": 42})
    time.sleep(0.002)
    _demo_profiler.end_line(5)
    _demo_profiler.end_function_call("compute")

    _demo_profiler.start_function_call("factorial", caller=None)
    _demo_profiler.start_function_call("factorial", caller="factorial")
    _demo_profiler.end_function_call("factorial")
    _demo_profiler.end_function_call("factorial")

    _demo_profiler.stop()

    _data = _demo_profiler.get_data()
    print("=" * 50)
    print("PROFILING RESULTS")
    print("=" * 50)
    print(f"Total time     : {_data.total_execution_time_ms:.3f} ms")
    print(f"Lines executed : {_data.total_lines_executed}")
    print(f"Peak memory    : {_data.peak_memory_bytes} bytes")
    print(f"Complexity     : {_data.complexity_estimate}")
    print()

    print("LINE STATS:")
    for _line, _ls in sorted(_data.line_stats.items()):
        print(
            f"  Line {_line}: {_ls.execution_count}x | "
            f"total={_ls.total_time_ms:.3f}ms | "
            f"avg={_ls.avg_time_ms:.3f}ms | "
            f"min={_ls.min_time_ms:.3f}ms | "
            f"max={_ls.max_time_ms:.3f}ms | "
            f"mem={_ls.memory_bytes}B"
        )

    print()
    print("FUNCTION STATS:")
    for _fn, _fs in _data.function_stats.items():
        print(
            f"  {_fn}: {_fs.call_count} calls | "
            f"total={_fs.total_time_ms:.3f}ms | "
            f"max_depth={_fs.max_recursion_depth} | "
            f"callers={_fs.callers}"
        )

    print()
    print("SUMMARY:")
    print(json.dumps(_demo_profiler.get_summary(), indent=2))
