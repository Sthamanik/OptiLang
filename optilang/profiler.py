"""
OptiLang Profiler - Tracks execution metrics for optimization analysis.

This module provides line-by-line profiling during code execution, collecting:
- Execution count per line
- Time spent on each line
- Function call statistics
- Memory estimation (variable count)
"""

import time
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class LineStats:
    """Statistics for a single line of code."""

    line_number: int
    execution_count: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    memory_vars: int = 0  # Number of variables in scope

    def update_time(self, elapsed_ms: float) -> None:
        """Update timing statistics."""
        self.total_time_ms += elapsed_ms
        self.execution_count += 1
        self.avg_time_ms = self.total_time_ms / self.execution_count

    def to_dict(self) -> dict:
        """Convert to dictionary for serialization."""
        return {
            "line": self.line_number,
            "count": self.execution_count,
            "total_time": round(self.total_time_ms, 3),
            "avg_time": round(self.avg_time_ms, 3),
            "memory_vars": self.memory_vars,
        }


@dataclass
class FunctionStats:
    """Statistics for a function."""

    name: str
    call_count: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    max_recursion_depth: int = 0

    def record_call(self, elapsed_ms: float, depth: int = 0) -> None:
        """Record a function call."""
        self.call_count += 1
        self.total_time_ms += elapsed_ms
        self.avg_time_ms = self.total_time_ms / self.call_count
        self.max_recursion_depth = max(self.max_recursion_depth, depth)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "calls": self.call_count,
            "total_time": round(self.total_time_ms, 3),
            "avg_time": round(self.avg_time_ms, 3),
            "max_depth": self.max_recursion_depth,
        }


@dataclass
class ProfilingData:
    """Complete profiling data for a code execution."""

    line_stats: Dict[int, LineStats] = field(default_factory=dict)
    function_stats: Dict[str, FunctionStats] = field(default_factory=dict)
    total_execution_time_ms: float = 0.0
    total_lines_executed: int = 0
    start_time: Optional[float] = None
    end_time: Optional[float] = None

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "line_stats": {
                line: stats.to_dict() for line, stats in sorted(self.line_stats.items())
            },
            "function_stats": {
                name: stats.to_dict() for name, stats in self.function_stats.items()
            },
            "total_time_ms": round(self.total_execution_time_ms, 3),
            "total_lines": self.total_lines_executed,
            "lines_profiled": len(self.line_stats),
        }


class Profiler:
    """
    Profiler that tracks execution metrics during code interpretation.

    Usage:
        profiler = Profiler()
        profiler.start()

        # In your executor, before executing each line:
        profiler.start_line(line_number, variables_in_scope)
        # ... execute line ...
        profiler.end_line(line_number)

        profiler.stop()
        data = profiler.get_data()
    """

    def __init__(self) -> None:
        self.data = ProfilingData()
        self._current_line_start: Optional[float] = None
        self._function_call_stack: List[tuple] = []  # [(name, start_time, depth)]
        self._enabled = True

    def start(self) -> None:
        """Start profiling session."""
        self.data.start_time = time.perf_counter()

    def stop(self) -> None:
        """Stop profiling and calculate totals."""
        if self.data.start_time is not None:
            self.data.end_time = time.perf_counter()
            self.data.total_execution_time_ms = (
                self.data.end_time - self.data.start_time
            ) * 1000

        # Calculate total lines executed
        self.data.total_lines_executed = sum(
            stats.execution_count for stats in self.data.line_stats.values()
        )

    def start_line(self, line_number: int, variable_count: int = 0) -> None:
        """
        Called before executing a line.

        Args:
            line_number: The line number being executed
            variable_count: Number of variables currently in scope
        """
        if not self._enabled:
            return

        # Initialize line stats if first time
        if line_number not in self.data.line_stats:
            self.data.line_stats[line_number] = LineStats(line_number)

        # Update memory info
        self.data.line_stats[line_number].memory_vars = variable_count

        # Start timing
        self._current_line_start = time.perf_counter()

    def end_line(self, line_number: int) -> None:
        """
        Called after executing a line.

        Args:
            line_number: The line number that was executed
        """
        if not self._enabled or self._current_line_start is None:
            return

        # Calculate elapsed time
        elapsed = (time.perf_counter() - self._current_line_start) * 1000  # ms

        # Update stats
        if line_number in self.data.line_stats:
            self.data.line_stats[line_number].update_time(elapsed)

        self._current_line_start = None

    def start_function_call(self, function_name: str) -> None:
        """
        Called when entering a function.

        Args:
            function_name: Name of the function being called
        """
        if not self._enabled:
            return

        depth = len(self._function_call_stack)
        start_time = time.perf_counter()
        self._function_call_stack.append((function_name, start_time, depth))

        # Initialize function stats if needed
        if function_name not in self.data.function_stats:
            self.data.function_stats[function_name] = FunctionStats(function_name)

    def end_function_call(self, function_name: str) -> None:
        """
        Called when exiting a function.

        Args:
            function_name: Name of the function returning
        """
        if not self._enabled or not self._function_call_stack:
            return

        # Pop from stack
        name, start_time, depth = self._function_call_stack.pop()

        # Verify it matches (simple check)
        if name != function_name:
            # Warning: mismatched function exit
            pass

        # Calculate time
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Record stats
        if function_name in self.data.function_stats:
            self.data.function_stats[function_name].record_call(elapsed_ms, depth)

    def get_data(self) -> ProfilingData:
        """Get the profiling data."""
        return self.data 

    def reset(self) -> None:
        """Reset all profiling data."""
        self.data = ProfilingData()
        self._current_line_start = None
        self._function_call_stack = []

    def enable(self) -> None:
        """Enable profiling."""
        self._enabled = True

    def disable(self) -> None:
        """Disable profiling (for performance)."""
        self._enabled = False

    def get_hottest_lines(self, top_n: int = 10) -> List[LineStats]:
        """
        Get the lines that took the most time.

        Args:
            top_n: Number of top lines to return

        Returns:
            List of LineStats sorted by total time (descending)
        """
        sorted_lines = sorted(
            self.data.line_stats.values(), key=lambda x: x.total_time_ms, reverse=True
        )
        return sorted_lines[:top_n]

    def get_most_executed_lines(self, top_n: int = 10) -> List[LineStats]:
        """
        Get the lines that were executed most often.

        Args:
            top_n: Number of top lines to return

        Returns:
            List of LineStats sorted by execution count (descending)
        """
        sorted_lines = sorted(
            self.data.line_stats.values(), key=lambda x: x.execution_count, reverse=True
        )
        return sorted_lines[:top_n]


# Convenience function for simple use cases
def profile_execution(executor_func, code: str, *args, **kwargs):
    """
    Convenience wrapper to profile a code execution.

    Args:
        executor_func: The executor function to profile
        code: The code to execute
        *args, **kwargs: Additional arguments for executor

    Returns:
        tuple: (execution_result, profiling_data)
    """
    profiler = Profiler()
    profiler.start()

    # You'll need to modify your executor to accept a profiler
    result = executor_func(code, profiler=profiler, *args, **kwargs)

    profiler.stop()
    return result, profiler.get_data()


if __name__ == "__main__":
    # Example usage (for testing)
    profiler = Profiler()
    profiler.start()

    # Simulate executing lines
    for i in range(1, 6):
        profiler.start_line(i, variable_count=2)
        time.sleep(0.001)  # Simulate work
        profiler.end_line(i)

    # Simulate function call
    profiler.start_function_call("my_function")
    profiler.start_line(10, variable_count=3)
    time.sleep(0.002)
    profiler.end_line(10)
    profiler.end_function_call("my_function")

    profiler.stop()

    # Print results
    data = profiler.get_data()
    print("Profiling Results:")
    print(f"Total time: {data.total_execution_time_ms:.3f}ms")
    print(f"Lines executed: {data.total_lines_executed}")
    print("\nLine stats:")
    for line, stats in sorted(data.line_stats.items()):
        print(
            f"  Line {line}: {stats.execution_count} executions, "
            f"{stats.total_time_ms:.3f}ms total"
        )
    print("\nFunction stats:")
    for name, stats in data.function_stats.items():
        print(
            f"  {name}: {stats.call_count} calls, " f"{stats.total_time_ms:.3f}ms total"
        )
