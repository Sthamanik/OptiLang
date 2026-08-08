"""
OptiLang Profiler - Tracks execution metrics for optimization analysis.

This module provides line-by-line profiling during code execution, collecting:
- Execution count per line
- Time spent on each line (total, average, min, max)
- Function call statistics (with caller tracking and recursion depth)
- Memory estimation (variable count + byte-level size estimation)
- Complexity detection (O(1), O(n), O(n²), etc.)
- High-level summary for web API consumption
"""

import math
import random
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, cast

__all__ = [
    "Profiler",
    "ProfilerConfig",
    "ProfilingData",
    "FunctionStats",
    "LineStats",
    "detect_complexity",
    "detect_complexity_with_confidence",
    "estimate_memory_bytes",
    "profile_execution",
    # Internal helpers for tests
    "_estimate_deep_object_size",
    "_safe_getsizeof",
]

from ..core.ast_nodes import (
    ASTNode,
    AssignmentNode,
    AugmentedAssignmentNode,
    BinaryOpNode,
    ForNode,
    IdentifierNode,
    ProgramNode,
    WhileNode,
)
from ..types.constants import (
    COMPLEXITY_EXP,
    COMPLEXITY_LOGN,
    COMPLEXITY_N,
    COMPLEXITY_N2,
    COMPLEXITY_N2LOGN,
    COMPLEXITY_N3,
    COMPLEXITY_N4,
    COMPLEXITY_NK,
    COMPLEXITY_NLOGN,
    COMPLEXITY_O1,
    COMPLEXITY_UNKNOWN,
)


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
    complexity_method: str = "heuristic"
    complexity_confidence: float = 1.0
    complexity_display: Optional[str] = None
    complexity_worst_case: Optional[str] = None
    complexity_best_case: Optional[str] = None
    complexity_bound_symbols: List[str] = field(default_factory=list)
    complexity_has_early_exit: bool = False
    complexity_fallback_reason: Optional[str] = None
    sampled_lines: int = 0
    skipped_lines: int = 0
    line_sampling_rate: float = 1.0
    memory_mode: str = "shallow"
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
            "complexity_method": self.complexity_method,
            "complexity_confidence": round(self.complexity_confidence, 3),
            "complexity_display": self.complexity_display or self.complexity_estimate,
            "complexity_worst_case": self.complexity_worst_case
            or self.complexity_estimate,
            "complexity_best_case": self.complexity_best_case,
            "complexity_bound_symbols": self.complexity_bound_symbols,
            "complexity_has_early_exit": self.complexity_has_early_exit,
            "complexity_fallback_reason": self.complexity_fallback_reason,
            "sampled_lines": self.sampled_lines,
            "skipped_lines": self.skipped_lines,
            "line_sampling_rate": self.line_sampling_rate,
            "memory_mode": self.memory_mode,
        }


@dataclass
class ProfilerConfig:
    """Runtime configuration for profiling overhead/precision tradeoffs."""

    memory_mode: str = "shallow"  # "off" | "shallow" | "deep"
    deep_max_depth: int = 3
    deep_max_items: int = 500
    line_sampling_rate: float = 1.0
    random_seed: Optional[int] = None

    def normalized_memory_mode(self) -> str:
        """Return a safe memory mode; defaults to shallow for invalid values."""
        mode = self.memory_mode.strip().lower()
        if mode in {"off", "shallow", "deep"}:
            return mode
        return "shallow"

    def normalized_sampling_rate(self) -> float:
        """Clamp sampling rate to [0.0, 1.0]."""
        return min(1.0, max(0.0, self.line_sampling_rate))

    def normalized_deep_max_depth(self) -> int:
        """Ensure deep profiling depth is a non-negative integer."""
        return max(0, int(self.deep_max_depth))

    def normalized_deep_max_items(self) -> int:
        """Ensure deep profiling item budget is non-negative."""
        return max(0, int(self.deep_max_items))


def _safe_getsizeof(value: Any) -> int:
    """Best-effort object size lookup with fallback."""
    try:
        return sys.getsizeof(value)
    except (TypeError, ValueError):
        return 28


def _estimate_memory_shallow(env_values: Dict[str, Any]) -> int:
    """Estimate memory in a shallow way (one level into list/dict)."""
    total = 0
    for value in env_values.values():
        total += _safe_getsizeof(value)

        if isinstance(value, list):
            for item in value:
                total += _safe_getsizeof(item)
        elif isinstance(value, dict):
            for key, item in value.items():
                total += _safe_getsizeof(key) + _safe_getsizeof(item)

    return total


def _estimate_deep_object_size(
    value: Any,
    max_depth: int,
    max_items: int,
) -> int:
    """Estimate object size recursively with cycle and budget protection."""
    seen: Set[int] = set()
    remaining = max_items

    def walk(current: Any, depth: int) -> int:
        nonlocal remaining

        obj_id = id(current)
        if obj_id in seen:
            return 0
        seen.add(obj_id)

        total_size = _safe_getsizeof(current)
        if depth >= max_depth or remaining <= 0:
            return total_size

        if isinstance(current, dict):
            for key, item in current.items():
                if remaining <= 0:
                    break
                remaining -= 1
                total_size += walk(key, depth + 1)
                if remaining <= 0:
                    break
                remaining -= 1
                total_size += walk(item, depth + 1)

        elif isinstance(current, (list, tuple, set, frozenset)):
            for item in current:
                if remaining <= 0:
                    break
                remaining -= 1
                total_size += walk(item, depth + 1)

        elif hasattr(current, "__dict__"):
            if remaining > 0:
                remaining -= 1
                total_size += walk(vars(current), depth + 1)

        return total_size

    return walk(value, depth=0)


def estimate_memory_bytes(
    env_values: Dict[str, Any],
    mode: str = "shallow",
    deep_max_depth: int = 3,
    deep_max_items: int = 500,
) -> int:
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
    normalized_mode = mode.strip().lower()
    if normalized_mode == "off":
        return 0
    if normalized_mode == "deep":
        max_depth = max(0, int(deep_max_depth))
        max_items = max(0, int(deep_max_items))
        return sum(
            _estimate_deep_object_size(value, max_depth=max_depth, max_items=max_items)
            for value in env_values.values()
        )
    return _estimate_memory_shallow(env_values)


def _analyze_execution_pattern(
    line_stats: Dict[int, LineStats],
) -> Dict[str, Any]:
    """
    Analyze execution count patterns to infer complexity class.

    Uses ratio-based analysis which is independent of actual input size n.
    Returns a dict with analysis results for complexity detection.
    """
    if not line_stats:
        return {
            "pattern": "empty",
            "max_ratio": 1.0,
            "second_ratio": 1.0,
            "hot_line_count": 0,
            "unique_hot_lines": 0,
            "max_count": 0,
            "total_lines": 0,
        }

    active_stats = [s for s in line_stats.values() if s.execution_count > 0]
    if not active_stats:
        return {
            "pattern": "empty",
            "max_ratio": 1.0,
            "second_ratio": 1.0,
            "hot_line_count": 0,
            "unique_hot_lines": 0,
            "max_count": 0,
            "total_lines": 0,
        }

    counts = sorted([s.execution_count for s in active_stats])
    max_count = counts[-1]
    total_lines = len(counts)

    # Calculate ratios between execution counts
    second_max = counts[-2] if len(counts) > 1 else 1
    third_max = counts[-3] if len(counts) > 2 else 1

    max_ratio = max_count / second_max if second_max > 0 else float(max_count)
    second_ratio = second_max / third_max if third_max > 0 else 1.0

    # Count hot lines (lines executing at >= 50% of max)
    hot_threshold = max_count * 0.5
    hot_counts = [c for c in counts if c >= hot_threshold]
    hot_line_count = len(hot_counts)

    # Count lines executing at >= 90% of max (truly hot lines)
    very_hot_threshold = max_count * 0.9
    unique_hot_lines = sum(1 for c in counts if c >= very_hot_threshold)

    # Determine pattern based on ratios
    if max_count <= 1:
        pattern = "constant"
    elif max_ratio <= 2.0 and hot_line_count >= max(3, total_lines * 0.5):
        # Many similar hot lines (ratio ≈ 1) → O(n log n) pattern
        pattern = "nlogn"
    elif max_ratio <= 2.0 and hot_line_count <= 1 and max_count >= 8:
        # One hot line, low max_ratio → O(log n) ONLY if max_count >= 8
        # [1, 2, 3] → linear (max_count=3 is too small)
        # [1, 8, 3] → log (max_count=8, ratio=2.67... wait, max_ratio=8/3=2.67 > 2.0)
        # Let me recalculate: counts=[1,2,3], max_ratio=3/2=1.5, hot_line_count=1
        # This matches the existing "log" branch, so we need max_count >= 8
        pattern = "log"
    elif (
        max_ratio <= 2.5
        and hot_line_count == 2
        and unique_hot_lines == 2
        and max_count >= 5
    ):
        # Two equal hot lines = check for quadratic vs linear
        # [1, 100, 10000, 10000, 1] → has n as intermediate
        #   (100 = sqrt(10000)) → quadratic
        # [1, 100, 100, 1] → no intermediate between 1 and max → linear
        hot_line_ratio = hot_line_count / total_lines if total_lines > 0 else 0
        # Check for sqrt(n) intermediate (nested loop signal)
        intermediate_counts = [c for c in counts if 1 < c < max_count]
        has_sqrt_intermediate = any(
            abs(c - math.sqrt(max_count)) <= max(2, c * 0.05)
            for c in intermediate_counts
        )
        # Quadratic if: hot lines dominate OR sqrt(n) intermediate exists
        if hot_line_ratio >= 0.5 or (has_sqrt_intermediate and max_count >= 100):
            pattern = "quadratic"
        else:
            pattern = "linear"
    elif (
        max_ratio > 2.0
        and hot_line_count <= 2
        and unique_hot_lines <= 1
        and max_count <= 12
        and max_count >= 8
    ):
        # One dominant hot line with small max_count → O(log n)
        # e.g., [1, 8, 3] → one hot line with count 8, rest much smaller (8 = 2^3)
        # [1, 2, 3] → linear (max_count = 3 is too small for log, could be n=3)
        pattern = "log"
    elif max_ratio > 2.0 and hot_line_count >= 2:
        # Multiple hot lines with ratio > 2 suggests quadratic
        pattern = "quadratic"
    elif (
        max_ratio <= 2.0
        and hot_line_count >= 2
        and unique_hot_lines >= 2
        and max_count > 20
    ):
        # Multiple hot lines with equal counts AND large max_count = quadratic signal
        # Small max_count (<= 20) with equal hot lines is likely just linear
        pattern = "quadratic"
    else:
        pattern = "linear"

    return {
        "pattern": pattern,
        "max_ratio": max_ratio,
        "second_ratio": second_ratio,
        "hot_line_count": hot_line_count,
        "unique_hot_lines": unique_hot_lines,
        "max_count": max_count,
        "total_lines": total_lines,
        "unique_lines": total_lines,  # Alias for clarity
    }


def _get_max_loop_depth_from_ast(
    node: Optional["ASTNode"], current_depth: int = 0
) -> int:
    """
    Recursively compute the maximum loop nesting depth in an AST.

    Returns the deepest nesting of ForNode / WhileNode nodes.
    Non-loop nodes are transparent (pass depth unchanged).
    """
    if node is None:
        return current_depth

    max_depth = current_depth

    if isinstance(node, (ForNode, WhileNode)):
        # Recurse into body with increased depth
        body_max = current_depth
        for child in getattr(node, "body", []):
            body_max = max(
                body_max, _get_max_loop_depth_from_ast(child, current_depth + 1)
            )
        max_depth = max(max_depth, body_max)
    else:
        # Recurse into all children (transparent pass-through)
        for child in vars(node).values():
            if isinstance(child, ASTNode):
                max_depth = max(
                    max_depth, _get_max_loop_depth_from_ast(child, current_depth)
                )
            elif isinstance(child, list):
                for item in child:
                    if isinstance(item, ASTNode):
                        max_depth = max(
                            max_depth, _get_max_loop_depth_from_ast(item, current_depth)
                        )
            elif isinstance(child, tuple):
                for elem in child:
                    if isinstance(elem, ASTNode):
                        max_depth = max(
                            max_depth, _get_max_loop_depth_from_ast(elem, current_depth)
                        )

    return max_depth


def _count_loop_iterations_at_depth(
    line_stats: Dict[int, LineStats],
    ast: "ASTNode",
    current_depth: int = 0,
) -> Dict[int, int]:
    """
    Walk the AST and record the maximum execution count seen at each loop depth.

    Returns {depth: max_execution_count_seen_at_that_depth}.
    depth 0 = top-level, depth 1 = first loop body, etc.
    """
    depth_counts: Dict[int, int] = {}

    def walk(node: Optional["ASTNode"], depth: int) -> None:
        if node is None:
            return

        if isinstance(node, (ForNode, WhileNode)):
            # Record execution count for this loop's body at incremented depth
            line_count = line_stats.get(node.line)
            if line_count and line_count.execution_count > 0:
                depth_counts[depth + 1] = max(
                    depth_counts.get(depth + 1, 0), line_count.execution_count
                )
            # Walk body at incremented depth
            for child in getattr(node, "body", []):
                walk(child, depth + 1)
        else:
            # Record line count at current depth
            line_count = line_stats.get(node.line)
            if line_count and line_count.execution_count > 0:
                depth_counts[depth] = max(
                    depth_counts.get(depth, 0), line_count.execution_count
                )
            # Transparent: pass depth unchanged
            for child in vars(node).values():
                if isinstance(child, ASTNode):
                    walk(child, depth)
                elif isinstance(child, list):
                    for item in child:
                        if isinstance(item, ASTNode):
                            walk(item, depth)

    walk(ast, 0)
    return depth_counts


def _infer_complexity_from_depth_counts(
    depth_counts: Dict[int, int],
    max_loop_depth: int,
) -> Tuple[str, float]:
    """
    Infer complexity from depth-to-execution-count mapping.

    Uses ratios between depths to determine algorithmic complexity.
    """
    if not depth_counts:
        return COMPLEXITY_O1, 0.95

    max_active_depth = max(depth_counts.keys()) if depth_counts else 0
    max_count = max(depth_counts.values()) if depth_counts else 0

    # No loops executed
    if max_active_depth == 0 or max_count <= 1:
        return COMPLEXITY_O1, 0.95

    # Nested loops: check ratios
    if max_active_depth >= 3:
        return COMPLEXITY_NK, 0.85
    elif max_active_depth == 2:
        d1 = depth_counts.get(1, 1)
        d2 = depth_counts.get(2, 1)
        if d2 >= d1 * 2:
            return COMPLEXITY_N2, 0.88
        elif d2 >= d1:
            return COMPLEXITY_N2, 0.80
        else:
            return COMPLEXITY_N, 0.75
    elif max_active_depth == 1:
        # Single loop: check for constant inner (linear) vs logarithmic
        d0 = depth_counts.get(0, 1)
        d1 = depth_counts.get(1, 1)
        if d1 <= d0 * 10 and max_count <= 20:
            # Small count, single loop - could be log n or linear
            return COMPLEXITY_N, 0.70
        else:
            return COMPLEXITY_N, 0.75

    return COMPLEXITY_N, 0.70


def _detect_recursion_and_complexity(
    ast: Optional["ASTNode"],
    line_stats: Dict[int, "LineStats"],
) -> Tuple[Optional[str], float]:
    """
    Detect recursion and calculate complexity based on accumulated work.

    Returns:
        Tuple of (complexity, confidence) or (None, 0) if no recursion detected
    """
    if ast is None:
        return None, 0

    from ..core.ast_nodes import FunctionDefNode, FunctionCallNode

    # Find all function definitions
    functions: Dict[str, "FunctionDefNode"] = {}

    def find_functions(node: "ASTNode") -> None:
        if isinstance(node, FunctionDefNode):
            # Get the name attribute which is an IdentifierNode
            name_node = getattr(node, "name", None)
            if name_node is not None:
                func_name = getattr(name_node, "name", "")
                if func_name:
                    functions[func_name] = node
        for child in vars(node).values():
            if isinstance(child, ASTNode):
                find_functions(child)
            elif isinstance(child, list):
                for item in child:
                    if isinstance(item, ASTNode):
                        find_functions(item)

    find_functions(ast)

    if not functions:
        return None, 0

    # Check for self-recursion in each function
    for func_name, func_def in functions.items():
        body = getattr(func_def, "body", [])
        call_count = 0  # Count of recursive calls to this function

        def find_calls(node: "ASTNode") -> None:
            nonlocal call_count
            if isinstance(node, FunctionCallNode):
                # Get function name - it might be in 'function'
                # attribute as IdentifierNode
                func_ident = getattr(node, "function", None)
                if func_ident is not None:
                    call_name = getattr(func_ident, "name", "")
                    if call_name == func_name:
                        call_count += 1
            for child in vars(node).values():
                if isinstance(child, ASTNode):
                    find_calls(child)
                elif isinstance(child, list):
                    for item in child:
                        if isinstance(item, ASTNode):
                            find_calls(item)

        for stmt in body:
            find_calls(stmt)

        # If this function calls itself recursively
        if call_count > 0:
            # Single recursive call → O(n) (e.g., tail recursion or linear recursion)
            # Multiple recursive calls (branching) → O(2^n) (e.g., fibonacci)
            if call_count >= 2:
                return COMPLEXITY_EXP, 0.90
            return COMPLEXITY_N, 0.85

    return None, 0


def _analyze_nested_loops(ast: Optional["ASTNode"]) -> Optional[str]:
    """
    Analyze nested loops to detect complexity like O(n log n).

    Returns:
        - "nlogn": nested loops where outer is O(n) and inner is O(log n)
        - "n2": nested loops where both are O(n) or similar
        - None: not a clear nested loop pattern
    """
    if ast is None:
        return None

    from ..core.ast_nodes import ForNode, WhileNode, AssignmentNode, IdentifierNode

    loop_info: List[Tuple[int, str, "ASTNode"]] = []  # (depth, pattern, node)

    def walk(node: "ASTNode", depth: int) -> None:
        if isinstance(node, (ForNode, WhileNode)):
            # Determine if this loop is O(n) or O(log n)
            pattern = "linear"  # default
            body = getattr(node, "body", [])

            # Check for halving pattern inside the loop
            for stmt in body:
                if isinstance(stmt, AssignmentNode):
                    rhs = getattr(stmt, "value", None)
                    if rhs is not None:
                        from ..core.ast_nodes import BinaryOpNode

                        if isinstance(rhs, BinaryOpNode):
                            op = getattr(rhs, "operator", None)
                            left = getattr(rhs, "left", None)
                            target = getattr(stmt, "target", None)

                            # Check for var //= const pattern
                            if op == "//":
                                if isinstance(left, IdentifierNode) and isinstance(
                                    target, IdentifierNode
                                ):
                                    if getattr(left, "name", "") == getattr(
                                        target, "name", ""
                                    ):
                                        pattern = "log"

            loop_info.append((depth + 1, pattern, node))
            for child in body:
                walk(child, depth + 1)
        else:
            for child in vars(node).values():
                if isinstance(child, ASTNode):
                    walk(child, depth)
                elif isinstance(child, list):
                    for item in child:
                        if isinstance(item, ASTNode):
                            walk(item, depth)

    walk(ast, 0)

    # Analyze nested loops
    if len(loop_info) >= 2:
        # Check if we have two levels of nesting
        max_depth = max(d for d, _, _ in loop_info) if loop_info else 0
        if max_depth >= 2:
            # Get patterns at different depths
            depth_patterns: Dict[int, str] = {}
            for depth, pattern, _ in loop_info:
                if depth not in depth_patterns:
                    depth_patterns[depth] = pattern

            # If outer is linear and inner is log, it's O(n log n)
            if 1 in depth_patterns and 2 in depth_patterns:
                if depth_patterns[1] == "linear" and depth_patterns[2] == "log":
                    return "nlogn"
                elif depth_patterns[1] == "linear" and depth_patterns[2] == "linear":
                    # Check for depth 3 with log pattern → O(n² log n)
                    if 3 in depth_patterns and depth_patterns[3] == "log":
                        return "n2logn"
                    # Check for 4+ nested loops → O(n⁴)
                    if max_depth == 4:
                        return "n4"
                    # Check for depth 3 (3 nested loops) → O(n³)
                    if max_depth == 3:
                        return "n3"
                    # More than 4 → O(n^k)
                    if max_depth > 4:
                        return "nk"
                    return "n2"

    return None


def _detect_algorithm_pattern(node: Optional["ASTNode"]) -> Optional[str]:
    """
    Detect specific algorithmic patterns from AST structure.

    Returns:
        - "binary_search": while loop with left/right/mid pattern
        - "binary_exponentiation": while loop with variable //= 2 pattern
        - "nested_loop": multiple nested loops at depth 2+
        - "single_loop": one loop at depth 1
        - None: unknown
    """
    if node is None:
        return None

    # Count loops at each depth and find first loop for pattern matching
    loop_depths = []
    first_loop = None

    def walk(n: Optional["ASTNode"], depth: int) -> None:
        nonlocal first_loop
        if n is None:
            return
        if isinstance(n, (ForNode, WhileNode)):
            loop_depths.append(depth + 1)
            if first_loop is None:
                first_loop = n
            for child in getattr(n, "body", []):
                walk(child, depth + 1)
        else:
            for child in vars(n).values():
                if isinstance(child, ASTNode):
                    walk(child, depth)
                elif isinstance(child, list):
                    for item in child:
                        if isinstance(item, ASTNode):
                            walk(item, depth)

    walk(node, 0)
    max_depth = max(loop_depths) if loop_depths else 0

    if max_depth == 0:
        return None

    if max_depth >= 2:
        return "nested_loop"

    # Single loop - check for patterns
    if first_loop is not None and isinstance(first_loop, WhileNode):
        # Check for O(log n) patterns: halving/reducing each iteration
        for stmt in getattr(first_loop, "body", []):
            if isinstance(stmt, AugmentedAssignmentNode):
                target = getattr(stmt, "target", None)
                operator = getattr(stmt, "operator", None)
                # Binary exponentiation: b //= 2
                if target is not None and operator == "//=":
                    return "binary_exponentiation"
                # GCD (Euclid): b %= a
                if target is not None and operator == "%=":
                    return "euclidean_gcd"

            # Check for regular assignment with // like b = b // 2
            if isinstance(stmt, AssignmentNode):
                rhs = getattr(stmt, "value", None)
                target_name = getattr(stmt, "target", None)

                if isinstance(rhs, BinaryOpNode):
                    op = getattr(rhs, "operator", None)
                    left = getattr(rhs, "left", None)
                    right = getattr(rhs, "right", None)

                    # Binary exponentiation: b = b // 2
                    if op == "//" and left is not None and right is not None:
                        if isinstance(left, IdentifierNode) and isinstance(
                            target_name, IdentifierNode
                        ):
                            if getattr(left, "name", "") == getattr(
                                target_name, "name", ""
                            ):
                                return "binary_exponentiation"

                    # Euclid GCD: b = a % b pattern
                    if op == "%" and left is not None and right is not None:
                        # Check if target appears on RHS (b = a % b)
                        if isinstance(right, IdentifierNode) and isinstance(
                            target_name, IdentifierNode
                        ):
                            if getattr(right, "name", "") == getattr(
                                target_name, "name", ""
                            ):
                                return "euclidean_gcd"

        # Legacy binary search detection (for programs using mid = (l+r)//2)
        for stmt in getattr(first_loop, "body", []):
            if isinstance(stmt, AssignmentNode):
                rhs = getattr(stmt, "value", None)
                if isinstance(rhs, BinaryOpNode):
                    op = getattr(rhs, "operator", None)
                    if op == "//":
                        return "binary_search"

    return "single_loop"


def detect_complexity_with_confidence(
    line_stats: Dict[int, LineStats],
    max_loop_depth: int = 0,
    ast: Optional["ASTNode"] = None,
) -> Tuple[str, float]:
    """
    Return complexity class using AST + profiling data.

    Priority:
    1. Recursion → O(n) (accumulated work)
    2. Nested loops with mixed patterns → O(n log n)
    3. AST depth >= 2 → use depth counts for accurate complexity
    4. AST depth == 1 → check for O(log n) patterns (binary search/exponentiation)
    5. No AST → fallback to count-based heuristics

    Confidence is highest when we have both AST structure and runtime data.
    """
    if not line_stats:
        return COMPLEXITY_O1, 0.95

    counts = [s.execution_count for s in line_stats.values()]
    max_count = max(counts)

    # Compute loop depth from AST if provided
    if ast is not None:
        max_loop_depth = _get_max_loop_depth_from_ast(ast)

    # Check for recursion first
    if ast is not None:
        recurse_complexity, recurse_conf = _detect_recursion_and_complexity(
            ast, line_stats
        )
        if recurse_complexity is not None:
            return recurse_complexity, recurse_conf

    # Check for nested loops with different complexity patterns
    # (O(n log n), O(n² log n), O(n³))
    if ast is not None:
        nested_pattern = _analyze_nested_loops(ast)
        if nested_pattern == "nlogn":
            return COMPLEXITY_NLOGN, 0.85
        elif nested_pattern == "n2logn":
            return COMPLEXITY_N2LOGN, 0.85
        elif nested_pattern == "n3":
            return COMPLEXITY_N3, 0.85
        elif nested_pattern == "n4":
            return COMPLEXITY_N4, 0.85
        elif nested_pattern == "nk":
            return COMPLEXITY_NK, 0.85
        elif nested_pattern == "n2":
            return COMPLEXITY_N2, 0.85

    # Use AST loop depth as primary signal (even if max_count is 1)
    if max_loop_depth >= 5:
        return COMPLEXITY_NK, 0.90
    elif max_loop_depth == 4:
        return COMPLEXITY_N4, 0.85
    elif max_loop_depth == 3:
        return COMPLEXITY_N3, 0.85
    elif max_loop_depth == 2:
        # For depth 2, check if inner loop is O(log n) pattern
        if ast is not None:
            nested_pattern = _analyze_nested_loops(ast)
            if nested_pattern == "nlogn":
                return COMPLEXITY_NLOGN, 0.85
        return COMPLEXITY_N2, 0.85
    elif max_loop_depth == 1:
        # Single loop - check for O(log n) patterns
        algo_pattern = _detect_algorithm_pattern(ast) if ast else None
        if algo_pattern in ("binary_search", "binary_exponentiation", "euclidean_gcd"):
            return COMPLEXITY_LOGN, 0.90

        # Disambiguate using pattern analysis
        pattern_analysis = _analyze_execution_pattern(line_stats)
        pattern = pattern_analysis["pattern"]

        if pattern == "log":
            return COMPLEXITY_LOGN, 0.75
        else:
            return COMPLEXITY_N, 0.80

    # No loops in AST: check if truly constant (max_count <= 1)
    if max_count <= 1:
        return COMPLEXITY_O1, 0.95

    # Use count-based heuristics for programs with no loops
    pattern_analysis = _analyze_execution_pattern(line_stats)
    pattern = pattern_analysis["pattern"]

    # Pattern overrides count thresholds when we have clear signal
    if pattern == "log":
        return COMPLEXITY_LOGN, 0.75
    elif pattern == "nlogn":
        return COMPLEXITY_NLOGN, 0.70
    elif pattern == "quadratic":
        return COMPLEXITY_N2, 0.55  # Low confidence - can't confirm without AST
    elif pattern == "linear":
        return COMPLEXITY_N, 0.60

    # Fallback for ambiguous patterns: default to O(n) with low confidence
    # We CANNOT determine O(n) vs O(n²) from counts alone without AST
    return COMPLEXITY_N, 0.50


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
        O(n²)       - Max count suggests nested loops
        O(n^3)+      - Very high execution counts

    Args:
        line_stats: Dictionary of line number to LineStats

    Returns:
        A string representing the estimated complexity class
    """
    complexity, _confidence = detect_complexity_with_confidence(line_stats, ast=None)
    return complexity


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

    def __init__(self, config: Optional[ProfilerConfig] = None) -> None:
        self.config = config or ProfilerConfig()
        self.data = ProfilingData()
        self._current_line_start: Optional[float] = None
        self._current_line_number: Optional[int] = None
        self._current_line_sampled = False
        # Each entry: (func_name, start_time, depth, caller)
        self._function_call_stack: List[Tuple[str, float, int, Optional[str]]] = []
        self._enabled = True
        self._rng = random.Random(self.config.random_seed)
        self.data.line_sampling_rate = self.config.normalized_sampling_rate()
        self.data.memory_mode = self.config.normalized_memory_mode()

    # ── Session Control ──

    def start(self) -> None:
        """Begin a profiling session. Call this before any code executes."""
        self.data.start_time = time.perf_counter()

    def stop(self, max_loop_depth: int = 0, ast: Optional["ASTNode"] = None) -> None:
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
        method = "empirical"
        display_complexity: Optional[str] = None
        worst_case: Optional[str] = None
        best_case: Optional[str] = None
        bound_symbols: List[str] = []
        has_early_exit = False
        fallback_reason: Optional[str] = None

        if ast is not None:
            from ..analysis.complexity import analyze_complexity

            static_result = analyze_complexity(cast(ProgramNode, ast))
            if static_result.complexity == COMPLEXITY_UNKNOWN:
                fallback_reason = static_result.fallback_reason
                complexity, confidence = detect_complexity_with_confidence(
                    self.data.line_stats,
                    max_loop_depth=max_loop_depth,
                    ast=ast,
                )
                method = "empirical"
            else:
                complexity = static_result.complexity
                confidence = static_result.confidence
                method = static_result.method
                display_complexity = static_result.display_complexity
                worst_case = static_result.worst_case
                best_case = static_result.best_case
                bound_symbols = static_result.bound_symbols or []
                has_early_exit = static_result.has_early_exit
                fallback_reason = static_result.fallback_reason
        else:
            complexity, confidence = detect_complexity_with_confidence(
                self.data.line_stats,
                max_loop_depth=max_loop_depth,
                ast=None,
            )
        sampling_rate = self.config.normalized_sampling_rate()
        sampling_adjusted_confidence = confidence * (0.5 + (0.5 * sampling_rate))
        self.data.complexity_estimate = complexity
        self.data.complexity_method = method
        cap = 1.0 if method in {"static", "unbounded"} else 0.99
        self.data.complexity_confidence = max(
            0.0 if complexity == COMPLEXITY_UNKNOWN else 0.05,
            min(cap, sampling_adjusted_confidence),
        )
        self.data.complexity_display = display_complexity or complexity
        self.data.complexity_worst_case = worst_case or complexity
        self.data.complexity_best_case = best_case
        self.data.complexity_bound_symbols = bound_symbols
        self.data.complexity_has_early_exit = has_early_exit
        self.data.complexity_fallback_reason = fallback_reason

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

        sampling_rate = self.config.normalized_sampling_rate()
        sampled = sampling_rate >= 1.0 or self._rng.random() < sampling_rate
        self._current_line_number = line_number
        self._current_line_sampled = sampled

        if not sampled:
            self.data.skipped_lines += 1
            self._current_line_start = None
            return

        self.data.sampled_lines += 1

        mode = self.config.normalized_memory_mode()
        if env_values is not None and mode != "off":
            var_count = len(env_values)
            mem_bytes = estimate_memory_bytes(
                env_values,
                mode=mode,
                deep_max_depth=self.config.normalized_deep_max_depth(),
                deep_max_items=self.config.normalized_deep_max_items(),
            )
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
        if not self._enabled:
            return

        if self._current_line_number is None:
            return

        resolved_line = self._current_line_number
        if resolved_line not in self.data.line_stats:
            self.data.line_stats[resolved_line] = LineStats(resolved_line)

        if not self._current_line_sampled:
            self.data.line_stats[resolved_line].execution_count += 1
            self._current_line_number = None
            self._current_line_start = None
            self._current_line_sampled = False
            return

        if self._current_line_start is None:
            self._current_line_number = None
            self._current_line_sampled = False
            return

        elapsed_ms = (time.perf_counter() - self._current_line_start) * 1000
        self.data.line_stats[resolved_line].update_time(elapsed_ms)

        self._current_line_number = None
        self._current_line_start = None
        self._current_line_sampled = False

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
            "complexity_method": self.data.complexity_method,
            "complexity_confidence": round(self.data.complexity_confidence, 3),
            "complexity_display": self.data.complexity_display
            or self.data.complexity_estimate,
            "complexity_worst_case": self.data.complexity_worst_case
            or self.data.complexity_estimate,
            "complexity_best_case": self.data.complexity_best_case,
            "complexity_bound_symbols": self.data.complexity_bound_symbols,
            "complexity_has_early_exit": self.data.complexity_has_early_exit,
            "complexity_fallback_reason": self.data.complexity_fallback_reason,
            "functions_called": len(self.data.function_stats),
            "total_function_calls": sum(
                f.call_count for f in self.data.function_stats.values()
            ),
            "sampled_lines": self.data.sampled_lines,
            "skipped_lines": self.data.skipped_lines,
            "line_sampling_rate": self.data.line_sampling_rate,
            "memory_mode": self.data.memory_mode,
            "hottest_lines": [s.to_dict() for s in hottest_lines],
            "most_executed_lines": [s.to_dict() for s in most_executed],
            "hottest_functions": [f.to_dict() for f in hottest_funcs],
        }

    # ── Control ──

    def reset(self) -> None:
        """Reset all profiling data for a fresh session."""
        self.data = ProfilingData()
        self._current_line_start = None
        self._current_line_number = None
        self._current_line_sampled = False
        self._function_call_stack = []
        self.data.line_sampling_rate = self.config.normalized_sampling_rate()
        self.data.memory_mode = self.config.normalized_memory_mode()

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


def main() -> None:
    """CLI entry point for the profiler."""
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


if __name__ == "__main__":
    main()
