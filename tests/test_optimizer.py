"""
Additional tests covering every branch and condition in optimizer.py.

33 gaps identified across all 10 patterns, grouped by pattern.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Callable

import pytest

from optilang.ast_nodes import ProgramNode
from optilang.executor import execute
from optilang.lexer import tokenize
from optilang.models import Suggestion
from optilang.optimizer import (
    detect_constant_folding,
    detect_dead_code,
    detect_early_return,
    detect_expensive_calls,
    detect_hot_loops,
    detect_loop_invariant,
    detect_nested_loops,
    detect_repeated_computation,
    detect_string_concat,
    detect_unused_vars,
)
from optilang.parser import parse
from optilang.profiler import ProfilingData


# Shared helpers
def _parse(source: str) -> ProgramNode:
    return parse(tokenize(source))


def _run(
    source: str,
) -> Tuple[ProgramNode, Optional[ProfilingData], Optional[Dict[str, Any]]]:
    result = execute(source)
    ast = _parse(source)
    return ast, result.profiling, result.symbol_table


def _detect(
    detector: Callable[[Any, Any, Any], List[Suggestion]], source: str
) -> List[Suggestion]:
    ast, profiling, symbol_table = _run(source)
    return detector(ast, profiling, symbol_table)


# Pattern 1 — Unused variables (3 new cases)


class TestUnusedVarsExtended:

    def test_variable_used_inside_while_loop_not_flagged(self) -> None:
        # x is read in the while condition — must not be flagged
        src = "x = 10\nwhile x > 0:\n    x = x - 1\nprint(x)"
        suggestions = _detect(detect_unused_vars, src)
        assert not any("x" in s.description for s in suggestions)

    def test_variable_used_inside_try_block_not_flagged(self) -> None:
        src = "value = 42\n" "try:\n" "    print(value)\n" "except:\n" "    pass\n"
        suggestions = _detect(detect_unused_vars, src)
        assert not any("value" in s.description for s in suggestions)

    def test_variable_reassigned_then_read_not_flagged(self) -> None:
        # x is reassigned but the final value is read by print
        src = "x = 1\nx = 2\nx = 3\nprint(x)"
        suggestions = _detect(detect_unused_vars, src)
        assert not any("x" in s.description for s in suggestions)


# Pattern 2 — Dead code (7 new cases)


class TestDeadCodeExtended:

    def test_dead_code_inside_elif_block(self) -> None:
        src = (
            "def f(x):\n"
            "    if x > 10:\n"
            "        return 1\n"
            "    elif x > 5:\n"
            "        return 2\n"
            "        y = 99\n"  # dead — after return in elif
            "    return 0\n"
            "f(7)"
        )
        ast = _parse(src)
        suggestions = detect_dead_code(ast)
        assert len(suggestions) >= 1

    def test_dead_code_inside_else_block(self) -> None:
        src = (
            "def f(x):\n"
            "    if x > 0:\n"
            "        return x\n"
            "    else:\n"
            "        return 0\n"
            "        y = 99\n"  # dead — after return in else
            "f(1)"
        )
        ast = _parse(src)
        suggestions = detect_dead_code(ast)
        assert len(suggestions) >= 1

    def test_dead_code_inside_try_block(self) -> None:
        src = (
            "def f():\n"
            "    try:\n"
            "        return 1\n"
            "        x = 2\n"  # dead — after return in try block
            "    except:\n"
            "        pass\n"
            "f()"
        )
        ast = _parse(src)
        suggestions = detect_dead_code(ast)
        assert len(suggestions) >= 1

    def test_dead_code_inside_except_block(self) -> None:
        src = (
            "def f():\n"
            "    try:\n"
            "        x = 1\n"
            "    except:\n"
            "        return 0\n"
            "        y = 99\n"  # dead — after return in except block
            "f()"
        )
        ast = _parse(src)
        suggestions = detect_dead_code(ast)
        assert len(suggestions) >= 1

    def test_dead_code_inside_finally_block(self) -> None:
        src = (
            "def f():\n"
            "    try:\n"
            "        x = 1\n"
            "    except:\n"
            "        pass\n"
            "    finally:\n"
            "        return 0\n"
            "        y = 99\n"  # dead — after return in finally
            "f()"
        )
        ast = _parse(src)
        suggestions = detect_dead_code(ast)
        assert len(suggestions) >= 1

    def test_multiple_dead_statements_after_one_return(self) -> None:
        src = (
            "def f():\n"
            "    return 1\n"
            "    x = 2\n"  # dead statement 1
            "    y = 3\n"  # dead statement 2
            "    print(x)\n"  # dead statement 3
            "f()"
        )
        ast = _parse(src)
        suggestions = detect_dead_code(ast)
        assert len(suggestions) >= 3

    def test_dead_code_in_while_loop_body(self) -> None:
        src = (
            "i = 0\n"
            "while i < 5:\n"
            "    break\n"
            "    i = i + 1\n"  # dead — after break in while
        )
        ast = _parse(src)
        suggestions = detect_dead_code(ast)
        assert len(suggestions) >= 1


# Pattern 3 — Constant folding (6 new cases)


class TestConstantFoldingExtended:

    def test_literal_subtraction_detected(self) -> None:
        ast = _parse("x = 20 - 8")
        suggestions = detect_constant_folding(ast)
        assert len(suggestions) >= 1
        assert "12" in suggestions[0].description

    def test_literal_floor_division_detected(self) -> None:
        ast = _parse("x = 17 // 3")
        suggestions = detect_constant_folding(ast)
        assert len(suggestions) >= 1
        assert "5" in suggestions[0].description

    def test_literal_modulo_detected(self) -> None:
        ast = _parse("x = 17 % 5")
        suggestions = detect_constant_folding(ast)
        assert len(suggestions) >= 1
        assert "2" in suggestions[0].description

    def test_literal_exponentiation_detected(self) -> None:
        ast = _parse("x = 2 ** 8")
        suggestions = detect_constant_folding(ast)
        assert len(suggestions) >= 1
        assert "256" in suggestions[0].description

    def test_boolean_literal_folding(self) -> None:
        # True is a BooleanNode — True + 1 = 2 is foldable
        ast = _parse("x = True + 1")
        suggestions = detect_constant_folding(ast)
        # BooleanNode is in _LITERAL_TYPES so this should fold
        assert len(suggestions) >= 1

    def test_reassigned_variable_excluded_from_propagation(self) -> None:
        # N is assigned twice so it must NOT enter const_map
        src = "N = 10\nN = 20\nlimit = N * 2\nprint(limit)"
        ast, _, st = _run(src)
        suggestions = detect_constant_folding(ast, symbol_table=st)
        # limit = N * 2 should NOT be flagged since N was reassigned
        assert not any("limit" in s.description for s in suggestions)


# Pattern 4 — Early return (1 new case)


class TestEarlyReturnExtended:

    def test_trivial_if_body_pass_not_flagged(self) -> None:
        # if_block is just a pass — not a real guard clause opportunity
        src = (
            "def f(x):\n"
            "    if x > 0:\n"
            "        pass\n"
            "    else:\n"
            "        return 0\n"
            "f(1)"
        )
        ast = _parse(src)
        suggestions = detect_early_return(ast)
        assert suggestions == []


# Pattern 5 — Loop-invariant code (3 new cases)


class TestLoopInvariantExtended:

    def test_while_loop_invariant_detected(self) -> None:
        src = (
            "n = 100\n"
            "i = 0\n"
            "while i < n:\n"
            "    limit = n * 3\n"
            "    i = i + 1\n"
            "print(limit)"
        )
        ast, profiling, st = _run(src)
        suggestions = detect_loop_invariant(ast, profiling, st)
        assert any("limit" in s.description for s in suggestions)

    def test_invariant_inside_nested_inner_loop(self) -> None:
        # The outer loop variable i is not written inside the inner loop,
        # but n is not written anywhere — inner body has invariant
        src = (
            "n = 50\n"
            "for i in range(n):\n"
            "    for j in range(n):\n"
            "        scale = n * 2\n"
            "        x = i + j + scale\n"
            "print(x)"
        )
        ast, profiling, st = _run(src)
        suggestions = detect_loop_invariant(ast, profiling, st)
        assert any("scale" in s.description for s in suggestions)

    def test_profiling_none_still_detects_invariant(self) -> None:
        # Without profiling the count gate is skipped — should still flag
        src = (
            "n = 50\n"
            "for i in range(n):\n"
            "    limit = n * 2\n"
            "    x = i + limit\n"
            "print(x)"
        )
        ast = _parse(src)
        suggestions = detect_loop_invariant(ast, profiling=None, symbol_table=None)
        assert any("limit" in s.description for s in suggestions)


# Pattern 6 — String concatenation in loops (2 new cases)


class TestStringConcatExtended:

    def test_medium_severity_for_moderate_iteration_count(self) -> None:
        # 100 iterations — above STR_CONCAT_MEDIUM_COUNT (50) but below HIGH (500)
        src = (
            'result = ""\n'
            "for i in range(100):\n"
            '    result += "x"\n'
            "print(result)"
        )
        ast, profiling, st = _run(src)
        suggestions = detect_string_concat(ast, profiling, st)
        assert len(suggestions) >= 1
        assert suggestions[0].severity == "medium"

    def test_string_concat_in_while_loop_detected(self) -> None:
        src = (
            'result = ""\n'
            "i = 0\n"
            "while i < 60:\n"
            '    result += "x"\n'
            "    i = i + 1\n"
            "print(result)"
        )
        ast, profiling, st = _run(src)
        suggestions = detect_string_concat(ast, profiling, st)
        assert len(suggestions) >= 1
        assert suggestions[0].pattern == "string_concat_loop"


# Pattern 7 — Nested loops (2 new cases)


class TestNestedLoopsExtended:

    def test_low_severity_nested_loop_small_count(self) -> None:
        # Only 3 × 3 = 9 iterations — below NESTED_MEDIUM_COUNT (100)
        src = (
            "for i in range(3):\n"
            "    for j in range(3):\n"
            "        x = i + j\n"
            "print(x)"
        )
        ast, profiling, st = _run(src)
        suggestions = detect_nested_loops(ast, profiling, st)
        assert len(suggestions) >= 1
        assert suggestions[0].severity == "low"

    def test_while_loop_nested_inside_for_detected(self) -> None:
        src = (
            "for i in range(10):\n"
            "    j = 0\n"
            "    while j < 10:\n"
            "        x = i + j\n"
            "        j = j + 1\n"
            "print(x)"
        )
        ast, profiling, st = _run(src)
        suggestions = detect_nested_loops(ast, profiling, st)
        assert len(suggestions) >= 1
        assert suggestions[0].pattern == "nested_loops"


# Pattern 8 — Hot loops (3 new cases)


class TestHotLoopsExtended:

    def test_high_severity_very_large_count(self) -> None:
        # HIGH requires max_body > hot_threshold * 10.
        # hot_threshold = max(mean * HOT_MULTIPLIER, MIN_HOT_COUNT) = max(mean*10, 1000)
        # Strategy: fill profiling with 999 background lines (count=1 each) so
        # mean stays low (~1), keeping threshold=1000. Then set the loop body
        # line (line 2) to count=15_000.
        # mean ≈ (999 + 15000) / 1000 ≈ 16  →  threshold = 1000
        # 15_000 > 1000 * 10 = 10_000  →  HIGH ✓
        from optilang.profiler import LineStats

        src = "for i in range(15000):\n    x = i * 2\nprint(x)"
        ast, profiling, st = _run(src)
        if profiling is None:
            pytest.skip("profiling unavailable")
        # Add 999 extra background lines so mean stays very low
        for bg_line in range(100, 1099):
            if bg_line not in profiling.line_stats:
                stats = LineStats(line_number=bg_line)
                stats.execution_count = 1
                stats.total_time_ms = 0.001
                stats.avg_time_ms = 0.001
                profiling.line_stats[bg_line] = stats
        suggestions = detect_hot_loops(ast, profiling=profiling)
        assert len(suggestions) >= 1
        assert suggestions[0].severity == "high"

    def test_hot_while_loop_detected(self) -> None:
        # While loop with 5000 iterations. Add background lines to keep mean low
        # so 5000 exceeds hot_threshold.
        # mean ≈ (999 + 5000) / 1000 ≈ 6  →  threshold = max(60, 1000) = 1000
        # 5000 > 1000 → MEDIUM ✓
        from optilang.profiler import LineStats

        src = (
            "i = 0\n" "while i < 5000:\n" "    x = i * 2\n" "    i = i + 1\n" "print(x)"
        )
        ast, profiling, st = _run(src)
        if profiling is None:
            pytest.skip("profiling unavailable")
        for bg_line in range(100, 1099):
            if bg_line not in profiling.line_stats:
                stats = LineStats(line_number=bg_line)
                stats.execution_count = 1
                stats.total_time_ms = 0.001
                stats.avg_time_ms = 0.001
                profiling.line_stats[bg_line] = stats
        suggestions = detect_hot_loops(ast, profiling=profiling)
        assert len(suggestions) >= 1
        assert suggestions[0].pattern == "hot_loop"

    def test_empty_profiling_line_stats_returns_empty(self) -> None:
        # No lines profiled — counts list is empty → return []
        from optilang.profiler import Profiler

        profiler = Profiler()
        profiler.start()
        profiler.stop()
        ast = _parse("for i in range(5000):\n    x = i\nprint(x)")
        suggestions = detect_hot_loops(ast, profiling=profiler.get_data())
        assert suggestions == []


# Pattern 9 — Repeated computation (3 new cases)


class TestRepeatedComputationExtended:

    def test_unary_operation_repeated(self) -> None:
        # -n computed twice with no change to n in between.
        # Pass profiling=None so the execution_count gate is skipped —
        # the detector relies purely on AST structure in static mode.
        src = "n = 50\n" "a = -n\n" "b = -n\n" "print(a + b)"
        ast = _parse(src)
        suggestions = detect_repeated_computation(
            ast, profiling=None, symbol_table=None
        )
        assert len(suggestions) >= 1
        assert suggestions[0].pattern == "repeated_computation"

    def test_function_call_expression_repeated(self) -> None:
        # len(items) computed twice with no modification to items.
        # Pass profiling=None so the execution_count gate is skipped —
        # the detector relies purely on AST fingerprinting in static mode.
        src = (
            "items = [1, 2, 3, 4, 5]\n"
            "x = len(items)\n"
            "y = len(items)\n"
            "print(x + y)"
        )
        ast = _parse(src)
        suggestions = detect_repeated_computation(
            ast, profiling=None, symbol_table=None
        )
        assert len(suggestions) >= 1
        assert suggestions[0].pattern == "repeated_computation"

    def test_profiling_none_still_detects_repetition(self) -> None:
        # Without profiling the count gate is skipped — repetition still detected
        src = "n = 100\n" "x = n * 2\n" "y = n * 2\n" "print(x + y)"
        ast = _parse(src)
        # Pass profiling=None — the count gate must be skipped entirely
        suggestions = detect_repeated_computation(
            ast, profiling=None, symbol_table=None
        )
        assert len(suggestions) >= 1
        assert suggestions[0].pattern == "repeated_computation"


# Pattern 10 — Expensive function calls (3 new cases)


class TestExpensiveCallsExtended:

    def test_medium_severity_not_in_loop_high_total_time(self) -> None:
        src = (
            "def process():\n"
            "    total = 0\n"
            "    for i in range(500):\n"
            "        total = total + i\n"
            "    return total\n"
            "result = 0\n"
            "for i in range(10):\n"
            "    result = result + process()\n"
            "print(result)"
        )
        ast, profiling, st = _run(src)
        if profiling and "process" in profiling.function_stats:
            # Patch to exceed thresholds and total_time > 50ms but NOT in loop
            # Use a call site outside a loop for the medium branch
            profiling.function_stats["process"].avg_time_ms = 6.0
            profiling.function_stats["process"].call_count = 10
            profiling.function_stats["process"].total_time_ms = 60.0
        # The call site is inside a for-loop so it will be high, which is fine —
        # we specifically test the medium branch with a standalone call site
        src2 = (
            "def compute():\n"
            "    total = 0\n"
            "    for i in range(500):\n"
            "        total = total + i\n"
            "    return total\n"
            "a = compute()\n"
            "b = compute()\n"
            "c = compute()\n"
            "d = compute()\n"
            "e = compute()\n"
            "f = compute()\n"
            "g = compute()\n"
            "h = compute()\n"
            "k = compute()\n"
            "l = compute()\n"
            "print(a)\n"
        )
        ast2, profiling2, st2 = _run(src2)
        if profiling2 and "compute" in profiling2.function_stats:
            profiling2.function_stats["compute"].avg_time_ms = 6.0
            profiling2.function_stats["compute"].call_count = 10
            profiling2.function_stats["compute"].total_time_ms = 60.0
            suggestions = detect_expensive_calls(ast2, profiling2, st2)
            medium = [s for s in suggestions if s.severity == "medium"]
            assert len(medium) >= 1

    def test_low_severity_not_in_loop_low_total_time(self) -> None:
        src = (
            "def quick():\n"
            "    return 1\n"
            "a = quick()\n"
            "b = quick()\n"
            "c = quick()\n"
            "d = quick()\n"
            "e = quick()\n"
            "f = quick()\n"
            "g = quick()\n"
            "h = quick()\n"
            "k = quick()\n"
            "l = quick()\n"
            "print(a)\n"
        )
        ast, profiling, st = _run(src)
        if profiling and "quick" in profiling.function_stats:
            profiling.function_stats["quick"].avg_time_ms = 1.5
            profiling.function_stats["quick"].call_count = 10
            profiling.function_stats["quick"].total_time_ms = 15.0
            suggestions = detect_expensive_calls(ast, profiling, st)
            low = [s for s in suggestions if s.severity == "low"]
            assert len(low) >= 1

    def test_expensive_function_with_no_call_site_in_ast(self) -> None:
        # profiling reports a function as expensive but it has no call site
        # in the AST (e.g. it was called dynamically) — should return empty
        from optilang.profiler import FunctionStats

        ast = _parse("x = 1\nprint(x)")
        from optilang.profiler import Profiler

        profiler = Profiler()
        profiler.start()
        profiler.stop()
        profiler.data.function_stats["ghost"] = FunctionStats(
            name="ghost",
            call_count=15,
            avg_time_ms=5.0,
            total_time_ms=75.0,
        )
        suggestions = detect_expensive_calls(ast, profiling=profiler.get_data())
        # ghost() does not appear anywhere in the AST — no suggestions
        assert suggestions == []
