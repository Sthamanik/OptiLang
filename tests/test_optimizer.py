"""
Additional tests covering every branch and condition in optimizer.py.

33 gaps identified across all 10 patterns, grouped by pattern.
"""

from __future__ import annotations

import builtins
from typing import Any, Dict, List, Optional, Tuple, Callable

import pytest

from optilang.ast_nodes import (
    AssignmentNode,
    BinaryOpNode,
    BooleanNode,
    DictNode,
    ForNode,
    FunctionCallNode,
    IdentifierNode,
    NullNode,
    NumberNode,
    PassNode,
    ProgramNode,
    StringNode,
)
from optilang.executor import execute
from optilang.lexer import tokenize
from optilang.models import OptimizationReport, Suggestion
from optilang.optimizer import (
    _UNRESOLVED,
    _build_const_map,
    _fold,
    _fp,
    _innermost_count,
    _repr_node,
    _resolve,
    _walk,
    Optimizer,
    analyze,
    analyze_source,
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
from optilang.profiler import FunctionStats, LineStats, ProfilingData


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


def _line_stats(line: int, count: int) -> LineStats:
    stats = LineStats(line_number=line)
    stats.execution_count = count
    stats.total_time_ms = float(count)
    stats.avg_time_ms = float(count)
    return stats


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


class TestOptimizerHelpers:

    def test_walk_visits_tuple_entries(self) -> None:
        ast = ProgramNode(
            1,
            1,
            statements=[
                AssignmentNode(
                    1,
                    1,
                    target=IdentifierNode(1, 1, "mapping"),
                    value=DictNode(
                        1,
                        11,
                        pairs=[(StringNode(1, 12, "k"), IdentifierNode(1, 18, "value"))],
                    ),
                )
            ],
        )

        names = [node.name for node in _walk(ast) if isinstance(node, IdentifierNode)]
        assert "mapping" in names
        assert "value" in names

    def test_unused_vars_without_symbol_table_excludes_loop_vars_and_params(
        self,
    ) -> None:
        ast = _parse(
            "for i in range(3):\n"
            "    pass\n"
            "def echo(value):\n"
            "    return value\n"
            "x = 1\n"
        )

        suggestions = detect_unused_vars(ast, profiling=None, symbol_table=None)

        assert len(suggestions) == 1
        assert suggestions[0].pattern == "unused_vars"
        assert "x" in suggestions[0].description

    def test_build_const_map_handles_none_augmented_assignments_and_symbol_table(
        self,
    ) -> None:
        ast = _parse("missing = None\ncount = 1\ncount += 2\nlabel = 'ok'\n")

        assert _build_const_map(ast, symbol_table=None) == {
            "missing": None,
            "label": "ok",
        }

        matched_ast = _parse("size = 5\n")
        assert _build_const_map(matched_ast, symbol_table={"size": 5}) == {"size": 5}
        assert _build_const_map(matched_ast, symbol_table={"size": 6}) == {}

    def test_constant_resolution_representation_and_folding_helpers(self) -> None:
        assert _resolve(NullNode(1, 1), {}) is None
        assert _resolve(PassNode(1, 1), {}) is _UNRESOLVED

        assert _repr_node(StringNode(1, 1, "hi"), {}) == "'hi'"
        assert _repr_node(IdentifierNode(1, 1, "n"), {"n": 2}) == "n(2)"
        assert _repr_node(IdentifierNode(1, 1, "n"), {}) == "n"

        assert _fold("*", 3, 4) == 12
        assert _fold("/", 8, 2) == 4
        assert _fold("/", 8, 0) is None
        assert _fold("??", 1, 2) is None

    def test_constant_folding_skips_invalid_operations(self) -> None:
        suggestions = detect_constant_folding(_parse('x = "a" - "b"\n'))
        assert suggestions == []

    def test_constant_folding_skips_none_results(self) -> None:
        assert detect_constant_folding(_parse("x = 1 / 0\n")) == []

    def test_detect_early_return_positive_case(self) -> None:
        ast = _parse(
            "def choose(x):\n"
            "    if x > 0:\n"
            "        return x\n"
            "    else:\n"
            "        return 0\n"
        )

        suggestions = detect_early_return(ast)

        assert len(suggestions) == 1
        assert suggestions[0].pattern == "early_return"

    def test_detect_early_return_rejects_non_matching_shapes(self) -> None:
        sources = [
            "def choose(x):\n"
            "    if x > 0:\n"
            "        return x\n"
            "    else:\n"
            "        return 0\n"
            "    print(x)\n",
            "def choose(x):\n"
            "    if x > 0:\n"
            "        return x\n",
            "def choose(x):\n"
            "    if x > 0:\n"
            "        return x\n"
            "    else:\n"
            "        y = 0\n",
        ]

        for source in sources:
            assert detect_early_return(_parse(source)) == []

    def test_loop_invariant_skips_literals_function_calls_and_low_counts(self) -> None:
        ast = _parse(
            "n = 3\n"
            "for i in range(n):\n"
            "    literal = 10\n"
            "    rendered = str(i)\n"
        )
        assert detect_loop_invariant(ast, profiling=None, symbol_table=None) == []

        ast = _parse(
            "n = 3\n"
            "for i in range(n):\n"
            "    limit = n * 2\n"
            "    print(limit)\n"
        )
        profiling = ProfilingData(line_stats={3: _line_stats(3, 3)})
        assert detect_loop_invariant(ast, profiling=profiling, symbol_table=None) == []

    def test_string_concat_skips_non_strings_and_scales_severity_and_recurses(
        self,
    ) -> None:
        non_string = _parse("total = 0\nfor i in range(5):\n    total += i\n")
        assert detect_string_concat(non_string, profiling=None, symbol_table=None) == []

        low = _parse('result = ""\nfor i in range(3):\n    result += "x"\n')
        low_profiling = ProfilingData(line_stats={3: _line_stats(3, 3)})
        assert detect_string_concat(low, low_profiling, {"result": ""})[0].severity == "low"

        high = _parse('result = ""\nfor i in range(600):\n    result += "x"\n')
        high_profiling = ProfilingData(line_stats={3: _line_stats(3, 600)})
        assert (
            detect_string_concat(high, high_profiling, {"result": ""})[0].severity
            == "high"
        )

        nested = _parse(
            'result = ""\n'
            "for i in range(2):\n"
            "    j = 0\n"
            "    while j < 2:\n"
            '        result += "x"\n'
            "        j = j + 1\n"
        )
        nested_profiling = ProfilingData(line_stats={5: _line_stats(5, 4)})
        suggestions = detect_string_concat(nested, nested_profiling, {"result": ""})
        assert any(s.line == 5 for s in suggestions)

    def test_innermost_count_and_hot_loop_edge_cases(self) -> None:
        ast = _parse(
            "for i in range(2):\n"
            "    for j in range(2):\n"
            "        x = i + j\n"
        )
        outer = ast.statements[0]
        assert isinstance(outer, ForNode)

        profiling = ProfilingData(line_stats={3: _line_stats(3, 4)})
        assert _innermost_count(outer, profiling) == 4
        assert _innermost_count(outer, profiling=None) == 0

        empty_body_ast = ProgramNode(
            1,
            1,
            statements=[
                ForNode(
                    1,
                    1,
                    iterator=IdentifierNode(1, 5, "i"),
                    iterable=IdentifierNode(1, 10, "items"),
                    body=[],
                )
            ],
        )
        counts_only = ProfilingData(line_stats={10: _line_stats(10, 2)})
        assert detect_hot_loops(empty_body_ast, profiling=counts_only) == []

        cool_loop_ast = ProgramNode(
            1,
            1,
            statements=[
                ForNode(
                    1,
                    1,
                    iterator=IdentifierNode(1, 5, "i"),
                    iterable=IdentifierNode(1, 10, "items"),
                    body=[
                        AssignmentNode(
                            2,
                            5,
                            target=IdentifierNode(2, 5, "x"),
                            value=NumberNode(2, 9, 1),
                        )
                    ],
                )
            ],
        )
        cool_profiling = ProfilingData(
            line_stats={2: _line_stats(2, 100), 20: _line_stats(20, 100)}
        )
        assert detect_hot_loops(cool_loop_ast, profiling=None) == []
        assert detect_hot_loops(cool_loop_ast, profiling=cool_profiling) == []

    def test_triple_nested_loops_are_high_severity(self) -> None:
        ast = _parse(
            "for i in range(2):\n"
            "    for j in range(2):\n"
            "        while j < 1:\n"
            "            j += 1\n"
        )

        suggestions = detect_nested_loops(ast, profiling=None, symbol_table=None)

        assert any(s.severity == "high" for s in suggestions)

    def test_repeated_computation_respects_same_line_writes_and_count_gate(
        self,
    ) -> None:
        same_line_ast = ProgramNode(
            1,
            1,
            statements=[
                AssignmentNode(
                    1,
                    1,
                    target=IdentifierNode(1, 1, "a"),
                    value=BinaryOpNode(
                        2,
                        5,
                        left=IdentifierNode(2, 5, "n"),
                        operator="+",
                        right=NumberNode(2, 9, 1),
                    ),
                ),
                AssignmentNode(
                    1,
                    10,
                    target=IdentifierNode(1, 10, "b"),
                    value=BinaryOpNode(
                        2,
                        15,
                        left=IdentifierNode(2, 15, "n"),
                        operator="+",
                        right=NumberNode(2, 19, 1),
                    ),
                ),
            ],
        )
        assert detect_repeated_computation(same_line_ast) == []

        with_intervening_write = _parse(
            "n = 1\n"
            "a = n * 2\n"
            "n = 3\n"
            "b = n * 2\n"
        )
        assert detect_repeated_computation(with_intervening_write) == []

        gated = _parse("n = 1\nx = n * 2\ny = n * 2\n")
        low_counts = ProfilingData(
            line_stats={2: _line_stats(2, 1), 3: _line_stats(3, 1)}
        )
        assert detect_repeated_computation(gated, profiling=low_counts) == []

        assert _fp(StringNode(1, 1, "hi")) == "'hi'"
        assert _fp(BooleanNode(1, 1, True)) == "True"
        assert _fp(PassNode(1, 1)) == "?"

    def test_repeated_computation_skips_already_reported_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake_ast = ProgramNode(line=1, column=1, statements=[])
        fake_nodes = [
            NumberNode(line=2, column=1, value=1),
            NumberNode(line=3, column=1, value=1),
            NumberNode(line=2, column=1, value=1),
            NumberNode(line=3, column=1, value=1),
        ]
        original_sorted = builtins.sorted

        def fake_walk(node: object):
            if node is fake_ast:
                yield from fake_nodes
            else:
                return
                yield

        def fake_sorted(iterable: object, *args: object, **kwargs: object):
            items = list(iterable)
            if items and all(isinstance(item, tuple) and len(item) == 2 for item in items):
                return items
            return original_sorted(items, *args, **kwargs)

        monkeypatch.setattr("optilang.optimizer._walk", fake_walk)
        monkeypatch.setattr("optilang.optimizer._nontrivial", lambda node: True)
        monkeypatch.setattr("optilang.optimizer._fp", lambda node: "same")
        monkeypatch.setattr(builtins, "sorted", fake_sorted)

        suggestions = detect_repeated_computation(
            fake_ast, profiling=None, symbol_table=None
        )

        assert suggestions

    def test_expensive_calls_handles_empty_non_expensive_and_duplicate_sites(
        self,
    ) -> None:
        ast = _parse("x = 1\n")
        assert detect_expensive_calls(ast, profiling=None) == []

        plain_call_ast = _parse("print(1)\n")
        not_expensive = ProfilingData(
            function_stats={
                "print": FunctionStats(
                    name="print",
                    call_count=2,
                    avg_time_ms=0.1,
                    total_time_ms=0.2,
                )
            }
        )
        assert detect_expensive_calls(plain_call_ast, profiling=not_expensive) == []

        duplicate_site_ast = ProgramNode(
            1,
            1,
            statements=[
                ForNode(
                    1,
                    1,
                    iterator=IdentifierNode(1, 5, "i"),
                    iterable=IdentifierNode(1, 10, "items"),
                    body=[
                        AssignmentNode(
                            3,
                            5,
                            target=IdentifierNode(3, 5, "a"),
                            value=FunctionCallNode(
                                3,
                                9,
                                function=IdentifierNode(3, 9, "slow"),
                                arguments=[IdentifierNode(3, 14, "i")],
                            ),
                        ),
                        AssignmentNode(
                            3,
                            20,
                            target=IdentifierNode(3, 20, "b"),
                            value=FunctionCallNode(
                                3,
                                24,
                                function=IdentifierNode(3, 24, "slow"),
                                arguments=[IdentifierNode(3, 29, "i")],
                            ),
                        ),
                    ],
                )
            ],
        )
        expensive = ProfilingData(
            function_stats={
                "slow": FunctionStats(
                    name="slow",
                    call_count=10,
                    avg_time_ms=5.0,
                    total_time_ms=50.0,
                )
            }
        )
        suggestions = detect_expensive_calls(duplicate_site_ast, profiling=expensive)
        assert len(suggestions) == 1
        assert suggestions[0].severity == "high"

    def test_optimizer_run_wrappers_and_exception_isolation(self) -> None:
        ast = _parse("x = 1\n")
        default_optimizer = Optimizer(ast)
        assert default_optimizer._ast is ast
        assert default_optimizer._profiling is None
        assert default_optimizer._symbol_table is None
        assert len(default_optimizer._detectors) > 0

        def broken_detector(*args: object) -> list[Suggestion]:
            raise RuntimeError("boom")

        def low_detector(*args: object) -> list[Suggestion]:
            return [
                Suggestion(
                    line=1,
                    pattern="low",
                    severity="low",
                    description="low impact",
                    suggestion="noop",
                    impact_score=1.0,
                )
            ]

        def high_detector(*args: object) -> list[Suggestion]:
            return [
                Suggestion(
                    line=1,
                    pattern="high",
                    severity="high",
                    description="high impact",
                    suggestion="noop",
                    impact_score=9.0,
                )
            ]

        report = Optimizer(
            ast, detectors=[broken_detector, low_detector, high_detector]
        ).run()
        assert [s.pattern for s in report.suggestions] == ["high", "low"]

        analyzed = analyze(ast)
        assert isinstance(analyzed, OptimizationReport)

        source_report = analyze_source("x = 1\n")
        assert isinstance(source_report, OptimizationReport)
        assert any(s.pattern == "unused_vars" for s in source_report.suggestions)
