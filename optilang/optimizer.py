"""
Optimization analysis engine for OptiLang.

This single file contains:
    - AST walker utility            (_walk)
    - All 10 pattern detectors      (one function per pattern)
    - The Optimizer orchestrator    (Optimizer class)
    - Public convenience functions  (analyze, analyze_source)

Pattern registry
----------------
Static  — AST + symbol table only, no execution needed:
    1.  detect_unused_vars          symbol table lookup
    2.  detect_dead_code            CFG reachability on AST blocks
    3.  detect_constant_folding     AST pattern matching + constant propagation
    4.  detect_early_return         guard clause detection

Hybrid  — AST structure + profiling data for severity scaling:
    5.  detect_loop_invariant       loop variable analysis (reaching definitions)
    6.  detect_string_concat        anti-pattern matching
    7.  detect_nested_loops         nesting depth tracking + execution count

Dynamic — profiling data primary, AST for structural context:
    8.  detect_hot_loops            execution frequency thresholding
    9.  detect_repeated_computation expression fingerprinting
    10. detect_expensive_calls      call frequency analysis

Note on tuning
--------------
All numeric thresholds (HOT_MULTIPLIER, LOOP_COUNT_THRESHOLD, etc.) are
empirically tuned on benchmark programs (loops, recursion, nested structures).
See CONTRIBUTING.md for the tuning guide.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Generator, List, Optional, Set, Tuple, Union

from .ast_nodes import (
    ASTNode,
    AssignmentNode,
    AugmentedAssignmentNode,
    BinaryOpNode,
    BooleanNode,
    BreakNode,
    ContinueNode,
    ForNode,
    FunctionCallNode,
    FunctionDefNode,
    IdentifierNode,
    IfNode,
    NullNode,
    NumberNode,
    PassNode,
    ProgramNode,
    ReturnNode,
    StringNode,
    TryNode,
    UnaryOpNode,
    WhileNode,
)
from .models import OptimizationReport, Suggestion
from .profiler import ProfilingData
import logging

_log = logging.getLogger(__name__)

# Tunable constants
LOOP_COUNT_THRESHOLD: int = 20
HOT_MULTIPLIER: int = 10
MIN_HOT_COUNT: int = 1_000
REPEAT_COUNT_THRESHOLD: int = 5
AVG_TIME_THRESHOLD_MS: float = 1.0
CALL_COUNT_THRESHOLD: int = 10
NESTED_MEDIUM_COUNT: int = 100
NESTED_HIGH_COUNT: int = 5_000
STR_CONCAT_MEDIUM_COUNT: int = 50
STR_CONCAT_HIGH_COUNT: int = 500


# AST walker utility
def _walk(node: ASTNode) -> Generator[ASTNode, None, None]:
    """
    Depth-first pre-order generator over every node in an AST subtree.
    Handles fields that are a single ASTNode, a list of ASTNode, or a
    list of tuples containing ASTNode (e.g. IfNode.elif_parts).
    """
    yield node
    for field in dataclasses.fields(node):
        value = getattr(node, field.name)
        if isinstance(value, ASTNode):
            yield from _walk(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, ASTNode):
                    yield from _walk(item)
                elif isinstance(item, tuple):
                    for element in item:
                        if isinstance(element, ASTNode):
                            yield from _walk(element)


def _get_count(profiling: ProfilingData, line: int) -> int:
    stats = profiling.line_stats.get(line)
    return stats.execution_count if stats else 0


def _make(
    pattern_id: str,
    severity: str,
    impact_score: float,
    line: int,
    description: str,
    suggestion: str,
) -> Suggestion:
    return Suggestion(
        line=line,
        pattern=pattern_id,
        severity=severity,
        description=description,
        suggestion=suggestion,
        impact_score=impact_score,
    )


# Pattern 1 — Unused variables
# Algorithm: Symbol table lookup


def detect_unused_vars(
    ast: ProgramNode,
    profiling: Optional[ProfilingData] = None,
    symbol_table: Optional[Dict[str, Any]] = None,
) -> List[Suggestion]:
    """
    Detect variables that are assigned but never read.

    Uses symbol_table (preferred) for the defined-names set so that
    dead-branch variables are excluded automatically. Falls back to an
    AST write-scan when no symbol table is available.

    A variable is flagged when:
        name in defined_names
        AND name not in expression_reads
        AND name is not a loop iterator or function parameter
    """
    if symbol_table is not None:
        defined: Set[str] = set(symbol_table.keys())
    else:
        defined = {
            node.target.name
            for node in _walk(ast)
            if isinstance(node, (AssignmentNode, AugmentedAssignmentNode))
        }

    # Mark node ids that are in definition position (not reads)
    excluded_ids: Set[int] = set()
    for node in _walk(ast):
        if isinstance(node, (AssignmentNode, AugmentedAssignmentNode)):
            excluded_ids.add(id(node.target))
        elif isinstance(node, ForNode):
            excluded_ids.add(id(node.iterator))
        elif isinstance(node, FunctionDefNode):
            excluded_ids.add(id(node.name))
            for param in node.parameters:
                excluded_ids.add(id(param))

    reads: Set[str] = {
        node.name
        for node in _walk(ast)
        if isinstance(node, IdentifierNode) and id(node) not in excluded_ids
    }

    loop_vars: Set[str] = {
        node.iterator.name for node in _walk(ast) if isinstance(node, ForNode)
    }

    params: Set[str] = {
        param.name
        for node in _walk(ast)
        if isinstance(node, FunctionDefNode)
        for param in node.parameters
    }

    first_line: Dict[str, int] = {}
    for node in _walk(ast):
        if isinstance(node, (AssignmentNode, AugmentedAssignmentNode)):
            name = node.target.name
            if name not in first_line:
                first_line[name] = node.line

    unused = defined - reads - loop_vars - params

    return [
        _make(
            pattern_id="unused_vars",
            severity="low",
            impact_score=3.0,
            line=first_line.get(name, 1),
            description=f"Variable '{name}' is assigned but never used",
            suggestion=(
                f"Remove the assignment to '{name}', or use its value "
                f"somewhere in the program."
            ),
        )
        for name in sorted(unused)
    ]


# Pattern 2 — Dead code
# Algorithm: CFG reachability (linear block scan)

_TERMINATORS = (ReturnNode, BreakNode, ContinueNode)


def detect_dead_code(
    ast: ProgramNode,
    profiling: Optional[ProfilingData] = None,
    symbol_table: Optional[Dict[str, Any]] = None,
) -> List[Suggestion]:
    """
    Detect statements that can never be executed.

    Models each block as a sequence of basic blocks. A terminator
    (return / break / continue) has no successor — any statement
    following it in the same block is unreachable.

    Terminator state is NOT propagated across block boundaries (an
    if-branch return does not make the statement after the if dead).
    """
    suggestions: List[Suggestion] = []
    _check_block_dead(ast.statements, suggestions)
    return suggestions


def _check_block_dead(
    statements: List[ASTNode],
    suggestions: List[Suggestion],
) -> None:
    terminator_line: Optional[int] = None
    for stmt in statements:
        if terminator_line is not None:
            suggestions.append(
                _make(
                    pattern_id="dead_code",
                    severity="medium",
                    impact_score=7.0,
                    line=stmt.line,
                    description=(
                        f"Unreachable code at line {stmt.line} "
                        f"(after terminator at line {terminator_line})"
                    ),
                    suggestion=(
                        "Remove or move this code — it can never execute because "
                        "a return, break, or continue precedes it in the same block."
                    ),
                )
            )
            continue
        if isinstance(stmt, _TERMINATORS):
            terminator_line = stmt.line
        _recurse_dead(stmt, suggestions)


def _recurse_dead(node: ASTNode, suggestions: List[Suggestion]) -> None:
    if isinstance(node, FunctionDefNode):
        _check_block_dead(node.body, suggestions)
    elif isinstance(node, IfNode):
        _check_block_dead(node.if_block, suggestions)
        for _, elif_block in node.elif_parts:
            _check_block_dead(elif_block, suggestions)
        if node.else_block:
            _check_block_dead(node.else_block, suggestions)
    elif isinstance(node, (WhileNode, ForNode)):
        _check_block_dead(node.body, suggestions)
    elif isinstance(node, TryNode):
        _check_block_dead(node.try_block, suggestions)
        if node.except_block:
            _check_block_dead(node.except_block, suggestions)
        if node.finally_block:
            _check_block_dead(node.finally_block, suggestions)


# Pattern 3 — Constant folding
# Algorithm: AST pattern matching + constant propagation

_LITERAL_TYPES = (NumberNode, StringNode, BooleanNode)
_FOLDABLE_OPS = {"+", "-", "*", "/", "//", "%", "**"}
_UNRESOLVED = object()


def detect_constant_folding(
    ast: ProgramNode,
    profiling: Optional[ProfilingData] = None,
    symbol_table: Optional[Dict[str, Any]] = None,
) -> List[Suggestion]:
    """
    Detect binary expressions whose result can be pre-computed.

    Level 1 (pure AST): both children are literal nodes.
        e.g.  x = 3 * 4  →  pre-computable to 12

    Level 2 (with symbol table — constant propagation): one or both
    children are single-assignment identifiers whose value is confirmed
    by the symbol table.
        e.g.  N = 100; limit = N * 2  →  pre-computable to 200
    """
    const_map = _build_const_map(ast, symbol_table)
    suggestions: List[Suggestion] = []

    for node in _walk(ast):
        if not isinstance(node, BinaryOpNode) or node.operator not in _FOLDABLE_OPS:
            continue

        left_val = _resolve(node.left, const_map)
        right_val = _resolve(node.right, const_map)
        if left_val is _UNRESOLVED or right_val is _UNRESOLVED:
            continue

        try:
            result = _fold(node.operator, left_val, right_val)
        except ZeroDivisionError, TypeError, ValueError, OverflowError:
            continue

        if result is None:
            continue

        lr = _repr_node(node.left, const_map)
        rr = _repr_node(node.right, const_map)
        suggestions.append(
            _make(
                pattern_id="constant_folding",
                severity="low",
                impact_score=2.0,
                line=node.line,
                description=(
                    f"Expression '{lr} {node.operator} {rr}' "
                    f"can be pre-computed to {result!r}"
                ),
                suggestion=(
                    f"Replace '{lr} {node.operator} {rr}' with "
                    f"the constant {result!r} to avoid recomputing it at runtime."
                ),
            )
        )

    return suggestions


def _build_const_map(
    ast: ProgramNode,
    symbol_table: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    assignment_count: Dict[str, int] = {}
    literal_value: Dict[str, Any] = {}
    for node in _walk(ast):
        if isinstance(node, AssignmentNode):
            name = node.target.name
            assignment_count[name] = assignment_count.get(name, 0) + 1

            if isinstance(node.value, _LITERAL_TYPES):
                literal_value[name] = node.value.value
            elif isinstance(node.value, NullNode):
                literal_value[name] = None
        elif isinstance(node, AugmentedAssignmentNode):
            assignment_count[node.target.name] = (
                assignment_count.get(node.target.name, 0) + 1
            )

    const_map: Dict[str, Any] = {}
    for name, count in assignment_count.items():
        if count == 1 and name in literal_value:
            val = literal_value[name]
            if symbol_table is not None:
                if symbol_table.get(name) == val:
                    const_map[name] = val
            else:
                const_map[name] = val
    return const_map


def _resolve(node: ASTNode, const_map: Dict[str, Any]) -> Any:
    if isinstance(node, _LITERAL_TYPES):
        return node.value
    if isinstance(node, NullNode):
        return None
    if isinstance(node, IdentifierNode):
        return const_map.get(node.name, _UNRESOLVED)
    return _UNRESOLVED


def _repr_node(node: ASTNode, const_map: Dict[str, Any]) -> str:
    if isinstance(node, NumberNode):
        return str(node.value)
    if isinstance(node, StringNode):
        return repr(node.value)
    if isinstance(node, IdentifierNode):
        if node.name in const_map:
            return f"{node.name}({const_map[node.name]!r})"
        return node.name
    return "..."


def _fold(op: str, left: Any, right: Any) -> Optional[Any]:
    if op == "+":
        return left + right
    if op == "-":
        return left - right
    if op == "*":
        return left * right
    if op == "/":
        return (left / right) if right != 0 else None
    if op == "//":
        return (left // right) if right != 0 else None
    if op == "%":
        return (left % right) if right != 0 else None
    if op == "**":
        return left**right
    return None


# Pattern 4 — Missing early return (guard clause detection)
# Algorithm: structural pattern matching on function bodies


def detect_early_return(
    ast: ProgramNode,
    profiling: Optional[ProfilingData] = None,
    symbol_table: Optional[Dict[str, Any]] = None,
) -> List[Suggestion]:
    """
    Detect functions whose entire body is a single if/else where the else
    ends with a return — a guard clause opportunity.

    Pattern:
        def f(x):
            if <condition>:
                <work>
                return <r>
            else:
                return <default>  ← hoist this to top as an early return

    Not flagged: functions with elif parts, no else block, trivial if body.
    """
    suggestions: List[Suggestion] = []
    for node in _walk(ast):
        if not isinstance(node, FunctionDefNode):
            continue
        body = node.body
        if len(body) != 1 or not isinstance(body[0], IfNode):
            continue
        if_node: IfNode = body[0]
        if if_node.elif_parts or not if_node.else_block:
            continue
        if_trivial = len(if_node.if_block) == 1 and isinstance(
            if_node.if_block[0], PassNode
        )
        if if_trivial:
            continue
        if not isinstance(if_node.else_block[-1], ReturnNode):
            continue
        suggestions.append(
            _make(
                pattern_id="early_return",
                severity="low",
                impact_score=3.0,
                line=if_node.line,
                description=(
                    f"Function '{node.name.name}' could use a guard clause "
                    f"to reduce nesting"
                ),
                suggestion=(
                    "Invert the if-condition and return early at the top of the "
                    "function. This removes one level of nesting. "
                    "Example: 'if not <condition>: return <default>'"
                ),
            )
        )
    return suggestions


# Pattern 5 — Loop-invariant code motion
# Algorithm: loop variable analysis (reaching definitions)


def detect_loop_invariant(
    ast: ProgramNode,
    profiling: Optional[ProfilingData] = None,
    symbol_table: Optional[Dict[str, Any]] = None,
) -> List[Suggestion]:
    """
    Detect assignments inside loops whose RHS does not depend on any
    variable written inside that same loop.

    Reaching definitions check:
        loop_writes = all names assigned anywhere in the loop body
        For each AssignmentNode in the body:
            reads = IdentifierNode names in the RHS
            if reads ∩ loop_writes == ∅  →  hoist above loop

    Dynamic gate: only flag when execution_count > LOOP_COUNT_THRESHOLD
    so we avoid noise on loops that barely ran.
    """
    suggestions: List[Suggestion] = []
    _scan_loop_invariant(ast, profiling, suggestions)
    return suggestions


def _scan_loop_invariant(
    node: ASTNode,
    profiling: Optional[ProfilingData],
    suggestions: List[Suggestion],
) -> None:
    for field in dataclasses.fields(node):
        value = getattr(node, field.name)
        children: List[ASTNode] = (
            [value]
            if isinstance(value, ASTNode)
            else (
                [v for v in value if isinstance(v, ASTNode)]
                if isinstance(value, list)
                else []
            )
        )
        for child in children:
            if isinstance(child, (ForNode, WhileNode)):
                _check_one_loop_invariant(child, profiling, suggestions)
            else:
                _scan_loop_invariant(child, profiling, suggestions)


def _check_one_loop_invariant(
    loop: Union[ForNode, WhileNode],
    profiling: Optional[ProfilingData],
    suggestions: List[Suggestion],
) -> None:
    body = loop.body
    loop_writes: Set[str] = {
        n.target.name
        for stmt in body
        for n in _walk(stmt)
        if isinstance(n, (AssignmentNode, AugmentedAssignmentNode))
    }
    if isinstance(loop, ForNode):
        loop_writes.add(loop.iterator.name)

    for stmt in body:
        if not isinstance(stmt, AssignmentNode):
            continue
        if isinstance(stmt.value, (FunctionCallNode,) + _LITERAL_TYPES):
            continue
        reads: Set[str] = {
            n.name for n in _walk(stmt.value) if isinstance(n, IdentifierNode)
        }
        if reads & loop_writes:
            continue
        if (
            profiling is not None
            and _get_count(profiling, stmt.line) < LOOP_COUNT_THRESHOLD
        ):
            continue
        suggestions.append(
            _make(
                pattern_id="loop_invariant",
                severity="medium",
                impact_score=8.0,
                line=stmt.line,
                description=(
                    f"Assignment to '{stmt.target.name}' at line {stmt.line} "
                    f"produces the same value on every iteration"
                ),
                suggestion=(
                    f"Move '{stmt.target.name} = ...' above the loop. "
                    f"Its value does not change between iterations."
                ),
            )
        )

    for stmt in body:
        if isinstance(stmt, (ForNode, WhileNode)):
            _check_one_loop_invariant(stmt, profiling, suggestions)


# Pattern 6 — String concatenation in loops
# Algorithm: anti-pattern matching


def detect_string_concat(
    ast: ProgramNode,
    profiling: Optional[ProfilingData] = None,
    symbol_table: Optional[Dict[str, Any]] = None,
) -> List[Suggestion]:
    """
    Detect string += inside a loop body (O(n²) string copies).

    Detected when RHS is a StringNode, a str() call, or the target
    variable is confirmed as a string via symbol_table.

    Severity scales with iteration count:
        count < STR_CONCAT_MEDIUM_COUNT  → low
        count < STR_CONCAT_HIGH_COUNT    → medium
        count >= STR_CONCAT_HIGH_COUNT   → high
    """
    suggestions: List[Suggestion] = []
    _scan_str_concat(ast, profiling, symbol_table, suggestions)
    return suggestions


def _scan_str_concat(
    node: ASTNode,
    profiling: Optional[ProfilingData],
    symbol_table: Optional[Dict[str, Any]],
    suggestions: List[Suggestion],
) -> None:
    for field in dataclasses.fields(node):
        value = getattr(node, field.name)
        children: List[ASTNode] = (
            [value]
            if isinstance(value, ASTNode)
            else (
                [v for v in value if isinstance(v, ASTNode)]
                if isinstance(value, list)
                else []
            )
        )
        for child in children:
            if isinstance(child, (ForNode, WhileNode)):
                _check_loop_str_concat(child, profiling, symbol_table, suggestions)
            else:
                _scan_str_concat(child, profiling, symbol_table, suggestions)


def _check_loop_str_concat(
    loop: Union[ForNode, WhileNode],
    profiling: Optional[ProfilingData],
    symbol_table: Optional[Dict[str, Any]],
    suggestions: List[Suggestion],
) -> None:
    for stmt in loop.body:
        if not (isinstance(stmt, AugmentedAssignmentNode) and stmt.operator == "+="):
            continue
        rhs_string = isinstance(stmt.value, StringNode)
        rhs_str_call = (
            isinstance(stmt.value, FunctionCallNode)
            and isinstance(stmt.value.function, IdentifierNode)
            and stmt.value.function.name == "str"
        )
        target_string = symbol_table is not None and isinstance(
            symbol_table.get(stmt.target.name), str
        )
        if not (rhs_string or rhs_str_call or target_string):
            continue

        count = _get_count(profiling, stmt.line) if profiling else 0
        if count >= STR_CONCAT_HIGH_COUNT:
            severity, impact = "high", 15.0
        elif count >= STR_CONCAT_MEDIUM_COUNT:
            severity, impact = "medium", 10.0
        else:
            severity, impact = "low", 5.0

        suggestions.append(
            _make(
                pattern_id="string_concat_loop",
                severity=severity,
                impact_score=impact,
                line=stmt.line,
                description=(
                    f"String concatenation with '+=' on '{stmt.target.name}' "
                    f"inside a loop creates O(n²) string copies"
                ),
                suggestion=(
                    f"Collect parts in a list and join after the loop. "
                    f"Example: append inside the loop, then "
                    f"'{stmt.target.name} = \"\".join(parts)' afterwards."
                ),
            )
        )

    for stmt in loop.body:
        if isinstance(stmt, (ForNode, WhileNode)):
            _check_loop_str_concat(stmt, profiling, symbol_table, suggestions)


# Pattern 7 — Nested loops
# Algorithm: nesting depth tracking + execution count scaling


def detect_nested_loops(
    ast: ProgramNode,
    profiling: Optional[ProfilingData] = None,
    symbol_table: Optional[Dict[str, Any]] = None,
) -> List[Suggestion]:
    """
    Detect loops nested inside other loops.

    Static: recursive DFS tracks nesting depth. Any loop at depth > 0 is nested.
    Dynamic: probe innermost body execution_count to scale severity:
        count >= NESTED_HIGH_COUNT or depth >= 2  → high
        count >= NESTED_MEDIUM_COUNT              → medium
        otherwise                                 → low
    """
    suggestions: List[Suggestion] = []
    _find_nested(ast, depth=0, profiling=profiling, suggestions=suggestions)
    return suggestions


def _find_nested(
    node: ASTNode,
    depth: int,
    profiling: Optional[ProfilingData],
    suggestions: List[Suggestion],
) -> None:
    for field in dataclasses.fields(node):
        value = getattr(node, field.name)
        children: List[ASTNode] = (
            [value]
            if isinstance(value, ASTNode)
            else (
                [v for v in value if isinstance(v, ASTNode)]
                if isinstance(value, list)
                else (
                    [v for v in value if isinstance(v, ASTNode)]
                    if isinstance(value, tuple)
                    else []
                )
            )
        )
        for child in children:
            if isinstance(child, (ForNode, WhileNode)):
                if depth > 0:
                    max_count = _innermost_count(child, profiling)
                    if max_count >= NESTED_HIGH_COUNT or depth >= 2:
                        severity, impact = "high", 18.0
                    elif max_count >= NESTED_MEDIUM_COUNT:
                        severity, impact = "medium", 12.0
                    else:
                        severity, impact = "low", 6.0
                    suggestions.append(
                        _make(
                            pattern_id="nested_loops",
                            severity=severity,
                            impact_score=impact,
                            line=child.line,
                            description=(
                                f"Nested loop at depth {depth + 1} "
                                f"(line {child.line})"
                            ),
                            suggestion=(
                                "Consider restructuring to reduce nesting. Options: "
                                "use a dictionary lookup, precompute values outside "
                                "the inner loop, or extract to a separate function."
                            ),
                        )
                    )
                _find_nested(child, depth + 1, profiling, suggestions)
            else:
                _find_nested(child, depth, profiling, suggestions)


def _innermost_count(
    loop: Union[ForNode, WhileNode],
    profiling: Optional[ProfilingData],
) -> int:
    body = loop.body
    nested = [s for s in body if isinstance(s, (ForNode, WhileNode))]
    if nested:
        return _innermost_count(nested[0], profiling)
    if profiling is None or not body:
        return 0
    return max((_get_count(profiling, s.line) for s in body), default=0)


# Pattern 8 — Hot loop detection
# Algorithm: execution frequency thresholding


def detect_hot_loops(
    ast: ProgramNode,
    profiling: Optional[ProfilingData] = None,
    symbol_table: Optional[Dict[str, Any]] = None,
) -> List[Suggestion]:
    """
    Detect loops whose body executes significantly above the program average.

    hot_threshold = max(mean(all counts) × HOT_MULTIPLIER, MIN_HOT_COUNT)

    A loop is hot when max(body_line_counts) > hot_threshold.
    This catches flat loops with high iteration counts that nested-loop
    detection would miss.

    Severity:
        count > hot_threshold × 10  → high
        count > hot_threshold       → medium
    """
    if profiling is None:
        return []

    counts = [
        s.execution_count
        for s in profiling.line_stats.values()
        if s.execution_count > 0
    ]
    if not counts:
        return []

    mean_count = sum(counts) / len(counts)
    hot_threshold = max(mean_count * HOT_MULTIPLIER, MIN_HOT_COUNT)

    suggestions: List[Suggestion] = []
    reported: Set[int] = set()

    for node in _walk(ast):
        if not isinstance(node, (ForNode, WhileNode)) or node.line in reported:
            continue
        body_counts = [
            _get_count(profiling, s.line) for s in node.body if hasattr(s, "line")
        ]
        if not body_counts:
            continue
        max_body = max(body_counts)
        if max_body <= hot_threshold:
            continue
        reported.add(node.line)
        severity = "high" if max_body > hot_threshold * 10 else "medium"
        impact = 18.0 if severity == "high" else 10.0
        suggestions.append(
            _make(
                pattern_id="hot_loop",
                severity=severity,
                impact_score=impact,
                line=node.line,
                description=(
                    f"Hot loop at line {node.line}: body executed {max_body:,} times "
                    f"({max_body / max(mean_count, 1):.0f}× above average)"
                ),
                suggestion=(
                    "This loop dominates execution time. Consider: "
                    "(1) reducing iterations, (2) moving invariant computations above "
                    "the loop, (3) using more efficient data structures, or "
                    "(4) caching repeated results."
                ),
            )
        )

    return suggestions


# Pattern 9 — Repeated computation
# Algorithm: expression fingerprinting


def detect_repeated_computation(
    ast: ProgramNode,
    profiling: Optional[ProfilingData] = None,
    symbol_table: Optional[Dict[str, Any]] = None,
) -> List[Suggestion]:
    """
    Detect the same non-trivial expression computed more than once with
    no intervening write to any variable it reads.

    Steps:
        1. Assign a canonical fingerprint string to every compound expression.
        2. Group by fingerprint; find pairs at different lines.
        3. For each pair (L1, L2): check no variable in the expression was
           reassigned between L1 and L2 (intervening write check).
        4. If profiling available, require execution_count > REPEAT_COUNT_THRESHOLD.
    """
    fingerprint_sites: Dict[str, List[Tuple[int, ASTNode]]] = {}
    for node in _walk(ast):
        if not _nontrivial(node):
            continue
        fp = _fp(node)
        fingerprint_sites.setdefault(fp, []).append((node.line, node))

    write_map: Dict[str, List[int]] = {}
    for node in _walk(ast):
        if isinstance(node, (AssignmentNode, AugmentedAssignmentNode)):
            write_map.setdefault(node.target.name, []).append(node.line)

    suggestions: List[Suggestion] = []
    reported: Set[str] = set()

    for fp, occurrences in fingerprint_sites.items():
        if len(occurrences) < 2:
            continue
        sorted_occ = sorted(occurrences, key=lambda x: x[0])
        for i in range(len(sorted_occ) - 1):
            line1, node1 = sorted_occ[i]
            line2, _ = sorted_occ[i + 1]
            if line1 == line2:
                continue
            key = f"{fp}:{line1}:{line2}"
            if key in reported:
                continue
            vars_in_expr: Set[str] = {
                n.name for n in _walk(node1) if isinstance(n, IdentifierNode)
            }
            # Intervening write check
            if any(
                line1 < wline < line2
                for name in vars_in_expr
                for wline in write_map.get(name, [])
            ):
                continue
            if profiling is not None:
                if (
                    _get_count(profiling, line1) < REPEAT_COUNT_THRESHOLD
                    and _get_count(profiling, line2) < REPEAT_COUNT_THRESHOLD
                ):
                    continue
            reported.add(key)
            suggestions.append(
                _make(
                    pattern_id="repeated_computation",
                    severity="medium",
                    impact_score=8.0,
                    line=line2,
                    description=(
                        f"Expression '{fp}' computed at line {line1} is repeated "
                        f"at line {line2} with no change to its variables"
                    ),
                    suggestion=(
                        f"Store the result after line {line1} and reuse it at "
                        f"line {line2}. Example: 'cached = {fp}'"
                    ),
                )
            )

    return suggestions


def _nontrivial(node: ASTNode) -> bool:
    if isinstance(node, BinaryOpNode):
        return not (
            isinstance(node.left, _LITERAL_TYPES)
            and isinstance(node.right, _LITERAL_TYPES)
        )
    if isinstance(node, UnaryOpNode):
        return not isinstance(node.operand, _LITERAL_TYPES)
    if isinstance(node, FunctionCallNode):
        return len(node.arguments) > 0
    return False


def _fp(node: ASTNode) -> str:
    if isinstance(node, NumberNode):
        return str(node.value)
    if isinstance(node, StringNode):
        return repr(node.value)
    if isinstance(node, BooleanNode):
        return str(node.value)
    if isinstance(node, IdentifierNode):
        return node.name
    if isinstance(node, BinaryOpNode):
        return f"({_fp(node.left)}{node.operator}{_fp(node.right)})"
    if isinstance(node, UnaryOpNode):
        return f"({node.operator}{_fp(node.operand)})"
    if isinstance(node, FunctionCallNode):
        return f"{node.function.name}({','.join(_fp(a) for a in node.arguments)})"
    return "?"


# Pattern 10 — Expensive function calls
# Algorithm: call frequency analysis


def detect_expensive_calls(
    ast: ProgramNode,
    profiling: Optional[ProfilingData] = None,
    symbol_table: Optional[Dict[str, Any]] = None,
) -> List[Suggestion]:
    """
    Detect frequently called slow functions, especially when called inside loops.

    From profiling.function_stats, identify functions where:
        call_count >= CALL_COUNT_THRESHOLD
        avg_time_ms >= AVG_TIME_THRESHOLD_MS

    Then find their call sites in the AST and check whether each site is
    inside a loop body (loop_lines set). Loop calls get "high" severity.
    """
    if profiling is None:
        return []

    expensive = {
        name: {
            "call_count": s.call_count,
            "avg_time_ms": s.avg_time_ms,
            "total_time_ms": s.total_time_ms,
        }
        for name, s in profiling.function_stats.items()
        if s.call_count >= CALL_COUNT_THRESHOLD
        and s.avg_time_ms >= AVG_TIME_THRESHOLD_MS
    }
    if not expensive:
        return []

    call_sites: Dict[str, List[int]] = {}
    for node in _walk(ast):
        if isinstance(node, FunctionCallNode):
            call_sites.setdefault(node.function.name, []).append(node.line)

    loop_lines: Set[int] = set()
    for node in _walk(ast):
        if isinstance(node, (ForNode, WhileNode)):
            for stmt in node.body:
                for inner in _walk(stmt):
                    if hasattr(inner, "line"):
                        loop_lines.add(inner.line)

    suggestions: List[Suggestion] = []
    reported: Set[str] = set()

    for func_name, stats in expensive.items():
        for line in call_sites.get(func_name, []):
            key = f"{func_name}:{line}"
            if key in reported:
                continue
            reported.add(key)
            in_loop = line in loop_lines

            if in_loop:
                severity, impact = "high", 18.0
                description = (
                    f"Slow function '{func_name}' called inside a loop "
                    f"at line {line} "
                    f"(avg {stats['avg_time_ms']:.2f}ms × "
                    f"{stats['call_count']} calls = "
                    f"{stats['total_time_ms']:.1f}ms total)"
                )
                suggestion = (
                    f"'{func_name}' is called repeatedly inside a loop. "
                    f"Cache its result if inputs do not change between iterations, "
                    f"or compute it once before the loop."
                )
            elif stats["total_time_ms"] > 50.0:
                severity, impact = "medium", 12.0
                description = (
                    f"Frequently called slow function '{func_name}' "
                    f"(avg {stats['avg_time_ms']:.2f}ms, {stats['call_count']} calls)"
                )
                suggestion = (
                    f"'{func_name}' accounts for significant execution time. "
                    f"Consider memoizing results if called with repeated arguments."
                )
            else:
                severity, impact = "low", 6.0
                description = (
                    f"Function '{func_name}' called {stats['call_count']} times "
                    f"(avg {stats['avg_time_ms']:.2f}ms per call)"
                )
                suggestion = (
                    f"Monitor '{func_name}' as call count grows. "
                    f"Consider caching if called with repeated arguments."
                )

            suggestions.append(
                _make(
                    pattern_id="expensive_calls",
                    severity=severity,
                    impact_score=impact,
                    line=line,
                    description=description,
                    suggestion=suggestion,
                )
            )

    return suggestions


# Optimizer orchestrator

_DETECTORS = [
    detect_unused_vars,
    detect_dead_code,
    detect_constant_folding,
    detect_early_return,
    detect_loop_invariant,
    detect_string_concat,
    detect_nested_loops,
    detect_hot_loops,
    detect_repeated_computation,
    detect_expensive_calls,
]


class Optimizer:
    """
    Runs all 10 pattern detectors and assembles an OptimizationReport.

    Args:
        ast:          ProgramNode from the parser (required).
        profiling:    ProfilingData from the profiler (optional).
        symbol_table: Final variable state from the executor (optional).
        detectors:    Override the detector list (for targeted testing).

    Example::

        result = execute(source)
        ast = parse(tokenize(source))
        report = Optimizer(ast, result.profiling, result.symbol_table).run()
    """

    def __init__(
        self,
        ast: ProgramNode,
        profiling: Optional[ProfilingData] = None,
        symbol_table: Optional[Dict[str, Any]] = None,
        detectors: Optional[List] = None,
    ) -> None:
        self._ast = ast
        self._profiling = profiling
        self._symbol_table = symbol_table
        self._detectors = detectors if detectors is not None else _DETECTORS

    def run(self) -> OptimizationReport:
        """
        Execute all detectors and return a complete OptimizationReport.
        Suggestions are sorted by impact_score descending.
        Individual detector exceptions are silently swallowed so one
        broken pattern cannot prevent the rest from running.
        """
        all_suggestions: List[Suggestion] = []
        for detector in self._detectors:
            try:
                all_suggestions.extend(
                    detector(self._ast, self._profiling, self._symbol_table)
                )
            except Exception as exc:
                _log.warning(
                    "Detector %s failed: %s",
                    detector.__name__,
                    exc,
                    exc_info=True,
                )
        all_suggestions.sort(key=lambda s: s.impact_score, reverse=True)
        return OptimizationReport(suggestions=all_suggestions)


# Public convenience functions


def analyze(
    ast: ProgramNode,
    profiling: Optional[ProfilingData] = None,
    symbol_table: Optional[Dict[str, Any]] = None,
) -> OptimizationReport:
    """
    Run all optimization patterns and return an OptimizationReport.

    Primary entry point when you already have the parsed AST and results.

    Args:
        ast:          ProgramNode from parse(tokenize(source)).
        profiling:    ProfilingData from execute(source).profiling.
        symbol_table: Dict from execute(source).symbol_table.

    Returns:
        OptimizationReport with suggestions sorted by impact_score.
    """
    return Optimizer(ast, profiling, symbol_table).run()


def analyze_source(source: str) -> OptimizationReport:
    """
    Full pipeline convenience: tokenize → parse → execute → analyze.

    The function the FastAPI interpreter service calls directly.

    Args:
        source: PyLite source code string.

    Returns:
        OptimizationReport with all suggestions.

    Example::

        from optilang.optimizer import analyze_source

        report = analyze_source(\"\"\"
        unused = 999
        for i in range(100):
            for j in range(100):
                x = i * j
        print(x)
        \"\"\")
        for s in report.suggestions:
            print(f"Line {s.line} [{s.severity}]: {s.description}")
    """
    from .executor import execute as _execute

    result = _execute(source)
    ast = result.ast
    if ast is None:
        return OptimizationReport(suggestions=[])
    return analyze(ast, result.profiling, result.symbol_table)
