"""Feature extraction from OptiLang pipeline result objects."""

from __future__ import annotations

import dataclasses
import logging
from typing import Dict, List, Optional, Tuple

from optilang.ast_nodes import ASTNode, ForNode, ProgramNode, WhileNode
from optilang.models import ExecutionResult, OptimizationReport
from optilang.scoring import ScoreReport

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _source_line_count(source: str) -> int:
    return len(source.splitlines()) if source else 0


def _profiling_time_ms(result: ExecutionResult) -> float:
    if result.profiling is not None:
        return result.profiling.total_execution_time_ms
    logger.warning("profiling unavailable — falling back to result.execution_time")
    return result.execution_time * 1000.0


def _loop_context(ast: Optional[ProgramNode]) -> Dict[int, Tuple[int, bool]]:
    """
    Walk AST and map each source line to (loop_depth, is_inside_loop).
    loop_depth=0 means top level. Keeps maximum depth when lines overlap.
    """
    context: Dict[int, Tuple[int, bool]] = {}
    if ast is None:
        return context

    def visit(node: ASTNode, depth: int) -> None:
        line = getattr(node, "line", None)
        if isinstance(line, int):
            previous_depth, _ = context.get(line, (0, False))
            effective_depth = max(previous_depth, depth)
            context[line] = (effective_depth, effective_depth > 0)

        child_depth = depth + 1 if isinstance(node, (ForNode, WhileNode)) else depth
        for field in dataclasses.fields(node):
            value = getattr(node, field.name)
            if isinstance(value, ASTNode):
                visit(value, child_depth)
            elif isinstance(value, list):
                for item in value:
                    if isinstance(item, ASTNode):
                        visit(item, child_depth)
                    elif isinstance(item, tuple):
                        for element in item:
                            if isinstance(element, ASTNode):
                                visit(element, child_depth)

    visit(ast, 0)
    return context


def _get(manifest_row: Dict[str, str], key: str, default: str = "") -> str:
    """Safely retrieve a manifest field with optional default."""
    value = manifest_row.get(key, default)
    if not value:
        logger.debug("manifest missing field '%s' — using default '%s'", key, default)
    return value


_COMPLEXITY_ORDINAL: Dict[str, int] = {
    "O(1)":              1,
    "O(log n)":          2,
    "O(n)":              3,
    "O(n log n)":        4,
    "O(n^2)":            5,
    "O(n²)":             5,   # unicode superscript variant from scorer
    "O(n^k)":            5,   # generic polynomial — treat as n^2 tier
    "O(n^3) or worse":   6,
    "O(n³)":             6,   # unicode superscript variant from scorer
    "O(2^n)":            7,
}


def _complexity_ordinal(complexity_class: str) -> int:
    return _COMPLEXITY_ORDINAL.get(complexity_class, 0)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract(
    source: str,
    result: ExecutionResult,
    report: Optional[OptimizationReport],
    score: ScoreReport,
    manifest_row: Dict[str, str],
    execution_id: str,
    ast: Optional[ProgramNode] = None,
) -> List[Dict[str, object]]:
    """
    Convert one pipeline run into flat suggestion rows for executions.csv.

    Skips runs with errors — errored programs carry no useful ML signal.
    Returns empty list when errors exist or no suggestions are found.

    Parameters
    ----------
    source          Raw source string of the executed program.
    result          ExecutionResult from interpreter pipeline.
    report          OptimizationReport from analyzer. May be None.
    score           ScoreReport from scoring stage.
    manifest_row    Metadata dict — family, strategy, etc.
    execution_id    UUID string for this run.
    ast             Parsed AST root. Used for loop depth resolution. May be None.
    """
    if result.errors:
        logger.debug(
            "skipping errored execution for program_id=%s",
            manifest_row.get("program_id"),
        )
        return []

    raw_suggestions = list(report.suggestions) if report is not None else []

    # Deduplicate by (pattern, line) — the constant_folding detector can
    # visit the same AST node multiple times via different walk paths,
    # producing identical rows that would corrupt training data.
    seen_keys: set = set()
    suggestions = []
    for s in raw_suggestions:
        key = (s.pattern, s.line)
        if key not in seen_keys:
            seen_keys.add(key)
            suggestions.append(s)

    if not suggestions:
        return []

    source_lines      = _source_line_count(source)
    execution_time_ms = _profiling_time_ms(result)
    total_suggestions = len(suggestions)
    co_occurring      = "|".join(sorted({s.pattern for s in suggestions}))
    loop_context      = _loop_context(ast)

    # --- Program-level profiling ---
    profiling            = result.profiling
    total_lines_executed = profiling.total_lines_executed if profiling else 0
    peak_memory_bytes    = profiling.peak_memory_bytes if profiling else 0

    program_context: Dict[str, object] = {
        # Identifiers / ground truth
        "execution_id":       execution_id,
        "family":             _get(manifest_row, "family"),
        "strategy":           _get(manifest_row, "strategy"),
        # Program-level context
        "source_lines":       source_lines,
        "complexity_class":   score.complexity_class,
        "complexity_ordinal": _complexity_ordinal(score.complexity_class),
        "execution_time_ms":  round(execution_time_ms, 3),
        "peak_memory_bytes":  peak_memory_bytes,
        "total_suggestions":  total_suggestions,
    }

    rows: List[Dict[str, object]] = []
    for suggestion in suggestions:
        loop_depth, is_inside_loop = loop_context.get(suggestion.line, (0, False))

        # --- Line-level profiling features ---
        line_stat = profiling.line_stats.get(suggestion.line) if profiling else None
        execution_count_at_line = line_stat.execution_count if line_stat else 0
        avg_time_ms_at_line     = round(line_stat.avg_time_ms, 3) if line_stat else 0.0
        total_time_ms_at_line   = round(line_stat.total_time_ms, 3) if line_stat else 0.0
        line_dominance = (
            round(execution_count_at_line / total_lines_executed, 6)
            if total_lines_executed > 0
            else 0.0
        )

        # --- Relative position (0–1 normalized) ---
        relative_line_position = (
            round(suggestion.line / source_lines, 4) if source_lines > 0 else 0.0
        )

        # --- Function-level profiling features ---
        # FunctionStats don't store line ranges, so we pick the function
        # with the highest call_count as the best proxy for the dominant
        # call context. Module-level suggestions fall back to zeros.
        # A richer mapping (storing def-line → end-line) can replace this
        # once FunctionDefNode line ranges are tracked in the executor.
        function_call_count = 0
        max_recursion_depth = 0
        if profiling and profiling.function_stats:
            dominant = max(
                profiling.function_stats.values(),
                key=lambda fs: fs.call_count,
            )
            function_call_count = dominant.call_count
            max_recursion_depth = dominant.max_recursion_depth

        rows.append(
            {
                **program_context,
                # Core suggestion identity
                "pattern":                 suggestion.pattern,
                "severity":                suggestion.severity,
                "impact_score":            suggestion.impact_score,
                # Structural (AST)
                "loop_depth":              loop_depth,
                "is_inside_loop":          is_inside_loop,
                "relative_line_position":  relative_line_position,
                "co_occurring_patterns":   co_occurring,
                # Dynamic — line level
                "execution_count_at_line": execution_count_at_line,
                "avg_time_ms_at_line":     avg_time_ms_at_line,
                "total_time_ms_at_line":   total_time_ms_at_line,
                "line_dominance":          line_dominance,
                # Dynamic — function level
                "function_call_count":     function_call_count,
                "max_recursion_depth":     max_recursion_depth,
            }
        )

    return rows