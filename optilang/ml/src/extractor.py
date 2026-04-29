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
    """
    Return execution time in milliseconds.

    Prefers result.profiling when available. Falls back to result.execution_time
    and logs a warning so the caller is aware the value is less precise.
    """
    if result.profiling is not None:
        return result.profiling.total_execution_time_ms

    logger.warning(
        "profiling data unavailable — falling back to result.execution_time"
    )
    return result.execution_time * 1000.0


def _resolve_status(result: ExecutionResult, report: Optional[OptimizationReport]) -> str:
    if result.errors:
        return "error"
    if report is None:
        return "analysis_unavailable"
    return "ok"


def _loop_context(ast: Optional[ProgramNode]) -> Dict[int, Tuple[int, bool]]:
    """
    Walk the AST and map each source line to (loop_depth, is_inside_loop).

    loop_depth=0 means the line is at top level (not inside any loop).
    loop_depth=1 means directly inside one loop, and so on.
    When a line appears under multiple nodes (e.g. shared line numbers),
    the maximum depth seen is kept.
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


def _manifest_field(manifest_row: Dict[str, str], key: str) -> str:
    """Safely retrieve a manifest field, warning when absent."""
    value = manifest_row.get(key, "")
    if not value:
        logger.warning("manifest row missing expected field: '%s'", key)
    return value


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
    Convert one pipeline run into a list of flat suggestion rows for executions.csv.

    Each row is one suggestion combined with its execution-level context.
    Returns an empty list when the report has no suggestions.

    Parameters
    ----------
    source:         Raw source string of the program that was executed.
    result:         ExecutionResult from the interpreter pipeline.
    report:         OptimizationReport produced by the analyzer. May be None.
    score:          ScoreReport from the scoring stage.
    manifest_row:   Dict of metadata from the program manifest (program_id, variant, etc.).
    execution_id:   Unique ID for this execution run (caller is responsible for generating).
    ast:            Parsed AST root node. Used for loop depth resolution. May be None.
    """
    suggestions = list(report.suggestions) if report is not None else []

    # Execution-level scalars — computed once, shared across all suggestion rows
    source_lines = _source_line_count(source)
    execution_time_ms = _profiling_time_ms(result)
    error_count = len(result.errors)
    total_suggestions = len(suggestions)
    status = _resolve_status(result, report)
    co_occurring_patterns = "|".join(sorted({s.pattern for s in suggestions}))
    loop_context = _loop_context(ast)

    # Shared execution context embedded in every suggestion row
    execution_context: Dict[str, object] = {
        "execution_id":       execution_id,
        "program_id":         _manifest_field(manifest_row, "program_id"),
        "variant":            _manifest_field(manifest_row, "variant"),
        "family":             _manifest_field(manifest_row, "family"),
        "strategy":           _manifest_field(manifest_row, "strategy"),
        "expected_patterns":  _manifest_field(manifest_row, "patterns"),
        "pathological":       _manifest_field(manifest_row, "pathological"),
        "source_path":        _manifest_field(manifest_row, "source_path"),
        "source_lines":       source_lines,
        "complexity_class":   score.complexity_class,
        "error_count":        error_count,
        "execution_time_ms":  round(execution_time_ms, 3),
        "total_suggestions":  total_suggestions,
        "score":              round(score.score, 2),
        "grade":              score.grade,
        "status":             status,
    }

    suggestion_rows: List[Dict[str, object]] = []
    for suggestion in suggestions:
        loop_depth, is_inside_loop = loop_context.get(suggestion.line, (0, False))
        suggestion_rows.append(
            {
                **execution_context,
                "pattern":              suggestion.pattern,
                "severity":             suggestion.severity,
                "impact_score":         suggestion.impact_score,
                "line_number":          suggestion.line,
                "loop_depth":           loop_depth,
                "is_inside_loop":       is_inside_loop,
                "co_occurring_patterns": co_occurring_patterns,
            }
        )

    return suggestion_rows