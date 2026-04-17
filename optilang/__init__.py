"""
OptiLang - A Python-inspired interpreter with real-time code analysis
and optimization suggestions.

Public API
----------
Core pipeline:
    execute(source)                 → ExecutionResult
    parse(tokens)                   → ProgramNode
    analyze(ast, profiling, syms)   → OptimizationReport
    analyze_source(source)          → OptimizationReport
    calculate_score(...)            → ScoreReport

Classes:
    Executor        — AST tree-walking interpreter
    Optimizer       — optimization pattern detector
    Scorer          — four-dimension scoring engine

Result models (useful for type hints):
    ExecutionResult     — output, errors, profiling, symbol table
    OptimizationReport  — ranked list of Suggestion objects
    Suggestion          — single optimization finding
    ScoreReport         — final score, grade, narrative, dimension breakdown
    DimensionScores     — per-dimension score breakdown with sub-scores
    ProfilingData       — execution profiling data (pass .to_dict() to Scorer)

Pattern classification constants (useful for external tools):
    EFFICIENCY_PATTERNS       — hot_loop, loop_invariant, repeated_computation,
                                expensive_calls
    QUALITY_PATTERNS          — dead_code, string_concat_loop
    MAINTAINABILITY_PATTERNS  — unused_vars, early_return, nested_loops,
                                constant_folding
"""

from __future__ import annotations

from .executor import Executor, execute
from .models import ExecutionResult, OptimizationReport, Suggestion
from .optimizer import Optimizer, analyze, analyze_source
from .parser import parse
from .profiler import ProfilingData
from .scoring import (
    EFFICIENCY_PATTERNS,
    MAINTAINABILITY_PATTERNS,
    QUALITY_PATTERNS,
    DimensionScores,
    Scorer,
    ScoreReport,
    calculate_score,
)

__version__ = "2.0.0"

__all__ = [
    # ── Core pipeline functions ──
    "execute",
    "parse",
    "analyze",
    "analyze_source",
    "calculate_score",
    # ── Classes ──
    "Executor",
    "Optimizer",
    "Scorer",
    # ── Result models ──
    "ExecutionResult",
    "OptimizationReport",
    "Suggestion",
    "ScoreReport",
    "DimensionScores",
    "ProfilingData",
    # ── Pattern classification constants ──
    "EFFICIENCY_PATTERNS",
    "QUALITY_PATTERNS",
    "MAINTAINABILITY_PATTERNS",
    # ── Package version ──
    "__version__",
]
