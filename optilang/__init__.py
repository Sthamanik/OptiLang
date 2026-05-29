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

import sys
import importlib
from typing import Any

# Re-export core modules for backward compatibility
from .core import ast_nodes as ast_nodes

from .analysis.complexity import (
    Complexity,
    ComplexityExpr,
    ComplexityResult,
    analyze_complexity,
    analyze_function_complexity,
)
from .runtime.executor import Executor, execute
from .types.models import ExecutionResult, OptimizationReport, Suggestion
from .analysis.optimizer import Optimizer, analyze, analyze_source
from .core.parser import parse
from .runtime.profiler import (
    ProfilingData,
    FunctionStats,
    LineStats,
    Profiler,
    ProfilerConfig,
    _safe_getsizeof,
    _estimate_deep_object_size,
    detect_complexity,
    detect_complexity_with_confidence,
    estimate_memory_bytes,
    profile_execution,
)
from .analysis.scoring import (
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
    # ── Complexity analysis ──
    "analyze_complexity",
    "analyze_function_complexity",
    # ── Classes ──
    "Complexity",
    "ComplexityExpr",
    "ComplexityResult",
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
    "FunctionStats",
    "LineStats",
    "Profiler",
    "ProfilerConfig",
    "detect_complexity",
    "detect_complexity_with_confidence",
    "estimate_memory_bytes",
    "profile_execution",
    "_safe_getsizeof",
    "_estimate_deep_object_size",
    # ── Pattern classification constants ──
    "EFFICIENCY_PATTERNS",
    "QUALITY_PATTERNS",
    "MAINTAINABILITY_PATTERNS",
    # ── Package version ──
    "__version__",
]

# Backward compatibility: lazy loading via __getattr__
# This allows `from optilang.lexer import tokenize` to work
# even though lexer.py is now in optilang/core/
_module_map = {
    "lexer": "optilang.core.lexer",
    "parser": "optilang.core.parser",
    "token": "optilang.core.token",
    "ast_nodes": "optilang.core.ast_nodes",
    "executor": "optilang.runtime.executor",
    "optimizer": "optilang.analysis.optimizer",
    "scoring": "optilang.analysis.scoring",
    "complexity": "optilang.analysis.complexity",
    "semantic_analyzer": "optilang.analysis.semantic_analyzer",
    "models": "optilang.types.models",
    "constants": "optilang.types.constants",
}

# Special handling for profiler - use __getattr__ for import, CLI via runtime
_extra_imports = {"profiler": "optilang.runtime.profiler"}


def __getattr__(name: str) -> Any:
    if name in _module_map:
        return importlib.import_module(_module_map[name])
    if name in _extra_imports:
        return importlib.import_module(_extra_imports[name])
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# Also register in sys.modules for deeper imports (skip profiler due to CLI conflict)
for name, path in _module_map.items():
    if f"optilang.{name}" not in sys.modules:
        try:
            sys.modules[f"optilang.{name}"] = importlib.import_module(path)
        except ImportError:
            pass
