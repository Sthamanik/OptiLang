"""
OptiLang - A Python-inspired interpreter with real-time code analysis
and optimization suggestions.
"""

from __future__ import annotations

from .executor import Executor, execute
from .models import ExecutionResult, OptimizationReport, Suggestion
from .optimizer import Optimizer, analyze, analyze_source
from .parser import parse
from .scoring import DimensionScores, Scorer, ScoreReport, calculate_score

__version__ = "1.0.0"

__all__ = [
    # Core functions
    "execute",
    "parse",
    "calculate_score",
    "analyze",
    "analyze_source",
    # Classes
    "Executor",
    "Optimizer",
    "Scorer",
    # Result models (useful for type hints)
    "ExecutionResult",
    "OptimizationReport",
    "Suggestion",
    "ScoreReport",
    "DimensionScores",
]
