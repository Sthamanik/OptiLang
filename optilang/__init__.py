"""
OptiLang - A Python-inspired interpreter with real-time code analysis
and optimization suggestions.
"""

from __future__ import annotations

from .executor import Executor, execute
from .models import ExecutionResult, OptimizationReport, Suggestion
from .parser import parse
from .scoring import (
    DynamicScorer,
    Scorer,
    ScoreReport,
    calculate_full_score,
    calculate_score,
)

__version__ = "0.3.0"

__all__ = [
    "execute",
    "parse",
    "calculate_score",
    "calculate_full_score",
    "Executor",
    "Scorer",
    "DynamicScorer",
    "ExecutionResult",
    "OptimizationReport",
    "Suggestion",
    "ScoreReport",
]
