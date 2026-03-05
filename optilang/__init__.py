"""
OptiLang - A Python-inspired interpreter with real-time code analysis
and optimization suggestions.
"""

from __future__ import annotations

from .executor import Executor, execute
from .models import ExecutionResult, OptimizationReport, Suggestion
from .parser import parse
from .scoring import Scorer, ScoreReport, calculate_score

__version__ = "0.3.0"

__all__ = [
    # Core functions
    "execute",
    "parse",
    "calculate_score",
    # Classes
    "Executor",
    "Scorer",
    # Result models (useful for type hints)
    "ExecutionResult",
    "OptimizationReport",
    "Suggestion",
    "ScoreReport",
]
