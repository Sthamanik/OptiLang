"""
Data models for OptiLang.

Scoring data is no longer stored on OptimizationReport.
Use ScoreReport (from optilang.scoring) for all score-related data.

Pipeline overview:
    execute(source)         → ExecutionResult   (output, errors, profiling)
    Optimizer(...).run()    → OptimizationReport (suggestions only)
    Scorer(...).calculate() → ScoreReport        (score, dimensions, narrative)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .profiler import ProfilingData


@dataclass
class ExecutionResult:
    """Result of code execution."""

    output: str
    errors: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    profiling: Optional[ProfilingData] = None
    symbol_table: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Suggestion:
    """A single optimization suggestion."""

    line: int
    pattern: str
    severity: str  # "low" | "medium" | "high"
    description: str
    suggestion: str
    impact_score: float


@dataclass
class OptimizationReport:
    """
    Output of the Optimizer — a ranked list of optimization suggestions.

    Scoring (final score, dimension breakdown, narrative) is handled
    separately by Scorer and returned as a ScoreReport. Do not add
    scoring fields here.
    """

    suggestions: List[Suggestion] = field(default_factory=list)
