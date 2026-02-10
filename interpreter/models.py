"""
Data models for OptiLang.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional


@dataclass
class ExecutionResult:
    """Result of code execution."""
    output: str
    errors: List[str] = field(default_factory=list)
    execution_time: float = 0.0
    profiling: Optional[Dict[str, Any]] = None


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
    """Complete optimization analysis report."""
    suggestions: List[Suggestion] = field(default_factory=list)
    optimization_score: float = 100.0
    score_breakdown: Dict[str, float] = field(default_factory=dict)
    complexity_analysis: Dict[str, Any] = field(default_factory=dict)
