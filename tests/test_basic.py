"""
Basic tests to verify setup.
"""

import pytest
from optilang.models import ExecutionResult, Suggestion, OptimizationReport


def test_execution_result_creation():
    """Test ExecutionResult can be created."""
    result = ExecutionResult(
        output="Hello, World!",
        errors=[],
        execution_time=0.5
    )
    assert result.output == "Hello, World!"
    assert result.errors == []
    assert result.execution_time == 0.5


def test_suggestion_creation():
    """Test Suggestion can be created."""
    suggestion = Suggestion(
        line=5,
        pattern="nested_loops",
        severity="high",
        description="Nested loops detected",
        suggestion="Consider using dictionary lookup",
        impact_score=18.5
    )
    assert suggestion.line == 5
    assert suggestion.severity == "high"


def test_optimization_report_creation():
    """Test OptimizationReport can be created."""
    report = OptimizationReport(
        suggestions=[],
        optimization_score=100.0,
        score_breakdown={},
        complexity_analysis={}
    )
    assert report.optimization_score == 100.0
    assert len(report.suggestions) == 0