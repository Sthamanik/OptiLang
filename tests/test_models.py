"""
Tests for OptiLang data models (Sprint 2)

Covers:
- ExecutionResult: fields, defaults, profiling integration
- Suggestion: fields and severity values
- OptimizationReport: defaults and structure
"""

from __future__ import annotations
from optilang.models import ExecutionResult, OptimizationReport, Suggestion
from optilang.profiler import ProfilingData

#  Unit Tests: ExecutionResult


class TestExecutionResult:
    """Tests for the ExecutionResult dataclass."""

    def test_required_field_output(self) -> None:
        result = ExecutionResult(output="hello")
        assert result.output == "hello"

    def test_default_errors_empty(self) -> None:
        result = ExecutionResult(output="")
        assert result.errors == []

    def test_default_execution_time_zero(self) -> None:
        result = ExecutionResult(output="")
        assert result.execution_time == 0.0

    def test_default_profiling_none(self) -> None:
        result = ExecutionResult(output="")
        assert result.profiling is None

    def test_default_symbol_table_empty(self) -> None:
        result = ExecutionResult(output="")
        assert result.symbol_table == {}

    def test_with_errors(self) -> None:
        result = ExecutionResult(output="", errors=["NameError: x not defined"])
        assert len(result.errors) == 1
        assert "NameError" in result.errors[0]

    def test_with_execution_time(self) -> None:
        result = ExecutionResult(output="", execution_time=1.23)
        assert result.execution_time == 1.23

    def test_with_symbol_table(self) -> None:
        result = ExecutionResult(output="", symbol_table={"x": 42})
        assert result.symbol_table["x"] == 42

    def test_with_profiling_data(self) -> None:
        profiling = ProfilingData()
        result = ExecutionResult(output="", profiling=profiling)
        assert result.profiling is not None
        assert isinstance(result.profiling, ProfilingData)

    def test_errors_list_is_independent(self) -> None:
        # Two instances should not share the same errors list
        r1 = ExecutionResult(output="")
        r2 = ExecutionResult(output="")
        r1.errors.append("error")
        assert r2.errors == []

    def test_symbol_table_is_independent(self) -> None:
        r1 = ExecutionResult(output="")
        r2 = ExecutionResult(output="")
        r1.symbol_table["x"] = 1
        assert "x" not in r2.symbol_table


#  Unit Tests: Suggestion


class TestSuggestion:
    """Tests for the Suggestion dataclass."""

    def _make_suggestion(self, severity: str = "medium") -> Suggestion:
        return Suggestion(
            line=5,
            pattern="nested_loops",
            severity=severity,
            description="Nested loops detected",
            suggestion="Consider using a lookup table",
            impact_score=15.0,
        )

    def test_all_fields_set(self) -> None:
        s = self._make_suggestion()
        assert s.line == 5
        assert s.pattern == "nested_loops"
        assert s.severity == "medium"
        assert s.description == "Nested loops detected"
        assert s.suggestion == "Consider using a lookup table"
        assert s.impact_score == 15.0

    def test_severity_high(self) -> None:
        s = self._make_suggestion(severity="high")
        assert s.severity == "high"

    def test_severity_low(self) -> None:
        s = self._make_suggestion(severity="low")
        assert s.severity == "low"

    def test_impact_score_float(self) -> None:
        s = self._make_suggestion()
        assert isinstance(s.impact_score, float)


#  Unit Tests: OptimizationReport


class TestOptimizationReport:
    """Tests for the OptimizationReport dataclass."""

    def test_default_suggestions_empty(self) -> None:
        report = OptimizationReport()
        assert report.suggestions == []

    def test_default_score_100(self) -> None:
        report = OptimizationReport()
        assert report.optimization_score == 100.0

    def test_default_breakdown_empty(self) -> None:
        report = OptimizationReport()
        assert report.score_breakdown == {}

    def test_default_complexity_analysis_empty(self) -> None:
        report = OptimizationReport()
        assert report.complexity_analysis == {}

    def test_with_suggestions(self) -> None:
        s = Suggestion(
            line=1,
            pattern="unused_vars",
            severity="low",
            description="Unused variable",
            suggestion="Remove it",
            impact_score=2.0,
        )
        report = OptimizationReport(suggestions=[s], optimization_score=95.0)
        assert len(report.suggestions) == 1
        assert report.optimization_score == 95.0

    def test_with_score_breakdown(self) -> None:
        report = OptimizationReport(
            score_breakdown={"severity_penalty": 2.0, "complexity_penalty": 3.0}
        )
        assert report.score_breakdown["severity_penalty"] == 2.0

    def test_suggestions_list_is_independent(self) -> None:
        r1 = OptimizationReport()
        r2 = OptimizationReport()
        r1.suggestions.append(Suggestion(1, "p", "low", "d", "s", 1.0))
        assert r2.suggestions == []
