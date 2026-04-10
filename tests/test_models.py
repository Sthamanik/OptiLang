"""
Tests for OptiLang data models (Sprint 2 — updated for v0.4.0)

Changes from previous version:
    OptimizationReport no longer carries scoring fields.
    The following fields were removed in v0.4.0:
        - optimization_score   (moved to ScoreReport.score)
        - score_breakdown      (moved to ScoreReport.dimensions)
        - complexity_analysis  (moved to ScoreReport.complexity_class)

Covers:
    - ExecutionResult: fields, defaults, profiling integration
    - Suggestion: fields and severity values
    - OptimizationReport: only holds suggestions; removed fields are absent
"""

from __future__ import annotations

import pytest

from optilang.models import ExecutionResult, OptimizationReport, Suggestion
from optilang.profiler import ProfilingData
from optilang.token import Token, TokenType

# ---------------------------------------------------------------------------
# Unit Tests: ExecutionResult
# ---------------------------------------------------------------------------


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
        # Two instances must not share the same errors list
        r1 = ExecutionResult(output="")
        r2 = ExecutionResult(output="")
        r1.errors.append("error")
        assert r2.errors == []

    def test_symbol_table_is_independent(self) -> None:
        r1 = ExecutionResult(output="")
        r2 = ExecutionResult(output="")
        r1.symbol_table["x"] = 1
        assert "x" not in r2.symbol_table


# ---------------------------------------------------------------------------
# Unit Tests: Suggestion
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Unit Tests: OptimizationReport
# ---------------------------------------------------------------------------


class TestOptimizationReport:
    """
    Tests for the OptimizationReport dataclass.

    As of v0.4.0, OptimizationReport only holds a list of Suggestion objects.
    Scoring data (score, breakdown, complexity) lives in ScoreReport.
    """

    # ── Structure ────────────────────────────────────────────────────────

    def test_default_suggestions_empty(self) -> None:
        report = OptimizationReport()
        assert report.suggestions == []

    def test_suggestions_list_is_independent(self) -> None:
        # Two instances must not share the same suggestions list
        r1 = OptimizationReport()
        r2 = OptimizationReport()
        r1.suggestions.append(Suggestion(1, "p", "low", "d", "s", 1.0))
        assert r2.suggestions == []

    def test_only_field_is_suggestions(self) -> None:
        # OptimizationReport must expose exactly one public data field
        report = OptimizationReport()
        fields = [f for f in vars(report) if not f.startswith("_")]
        assert fields == ["suggestions"]

    # ── Removed fields are gone ───────────────────────────────────────────

    def test_optimization_score_field_does_not_exist(self) -> None:
        report = OptimizationReport()
        assert not hasattr(
            report, "optimization_score"
        ), "use ScoreReport.score"

    def test_score_breakdown_field_does_not_exist(self) -> None:
        report = OptimizationReport()
        assert not hasattr(
            report, "score_breakdown"
        ), "use ScoreReport.dimensions"

    def test_complexity_analysis_field_does_not_exist(self) -> None:
        report = OptimizationReport()
        assert not hasattr(
            report, "complexity_analysis"
        ), "use ScoreReport.complexity_class"

    def test_constructor_rejects_optimization_score_kwarg(self) -> None:
        with pytest.raises(TypeError):
            OptimizationReport(optimization_score=95.0)  # type: ignore[call-arg]

    def test_constructor_rejects_score_breakdown_kwarg(self) -> None:
        with pytest.raises(TypeError):
            OptimizationReport(score_breakdown={"x": 1.0})  # type: ignore[call-arg]

    def test_constructor_rejects_complexity_analysis_kwarg(self) -> None:
        with pytest.raises(TypeError):
            OptimizationReport(
                complexity_analysis={"class": "O(n)"})  # type: ignore[call-arg]

    # ── Suggestions behaviour ─────────────────────────────────────────────

    def test_with_single_suggestion(self) -> None:
        s = Suggestion(
            line=1,
            pattern="unused_vars",
            severity="low",
            description="Unused variable",
            suggestion="Remove it",
            impact_score=2.0,
        )
        report = OptimizationReport(suggestions=[s])
        assert len(report.suggestions) == 1
        assert report.suggestions[0].pattern == "unused_vars"

    def test_with_multiple_suggestions(self) -> None:
        suggestions = [
            Suggestion(1, "hot_loop", "high", "d", "s", 18.0),
            Suggestion(2, "dead_code", "medium", "d", "s", 7.0),
            Suggestion(3, "unused_vars", "low", "d", "s", 3.0),
        ]
        report = OptimizationReport(suggestions=suggestions)
        assert len(report.suggestions) == 3

    def test_suggestions_preserve_order(self) -> None:
        suggestions = [
            Suggestion(3, "hot_loop", "high", "d", "s", 18.0),
            Suggestion(1, "unused_vars", "low", "d", "s", 3.0),
            Suggestion(2, "dead_code", "medium", "d", "s", 7.0),
        ]
        report = OptimizationReport(suggestions=suggestions)
        assert [s.line for s in report.suggestions] == [3, 1, 2]

    def test_suggestions_severity_values(self) -> None:
        for severity in ("low", "medium", "high"):
            s = Suggestion(1, "p", severity, "d", "s", 1.0)
            report = OptimizationReport(suggestions=[s])
            assert report.suggestions[0].severity == severity

    def test_append_suggestion_after_construction(self) -> None:
        report = OptimizationReport()
        report.suggestions.append(Suggestion(5, "nested_loops", "high", "d", "s", 12.0))
        assert len(report.suggestions) == 1
        assert report.suggestions[0].pattern == "nested_loops"

    def test_suggestions_can_be_filtered_by_severity(self) -> None:
        suggestions = [
            Suggestion(1, "hot_loop", "high", "d", "s", 18.0),
            Suggestion(2, "dead_code", "medium", "d", "s", 7.0),
            Suggestion(3, "unused_vars", "low", "d", "s", 3.0),
        ]
        report = OptimizationReport(suggestions=suggestions)
        high_only = [s for s in report.suggestions if s.severity == "high"]
        assert len(high_only) == 1
        assert high_only[0].pattern == "hot_loop"

    def test_suggestions_can_be_sorted_by_impact_score(self) -> None:
        suggestions = [
            Suggestion(3, "unused_vars", "low", "d", "s", 3.0),
            Suggestion(1, "hot_loop", "high", "d", "s", 18.0),
            Suggestion(2, "dead_code", "medium", "d", "s", 7.0),
        ]
        report = OptimizationReport(suggestions=suggestions)
        sorted_s = sorted(
            report.suggestions, key=lambda s: s.impact_score, reverse=True
        )
        assert sorted_s[0].pattern == "hot_loop"
        assert sorted_s[-1].pattern == "unused_vars"


class TestTokenRepr:

    def test_token_repr_includes_type_value_and_position(self) -> None:
        token = Token(TokenType.NUMBER, 42, 3, 7)
        assert repr(token) == "Token(TokenType.NUMBER, 42, 3:7)"
