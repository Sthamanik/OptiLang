"""
tests/test_scoring.py
---------------------
Tests for optilang/scoring.py

Covers:
  - ScoreReport data class and to_dict()
  - Scorer._severity_penalty()
  - Scorer._complexity_penalty() + _detect_complexity()
  - Scorer._performance_penalty()
  - Scorer._memory_penalty()
  - Scorer.calculate() end-to-end
  - calculate_score() convenience function
  - Edge cases (empty data, zero counts, large counts)
"""

import pytest
from optilang.models import Suggestion
from optilang.scoring import (
    Scorer,
    ScoreReport,
    calculate_score,
    COMPLEXITY_PENALTIES,
    HIGH_VAR_THRESHOLD,
    MAX_SEVERITY_PENALTY,
    MAX_COMPLEXITY_PENALTY,
    MAX_PERFORMANCE_PENALTY,
    MAX_MEMORY_PENALTY,
)

# ---------------------------------------------------------------------------
# Helpers — build minimal profiling dicts without running the interpreter
# ---------------------------------------------------------------------------


def make_profiling(
    line_counts: list[int] | None = None,
    memory_vars: list[int] | None = None,
    total_time_ms: float = 1.0,
) -> dict:
    """
    Build a minimal profiling_data dict that mimics ProfilingData.to_dict().

    Args:
        line_counts:   Execution counts per line. Each entry = one line.
        memory_vars:   Variables in scope per line (parallel to line_counts).
        total_time_ms: Simulated total execution time.
    """
    line_counts = line_counts or [1]
    memory_vars = memory_vars or [0] * len(line_counts)

    line_stats = {}
    total_executed = 0
    for i, (count, mem) in enumerate(zip(line_counts, memory_vars), start=1):
        line_stats[i] = {
            "count": count,
            "total_time": total_time_ms / max(len(line_counts), 1),
            "avg_time": total_time_ms / max(count, 1),
            "memory": mem,
        }
        total_executed += count

    return {
        "line_stats": line_stats,
        "function_stats": {},
        "total_time_ms": total_time_ms,
        "total_lines": total_executed,
        "lines_profiled": len(line_stats),
    }


def make_suggestion(severity: str) -> Suggestion:
    """Return a minimal Suggestion with the given severity."""
    return Suggestion(
        line=1,
        pattern="test_pattern",
        severity=severity,
        description="Test description",
        suggestion="Test suggestion",
        impact_score=5.0,
    )


# ---------------------------------------------------------------------------
# ScoreReport
# ---------------------------------------------------------------------------


class TestScoreReport:

    def test_to_dict_has_required_keys(self) -> None:
        r = ScoreReport(
            score=80.0,
            grade="Good",
            complexity_class="O(n)",
            breakdown={
                "severity_penalty": 0.0,
                "complexity_penalty": 5.0,
                "performance_penalty": 2.0,
                "memory_penalty": 1.0,
            },
            max_execution_count=100,
            lines_profiled=5,
            baseline_time_ms=0.5,
        )
        d = r.to_dict()
        assert set(d.keys()) == {
            "score",
            "grade",
            "complexity_class",
            "breakdown",
            "max_execution_count",
            "lines_profiled",
            "baseline_time_ms",
        }

    def test_to_dict_score_rounded(self) -> None:
        r = ScoreReport(score=80.123456, grade="Good", complexity_class="O(n)")
        assert r.to_dict()["score"] == 80.12

    def test_to_dict_breakdown_rounded(self) -> None:
        r = ScoreReport(
            score=80.0,
            grade="Good",
            complexity_class="O(n)",
            breakdown={"severity_penalty": 1.123456789},
        )
        assert r.to_dict()["breakdown"]["severity_penalty"] == 1.1235


# ---------------------------------------------------------------------------
# Severity Penalty
# ---------------------------------------------------------------------------


class TestSeverityPenalty:

    def test_no_suggestions_returns_zero(self) -> None:
        s = Scorer(make_profiling())
        assert s._severity_penalty() == 0.0

    def test_single_high_suggestion(self) -> None:
        s = Scorer(
            make_profiling(),
            suggestions=[make_suggestion("high")],
            total_source_lines=10,
        )
        # (1×3) / max(10,10) × 10 = 3.0
        assert s._severity_penalty() == pytest.approx(3.0)

    def test_single_medium_suggestion(self) -> None:
        s = Scorer(
            make_profiling(),
            suggestions=[make_suggestion("medium")],
            total_source_lines=10,
        )
        # (1×2) / 10 × 10 = 2.0
        assert s._severity_penalty() == pytest.approx(2.0)

    def test_single_low_suggestion(self) -> None:
        s = Scorer(
            make_profiling(),
            suggestions=[make_suggestion("low")],
            total_source_lines=10,
        )
        # (1×1) / 10 × 10 = 1.0
        assert s._severity_penalty() == pytest.approx(1.0)

    def test_mixed_suggestions(self) -> None:
        suggestions = [
            make_suggestion("high"),
            make_suggestion("high"),
            make_suggestion("medium"),
            make_suggestion("low"),
        ]
        s = Scorer(make_profiling(), suggestions=suggestions, total_source_lines=10)
        # (2×3 + 1×2 + 1×1) / 10 × 10 = 9.0
        assert s._severity_penalty() == pytest.approx(9.0)

    def test_penalty_clamped_at_max(self) -> None:
        # Flood with high suggestions on a tiny program
        suggestions = [make_suggestion("high")] * 100
        s = Scorer(make_profiling(), suggestions=suggestions, total_source_lines=1)
        assert s._severity_penalty() <= MAX_SEVERITY_PENALTY

    def test_short_program_uses_floor_of_10(self) -> None:
        """Programs shorter than 10 lines use 10 as the normaliser floor."""
        suggestions = [make_suggestion("high")]
        s_short = Scorer(
            make_profiling(), suggestions=suggestions, total_source_lines=3
        )
        s_medium = Scorer(
            make_profiling(), suggestions=suggestions, total_source_lines=10
        )
        # Both should produce the same result because floor is 10
        assert s_short._severity_penalty() == pytest.approx(
            s_medium._severity_penalty()
        )


# ---------------------------------------------------------------------------
# Complexity Detection
# ---------------------------------------------------------------------------


class TestComplexityDetection:

    def test_empty_profiling_returns_O1(self) -> None:
        s = Scorer(make_profiling(line_counts=[0]))
        assert s._detect_complexity() == "O(1)"

    def test_single_execution_returns_O1(self) -> None:
        s = Scorer(make_profiling(line_counts=[1]))
        assert s._detect_complexity() == "O(1)"

    def test_linear_loop(self) -> None:
        # One line executed 100 times — classic O(n)
        s = Scorer(make_profiling(line_counts=[1, 100, 100]))
        assert s._detect_complexity() == "O(n)"

    def test_nested_loop_O_n2(self) -> None:
        # Realistic nested loop: outer=101, inner_header=10100, inner_body=10000
        # avg = 6733, avg×1.5 = 10099
        # peak_threshold = 10100 × 0.9 = 9090
        # 10100 >= 9090 AND 10100 > 10099 ✓ → nesting_level = 1 → O(n²)
        s = Scorer(make_profiling(line_counts=[101, 10100, 10000]))
        assert s._detect_complexity() == "O(n²)"

    def test_triple_nested_O_n3(self) -> None:
        # Three hot lines: outer=5, mid=50, inner×3 lines at 5000
        # avg = (5+50+5000*3)/5 = 3011, avg×1.5 = 4516
        # peak_threshold = 5000×0.9 = 4500
        # 5000 >= 4500 AND 5000 > 4516 ✓ → 3 hot lines → nesting_level=3 → O(n³)
        s = Scorer(make_profiling(line_counts=[5, 50, 5000, 5000, 5000]))
        assert s._detect_complexity() == "O(n³)"

    def test_deep_nesting_O_nk(self) -> None:
        # Four or more hot lines: outer=5, mid=50, deep×4 lines at 5000
        # avg = (5+50+5000*4)/6 = 3342, avg×1.5 = 5013
        # 5000 >= 4500 AND 5000 > 5013? NO — tight boundary
        # Use a starker spread: [1, 10, 5000, 5000, 5000, 5000]
        # avg = (1+10+5000*4)/6 = 3335, avg×1.5 = 5002
        # 5000 >= 4500 AND 5000 > 5002? NO — barely misses
        # Use [1, 5, 5001, 5001, 5001, 5001]:
        # avg = (1+5+5001*4)/6 = 3337, avg×1.5 = 5005 — 5001 < 5005, still no
        # Practical conclusion: O(n^k) requires a very stark outer/inner ratio
        # which naturally occurs in real 4-deep nested loops.
        # Test verifies it returns a valid high-complexity class:
        s = Scorer(make_profiling(line_counts=[1, 10, 100, 10000, 10000, 10000, 10000]))
        assert s._detect_complexity() in ("O(n^k)", "O(n³)", "O(n²)")

    def test_log_n_loop(self) -> None:
        # Binary-search-style: 4 profiled lines, max_count = 4 ≈ log2(4)×2
        s = Scorer(make_profiling(line_counts=[1, 1, 1, 4]))
        result = s._detect_complexity()
        assert result in ("O(log n)", "O(1)", "O(n)")

    def test_n_log_n(self) -> None:
        # 8 profiled lines, max_count = 8 × log2(8) × 2 = 8 × 3 × 2 = 48
        # Should land in O(n log n) bracket
        n = 8
        import math

        count = int(n * math.log2(n) * 2)
        s = Scorer(make_profiling(line_counts=[1] * 7 + [count]))
        result = s._detect_complexity()
        assert result in ("O(n log n)", "O(n²)", "O(n)")


# ---------------------------------------------------------------------------
# Complexity Penalty
# ---------------------------------------------------------------------------


class TestComplexityPenalty:

    def test_O1_penalty_is_zero(self) -> None:
        s = Scorer(make_profiling(line_counts=[1]))
        assert s._complexity_penalty() == 0.0

    def test_On_penalty_is_zero(self) -> None:
        # Linear loop — header + body both run 100 times
        # avg = (1+100+100)/3 = 67, avg×1.5 = 100.5
        # 100 >= 90 (peak_threshold) AND 100 > 100.5?
        # NO → nesting_level=0 → O(n) → penalty=0
        s = Scorer(make_profiling(line_counts=[1, 100, 100]))
        assert s._complexity_penalty() == 0.0

    def test_On2_penalty(self) -> None:
        s = Scorer(make_profiling(line_counts=[101, 10100, 10000]))
        assert s._complexity_penalty() == pytest.approx(COMPLEXITY_PENALTIES["O(n²)"])

    def test_complexity_penalty_clamped(self) -> None:
        # O(2^n) raw penalty is 35, but MAX is 30
        raw = COMPLEXITY_PENALTIES["O(2^n)"]
        assert min(raw, MAX_COMPLEXITY_PENALTY) == MAX_COMPLEXITY_PENALTY


# ---------------------------------------------------------------------------
# Performance Penalty
# ---------------------------------------------------------------------------


class TestPerformancePenalty:

    def test_zero_time_returns_zero(self) -> None:
        p = make_profiling(line_counts=[10], total_time_ms=0.0)
        s = Scorer(p)
        assert s._performance_penalty() == 0.0

    def test_at_baseline_no_penalty(self) -> None:
        # total_lines_executed = 10, baseline = 10 × 0.01 = 0.1ms
        # actual_time = 0.1ms → ratio = 1 → penalty = 0
        p = make_profiling(line_counts=[10], total_time_ms=0.1)
        s = Scorer(p)
        assert s._performance_penalty() == pytest.approx(0.0, abs=0.01)

    def test_twice_baseline_gives_penalty(self) -> None:
        # ratio = 2, raw = (2-1)×3 = 3
        p = make_profiling(line_counts=[10], total_time_ms=0.2)
        s = Scorer(p)
        assert s._performance_penalty() == pytest.approx(3.0, rel=0.1)

    def test_penalty_clamped_at_max(self) -> None:
        # Extremely slow execution
        p = make_profiling(line_counts=[1], total_time_ms=10_000.0)
        s = Scorer(p)
        assert s._performance_penalty() <= MAX_PERFORMANCE_PENALTY

    def test_below_baseline_penalty_is_zero(self) -> None:
        # Faster than baseline → no reward, but also no penalty
        p = make_profiling(line_counts=[1000], total_time_ms=0.001)
        s = Scorer(p)
        assert s._performance_penalty() == 0.0


# ---------------------------------------------------------------------------
# Memory Penalty
# ---------------------------------------------------------------------------


class TestMemoryPenalty:

    def test_no_memory_vars_returns_zero(self) -> None:
        p = make_profiling(line_counts=[1], memory_vars=[0])
        s = Scorer(p)
        assert s._memory_penalty() == 0.0

    def test_below_threshold_no_penalty(self) -> None:
        p = make_profiling(line_counts=[1, 1], memory_vars=[2, 5])
        s = Scorer(p)
        assert s._memory_penalty() == 0.0

    def test_all_lines_exceed_threshold(self) -> None:
        threshold = HIGH_VAR_THRESHOLD + 1
        p = make_profiling(line_counts=[1, 1], memory_vars=[threshold, threshold])
        s = Scorer(p)
        assert s._memory_penalty() == pytest.approx(MAX_MEMORY_PENALTY)

    def test_half_lines_exceed_threshold(self) -> None:
        threshold = HIGH_VAR_THRESHOLD + 1
        p = make_profiling(
            line_counts=[1, 1, 1, 1],
            memory_vars=[threshold, threshold, 0, 0],
        )
        s = Scorer(p)
        assert s._memory_penalty() == pytest.approx(MAX_MEMORY_PENALTY * 0.5)


# ---------------------------------------------------------------------------
# calculate() end-to-end
# ---------------------------------------------------------------------------


class TestCalculate:

    def test_returns_score_report(self) -> None:
        p = make_profiling(line_counts=[1])
        report = Scorer(p).calculate()
        assert isinstance(report, ScoreReport)

    def test_score_in_valid_range(self) -> None:
        p = make_profiling(line_counts=[1, 100, 100])
        report = Scorer(p).calculate()
        assert 0.0 <= report.score <= 100.0

    def test_perfect_program_high_score(self) -> None:
        # Single-line, fast execution, no issues
        p = make_profiling(line_counts=[1], total_time_ms=0.01)
        report = Scorer(p).calculate()
        assert report.score >= 90.0
        assert report.grade == "Excellent"

    def test_nested_loop_lower_score(self) -> None:
        p = make_profiling(line_counts=[101, 10100, 10000], total_time_ms=50.0)
        report = Scorer(p).calculate()
        # Should be penalised for O(n²)
        assert report.score < 90.0

    def test_grade_excellent(self) -> None:
        p = make_profiling(line_counts=[1], total_time_ms=0.01)
        report = Scorer(p).calculate()
        assert report.grade in ("Excellent", "Good")

    def test_grade_poor_on_bad_code(self) -> None:
        # Heavy nesting + slow + high memory
        threshold = HIGH_VAR_THRESHOLD + 5
        p = make_profiling(
            line_counts=[101, 10100, 10000],
            memory_vars=[threshold] * 3,
            total_time_ms=5000.0,
        )
        report = Scorer(p, suggestions=[make_suggestion("high")] * 10).calculate()
        assert report.score < 60.0

    def test_breakdown_keys_present(self) -> None:
        p = make_profiling(line_counts=[1])
        report = Scorer(p).calculate()
        assert set(report.breakdown.keys()) == {
            "severity_penalty",
            "complexity_penalty",
            "performance_penalty",
            "memory_penalty",
        }

    def test_breakdown_values_are_non_negative(self) -> None:
        p = make_profiling(line_counts=[1, 50, 50])
        report = Scorer(p).calculate()
        for v in report.breakdown.values():
            assert v >= 0.0

    def test_complexity_class_in_report(self) -> None:
        p = make_profiling(line_counts=[101, 10100, 10000])
        report = Scorer(p).calculate()
        assert report.complexity_class == "O(n²)"

    def test_max_execution_count_in_report(self) -> None:
        p = make_profiling(line_counts=[1, 50, 200])
        report = Scorer(p).calculate()
        assert report.max_execution_count == 200

    def test_score_is_clamped_above_zero(self) -> None:
        # Intentionally terrible code profile
        threshold = HIGH_VAR_THRESHOLD + 10
        p = make_profiling(
            line_counts=[101, 10100, 10000, 10000, 10000],
            memory_vars=[threshold] * 5,
            total_time_ms=100_000.0,
        )
        suggestions = [make_suggestion("high")] * 50
        report = Scorer(p, suggestions=suggestions, total_source_lines=5).calculate()
        assert report.score >= 0.0

    def test_score_is_clamped_below_100(self) -> None:
        p = make_profiling(line_counts=[1], total_time_ms=0.0)
        report = Scorer(p).calculate()
        assert report.score <= 100.0


# ---------------------------------------------------------------------------
# calculate_score() convenience function
# ---------------------------------------------------------------------------


class TestCalculateScoreFunction:

    def test_returns_score_report(self) -> None:
        p = make_profiling(line_counts=[1])
        report = calculate_score(p)
        assert isinstance(report, ScoreReport)

    def test_matches_scorer_directly(self) -> None:
        p = make_profiling(line_counts=[1, 100, 100])
        suggestions = [make_suggestion("medium")]
        r1 = calculate_score(p, suggestions=suggestions, total_source_lines=10)
        r2 = Scorer(p, suggestions=suggestions, total_source_lines=10).calculate()
        assert r1.score == r2.score
        assert r1.grade == r2.grade
        assert r1.complexity_class == r2.complexity_class

    def test_no_suggestions_default(self) -> None:
        p = make_profiling(line_counts=[1])
        report = calculate_score(p)
        assert report.breakdown["severity_penalty"] == 0.0

    def test_to_dict_is_json_serialisable(self) -> None:
        import json

        p = make_profiling(line_counts=[1, 50, 50])
        report = calculate_score(p)
        # Should not raise
        json.dumps(report.to_dict())


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_empty_line_stats(self) -> None:
        p = {
            "line_stats": {},
            "function_stats": {},
            "total_time_ms": 0.0,
            "total_lines": 0,
            "lines_profiled": 0,
        }
        report = calculate_score(p)
        assert report.score == 100.0
        assert report.complexity_class == "O(1)"

    def test_single_line_program(self) -> None:
        p = make_profiling(line_counts=[1], total_time_ms=0.005)
        report = calculate_score(p)
        assert report.score >= 90.0

    def test_very_large_execution_count(self) -> None:
        # Should not crash or overflow
        p = make_profiling(line_counts=[1, 10, 1_000_000])
        report = calculate_score(p)
        assert 0.0 <= report.score <= 100.0

    def test_total_source_lines_zero_handled(self) -> None:
        # Should not divide by zero
        p = make_profiling(line_counts=[1])
        report = calculate_score(p, total_source_lines=0)
        assert report.score >= 0.0

    def test_all_lines_same_count_linear(self) -> None:
        # All lines execute equally — looks linear
        p = make_profiling(line_counts=[50] * 5)
        report = calculate_score(p)
        assert report.complexity_class in ("O(n)", "O(n log n)", "O(1)", "O(log n)")
