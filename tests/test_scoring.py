"""
tests/test_scoring.py
---------------------
Comprehensive test suite for optilang/scoring.py.

Covers every testable behaviour introduced or changed in the current version:

    Correctness scoring          — smooth 5-step scale (fix #5)
    Pattern classification       — EFFICIENCY / QUALITY / MAINTAINABILITY sets
                                   including constant_folding in MAINTAINABILITY
                                   (fix #3)
    Partial-credit flags         — profiling_partial, optimizer_partial (fix #1)
    Density-to-score             — linear interpolation, no step cliffs (fix #6)
    Complexity detection         — all eight Big-O classes
    Efficiency sub-score         — independent of complexity sub-score
    Quality scoring              — dead_code, string_concat_loop only
    Maintainability scoring      — unused_vars, early_return, nested_loops,
                                   constant_folding
    Grade assignment             — all five grade bands
    Lowest-dimension selection   — normalised ratio + tie-breaking (fix #7)
    Narrative generation         — headline tier, hint content, partial notes
                                   (fixes #2, #4)
    ScoreReport.to_dict          — serialisable output shape
    calculate_score              — public API end-to-end
    Score clamping               — final score never exceeds 100 or goes below 0
    CV computation               — informational only, not used in scoring

Each test class is self-contained. Helper factories at the top remove
boilerplate so individual tests stay readable.

Run with:
    pytest tests/test_scoring.py -v
"""

from __future__ import annotations

import random
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest

# ---------------------------------------------------------------------------
# Path setup — allows running from the project root without installing the
# package, as long as scoring.py is in optilang/ or on PYTHONPATH.
# ---------------------------------------------------------------------------
sys.path.insert(0, ".")

import optilang.scoring as sc  # noqa: E402  (import after sys.path tweak)
from optilang.scoring import (  # noqa: E402
    COMPLEXITY_POINTS,
    EFFICIENCY_PATTERNS,
    MAINTAINABILITY_PATTERNS,
    MAX_MAINTAINABILITY,
    MAX_QUALITY,
    PARTIAL_COMPLEXITY,
    PARTIAL_EFFICIENCY,
    PARTIAL_MAINTAINABILITY,
    PARTIAL_QUALITY,
    QUALITY_PATTERNS,
    DimensionScores,
    Scorer,
    ScoreReport,
    _assign_grade,
    _density_to_score,
    _detect_complexity,
    _generate_narrative,
    _rank_dimensions,
    calculate_score,
)

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


@dataclass
class _Suggestion:
    """Minimal stub that satisfies Scorer's duck-typed suggestion access."""
    pattern: str
    severity: str   # "low" | "medium" | "high"


class _Report:
    """Minimal stub for OptimizationReport."""

    def __init__(self, suggestions: List[_Suggestion]) -> None:
        self.suggestions = suggestions


def _make_line_stats(*counts: int) -> Dict[str, Any]:
    """
    Build a profiling line_stats dict from a sequence of execution counts.

    Each positional argument becomes one line entry.
    Example: _make_line_stats(1, 100, 100, 1)  →  4 lines with those counts.
    """
    return {str(i + 1): {"count": c} for i, c in enumerate(counts)}


def _clean_report() -> _Report:
    """An optimizer report with zero suggestions."""
    return _Report([])


def _score(
    *,
    counts: tuple = (1,),
    suggestions: Optional[List[_Suggestion]] = None,
    source_lines: int = 10,
    errors: Optional[List[str]] = None,
) -> ScoreReport:
    """
    Convenience wrapper around calculate_score with sensible defaults.

    Args:
        counts:       Tuple of execution counts for synthetic line_stats.
        suggestions:  Optimizer suggestion list (default: empty = clean).
        source_lines: Source line count for density normalisation.
        errors:       List of error strings.
    """
    profiling = {"line_stats": _make_line_stats(*counts)}
    report = _Report(suggestions or [])
    return calculate_score(
        profiling_data=profiling,
        optimizer_report=report,
        source_lines=source_lines,
        errors=errors or [],
    )


# ---------------------------------------------------------------------------
# 1. Pattern classification sets
# ---------------------------------------------------------------------------


class TestPatternSets:
    """Verify each pattern belongs to exactly one set and to the right one."""

    def test_efficiency_patterns_membership(self) -> None:
        assert "hot_loop" in EFFICIENCY_PATTERNS
        assert "loop_invariant" in EFFICIENCY_PATTERNS
        assert "repeated_computation" in EFFICIENCY_PATTERNS
        assert "expensive_calls" in EFFICIENCY_PATTERNS

    def test_quality_patterns_membership(self) -> None:
        assert "dead_code" in QUALITY_PATTERNS
        assert "string_concat_loop" in QUALITY_PATTERNS

    def test_maintainability_patterns_membership(self) -> None:
        assert "unused_vars" in MAINTAINABILITY_PATTERNS
        assert "early_return" in MAINTAINABILITY_PATTERNS
        assert "nested_loops" in MAINTAINABILITY_PATTERNS
        # Fix #3: constant_folding is a write-time concern → Maintainability
        assert "constant_folding" in MAINTAINABILITY_PATTERNS

    def test_constant_folding_not_in_quality(self) -> None:
        """constant_folding must NOT be in Quality (fix #3)."""
        assert "constant_folding" not in QUALITY_PATTERNS

    def test_no_pattern_in_multiple_sets(self) -> None:
        """Every pattern belongs to exactly one set."""
        all_sets = [EFFICIENCY_PATTERNS, QUALITY_PATTERNS, MAINTAINABILITY_PATTERNS]
        all_patterns: List[str] = []
        for s in all_sets:
            all_patterns.extend(s)
        assert len(all_patterns) == len(set(all_patterns)), (
            "At least one pattern appears in more than one classification set"
        )

    def test_sets_are_disjoint(self) -> None:
        assert EFFICIENCY_PATTERNS.isdisjoint(QUALITY_PATTERNS)
        assert EFFICIENCY_PATTERNS.isdisjoint(MAINTAINABILITY_PATTERNS)
        assert QUALITY_PATTERNS.isdisjoint(MAINTAINABILITY_PATTERNS)


# ---------------------------------------------------------------------------
# 2. Correctness scoring — smooth 5-step scale (fix #5)
# ---------------------------------------------------------------------------


class TestCorrectnessScoring:
    """
    The new scale is: 35 / 25 / 15 / 5 / 0 for 0 / 1 / 2 / 3 / 4+ errors.

    The old scale (35 → 10 → 0) caused a 25-point cliff for one error.
    """

    def test_zero_errors_full_marks(self) -> None:
        sr = _score(errors=[])
        assert sr.dimensions.correctness == 35.0

    def test_one_error_smooth_deduction(self) -> None:
        sr = _score(errors=["err1"])
        assert sr.dimensions.correctness == 25.0

    def test_two_errors(self) -> None:
        sr = _score(errors=["err1", "err2"])
        assert sr.dimensions.correctness == 15.0

    def test_three_errors(self) -> None:
        sr = _score(errors=["e1", "e2", "e3"])
        assert sr.dimensions.correctness == 5.0

    def test_four_errors_zero(self) -> None:
        sr = _score(errors=["e1", "e2", "e3", "e4"])
        assert sr.dimensions.correctness == 0.0

    def test_many_errors_zero(self) -> None:
        sr = _score(errors=["e"] * 20)
        assert sr.dimensions.correctness == 0.0

    def test_error_count_matches_list_length(self) -> None:
        sr = _score(errors=["a", "b", "c"])
        assert sr.error_count == 3

    def test_step_between_0_and_1_error_is_10(self) -> None:
        """The step from 0→1 error must be exactly 10 points, not 25 (old cliff)."""
        sr0 = _score(errors=[])
        sr1 = _score(errors=["x"])
        assert sr0.dimensions.correctness - sr1.dimensions.correctness == 10.0

    def test_each_step_is_equal(self) -> None:
        """Each additional error costs exactly 10 points up to 3 errors."""
        scores = [
            _score(errors=["e"] * n).dimensions.correctness
            for n in range(4)
        ]
        steps = [scores[i] - scores[i + 1] for i in range(3)]
        assert all(s == 10.0 for s in steps), f"Steps are not uniform: {steps}"


# ---------------------------------------------------------------------------
# 3. Density-to-score: linear interpolation, no step cliffs (fix #6)
# ---------------------------------------------------------------------------


class TestDensityToScore:
    """
    _density_to_score must:
        - return max_score at density=0
        - return max_score * 0.10 at density ≥ 2.0 (clamped)
        - be strictly monotonically decreasing as density rises
        - vary continuously — no jump > 5 % of max_score per 0.01 density step
    """

    MAX = 20.0   # use Quality max for these tests

    def test_zero_density_full_marks(self) -> None:
        assert _density_to_score(0.0, self.MAX) == self.MAX

    def test_at_exact_anchors(self) -> None:
        # Anchor fractions: (0.0, 1.0), (0.3, 0.8), (0.6, 0.6), (1.0, 0.35), (2.0, 0.1)
        assert _density_to_score(0.0, self.MAX) == pytest.approx(self.MAX * 1.00, abs=0.01)
        assert _density_to_score(0.3, self.MAX) == pytest.approx(self.MAX * 0.80, abs=0.01)
        assert _density_to_score(0.6, self.MAX) == pytest.approx(self.MAX * 0.60, abs=0.01)
        assert _density_to_score(1.0, self.MAX) == pytest.approx(self.MAX * 0.35, abs=0.01)
        assert _density_to_score(2.0, self.MAX) == pytest.approx(self.MAX * 0.10, abs=0.01)

    def test_beyond_last_anchor_clamped(self) -> None:
        assert _density_to_score(3.0, self.MAX) == pytest.approx(self.MAX * 0.10, abs=0.01)
        assert _density_to_score(10.0, self.MAX) == pytest.approx(self.MAX * 0.10, abs=0.01)

    def test_midpoint_interpolated(self) -> None:
        """Midpoint between (0.3, 0.8) and (0.6, 0.6) should be ~0.7 fraction."""
        mid = _density_to_score(0.45, self.MAX)
        expected = self.MAX * 0.70
        assert mid == pytest.approx(expected, abs=0.02)

    def test_strictly_monotonically_decreasing(self) -> None:
        densities = [i * 0.1 for i in range(25)]   # 0.0 … 2.4
        scores = [_density_to_score(d, self.MAX) for d in densities]
        for i in range(len(scores) - 1):
            assert scores[i] >= scores[i + 1], (
                f"Score increased from density {densities[i]:.1f} to "
                f"{densities[i + 1]:.1f}: {scores[i]:.3f} → {scores[i + 1]:.3f}"
            )

    def test_no_score_cliff_at_band_boundaries(self) -> None:
        """
        At any density boundary, a 0.01 step must not cause a drop
        larger than 5 % of max_score. This was the core problem with the
        old step-band implementation.
        """
        max_allowed_jump = self.MAX * 0.05
        boundaries = [0.3, 0.6, 1.0]
        for b in boundaries:
            below = _density_to_score(b - 0.01, self.MAX)
            above = _density_to_score(b + 0.01, self.MAX)
            jump = abs(below - above)
            assert jump <= max_allowed_jump, (
                f"Score cliff of {jump:.3f} at density boundary {b} "
                f"(allowed ≤ {max_allowed_jump:.3f})"
            )

    def test_score_never_below_minimum(self) -> None:
        for density in [0.0, 0.5, 1.0, 2.0, 5.0, 100.0]:
            assert _density_to_score(density, self.MAX) >= self.MAX * 0.10 - 0.01

    def test_score_never_above_maximum(self) -> None:
        for density in [0.0, 0.001, 0.1]:
            assert _density_to_score(density, self.MAX) <= self.MAX + 0.01

    def test_works_for_all_dimension_maxes(self) -> None:
        """_density_to_score must work correctly for all three dimension maxes."""
        for max_score in [15.0, 20.0]:
            result = _density_to_score(0.0, max_score)
            assert result == pytest.approx(max_score, abs=0.01)
            result_clamped = _density_to_score(5.0, max_score)
            assert result_clamped == pytest.approx(max_score * 0.10, abs=0.01)


# ---------------------------------------------------------------------------
# 4. Complexity detection
# ---------------------------------------------------------------------------


class TestComplexityDetection:
    """_detect_complexity maps execution-count profiles to Big-O classes."""

    def test_empty_returns_o1(self) -> None:
        assert _detect_complexity({}) == "O(1)"

    def test_all_zero_counts_returns_o1(self) -> None:
        stats = {"1": {"count": 0}, "2": {"count": 0}}
        assert _detect_complexity(stats) == "O(1)"

    def test_single_execution_returns_o1(self) -> None:
        stats = _make_line_stats(1, 1, 1)
        assert _detect_complexity(stats) == "O(1)"

    def test_linear_profile_returns_on(self) -> None:
        # Flat loop: body runs n times, preamble/postamble run once
        stats = _make_line_stats(1, 1, 100, 100, 1)
        result = _detect_complexity(stats)
        assert result == "O(n)", f"Expected O(n), got {result}"

    def test_quadratic_profile_returns_on2(self) -> None:
        # Nested loop: outer=100, inner=10000
        stats = _make_line_stats(1, 100, 10000, 10000, 1)
        result = _detect_complexity(stats)
        assert result == "O(n²)", f"Expected O(n²), got {result}"

    def test_complexity_points_covers_all_classes(self) -> None:
        expected_classes = {
            "O(1)", "O(log n)", "O(n)", "O(n log n)",
            "O(n²)", "O(n³)", "O(n^k)", "O(2^n)",
        }
        assert set(COMPLEXITY_POINTS.keys()) == expected_classes

    def test_complexity_points_values_in_range(self) -> None:
        for cls, pts in COMPLEXITY_POINTS.items():
            assert 0.0 <= pts <= 15.0, (
                f"COMPLEXITY_POINTS['{cls}'] = {pts} is outside [0, 15]"
            )

    def test_better_complexity_earns_more_points(self) -> None:
        """Lower-complexity classes must score at least as many points."""
        order = ["O(1)", "O(log n)", "O(n)", "O(n log n)",
                 "O(n²)", "O(n³)", "O(n^k)", "O(2^n)"]
        for i in range(len(order) - 1):
            assert COMPLEXITY_POINTS[order[i]] >= COMPLEXITY_POINTS[order[i + 1]], (
                f"{order[i]} should score ≥ {order[i + 1]}"
            )


# ---------------------------------------------------------------------------
# 5. Efficiency sub-score — independent of complexity (core redesign goal)
# ---------------------------------------------------------------------------


class TestEfficiencySubScore:
    """
    The efficiency sub-score measures wasted work (EFFICIENCY_PATTERNS),
    not algorithmic complexity. The two sub-scores must be independent.
    """

    def test_no_efficiency_issues_full_marks(self) -> None:
        sr = _score(counts=(1, 100, 100, 1), suggestions=[])
        assert sr.dimensions.efficiency_subscore == 15.0

    def test_efficiency_issues_reduce_subscore(self) -> None:
        suggestions = [_Suggestion("loop_invariant", "high")] * 2
        sr = _score(counts=(1, 100, 100, 1), suggestions=suggestions, source_lines=5)
        assert sr.dimensions.efficiency_subscore < 15.0

    def test_high_severity_penalises_more_than_low(self) -> None:
        high_sr = _score(
            suggestions=[_Suggestion("hot_loop", "high")],
            source_lines=5,
        )
        low_sr = _score(
            suggestions=[_Suggestion("hot_loop", "low")],
            source_lines=5,
        )
        assert high_sr.dimensions.efficiency_subscore < low_sr.dimensions.efficiency_subscore

    def test_good_complexity_poor_efficiency_independent(self) -> None:
        """
        O(n) complexity + loop_invariant violation:
        complexity sub-score should be good (13), efficiency sub-score bad.
        The two values must differ — they measure different things.
        """
        suggestions = [_Suggestion("loop_invariant", "high")] * 5
        sr = _score(
            counts=(1, 100, 100, 1),
            suggestions=suggestions,
            source_lines=4,
        )
        assert sr.dimensions.complexity_subscore == 13.0   # O(n) = 13 pts
        assert sr.dimensions.efficiency_subscore < 13.0    # penalised

    def test_all_efficiency_patterns_counted(self) -> None:
        """Every pattern in EFFICIENCY_PATTERNS must reduce the efficiency sub-score."""
        for pattern in EFFICIENCY_PATTERNS:
            suggestions = [_Suggestion(pattern, "high")]
            sr = _score(suggestions=suggestions, source_lines=1)
            assert sr.dimensions.efficiency_subscore < 15.0, (
                f"Pattern '{pattern}' did not reduce efficiency sub-score"
            )

    def test_quality_patterns_do_not_affect_efficiency(self) -> None:
        """QUALITY_PATTERNS suggestions must not alter the efficiency sub-score."""
        for pattern in QUALITY_PATTERNS:
            suggestions = [_Suggestion(pattern, "high")] * 3
            sr = _score(suggestions=suggestions, source_lines=5)
            assert sr.dimensions.efficiency_subscore == 15.0, (
                f"Quality pattern '{pattern}' incorrectly reduced efficiency sub-score"
            )

    def test_maintainability_patterns_do_not_affect_efficiency(self) -> None:
        """MAINTAINABILITY_PATTERNS suggestions must not alter the efficiency sub-score."""
        for pattern in MAINTAINABILITY_PATTERNS:
            suggestions = [_Suggestion(pattern, "high")] * 3
            sr = _score(suggestions=suggestions, source_lines=5)
            assert sr.dimensions.efficiency_subscore == 15.0, (
                f"Maintainability pattern '{pattern}' incorrectly reduced efficiency sub-score"
            )

    def test_efficiency_and_complexity_sum_to_efficiency_complexity(self) -> None:
        sr = _score()
        total = sr.dimensions.complexity_subscore + sr.dimensions.efficiency_subscore
        assert total == pytest.approx(sr.dimensions.efficiency_complexity, abs=0.01)


# ---------------------------------------------------------------------------
# 6. Quality scoring — dead_code and string_concat_loop only
# ---------------------------------------------------------------------------


class TestQualityScoring:
    """Quality is sourced only from QUALITY_PATTERNS."""

    def test_no_quality_issues_full_marks(self) -> None:
        sr = _score(suggestions=[])
        assert sr.dimensions.quality == MAX_QUALITY

    def test_dead_code_reduces_quality(self) -> None:
        suggestions = [_Suggestion("dead_code", "medium")]
        sr = _score(suggestions=suggestions, source_lines=5)
        assert sr.dimensions.quality < MAX_QUALITY

    def test_string_concat_loop_reduces_quality(self) -> None:
        suggestions = [_Suggestion("string_concat_loop", "high")]
        sr = _score(suggestions=suggestions, source_lines=5)
        assert sr.dimensions.quality < MAX_QUALITY

    def test_constant_folding_does_not_reduce_quality(self) -> None:
        """constant_folding is Maintainability, not Quality (fix #3)."""
        suggestions = [_Suggestion("constant_folding", "high")] * 5
        sr = _score(suggestions=suggestions, source_lines=5)
        assert sr.dimensions.quality == MAX_QUALITY

    def test_efficiency_patterns_do_not_reduce_quality(self) -> None:
        for pattern in EFFICIENCY_PATTERNS:
            suggestions = [_Suggestion(pattern, "high")] * 3
            sr = _score(suggestions=suggestions, source_lines=5)
            assert sr.dimensions.quality == MAX_QUALITY, (
                f"Efficiency pattern '{pattern}' incorrectly reduced quality"
            )

    def test_quality_never_exceeds_max(self) -> None:
        sr = _score(suggestions=[])
        assert sr.dimensions.quality <= MAX_QUALITY

    def test_quality_never_below_floor(self) -> None:
        suggestions = [_Suggestion("dead_code", "high")] * 100
        sr = _score(suggestions=suggestions, source_lines=1)
        assert sr.dimensions.quality >= MAX_QUALITY * 0.10 - 0.01

    def test_high_severity_penalises_more_than_low(self) -> None:
        high_sr = _score(
            suggestions=[_Suggestion("dead_code", "high")],
            source_lines=5,
        )
        low_sr = _score(
            suggestions=[_Suggestion("dead_code", "low")],
            source_lines=5,
        )
        assert high_sr.dimensions.quality < low_sr.dimensions.quality


# ---------------------------------------------------------------------------
# 7. Maintainability scoring
# ---------------------------------------------------------------------------


class TestMaintainabilityScoring:
    """Maintainability is sourced only from MAINTAINABILITY_PATTERNS."""

    def test_no_issues_full_marks(self) -> None:
        sr = _score(suggestions=[])
        assert sr.dimensions.maintainability == MAX_MAINTAINABILITY

    def test_unused_vars_reduces_maintainability(self) -> None:
        suggestions = [_Suggestion("unused_vars", "low")]
        sr = _score(suggestions=suggestions, source_lines=5)
        assert sr.dimensions.maintainability < MAX_MAINTAINABILITY

    def test_early_return_reduces_maintainability(self) -> None:
        suggestions = [_Suggestion("early_return", "low")]
        sr = _score(suggestions=suggestions, source_lines=5)
        assert sr.dimensions.maintainability < MAX_MAINTAINABILITY

    def test_nested_loops_reduces_maintainability(self) -> None:
        suggestions = [_Suggestion("nested_loops", "medium")]
        sr = _score(suggestions=suggestions, source_lines=5)
        assert sr.dimensions.maintainability < MAX_MAINTAINABILITY

    def test_constant_folding_reduces_maintainability(self) -> None:
        """constant_folding now lives in Maintainability (fix #3)."""
        suggestions = [_Suggestion("constant_folding", "low")]
        sr = _score(suggestions=suggestions, source_lines=5)
        assert sr.dimensions.maintainability < MAX_MAINTAINABILITY

    def test_quality_patterns_do_not_affect_maintainability(self) -> None:
        for pattern in QUALITY_PATTERNS:
            suggestions = [_Suggestion(pattern, "high")] * 3
            sr = _score(suggestions=suggestions, source_lines=5)
            assert sr.dimensions.maintainability == MAX_MAINTAINABILITY, (
                f"Quality pattern '{pattern}' incorrectly reduced maintainability"
            )

    def test_efficiency_patterns_do_not_affect_maintainability(self) -> None:
        for pattern in EFFICIENCY_PATTERNS:
            suggestions = [_Suggestion(pattern, "high")] * 3
            sr = _score(suggestions=suggestions, source_lines=5)
            assert sr.dimensions.maintainability == MAX_MAINTAINABILITY, (
                f"Efficiency pattern '{pattern}' incorrectly reduced maintainability"
            )

    def test_maintainability_never_exceeds_max(self) -> None:
        sr = _score(suggestions=[])
        assert sr.dimensions.maintainability <= MAX_MAINTAINABILITY

    def test_maintainability_never_below_floor(self) -> None:
        suggestions = [_Suggestion("unused_vars", "high")] * 100
        sr = _score(suggestions=suggestions, source_lines=1)
        assert sr.dimensions.maintainability >= MAX_MAINTAINABILITY * 0.10 - 0.01


# ---------------------------------------------------------------------------
# 8. Partial credit and partial flags (fixes #1 and #2)
# ---------------------------------------------------------------------------


class TestPartialCredit:
    """
    When profiling or optimizer data is absent, partial credit is awarded
    and the corresponding flag is set on DimensionScores.
    """

    # ── profiling absent ──────────────────────────────────────────────

    def test_no_profiling_complexity_partial_credit(self) -> None:
        sr = calculate_score(
            profiling_data=None,
            optimizer_report=_clean_report(),
            source_lines=10,
        )
        assert sr.dimensions.complexity_subscore == PARTIAL_COMPLEXITY
        assert sr.dimensions.profiling_partial is True
        assert sr.complexity_class == "Unknown"

    def test_no_profiling_does_not_affect_efficiency_sub(self) -> None:
        """Efficiency sub-score comes from the optimizer, not profiling."""
        sr = calculate_score(
            profiling_data=None,
            optimizer_report=_clean_report(),
            source_lines=10,
        )
        assert sr.dimensions.efficiency_subscore == 15.0   # clean report → full marks

    def test_no_profiling_optimizer_partial_false(self) -> None:
        sr = calculate_score(
            profiling_data=None,
            optimizer_report=_clean_report(),
            source_lines=10,
        )
        assert sr.dimensions.optimizer_partial is False

    # ── optimizer absent ──────────────────────────────────────────────

    def test_no_optimizer_efficiency_partial_credit(self) -> None:
        """Fix #1: optimizer_partial must be True when optimizer is None."""
        sr = calculate_score(
            profiling_data={"line_stats": _make_line_stats(1, 100, 1)},
            optimizer_report=None,
            source_lines=10,
        )
        assert sr.dimensions.efficiency_subscore == PARTIAL_EFFICIENCY
        assert sr.dimensions.optimizer_partial is True

    def test_no_optimizer_quality_partial_credit(self) -> None:
        sr = calculate_score(
            profiling_data={"line_stats": _make_line_stats(1, 100, 1)},
            optimizer_report=None,
            source_lines=10,
        )
        assert sr.dimensions.quality == PARTIAL_QUALITY
        assert sr.dimensions.optimizer_partial is True

    def test_no_optimizer_maintainability_partial_credit(self) -> None:
        sr = calculate_score(
            profiling_data={"line_stats": _make_line_stats(1, 100, 1)},
            optimizer_report=None,
            source_lines=10,
        )
        assert sr.dimensions.maintainability == PARTIAL_MAINTAINABILITY

    def test_no_optimizer_profiling_partial_false(self) -> None:
        sr = calculate_score(
            profiling_data={"line_stats": _make_line_stats(1, 100, 1)},
            optimizer_report=None,
            source_lines=10,
        )
        assert sr.dimensions.profiling_partial is False

    # ── both absent ───────────────────────────────────────────────────

    def test_both_absent_all_partial_flags_set(self) -> None:
        sr = calculate_score(
            profiling_data=None,
            optimizer_report=None,
            source_lines=10,
        )
        assert sr.dimensions.profiling_partial is True
        assert sr.dimensions.optimizer_partial is True

    def test_both_absent_all_sub_scores_are_partial(self) -> None:
        sr = calculate_score(
            profiling_data=None,
            optimizer_report=None,
            source_lines=10,
        )
        assert sr.dimensions.complexity_subscore == PARTIAL_COMPLEXITY
        assert sr.dimensions.efficiency_subscore == PARTIAL_EFFICIENCY
        assert sr.dimensions.quality == PARTIAL_QUALITY
        assert sr.dimensions.maintainability == PARTIAL_MAINTAINABILITY

    # ── both present ──────────────────────────────────────────────────

    def test_both_present_no_partial_flags(self) -> None:
        sr = _score()
        assert sr.dimensions.profiling_partial is False
        assert sr.dimensions.optimizer_partial is False


# ---------------------------------------------------------------------------
# 9. Grade assignment
# ---------------------------------------------------------------------------


class TestGradeAssignment:
    """_assign_grade maps numeric scores to the five grade labels."""

    @pytest.mark.parametrize("score,expected_grade", [
        (100.0, "Excellent"),
        (90.0,  "Excellent"),
        (89.9,  "Good"),
        (75.0,  "Good"),
        (74.9,  "Fair"),
        (60.0,  "Fair"),
        (59.9,  "Poor"),
        (40.0,  "Poor"),
        (39.9,  "Critical"),
        (0.0,   "Critical"),
    ])
    def test_grade_boundaries(self, score: float, expected_grade: str) -> None:
        assert _assign_grade(score) == expected_grade

    def test_full_pipeline_perfect_score_grade(self) -> None:
        sr = _score(counts=(1,), suggestions=[], errors=[])
        assert sr.grade == "Excellent"

    def test_full_pipeline_many_errors_grade(self) -> None:
        sr = _score(errors=["e"] * 10, suggestions=[_Suggestion("dead_code", "high")] * 20)
        # With 0 correctness, very low quality → should be Critical or Poor
        assert sr.grade in ("Critical", "Poor")


# ---------------------------------------------------------------------------
# 10. Dimension ranking — _rank_dimensions (replaces _lowest_dimension)
# ---------------------------------------------------------------------------


class TestRankDimensions:
    """
    _rank_dimensions returns all four dimensions sorted by absolute points
    missing descending. Ties break by higher dimension max.

    This replaces the old _lowest_dimension which returned only one name
    and could not surface multiple weak areas.
    """

    def _dims(self, c=35.0, ec=30.0, q=20.0, m=15.0) -> DimensionScores:
        return DimensionScores(
            correctness=c,
            efficiency_complexity=ec,
            quality=q,
            maintainability=m,
        )

    def test_returns_four_entries(self) -> None:
        ranked = _rank_dimensions(self._dims())
        assert len(ranked) == 4

    def test_each_entry_is_tuple_of_three(self) -> None:
        for entry in _rank_dimensions(self._dims()):
            assert len(entry) == 3
            name, ratio, missing = entry
            assert isinstance(name, str)
            assert isinstance(ratio, float)
            assert isinstance(missing, float)

    def test_sorted_by_missing_descending(self) -> None:
        """Dimension with most absolute points missing comes first."""
        dims = self._dims(c=0.0, ec=30.0, q=20.0, m=15.0)
        ranked = _rank_dimensions(dims)
        # Correctness missing 35, all others missing 0
        assert ranked[0][0] == "Correctness"
        assert ranked[0][2] == pytest.approx(35.0)

    def test_all_perfect_missing_is_zero(self) -> None:
        ranked = _rank_dimensions(self._dims())
        for _, ratio, missing in ranked:
            assert ratio == pytest.approx(1.0)
            assert missing == pytest.approx(0.0)

    def test_tie_breaking_by_higher_max(self) -> None:
        """
        Two dimensions both missing 10 points:
        Correctness: 25/35 → missing 10
        Efficiency:  20/30 → missing 10
        Tie → Correctness wins (max=35 > max=30).
        """
        dims = self._dims(c=25.0, ec=20.0, q=20.0, m=15.0)
        ranked = _rank_dimensions(dims)
        assert ranked[0][0] == "Correctness"

    def test_normalised_ratio_correct(self) -> None:
        dims = self._dims(c=17.5, ec=30.0, q=20.0, m=15.0)
        ranked = _rank_dimensions(dims)
        correctness_entry = next(e for e in ranked if e[0] == "Correctness")
        assert correctness_entry[1] == pytest.approx(0.5)   # 17.5 / 35

    def test_all_names_present(self) -> None:
        valid = {
            "Correctness", "Efficiency & Complexity", "Quality", "Maintainability"
        }
        ranked = _rank_dimensions(self._dims())
        assert {e[0] for e in ranked} == valid

    def test_random_inputs_always_four_entries(self) -> None:
        for _ in range(20):
            dims = self._dims(
                c=random.uniform(0, 35),
                ec=random.uniform(0, 30),
                q=random.uniform(0, 20),
                m=random.uniform(0, 15),
            )
            assert len(_rank_dimensions(dims)) == 4


# ---------------------------------------------------------------------------
# 11. Narrative generation — multi-dimension, perfect-score-aware
# ---------------------------------------------------------------------------


class TestNarrativeGeneration:
    """
    _generate_narrative must:
        - open with the correct grade-tier headline
        - say nothing about a "weakest area" when every dimension is perfect
        - mention ALL actionable dimensions (< 90 % of max), not just one
        - list actionable dimensions in order of most absolute points missing
        - give a gentle single-dimension note when all dims are healthy but
          not all perfect
        - include accurate partial-credit notes when data was unavailable
        - include the right hint keywords for each dimension mentioned
    """

    def _dims_with_lowest(self, lowest: str) -> DimensionScores:
        """Create DimensionScores where only the specified dimension is bad."""
        d = DimensionScores(
            correctness=35.0,
            efficiency_complexity=30.0,
            quality=20.0,
            maintainability=15.0,
        )
        if lowest == "Correctness":
            d.correctness = 0.0
        elif lowest == "Efficiency & Complexity":
            d.efficiency_complexity = 0.0
        elif lowest == "Quality":
            d.quality = 0.0
        elif lowest == "Maintainability":
            d.maintainability = 0.0
        return d

    # ── Grade headlines ───────────────────────────────────────────────

    def test_excellent_headline(self) -> None:
        dims = DimensionScores(
            correctness=35.0, efficiency_complexity=30.0,
            quality=20.0, maintainability=15.0,
        )
        narrative = _generate_narrative(95.0, dims)
        assert "Excellent" in narrative

    def test_good_headline(self) -> None:
        dims = self._dims_with_lowest("Maintainability")
        narrative = _generate_narrative(80.0, dims)
        assert "Good" in narrative

    def test_fair_headline(self) -> None:
        dims = self._dims_with_lowest("Quality")
        narrative = _generate_narrative(65.0, dims)
        assert "Fair" in narrative

    def test_poor_headline(self) -> None:
        dims = self._dims_with_lowest("Correctness")
        narrative = _generate_narrative(45.0, dims)
        assert "needs some work" in narrative.lower() or "poor" in narrative.lower()

    def test_critical_headline(self) -> None:
        dims = self._dims_with_lowest("Correctness")
        narrative = _generate_narrative(20.0, dims)
        assert "significant issues" in narrative.lower()

    # ── Perfect score: no "weakest area" mentioned ────────────────────

    def test_perfect_score_no_weakest_area_mention(self) -> None:
        """
        Sample 1 fix: score=100 with all dims perfect must NOT say
        'your weakest area is Correctness' or any similar false claim.
        """
        dims = DimensionScores(
            correctness=35.0, efficiency_complexity=30.0,
            quality=20.0, maintainability=15.0,
        )
        narrative = _generate_narrative(100.0, dims)
        assert "weakest area" not in narrative.lower()
        assert "area to improve" not in narrative.lower()
        assert "full marks" in narrative.lower()

    def test_perfect_score_positive_only_narrative(self) -> None:
        dims = DimensionScores(
            correctness=35.0, efficiency_complexity=30.0,
            quality=20.0, maintainability=15.0,
        )
        narrative = _generate_narrative(100.0, dims)
        # Should be purely congratulatory
        assert "Excellent" in narrative
        assert "every dimension" in narrative.lower() or "full marks" in narrative.lower()

    # ── Multiple actionable dimensions all mentioned ──────────────────

    def test_two_bad_dims_both_mentioned(self) -> None:
        """
        Sample 2 fix: Maintainability at 67% AND Efficiency at 82% must
        both appear in the narrative, not just Maintainability.
        """
        dims = DimensionScores(
            correctness=35.0,
            efficiency_complexity=24.7,   # 82 % — actionable
            quality=20.0,
            maintainability=10.0,         # 67 % — actionable
        )
        narrative = _generate_narrative(89.7, dims)
        assert "Efficiency" in narrative
        assert "Maintainability" in narrative

    def test_two_bad_dims_efficiency_listed_first(self) -> None:
        """
        Efficiency missing 5.3 pts, Maintainability missing 5.0 pts.
        Efficiency should appear before Maintainability (more pts missing).
        """
        dims = DimensionScores(
            correctness=35.0,
            efficiency_complexity=24.7,   # missing 5.3
            quality=20.0,
            maintainability=10.0,         # missing 5.0
        )
        narrative = _generate_narrative(89.7, dims)
        eff_pos = narrative.find("Efficiency")
        maint_pos = narrative.find("Maintainability")
        assert eff_pos < maint_pos, (
            "Efficiency (more points missing) must appear before Maintainability"
        )

    def test_single_bad_dim_only_that_dim_mentioned(self) -> None:
        dims = self._dims_with_lowest("Maintainability")
        narrative = _generate_narrative(80.0, dims)
        assert "Maintainability" in narrative

    def test_all_bad_dims_all_mentioned(self) -> None:
        dims = DimensionScores(
            correctness=25.0,              # all below 90 %
            efficiency_complexity=15.0,
            quality=8.0,
            maintainability=5.0,
        )
        narrative = _generate_narrative(53.0, dims)
        assert "Correctness" in narrative
        assert "Efficiency" in narrative
        assert "Quality" in narrative
        assert "Maintainability" in narrative

    # ── Healthy but not perfect: gentle single note ───────────────────

    def test_all_healthy_not_perfect_gives_gentle_note(self) -> None:
        """All dims ≥ 90 % but not 100 % → gentle 'squeeze out' tone."""
        dims = DimensionScores(
            correctness=33.0,             # 94 % — healthy
            efficiency_complexity=28.0,   # 93 % — healthy
            quality=19.0,                 # 95 % — healthy
            maintainability=14.0,         # 93 % — healthy
        )
        narrative = _generate_narrative(94.0, dims)
        assert "great shape" in narrative.lower() or "squeeze" in narrative.lower()

    def test_all_healthy_not_perfect_no_alarm_language(self) -> None:
        dims = DimensionScores(
            correctness=33.0,
            efficiency_complexity=28.0,
            quality=19.0,
            maintainability=14.0,
        )
        narrative = _generate_narrative(94.0, dims)
        # Should not say "needs work" or "significant issues"
        assert "needs some work" not in narrative.lower()
        assert "significant issues" not in narrative.lower()

    # ── Hint content covers all patterns (fix #4) ─────────────────────

    def test_maintainability_hint_mentions_early_return(self) -> None:
        dims = self._dims_with_lowest("Maintainability")
        narrative = _generate_narrative(70.0, dims)
        assert "early" in narrative.lower()

    def test_maintainability_hint_mentions_constant_folding(self) -> None:
        dims = self._dims_with_lowest("Maintainability")
        narrative = _generate_narrative(70.0, dims)
        assert "constant" in narrative.lower() or "pre-computed" in narrative.lower()

    def test_maintainability_hint_mentions_unused_vars(self) -> None:
        dims = self._dims_with_lowest("Maintainability")
        narrative = _generate_narrative(70.0, dims)
        assert "unused" in narrative.lower()

    def test_maintainability_hint_mentions_nested_loops(self) -> None:
        dims = self._dims_with_lowest("Maintainability")
        narrative = _generate_narrative(70.0, dims)
        assert "nested" in narrative.lower() or "loop" in narrative.lower()

    def test_quality_hint_mentions_string_concat(self) -> None:
        dims = self._dims_with_lowest("Quality")
        narrative = _generate_narrative(65.0, dims)
        assert "string" in narrative.lower() or "concat" in narrative.lower()

    def test_quality_hint_mentions_dead_code(self) -> None:
        dims = self._dims_with_lowest("Quality")
        narrative = _generate_narrative(65.0, dims)
        assert "dead" in narrative.lower()

    def test_efficiency_hint_mentions_loop_invariant(self) -> None:
        dims = self._dims_with_lowest("Efficiency & Complexity")
        narrative = _generate_narrative(65.0, dims)
        assert "invariant" in narrative.lower() or "loop" in narrative.lower()

    # ── Partial-credit notes (fix #2) ────────────────────────────────

    def test_no_partial_note_when_all_data_present(self) -> None:
        dims = DimensionScores(
            correctness=35.0, efficiency_complexity=30.0,
            quality=20.0, maintainability=15.0,
            profiling_partial=False, optimizer_partial=False,
        )
        narrative = _generate_narrative(95.0, dims)
        assert "Note:" not in narrative

    def test_profiling_partial_note_mentions_complexity(self) -> None:
        dims = DimensionScores(
            correctness=35.0, efficiency_complexity=22.0,
            quality=20.0, maintainability=15.0,
            profiling_partial=True, optimizer_partial=False,
        )
        narrative = _generate_narrative(85.0, dims)
        assert "Note:" in narrative
        assert "Complexity" in narrative
        assert "Efficiency sub-score" not in narrative or "Complexity sub-score only" in narrative

    def test_optimizer_partial_note_mentions_efficiency_quality_maintainability(
        self,
    ) -> None:
        dims = DimensionScores(
            correctness=35.0, efficiency_complexity=21.0,
            quality=10.0, maintainability=7.0,
            profiling_partial=False, optimizer_partial=True,
        )
        narrative = _generate_narrative(73.0, dims)
        assert "Note:" in narrative
        assert "Efficiency" in narrative
        assert "Quality" in narrative
        assert "Maintainability" in narrative

    def test_both_partial_note_mentions_all_four(self) -> None:
        dims = DimensionScores(
            correctness=35.0, efficiency_complexity=15.0,
            quality=10.0, maintainability=7.0,
            profiling_partial=True, optimizer_partial=True,
        )
        narrative = _generate_narrative(67.0, dims)
        assert "Note:" in narrative
        assert "Complexity" in narrative
        assert "Efficiency" in narrative
        assert "Quality" in narrative
        assert "Maintainability" in narrative


# ---------------------------------------------------------------------------
# 12. ScoreReport serialisation
# ---------------------------------------------------------------------------


class TestScoreReportSerialisaton:
    """ScoreReport.to_dict must return a JSON-friendly dict with all keys."""

    EXPECTED_KEYS = {
        "score", "grade", "complexity_class", "dimensions",
        "narrative", "error_count", "lines_profiled", "cv",
    }
    EXPECTED_DIM_KEYS = {
        "correctness", "efficiency_complexity", "quality", "maintainability",
        "complexity_subscore", "efficiency_subscore",
        "profiling_partial", "optimizer_partial",
    }

    def test_to_dict_top_level_keys(self) -> None:
        sr = _score()
        d = sr.to_dict()
        assert set(d.keys()) == self.EXPECTED_KEYS

    def test_to_dict_dimensions_keys(self) -> None:
        sr = _score()
        d = sr.to_dict()
        assert set(d["dimensions"].keys()) == self.EXPECTED_DIM_KEYS

    def test_score_is_rounded_to_2dp(self) -> None:
        sr = _score()
        d = sr.to_dict()
        # Verify it's a float with at most 2 decimal places
        assert round(d["score"], 2) == d["score"]

    def test_cv_present_and_float(self) -> None:
        sr = _score(counts=(1, 50, 100, 50, 1))
        d = sr.to_dict()
        assert isinstance(d["cv"], float)

    def test_lines_profiled_matches_line_stats(self) -> None:
        sr = _score(counts=(1, 2, 3, 4, 5))   # 5 lines
        assert sr.lines_profiled == 5

    def test_error_count_in_dict(self) -> None:
        sr = _score(errors=["e1", "e2"])
        d = sr.to_dict()
        assert d["error_count"] == 2

    def test_grade_is_string(self) -> None:
        sr = _score()
        assert isinstance(sr.to_dict()["grade"], str)

    def test_narrative_is_string(self) -> None:
        sr = _score()
        assert isinstance(sr.to_dict()["narrative"], str)


# ---------------------------------------------------------------------------
# 13. Score clamping and total
# ---------------------------------------------------------------------------


class TestScoreClamping:
    """Final score must always be in [0.0, 100.0]."""

    def test_perfect_program_does_not_exceed_100(self) -> None:
        sr = _score(counts=(1,), suggestions=[], errors=[])
        assert sr.score <= 100.0

    def test_worst_case_does_not_go_below_0(self) -> None:
        suggestions = [_Suggestion(p, "high") for p in QUALITY_PATTERNS] * 50
        sr = _score(
            counts=(1,),
            suggestions=suggestions,
            source_lines=1,
            errors=["e"] * 10,
        )
        assert sr.score >= 0.0

    def test_total_equals_sum_of_dimensions(self) -> None:
        sr = _score(counts=(1, 50, 50, 1), suggestions=[], errors=[])
        expected = (
            sr.dimensions.correctness
            + sr.dimensions.efficiency_complexity
            + sr.dimensions.quality
            + sr.dimensions.maintainability
        )
        assert sr.score == pytest.approx(min(100.0, max(0.0, expected)), abs=0.01)


# ---------------------------------------------------------------------------
# 14. CV computation (informational, not used in scoring)
# ---------------------------------------------------------------------------


class TestCVComputation:
    """CV is present in ScoreReport but must not affect any dimension score."""

    def test_cv_present_in_report(self) -> None:
        sr = _score(counts=(1, 100, 100, 1))
        assert hasattr(sr, "cv")
        assert isinstance(sr.cv, float)

    def test_cv_zero_for_single_line(self) -> None:
        sr = _score(counts=(5,))
        assert sr.cv == 0.0

    def test_cv_does_not_influence_efficiency_score(self) -> None:
        """
        Two programs with the same optimizer suggestions but different CV
        values (flat vs. spiked) must score identically on efficiency.
        """
        flat_sr = _score(counts=(10, 10, 10, 10), suggestions=[])
        spiked_sr = _score(counts=(1, 1, 10000, 1), suggestions=[])
        assert flat_sr.dimensions.efficiency_subscore == (
            spiked_sr.dimensions.efficiency_subscore
        )

    def test_cv_positive_for_uneven_profile(self) -> None:
        sr = _score(counts=(1, 1, 10000, 1))
        assert sr.cv > 0.0


# ---------------------------------------------------------------------------
# 15. Weighted density helper
# ---------------------------------------------------------------------------


class TestWeightedDensity:
    """Scorer._weighted_density must weight severities 3:2:1 (high:med:low)."""

    def _density(
        self,
        suggestions: List[_Suggestion],
        source_lines: int,
    ) -> float:
        scorer = Scorer(
            profiling_data={"line_stats": _make_line_stats(1)},
            optimizer_report=_Report(suggestions),
            source_lines=source_lines,
        )
        return scorer._weighted_density(suggestions)

    def test_empty_suggestions_zero_density(self) -> None:
        assert self._density([], 10) == 0.0

    def test_one_low_suggestion(self) -> None:
        assert self._density([_Suggestion("unused_vars", "low")], 10) == pytest.approx(1 / 10)

    def test_one_medium_suggestion(self) -> None:
        assert self._density([_Suggestion("unused_vars", "medium")], 10) == pytest.approx(2 / 10)

    def test_one_high_suggestion(self) -> None:
        assert self._density([_Suggestion("unused_vars", "high")], 10) == pytest.approx(3 / 10)

    def test_mixed_severities(self) -> None:
        suggestions = [
            _Suggestion("hot_loop", "high"),      # 3
            _Suggestion("dead_code", "medium"),   # 2
            _Suggestion("unused_vars", "low"),    # 1
        ]
        # Total weight = 6, source_lines = 6 → density = 1.0
        assert self._density(suggestions, 6) == pytest.approx(1.0)

    def test_larger_source_reduces_density(self) -> None:
        s = [_Suggestion("unused_vars", "high")]
        small = self._density(s, 5)
        large = self._density(s, 50)
        assert large < small

    def test_more_suggestions_increases_density(self) -> None:
        one = self._density([_Suggestion("unused_vars", "low")], 10)
        five = self._density([_Suggestion("unused_vars", "low")] * 5, 10)
        assert five > one


# ---------------------------------------------------------------------------
# 16. calculate_score public API
# ---------------------------------------------------------------------------


class TestCalculateScoreAPI:
    """Public API integration tests."""

    def test_returns_score_report_instance(self) -> None:
        sr = calculate_score(
            profiling_data={"line_stats": _make_line_stats(1, 100, 1)},
            optimizer_report=_clean_report(),
            source_lines=5,
        )
        assert isinstance(sr, ScoreReport)

    def test_defaults_errors_to_empty(self) -> None:
        sr = calculate_score(
            profiling_data={"line_stats": _make_line_stats(1)},
            optimizer_report=_clean_report(),
            source_lines=5,
        )
        assert sr.error_count == 0
        assert sr.dimensions.correctness == 35.0

    def test_none_profiling_and_none_optimizer(self) -> None:
        sr = calculate_score(
            profiling_data=None,
            optimizer_report=None,
            source_lines=10,
        )
        assert isinstance(sr, ScoreReport)
        assert 0.0 <= sr.score <= 100.0

    def test_source_lines_default_of_1_does_not_crash(self) -> None:
        sr = calculate_score(
            profiling_data={"line_stats": _make_line_stats(1)},
            optimizer_report=_clean_report(),
        )
        assert isinstance(sr, ScoreReport)

    def test_score_in_valid_range(self) -> None:
        for _ in range(10):
            n_errors = random.randint(0, 5)
            n_suggestions = random.randint(0, 10)
            all_patterns = list(
                EFFICIENCY_PATTERNS | QUALITY_PATTERNS | MAINTAINABILITY_PATTERNS
            )
            suggestions = [
                _Suggestion(
                    random.choice(all_patterns),
                    random.choice(["low", "medium", "high"]),
                )
                for _ in range(n_suggestions)
            ]
            sr = calculate_score(
                profiling_data={"line_stats": _make_line_stats(*[random.randint(1, 1000) for _ in range(5)])},
                optimizer_report=_Report(suggestions),
                source_lines=random.randint(1, 50),
                errors=["err"] * n_errors,
            )
            assert 0.0 <= sr.score <= 100.0, f"Score out of range: {sr.score}"

    def test_complexity_class_in_known_set(self) -> None:
        known = set(COMPLEXITY_POINTS.keys()) | {"Unknown"}
        sr = calculate_score(
            profiling_data={"line_stats": _make_line_stats(1, 100, 100, 1)},
            optimizer_report=_clean_report(),
            source_lines=4,
        )
        assert sr.complexity_class in known
