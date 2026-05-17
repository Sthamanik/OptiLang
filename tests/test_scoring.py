"""
tests/test_scoring.py
---------------------
Comprehensive test suite for optilang/scoring.py.

Covers every testable behaviour in the current version:

    Pattern classification       — EFFICIENCY / QUALITY / MAINTAINABILITY sets
                                   including constant_folding in MAINTAINABILITY
    Correctness scoring          — density-based (errors / source_lines),
                                   same-size program judged proportionally
    Density-to-score             — linear interpolation, no step cliffs
    Complexity detection         — all eight Big-O classes
    Coverage-weighted complexity — hot_coverage blending: tiny hot path in a
                                   large program is penalised far less than a
                                   fully quadratic program
    Efficiency sub-score         — independent of complexity sub-score,
                                   sourced from EFFICIENCY_PATTERNS only
    Quality scoring              — dead_code, string_concat_loop only
    Maintainability scoring      — unused_vars, early_return, nested_loops,
                                   constant_folding
    Partial-credit flags         — profiling_partial, optimizer_partial
    Grade assignment             — all five grade bands
    Dimension ranking            — normalised ratio + tie-breaking by max
    Narrative generation         — dynamic hints from actual findings only,
                                   perfect score says no false weakest area,
                                   multiple weak dims all mentioned
    Dynamic hint building        — _build_dimension_hint specificity
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

import sys
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import pytest

# ---------------------------------------------------------------------------
# Path setup — allows running from the project root without installing the
# package, as long as scoring.py is in optilang/ or on PYTHONPATH.
# ---------------------------------------------------------------------------
sys.path.insert(0, ".")

from optilang.scoring import (   # noqa: E402
    COMPLEXITY_POINTS,
    EFFICIENCY_PATTERNS,
    MAX_MAINTAINABILITY,
    MAX_QUALITY,
    MAINTAINABILITY_PATTERNS,
    PARTIAL_COMPLEXITY,
    PARTIAL_EFFICIENCY,
    PARTIAL_MAINTAINABILITY,
    PARTIAL_QUALITY,
    QUALITY_PATTERNS,
    DimensionScores,
    ScoreReport,
    Scorer,
    _assign_grade,
    _build_dimension_hint,
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
    Correctness uses error density (errors / source_lines) fed through
    _density_to_score, so the same number of errors scores very differently
    depending on program size. This is the core dynamic scoring behaviour.
    """

    def test_zero_errors_full_marks(self) -> None:
        sr = _score(errors=[])
        assert sr.dimensions.correctness == 35.0

    def test_zero_errors_any_size_full_marks(self) -> None:
        for lines in [1, 5, 50, 200]:
            sr = calculate_score(
                profiling_data={"line_stats": _make_line_stats(*[1] * lines)},
                optimizer_report=_clean_report(),
                source_lines=lines,
                errors=[],
            )
            assert sr.dimensions.correctness == 35.0, (
                f"Zero errors in {lines}-line program must score 35.0"
            )

    def test_more_errors_lower_score(self) -> None:
        """More errors in same-size program → lower correctness."""
        sr1 = _score(errors=["e"] * 1, source_lines=20)
        sr3 = _score(errors=["e"] * 3, source_lines=20)
        assert sr1.dimensions.correctness > sr3.dimensions.correctness

    def test_same_errors_larger_program_scores_higher(self) -> None:
        """
        Core dynamic scoring test: 4 errors in 100 lines is much better
        than 4 errors in 5 lines. Larger program must score higher.
        """
        sr_small = calculate_score(
            profiling_data={"line_stats": _make_line_stats(*[1] * 5)},
            optimizer_report=_clean_report(),
            source_lines=5,
            errors=["e"] * 4,
        )
        sr_large = calculate_score(
            profiling_data={"line_stats": _make_line_stats(*[1] * 100)},
            optimizer_report=_clean_report(),
            source_lines=100,
            errors=["e"] * 4,
        )
        assert sr_large.dimensions.correctness > sr_small.dimensions.correctness, (
            "100-line program with 4 errors should score higher than "
            "5-line program with 4 errors"
        )

    def test_high_error_density_low_score(self) -> None:
        """A program where errors > 60% of lines should score poorly."""
        sr = calculate_score(
            profiling_data={"line_stats": _make_line_stats(1, 1, 1, 1, 1)},
            optimizer_report=_clean_report(),
            source_lines=5,
            errors=["e"] * 3,   # 3/5 = 0.60 density
        )
        # density=0.6 → 60% of max = 21.0
        assert sr.dimensions.correctness <= 21.5

    def test_low_error_density_high_score(self) -> None:
        """A program with < 5% error density should score near-perfect."""
        sr = calculate_score(
            profiling_data={"line_stats": _make_line_stats(*[1] * 100)},
            optimizer_report=_clean_report(),
            source_lines=100,
            errors=["e"] * 4,   # 4/100 = 0.04 density
        )
        # density=0.04 is between 0.0 and 0.3 anchors → near 80% of max
        assert sr.dimensions.correctness >= 28.0

    def test_correctness_uses_density_not_raw_count(self) -> None:
        """
        Prove it's density-based: 3 errors in 5 lines must score LOWER
        than 3 errors in 50 lines.
        """
        sr_dense = calculate_score(
            profiling_data={"line_stats": _make_line_stats(*[1] * 5)},
            optimizer_report=_clean_report(),
            source_lines=5,
            errors=["e"] * 3,
        )
        sr_sparse = calculate_score(
            profiling_data={"line_stats": _make_line_stats(*[1] * 50)},
            optimizer_report=_clean_report(),
            source_lines=50,
            errors=["e"] * 3,
        )
        assert sr_sparse.dimensions.correctness > sr_dense.dimensions.correctness

    def test_correctness_never_exceeds_max(self) -> None:
        for errs in [0, 1, 5, 10]:
            sr = _score(errors=["e"] * errs, source_lines=20)
            assert sr.dimensions.correctness <= 35.0

    def test_correctness_never_below_floor(self) -> None:
        """Even catastrophic error rate stays at or above 10% of max."""
        sr = calculate_score(
            profiling_data={"line_stats": _make_line_stats(1, 1, 1)},
            optimizer_report=_clean_report(),
            source_lines=3,
            errors=["e"] * 100,   # density >> 2.0
        )
        assert sr.dimensions.correctness >= 35.0 * 0.10 - 0.01

    def test_error_count_in_report(self) -> None:
        sr = _score(errors=["a", "b", "c"])
        assert sr.error_count == 3


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
            "O(n²)", "O(n² log n)", "O(n³)", "O(n⁴)", "O(n^k)", "O(2^n)",
            "O(k^n)", "O(n!)", "O(n + m)", "O(n*m)", "O(?)", "O(∞)",
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
                 "O(n²)", "O(n³)", "O(n⁴)", "O(n^k)", "O(2^n)",
                 "O(k^n)", "O(n!)", "O(∞)"]
        for i in range(len(order) - 1):
            assert COMPLEXITY_POINTS[order[i]] >= COMPLEXITY_POINTS[order[i + 1]], (
                f"{order[i]} should score ≥ {order[i + 1]}"
            )


# ---------------------------------------------------------------------------
# 4b. Coverage-weighted complexity scoring
# ---------------------------------------------------------------------------


class TestCoverageWeightedComplexity:
    """
    _coverage_weighted_complexity blends the base complexity penalty with
    hot_coverage (fraction of lines in the hot path). A tiny quadratic
    section in a large program is penalised far less than a fully
    quadratic program.
    """

    def _cwc(self, cls: str, *counts: int) -> float:
        from optilang.scoring import _coverage_weighted_complexity
        stats = _make_line_stats(*counts)
        return _coverage_weighted_complexity(cls, stats)

    def test_perfect_classes_always_15(self) -> None:
        for cls in ["O(1)", "O(log n)"]:
            assert self._cwc(cls, 1, 1, 1) == 15.0

    def test_tiny_hot_path_near_perfect(self) -> None:
        """2 quadratic lines in 200-line program → near full marks."""
        counts = [1] * 198 + [10000, 10000]
        score = self._cwc("O(n²)", *counts)
        assert score >= 14.0, (
            f"Tiny hot path should score near 15, got {score}"
        )

    def test_fully_hot_program_full_penalty(self) -> None:
        """Program fully in hot path → close to base COMPLEXITY_POINTS value."""
        counts = [10000, 10000, 10000, 10000, 10000]
        score = self._cwc("O(n²)", *counts)
        base = COMPLEXITY_POINTS["O(n²)"]   # 6.0
        assert score <= base + 1.5, (
            f"Fully hot O(n²) should score near {base}, got {score}"
        )

    def test_larger_hot_coverage_lower_score(self) -> None:
        """More of the program in the hot path → lower score."""
        small_hot = self._cwc("O(n²)", *([1] * 18 + [10000, 10000]))  # 2/20
        large_hot = self._cwc("O(n²)", *([10000] * 10 + [1] * 10))    # 10/20
        assert large_hot < small_hot

    def test_score_in_valid_range(self) -> None:
        for cls in COMPLEXITY_POINTS:
            score = self._cwc(cls, 1, 100, 10000, 1)
            assert 0.0 <= score <= 15.0, (
                f"{cls} produced out-of-range score {score}"
            )

    def test_empty_line_stats_returns_base(self) -> None:
        from optilang.scoring import _coverage_weighted_complexity
        score = _coverage_weighted_complexity("O(n²)", {})
        assert score == COMPLEXITY_POINTS["O(n²)"]

    def test_all_zero_counts_returns_perfect(self) -> None:
        """No execution → treat as O(1) (no evidence of complexity)."""
        score = self._cwc("O(n²)", 0, 0, 0)
        assert score == 15.0

    def test_same_class_larger_program_penalised_less(self) -> None:
        """
        Same O(n²) class, but small program is fully quadratic while large
        program has only a tiny quadratic section. Large program scores higher.
        """
        small_score = self._cwc("O(n²)", 1, 100, 10000, 10000, 1)
        large_score = self._cwc("O(n²)", *([1] * 195 + [10000, 10000, 10000, 10000, 1]))
        assert large_score > small_score


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
        complexity sub-score should be good, efficiency sub-score bad.
        The two values must differ — they measure different things.
        Coverage-weighted: O(n) with a loop gives moderate hot_coverage,
        so complexity_sub will be between 13 and 15, not exactly 13.
        """
        suggestions = [_Suggestion("loop_invariant", "high")] * 5
        sr = _score(
            counts=(1, 100, 100, 1),
            suggestions=suggestions,
            source_lines=4,
        )
        # Complexity sub should reflect O(n) class but be coverage-weighted
        assert sr.dimensions.complexity_subscore >= 10.0    # O(n) is good
        assert sr.dimensions.efficiency_subscore < 10.0     # penalised heavily

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
        assert sr.complexity_class == "O(?)"

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
# 11. Narrative generation — dynamic, suggestion-specific hints
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
        - include ONLY the issues that were actually detected — no generic
          laundry lists of every possible problem in a dimension
        - include accurate partial-credit notes when data was unavailable
    """

    @dataclass
    class _Sugg:
        pattern: str
        severity: str

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

    def _narrative(
        self,
        score: float,
        dims: DimensionScores,
        complexity_class: str = "O(n)",
        suggestions: Optional[List] = None,
    ) -> str:
        return _generate_narrative(
            score,
            dims,
            complexity_class=complexity_class,
            all_suggestions=suggestions or [],
        )

    # ── Grade headlines ───────────────────────────────────────────────

    def test_excellent_headline(self) -> None:
        dims = DimensionScores(
            correctness=35.0, efficiency_complexity=30.0,
            quality=20.0, maintainability=15.0,
        )
        assert "Excellent" in self._narrative(95.0, dims)

    def test_good_headline(self) -> None:
        dims = self._dims_with_lowest("Maintainability")
        assert "Good" in self._narrative(80.0, dims)

    def test_fair_headline(self) -> None:
        dims = self._dims_with_lowest("Quality")
        assert "Fair" in self._narrative(65.0, dims)

    def test_poor_headline(self) -> None:
        dims = self._dims_with_lowest("Correctness")
        n = self._narrative(45.0, dims)
        assert "needs some work" in n.lower() or "poor" in n.lower()

    def test_critical_headline(self) -> None:
        dims = self._dims_with_lowest("Correctness")
        n = self._narrative(20.0, dims)
        assert "significant issues" in n.lower()

    # ── Perfect score: no false "weakest area" ────────────────────────

    def test_perfect_score_no_weakest_area_mention(self) -> None:
        dims = DimensionScores(
            correctness=35.0, efficiency_complexity=30.0,
            quality=20.0, maintainability=15.0,
        )
        n = self._narrative(100.0, dims)
        assert "weakest area" not in n.lower()
        assert "area to improve" not in n.lower()
        assert "full marks" in n.lower()

    def test_perfect_score_positive_only_narrative(self) -> None:
        dims = DimensionScores(
            correctness=35.0, efficiency_complexity=30.0,
            quality=20.0, maintainability=15.0,
        )
        n = self._narrative(100.0, dims)
        assert "Excellent" in n
        assert "every dimension" in n.lower() or "full marks" in n.lower()

    # ── Specificity: only detected issues are mentioned ───────────────

    def test_no_nested_loop_mention_when_none_detected(self) -> None:
        """
        Core fix: a program with O(n) complexity and only an unused_vars
        suggestion must NOT mention nested loops in the narrative.
        """
        dims = DimensionScores(
            correctness=35.0,
            efficiency_complexity=28.0,   # O(n)=13 + efficiency=15
            quality=20.0,
            maintainability=13.9,         # unused_vars deduction
            complexity_subscore=13.0,
            efficiency_subscore=15.0,
        )
        suggs = [self._Sugg("unused_vars", "low")]
        n = self._narrative(96.9, dims, complexity_class="O(n)", suggestions=suggs)
        assert "nested loop" not in n.lower(), (
            "Narrative must not mention nested loops when none were detected"
        )

    def test_loop_invariant_mentioned_only_when_detected(self) -> None:
        dims = DimensionScores(
            correctness=35.0, efficiency_complexity=20.0,
            quality=20.0, maintainability=15.0,
            complexity_subscore=13.0, efficiency_subscore=7.0,
        )
        suggs = [self._Sugg("loop_invariant", "medium")]
        n = self._narrative(90.0, dims, complexity_class="O(n)", suggestions=suggs)
        assert "invariant" in n.lower() or "never changes" in n.lower()

    def test_loop_invariant_not_mentioned_when_not_detected(self) -> None:
        dims = DimensionScores(
            correctness=35.0, efficiency_complexity=20.0,
            quality=20.0, maintainability=15.0,
            complexity_subscore=13.0, efficiency_subscore=7.0,
        )
        # No suggestions — only complexity sub-score is below perfect
        n = self._narrative(90.0, dims, complexity_class="O(n)", suggestions=[])
        assert "invariant" not in n.lower()

    def test_unused_vars_mentioned_only_when_detected(self) -> None:
        dims = self._dims_with_lowest("Maintainability")
        suggs = [self._Sugg("unused_vars", "low")]
        n = self._narrative(80.0, dims, suggestions=suggs)
        assert "never used" in n.lower() or "assigned but never" in n.lower()

    def test_dead_code_mentioned_only_when_detected(self) -> None:
        dims = self._dims_with_lowest("Quality")
        suggs = [self._Sugg("dead_code", "medium")]
        n = self._narrative(65.0, dims, suggestions=suggs)
        assert "dead" in n.lower() or "never execute" in n.lower()

    def test_complexity_class_mentioned_in_efficiency_hint(self) -> None:
        """Complexity class is surfaced when complexity_sub < 15."""
        dims = DimensionScores(
            correctness=35.0, efficiency_complexity=28.0,
            quality=20.0, maintainability=15.0,
            complexity_subscore=13.0, efficiency_subscore=15.0,
        )
        n = self._narrative(96.0, dims, complexity_class="O(n)", suggestions=[])
        assert "O(n)" in n or "linear" in n.lower()

    def test_quadratic_complexity_mentioned_when_detected(self) -> None:
        dims = DimensionScores(
            correctness=35.0, efficiency_complexity=21.0,
            quality=20.0, maintainability=15.0,
            complexity_subscore=6.0, efficiency_subscore=15.0,
        )
        n = self._narrative(91.0, dims, complexity_class="O(n²)", suggestions=[])
        assert "quadratic" in n.lower() or "O(n²)" in n or "nested" in n.lower()

    def test_no_complexity_hint_when_complexity_perfect(self) -> None:
        """When complexity_sub=15 (O(1) or O(log n)), no complexity message needed."""
        dims = DimensionScores(
            correctness=35.0, efficiency_complexity=22.0,
            quality=20.0, maintainability=15.0,
            complexity_subscore=15.0, efficiency_subscore=7.0,
        )
        suggs = [self._Sugg("repeated_computation", "medium")]
        n = self._narrative(92.0, dims, complexity_class="O(1)", suggestions=suggs)
        # Should mention the repeated_computation but not any complexity class
        assert "repeated" in n.lower() or "same expression" in n.lower()
        assert "quadratic" not in n.lower()
        assert "linear" not in n.lower()

    # ── Multiple actionable dimensions all mentioned ──────────────────

    def test_two_bad_dims_both_mentioned(self) -> None:
        dims = DimensionScores(
            correctness=35.0,
            efficiency_complexity=24.7,
            quality=20.0,
            maintainability=10.0,
            complexity_subscore=13.0,
            efficiency_subscore=11.7,
        )
        suggs = [self._Sugg("unused_vars", "low"), self._Sugg("loop_invariant", "medium")]
        n = self._narrative(89.7, dims, complexity_class="O(n)", suggestions=suggs)
        assert "Efficiency" in n
        assert "Maintainability" in n

    def test_two_bad_dims_higher_missing_listed_first(self) -> None:
        dims = DimensionScores(
            correctness=35.0,
            efficiency_complexity=24.7,   # missing 5.3
            quality=20.0,
            maintainability=10.0,         # missing 5.0
            complexity_subscore=13.0,
            efficiency_subscore=11.7,
        )
        suggs = [self._Sugg("unused_vars", "low")]
        n = self._narrative(89.7, dims, complexity_class="O(n)", suggestions=suggs)
        eff_pos = n.find("Efficiency")
        maint_pos = n.find("Maintainability")
        assert eff_pos < maint_pos

    def test_all_bad_dims_all_mentioned(self) -> None:
        dims = DimensionScores(
            correctness=25.0,
            efficiency_complexity=15.0,
            quality=8.0,
            maintainability=5.0,
            complexity_subscore=6.0,
            efficiency_subscore=9.0,
        )
        suggs = [
            self._Sugg("loop_invariant", "high"),
            self._Sugg("dead_code", "medium"),
            self._Sugg("unused_vars", "low"),
        ]
        n = self._narrative(53.0, dims, complexity_class="O(n²)", suggestions=suggs)
        assert "Correctness" in n
        assert "Efficiency" in n
        assert "Quality" in n
        assert "Maintainability" in n

    # ── Healthy but not perfect: gentle note ─────────────────────────

    def test_all_healthy_not_perfect_gives_gentle_note(self) -> None:
        dims = DimensionScores(
            correctness=33.0,
            efficiency_complexity=28.0,
            quality=19.0,
            maintainability=14.0,
            complexity_subscore=13.0,
            efficiency_subscore=15.0,
        )
        n = self._narrative(94.0, dims, complexity_class="O(n)", suggestions=[])
        assert "great shape" in n.lower() or "squeeze" in n.lower()

    def test_all_healthy_not_perfect_no_alarm_language(self) -> None:
        dims = DimensionScores(
            correctness=33.0,
            efficiency_complexity=28.0,
            quality=19.0,
            maintainability=14.0,
            complexity_subscore=13.0,
            efficiency_subscore=15.0,
        )
        n = self._narrative(94.0, dims, complexity_class="O(n)", suggestions=[])
        assert "needs some work" not in n.lower()
        assert "significant issues" not in n.lower()

    # ── Partial-credit notes ──────────────────────────────────────────

    def test_no_partial_note_when_all_data_present(self) -> None:
        dims = DimensionScores(
            correctness=35.0, efficiency_complexity=30.0,
            quality=20.0, maintainability=15.0,
            profiling_partial=False, optimizer_partial=False,
        )
        assert "Note:" not in self._narrative(95.0, dims)

    def test_profiling_partial_note_mentions_complexity(self) -> None:
        dims = DimensionScores(
            correctness=35.0, efficiency_complexity=22.0,
            quality=20.0, maintainability=15.0,
            profiling_partial=True, optimizer_partial=False,
        )
        n = self._narrative(85.0, dims)
        assert "Note:" in n
        assert "Complexity" in n
        assert "Efficiency sub-score" not in n or "Complexity sub-score only" in n

    def test_optimizer_partial_note_mentions_all_three(self) -> None:
        dims = DimensionScores(
            correctness=35.0, efficiency_complexity=21.0,
            quality=10.0, maintainability=7.0,
            profiling_partial=False, optimizer_partial=True,
        )
        n = self._narrative(73.0, dims)
        assert "Note:" in n
        assert "Efficiency" in n
        assert "Quality" in n
        assert "Maintainability" in n

    def test_both_partial_note_mentions_all_four(self) -> None:
        dims = DimensionScores(
            correctness=35.0, efficiency_complexity=15.0,
            quality=10.0, maintainability=7.0,
            profiling_partial=True, optimizer_partial=True,
        )
        n = self._narrative(67.0, dims)
        assert "Note:" in n
        assert "Complexity" in n
        assert "Efficiency" in n
        assert "Quality" in n
        assert "Maintainability" in n


# ---------------------------------------------------------------------------
# 11b. _build_dimension_hint — dynamic hint specificity
# ---------------------------------------------------------------------------


class TestBuildDimensionHint:
    """_build_dimension_hint must mention only detected issues."""

    @dataclass
    class _S:
        pattern: str
        severity: str

    def test_correctness_hint_always_same(self) -> None:
        h = _build_dimension_hint("Correctness", "O(n)", 13.0, [])
        assert "error" in h.lower()

    def test_efficiency_no_issues_generic_encouragement(self) -> None:
        """complexity=15 (perfect), no suggestions → no alarm, just encouragement."""
        h = _build_dimension_hint("Efficiency & Complexity", "O(1)", 15.0, [])
        assert "good" in h.lower() or "keep" in h.lower()

    def test_efficiency_complexity_class_surfaced(self) -> None:
        h = _build_dimension_hint("Efficiency & Complexity", "O(n²)", 6.0, [])
        assert "quadratic" in h.lower() or "O(n²)" in h or "nested" in h.lower()

    def test_efficiency_suggestion_surfaced(self) -> None:
        suggs = [self._S("loop_invariant", "medium")]
        h = _build_dimension_hint("Efficiency & Complexity", "O(n)", 13.0, suggs)
        assert "invariant" in h.lower() or "never changes" in h.lower()

    def test_efficiency_no_nested_loop_when_not_in_suggestions(self) -> None:
        """nested_loops is a MAINTAINABILITY pattern — must never appear in efficiency hint."""
        h = _build_dimension_hint("Efficiency & Complexity", "O(n)", 13.0, [])
        assert "nested loop" not in h.lower()

    def test_quality_only_detected_patterns(self) -> None:
        suggs = [self._S("dead_code", "medium")]
        h = _build_dimension_hint("Quality", "O(n)", 13.0, suggs)
        assert "dead" in h.lower() or "never execute" in h.lower()
        assert "string" not in h.lower()   # string_concat_loop not in suggestions

    def test_maintainability_only_detected_patterns(self) -> None:
        suggs = [self._S("unused_vars", "low")]
        h = _build_dimension_hint("Maintainability", "O(n)", 13.0, suggs)
        assert "never used" in h.lower() or "assigned but never" in h.lower()
        assert "nested" not in h.lower()   # nested_loops not in suggestions
        assert "early" not in h.lower()    # early_return not in suggestions

    def test_unknown_pattern_does_not_crash(self) -> None:
        suggs = [self._S("brand_new_pattern", "low")]
        h = _build_dimension_hint("Quality", "O(n)", 13.0, suggs)
        assert isinstance(h, str)


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
        known = set(COMPLEXITY_POINTS.keys()) | {"Unknown", "O(?)"}
        sr = calculate_score(
            profiling_data={"line_stats": _make_line_stats(1, 100, 100, 1)},
            optimizer_report=_clean_report(),
            source_lines=4,
        )
        assert sr.complexity_class in known
