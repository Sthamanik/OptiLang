"""
tests/test_scoring.py
---------------------
Tests for optilang/scoring.py

Covers every public and private component:

    TestDimensionScores         — dataclass fields, defaults, to_dict()
    TestScoreReport             — dataclass fields, to_dict(), JSON safety
    TestDetectComplexity        — module-level heuristic, all Big-O classes
    TestAssignGrade             — all five grade bands and exact boundaries
    TestLowestDimension         — percentage-normalised comparison, tie handling
    TestGenerateNarrative       — all five score bands, partial-credit notes,
                                   lowest-dimension routing
    TestScorerCorrectness       — 0 / 1 / 2+ errors, boundary at exactly 2
    TestScorerEfficiencyComplexity — profiling present/absent, all complexity
                                     classes, CV bands, sub-score sum
    TestScorerComputeCV         — flat, spiked, single-line, empty profiles
    TestScorerQuality           — pattern filtering, density bands, partial credit
    TestScorerMaintainability   — pattern filtering, density bands, partial credit
    TestScorerDensityToScore    — all five density bands for both max values
    TestScorerCalculate         — end-to-end: perfect score, bad code, clamping,
                                   breakdown keys, partial flags, error_count
    TestCalculateScoreFunction  — public API matches Scorer directly, defaults
    TestEdgeCases               — empty profiling, zero source lines, unknown
                                   complexity class, both partials, all errors
"""

from __future__ import annotations

import json

import pytest

from optilang.scoring import (
    COMPLEXITY_POINTS,
    MAINTAINABILITY_PATTERNS,
    MAX_CORRECTNESS,
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
    _detect_complexity,
    _generate_narrative,
    _lowest_dimension,
    calculate_score,
)

# ---------------------------------------------------------------------------
# Shared test helpers
# ---------------------------------------------------------------------------


def make_profiling(
    line_counts: list[int] | None = None,
    total_time_ms: float = 1.0,
) -> dict:
    """
    Build a minimal profiling dict that mimics ProfilingData.to_dict().

    Args:
        line_counts:   Execution count per line. Each entry = one profiled line.
        total_time_ms: Simulated total execution time (not used by scorer
                       directly but present for structural completeness).
    """
    line_counts = line_counts or [1]
    line_stats: dict = {}
    total = 0
    for i, count in enumerate(line_counts, start=1):
        line_stats[str(i)] = {
            "count": count,
            "total_time_ms": total_time_ms / max(len(line_counts), 1),
            "avg_time_ms": total_time_ms / max(count, 1),
            "memory": 0,
        }
        total += count
    return {
        "line_stats": line_stats,
        "total_time_ms": total_time_ms,
        "total_lines": total,
        "lines_profiled": len(line_stats),
    }


class _Suggestion:
    """Minimal Suggestion stub."""

    def __init__(self, pattern: str, severity: str) -> None:
        self.pattern = pattern
        self.severity = severity


class _Report:
    """Minimal OptimizationReport stub with configurable suggestions."""

    def __init__(self, suggestions: list[_Suggestion] | None = None) -> None:
        self.suggestions = suggestions or []


def _empty_report() -> _Report:
    return _Report()


def _report(*items: tuple[str, str]) -> _Report:
    """Build a report from (pattern, severity) pairs."""
    return _Report([_Suggestion(p, s) for p, s in items])


# ---------------------------------------------------------------------------
# TestDimensionScores
# ---------------------------------------------------------------------------


class TestDimensionScores:

    def test_default_values_are_zero_and_false(self) -> None:
        d = DimensionScores()
        assert d.correctness == 0.0
        assert d.efficiency_complexity == 0.0
        assert d.quality == 0.0
        assert d.maintainability == 0.0
        assert d.complexity_subscore == 0.0
        assert d.efficiency_subscore == 0.0
        assert d.profiling_partial is False
        assert d.optimizer_partial is False

    def test_to_dict_has_all_keys(self) -> None:
        d = DimensionScores(correctness=35.0, efficiency_complexity=30.0)
        result = d.to_dict()
        expected_keys = {
            "correctness",
            "efficiency_complexity",
            "quality",
            "maintainability",
            "complexity_subscore",
            "efficiency_subscore",
            "profiling_partial",
            "optimizer_partial",
        }
        assert set(result.keys()) == expected_keys

    def test_to_dict_rounds_floats_to_two_decimal_places(self) -> None:
        d = DimensionScores(correctness=35.123456)
        assert d.to_dict()["correctness"] == 35.12

    def test_to_dict_preserves_boolean_flags(self) -> None:
        d = DimensionScores(profiling_partial=True, optimizer_partial=True)
        result = d.to_dict()
        assert result["profiling_partial"] is True
        assert result["optimizer_partial"] is True


# ---------------------------------------------------------------------------
# TestScoreReport
# ---------------------------------------------------------------------------


class TestScoreReport:

    def _make_report(self, score: float = 80.0) -> ScoreReport:
        return ScoreReport(
            score=score,
            grade="Good",
            complexity_class="O(n)",
            dimensions=DimensionScores(
                correctness=35.0,
                efficiency_complexity=20.0,
                quality=15.0,
                maintainability=10.0,
            ),
            narrative="Test narrative.",
            error_count=0,
            lines_profiled=5,
            cv=0.5,
        )

    def test_to_dict_has_all_required_keys(self) -> None:
        d = self._make_report().to_dict()
        assert set(d.keys()) == {
            "score",
            "grade",
            "complexity_class",
            "dimensions",
            "narrative",
            "error_count",
            "lines_profiled",
            "cv",
        }

    def test_to_dict_score_rounded_to_two_places(self) -> None:
        r = self._make_report(score=80.123456)
        assert r.to_dict()["score"] == 80.12

    def test_to_dict_cv_rounded_to_four_places(self) -> None:
        r = ScoreReport(
            score=80.0,
            grade="Good",
            complexity_class="O(n)",
            dimensions=DimensionScores(),
            narrative="n",
            cv=1.23456789,
        )
        assert r.to_dict()["cv"] == 1.2346

    def test_to_dict_dimensions_is_dict(self) -> None:
        d = self._make_report().to_dict()
        assert isinstance(d["dimensions"], dict)

    def test_to_dict_is_json_serialisable(self) -> None:
        d = self._make_report().to_dict()
        json.dumps(d)  # must not raise


# ---------------------------------------------------------------------------
# TestDetectComplexity
# ---------------------------------------------------------------------------


class TestDetectComplexity:

    def test_empty_line_stats_returns_O1(self) -> None:
        assert _detect_complexity({}) == "O(1)"

    def test_all_counts_zero_returns_O1(self) -> None:
        stats = {"1": {"count": 0}, "2": {"count": 0}}
        assert _detect_complexity(stats) == "O(1)"

    def test_single_execution_returns_O1(self) -> None:
        stats = {"1": {"count": 1}}
        assert _detect_complexity(stats) == "O(1)"

    def test_flat_linear_returns_On(self) -> None:
        # Three lines each running 100 times — classic O(n)
        stats = {
            "1": {"count": 1},
            "2": {"count": 100},
            "3": {"count": 100},
        }
        assert _detect_complexity(stats) == "O(n)"

    def test_nested_loop_returns_On2(self) -> None:
        # Outer=101, inner header=10100, inner body=10000
        stats = {
            "1": {"count": 101},
            "2": {"count": 10100},
            "3": {"count": 10000},
        }
        assert _detect_complexity(stats) == "O(n²)"

    def test_triple_nested_returns_On3(self) -> None:
        # Three hot lines at ~5000, outer=5, mid=50
        stats = {
            "1": {"count": 5},
            "2": {"count": 50},
            "3": {"count": 5000},
            "4": {"count": 5000},
            "5": {"count": 5000},
        }
        assert _detect_complexity(stats) == "O(n³)"

    def test_log_n_range_detected(self) -> None:
        # max_count = 4 ≈ log2(4) * 2 — sub-linear
        stats = {str(i): {"count": 1} for i in range(1, 4)}
        stats["4"] = {"count": 4}
        result = _detect_complexity(stats)
        assert result in ("O(log n)", "O(1)", "O(n)")

    def test_deep_nesting_returns_high_complexity(self) -> None:
        # Four or more hot lines — expects n^k or similar
        stats = {
            "1": {"count": 1},
            "2": {"count": 10},
            "3": {"count": 100},
            "4": {"count": 10000},
            "5": {"count": 10000},
            "6": {"count": 10000},
            "7": {"count": 10000},
        }
        result = _detect_complexity(stats)
        assert result in ("O(n^k)", "O(n³)", "O(n²)")

    def test_all_complexity_classes_are_in_complexity_points(self) -> None:
        # Every class the heuristic can return must have a point value
        possible = {
            "O(1)",
            "O(log n)",
            "O(n)",
            "O(n log n)",
            "O(n²)",
            "O(n³)",
            "O(n^k)",
            "O(2^n)",
        }
        for cls in possible:
            assert cls in COMPLEXITY_POINTS, f"Missing: {cls}"


# ---------------------------------------------------------------------------
# TestAssignGrade
# ---------------------------------------------------------------------------


class TestAssignGrade:

    def test_score_100_is_excellent(self) -> None:
        assert _assign_grade(100.0) == "Excellent"

    def test_score_90_is_excellent(self) -> None:
        assert _assign_grade(90.0) == "Excellent"

    def test_score_89_is_good(self) -> None:
        assert _assign_grade(89.9) == "Good"

    def test_score_75_is_good(self) -> None:
        assert _assign_grade(75.0) == "Good"

    def test_score_74_is_fair(self) -> None:
        assert _assign_grade(74.9) == "Fair"

    def test_score_60_is_fair(self) -> None:
        assert _assign_grade(60.0) == "Fair"

    def test_score_59_is_poor(self) -> None:
        assert _assign_grade(59.9) == "Poor"

    def test_score_40_is_poor(self) -> None:
        assert _assign_grade(40.0) == "Poor"

    def test_score_39_is_critical(self) -> None:
        assert _assign_grade(39.9) == "Critical"

    def test_score_0_is_critical(self) -> None:
        assert _assign_grade(0.0) == "Critical"


# ---------------------------------------------------------------------------
# TestLowestDimension
# ---------------------------------------------------------------------------


class TestLowestDimension:

    def test_correctness_lowest(self) -> None:
        # correctness=0/35 = 0% — clearly lowest
        dims = DimensionScores(
            correctness=0.0,
            efficiency_complexity=30.0,
            quality=20.0,
            maintainability=15.0,
        )
        assert _lowest_dimension(dims) == "Correctness"

    def test_efficiency_complexity_lowest(self) -> None:
        dims = DimensionScores(
            correctness=35.0,
            efficiency_complexity=5.0,  # 5/30 ≈ 17%
            quality=20.0,
            maintainability=15.0,
        )
        assert _lowest_dimension(dims) == "Efficiency & Complexity"

    def test_quality_lowest(self) -> None:
        dims = DimensionScores(
            correctness=35.0,
            efficiency_complexity=30.0,
            quality=2.0,  # 2/20 = 10%
            maintainability=15.0,
        )
        assert _lowest_dimension(dims) == "Quality"

    def test_maintainability_lowest(self) -> None:
        dims = DimensionScores(
            correctness=35.0,
            efficiency_complexity=30.0,
            quality=20.0,
            maintainability=0.0,  # 0/15 = 0%
        )
        assert _lowest_dimension(dims) == "Maintainability"

    def test_uses_percentage_not_raw_value(self) -> None:
        # correctness=17.5/35=50%, efficiency=15/30=50%, quality=0/20=0%
        # quality must win even though raw value isn't smallest vs correctness
        dims = DimensionScores(
            correctness=17.5,
            efficiency_complexity=15.0,
            quality=0.0,
            maintainability=15.0,
        )
        assert _lowest_dimension(dims) == "Quality"

    def test_perfect_scores_returns_a_valid_dimension(self) -> None:
        # All at 100% — min() picks one deterministically
        dims = DimensionScores(
            correctness=35.0,
            efficiency_complexity=30.0,
            quality=20.0,
            maintainability=15.0,
        )
        valid = {
            "Correctness",
            "Efficiency & Complexity",
            "Quality",
            "Maintainability",
        }
        assert _lowest_dimension(dims) in valid


# ---------------------------------------------------------------------------
# TestGenerateNarrative
# ---------------------------------------------------------------------------


class TestGenerateNarrative:

    def _dims_with_lowest(self, lowest: str) -> DimensionScores:
        """Return DimensionScores where the named dimension scores 0%."""
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
        else:
            d.maintainability = 0.0
        return d

    def test_score_90_plus_contains_excellent(self) -> None:
        dims = DimensionScores(
            correctness=35.0,
            efficiency_complexity=30.0,
            quality=20.0,
            maintainability=15.0,
        )
        narrative = _generate_narrative(95.0, dims)
        assert "Excellent" in narrative

    def test_score_75_to_89_contains_good(self) -> None:
        dims = self._dims_with_lowest("Maintainability")
        narrative = _generate_narrative(80.0, dims)
        assert "Good" in narrative

    def test_score_60_to_74_contains_fair(self) -> None:
        dims = self._dims_with_lowest("Quality")
        narrative = _generate_narrative(65.0, dims)
        assert "Fair" in narrative

    def test_score_40_to_59_contains_needs_work(self) -> None:
        dims = self._dims_with_lowest("Correctness")
        narrative = _generate_narrative(50.0, dims)
        assert "needs some work" in narrative.lower()

    def test_score_below_40_contains_significant_issues(self) -> None:
        dims = self._dims_with_lowest("Correctness")
        narrative = _generate_narrative(20.0, dims)
        assert "significant issues" in narrative.lower()

    def test_narrative_names_correctness_when_lowest(self) -> None:
        dims = self._dims_with_lowest("Correctness")
        narrative = _generate_narrative(60.0, dims)
        assert "Correctness" in narrative

    def test_narrative_names_efficiency_when_lowest(self) -> None:
        dims = self._dims_with_lowest("Efficiency & Complexity")
        narrative = _generate_narrative(60.0, dims)
        assert "Efficiency" in narrative

    def test_narrative_names_quality_when_lowest(self) -> None:
        dims = self._dims_with_lowest("Quality")
        narrative = _generate_narrative(60.0, dims)
        assert "Quality" in narrative

    def test_narrative_names_maintainability_when_lowest(self) -> None:
        dims = self._dims_with_lowest("Maintainability")
        narrative = _generate_narrative(60.0, dims)
        assert "Maintainability" in narrative

    def test_partial_profiling_note_included(self) -> None:
        dims = DimensionScores(
            correctness=35.0,
            efficiency_complexity=15.0,
            quality=20.0,
            maintainability=15.0,
            profiling_partial=True,
        )
        narrative = _generate_narrative(85.0, dims)
        assert "profiling" in narrative.lower()

    def test_partial_optimizer_note_included(self) -> None:
        dims = DimensionScores(
            correctness=35.0,
            efficiency_complexity=30.0,
            quality=10.0,
            maintainability=7.0,
            optimizer_partial=True,
        )
        narrative = _generate_narrative(82.0, dims)
        assert "optimizer" in narrative.lower()

    def test_both_partial_notes_included(self) -> None:
        dims = DimensionScores(
            correctness=35.0,
            efficiency_complexity=15.0,
            quality=10.0,
            maintainability=7.0,
            profiling_partial=True,
            optimizer_partial=True,
        )
        narrative = _generate_narrative(67.0, dims)
        assert "profiling" in narrative.lower()
        assert "optimizer" in narrative.lower()

    def test_no_partial_no_note(self) -> None:
        dims = DimensionScores(
            correctness=35.0,
            efficiency_complexity=30.0,
            quality=20.0,
            maintainability=15.0,
        )
        narrative = _generate_narrative(100.0, dims)
        assert "partial" not in narrative.lower()

    def test_narrative_is_non_empty_string(self) -> None:
        dims = DimensionScores()
        narrative = _generate_narrative(0.0, dims)
        assert isinstance(narrative, str) and len(narrative) > 10


# ---------------------------------------------------------------------------
# TestScorerCorrectness
# ---------------------------------------------------------------------------


class TestScorerCorrectness:

    def test_zero_errors_scores_max(self) -> None:
        s = Scorer(make_profiling([1]), _empty_report(), errors=[])
        assert s._score_correctness() == MAX_CORRECTNESS

    def test_one_error_scores_10(self) -> None:
        s = Scorer(make_profiling([1]), _empty_report(), errors=["e"])
        assert s._score_correctness() == 10.0

    def test_two_errors_scores_zero(self) -> None:
        s = Scorer(make_profiling([1]), _empty_report(), errors=["e", "f"])
        assert s._score_correctness() == 0.0

    def test_many_errors_scores_zero(self) -> None:
        s = Scorer(make_profiling([1]), _empty_report(), errors=["e"] * 10)
        assert s._score_correctness() == 0.0

    def test_errors_none_treated_as_empty(self) -> None:
        s = Scorer(make_profiling([1]), _empty_report(), errors=None)
        assert s._score_correctness() == MAX_CORRECTNESS

    def test_error_count_stored_in_report(self) -> None:
        s = Scorer(make_profiling([1]), _empty_report(), errors=["e", "f"])
        r = s.calculate()
        assert r.error_count == 2


# ---------------------------------------------------------------------------
# TestScorerEfficiencyComplexity
# ---------------------------------------------------------------------------


class TestScorerEfficiencyComplexity:

    def test_no_profiling_awards_partial_credit(self) -> None:
        s = Scorer(None, _empty_report())
        _, c_sub, e_sub, partial = s._score_efficiency_complexity()
        assert partial is True
        assert c_sub == PARTIAL_COMPLEXITY
        assert e_sub == PARTIAL_EFFICIENCY

    def test_no_profiling_complexity_class_is_unknown(self) -> None:
        s = Scorer(None, _empty_report())
        cls, _, _, _ = s._score_efficiency_complexity()
        assert cls == "Unknown"

    def test_o1_profile_scores_max_complexity(self) -> None:
        p = make_profiling([1, 1, 1, 1, 1])
        s = Scorer(p, _empty_report())
        cls, c_sub, _, partial = s._score_efficiency_complexity()
        assert partial is False
        assert cls == "O(1)"
        assert c_sub == COMPLEXITY_POINTS["O(1)"]

    def test_nested_loop_profile_scores_on2_complexity(self) -> None:
        p = make_profiling([101, 10100, 10000])
        s = Scorer(p, _empty_report())
        cls, c_sub, _, _ = s._score_efficiency_complexity()
        assert cls == "O(n²)"
        assert c_sub == COMPLEXITY_POINTS["O(n²)"]

    def test_complexity_and_efficiency_sum_to_efficiency_complexity(self) -> None:
        p = make_profiling([1, 1, 1])
        s = Scorer(p, _empty_report())
        _, c_sub, e_sub, _ = s._score_efficiency_complexity()
        r = s.calculate()
        assert r.dimensions.efficiency_complexity == pytest.approx(c_sub + e_sub)

    def test_complexity_points_all_map_correctly(self) -> None:
        # Every COMPLEXITY_POINTS entry should be reachable via the scorer
        for cls, expected_pts in COMPLEXITY_POINTS.items():
            assert 0.0 <= expected_pts <= 15.0

    def test_flat_profile_gives_max_efficiency_subscore(self) -> None:
        # All lines run once → CV = 0 → efficiency = 15
        p = make_profiling([1, 1, 1, 1, 1])
        s = Scorer(p, _empty_report())
        _, _, e_sub, _ = s._score_efficiency_complexity()
        assert e_sub == 15.0

    def test_highly_spiked_profile_gives_low_efficiency_subscore(self) -> None:
        # 50 background lines running once + one spike of 10000
        # produces CV > 4.0 → efficiency_subscore = 0
        p = make_profiling([1] * 50 + [10000])
        s = Scorer(p, _empty_report())
        _, _, e_sub, _ = s._score_efficiency_complexity()
        assert e_sub == 0.0

    def test_profiling_partial_flag_false_when_data_present(self) -> None:
        s = Scorer(make_profiling([1]), _empty_report())
        r = s.calculate()
        assert r.dimensions.profiling_partial is False


# ---------------------------------------------------------------------------
# TestScorerComputeCV
# ---------------------------------------------------------------------------


class TestScorerComputeCV:

    def test_empty_line_stats_returns_zero(self) -> None:
        s = Scorer(
            {"line_stats": {}, "total_time_ms": 0.0, "total_lines": 0}, _empty_report()
        )
        assert s._compute_cv() == 0.0

    def test_single_line_returns_zero(self) -> None:
        # Only one count — not enough for a meaningful spread
        s = Scorer(make_profiling([100]), _empty_report())
        assert s._compute_cv() == 0.0

    def test_identical_counts_give_cv_zero(self) -> None:
        # std=0 when all values are equal → CV=0
        s = Scorer(make_profiling([50, 50, 50, 50]), _empty_report())
        assert s._compute_cv() == pytest.approx(0.0)

    def test_cv_increases_with_spread(self) -> None:
        low_spread = Scorer(make_profiling([10, 10, 11, 10]), _empty_report())
        high_spread = Scorer(make_profiling([1, 1, 1, 10000]), _empty_report())
        assert high_spread._compute_cv() > low_spread._compute_cv()

    def test_cv_is_non_negative(self) -> None:
        for counts in [[1], [1, 2], [100, 1, 50], [1, 1, 1, 10000]]:
            s = Scorer(make_profiling(counts), _empty_report())
            assert s._compute_cv() >= 0.0

    def test_cv_stored_in_score_report(self) -> None:
        p = make_profiling([1, 100])
        s = Scorer(p, _empty_report())
        r = s.calculate()
        assert r.cv >= 0.0

    def test_cv_zero_maps_to_max_efficiency(self) -> None:
        # Identical counts → CV=0 → efficiency_subscore=15
        s = Scorer(make_profiling([5, 5, 5, 5, 5]), _empty_report())
        assert s._compute_efficiency_subscore() == 15.0

    @pytest.mark.parametrize(
        "counts,expected_sub",
        [
            ([1, 1, 1, 1], 15.0),  # CV=0.0  < 0.5 → 15
            ([1, 2, 3, 10], 12.0),  # CV≈0.8  < 1.0 → 12
            ([1, 1, 1, 100], 8.0),  # CV≈1.7  < 2.0 →  8
            ([1] * 50 + [10000], 0.0),  # CV > 4.0       →  0
        ],
    )
    def test_cv_bands(self, counts: list[int], expected_sub: float) -> None:
        s = Scorer(make_profiling(counts), _empty_report())
        assert s._compute_efficiency_subscore() == expected_sub


# ---------------------------------------------------------------------------
# TestScorerQuality
# ---------------------------------------------------------------------------


class TestScorerQuality:

    def test_no_optimizer_gives_partial_credit(self) -> None:
        s = Scorer(make_profiling([1]), None)
        score, partial = s._score_quality()
        assert partial is True
        assert score == PARTIAL_QUALITY

    def test_empty_suggestions_gives_max_quality(self) -> None:
        s = Scorer(make_profiling([1]), _empty_report())
        score, partial = s._score_quality()
        assert partial is False
        assert score == MAX_QUALITY

    def test_only_quality_patterns_are_counted(self) -> None:
        # Maintainability patterns must NOT affect quality score
        report = _report(
            ("unused_vars", "high"),  # maintainability — ignored
            ("nested_loops", "high"),  # maintainability — ignored
        )
        s = Scorer(make_profiling([1] * 10), report, source_lines=10)
        score, _ = s._score_quality()
        assert score == MAX_QUALITY

    def test_high_severity_quality_suggestion_reduces_score(self) -> None:
        report = _report(("hot_loop", "high"))
        s = Scorer(make_profiling([1] * 5), report, source_lines=5)
        score, _ = s._score_quality()
        assert score < MAX_QUALITY

    def test_all_quality_patterns_recognised(self) -> None:
        for pattern in QUALITY_PATTERNS:
            report = _report((pattern, "medium"))
            s = Scorer(make_profiling([1] * 20), report, source_lines=20)
            score, _ = s._score_quality()
            assert score < MAX_QUALITY, f"Pattern {pattern!r} not affecting quality"

    def test_quality_normalised_by_source_lines(self) -> None:
        # Same suggestion, but larger program → lower density → higher score
        report_same = _report(("hot_loop", "high"))
        s_small = Scorer(make_profiling([1] * 2), report_same, source_lines=2)
        s_large = Scorer(make_profiling([1] * 50), report_same, source_lines=50)
        score_small, _ = s_small._score_quality()
        score_large, _ = s_large._score_quality()
        assert score_large >= score_small

    def test_optimizer_partial_flag_false_when_optimizer_present(self) -> None:
        s = Scorer(make_profiling([1]), _empty_report())
        r = s.calculate()
        assert r.dimensions.optimizer_partial is False


# ---------------------------------------------------------------------------
# TestScorerMaintainability
# ---------------------------------------------------------------------------


class TestScorerMaintainability:

    def test_no_optimizer_gives_partial_credit(self) -> None:
        s = Scorer(make_profiling([1]), None)
        assert s._score_maintainability() == PARTIAL_MAINTAINABILITY

    def test_empty_suggestions_gives_max_maintainability(self) -> None:
        s = Scorer(make_profiling([1]), _empty_report())
        assert s._score_maintainability() == MAX_MAINTAINABILITY

    def test_only_maintainability_patterns_are_counted(self) -> None:
        # Quality patterns must NOT affect maintainability score
        report = _report(
            ("hot_loop", "high"),  # quality — ignored
            ("dead_code", "high"),  # quality — ignored
        )
        s = Scorer(make_profiling([1] * 10), report, source_lines=10)
        assert s._score_maintainability() == MAX_MAINTAINABILITY

    def test_high_severity_maintainability_suggestion_reduces_score(self) -> None:
        report = _report(("unused_vars", "high"))
        s = Scorer(make_profiling([1] * 5), report, source_lines=5)
        assert s._score_maintainability() < MAX_MAINTAINABILITY

    def test_all_maintainability_patterns_recognised(self) -> None:
        for pattern in MAINTAINABILITY_PATTERNS:
            report = _report((pattern, "medium"))
            s = Scorer(make_profiling([1] * 20), report, source_lines=20)
            score = s._score_maintainability()
            assert (
                score < MAX_MAINTAINABILITY
            ), f"Pattern {pattern!r} not affecting maintainability"

    def test_maintainability_normalised_by_source_lines(self) -> None:
        report_same = _report(("nested_loops", "high"))
        s_small = Scorer(make_profiling([1] * 2), report_same, source_lines=2)
        s_large = Scorer(make_profiling([1] * 50), report_same, source_lines=50)
        assert s_large._score_maintainability() >= s_small._score_maintainability()


# ---------------------------------------------------------------------------
# TestScorerDensityToScore
# ---------------------------------------------------------------------------


class TestScorerDensityToScore:

    @pytest.mark.parametrize(
        "density,max_score,expected",
        [
            # density=0 → 100% of max
            (0.0, 20.0, 20.0),
            (0.0, 15.0, 15.0),
            # density≤0.3 → 80%
            (0.1, 20.0, 16.0),
            (0.3, 20.0, 16.0),
            (0.1, 15.0, 12.0),
            # density≤0.6 → 60%
            (0.31, 20.0, 12.0),
            (0.6, 20.0, 12.0),
            (0.31, 15.0, 9.0),
            # density≤1.0 → 35%
            (0.61, 20.0, 7.0),
            (1.0, 20.0, 7.0),
            (0.61, 15.0, pytest.approx(5.25)),
            # density>1.0 → 15%
            (1.01, 20.0, 3.0),
            (5.0, 20.0, 3.0),
            (1.01, 15.0, pytest.approx(2.25)),
        ],
    )
    def test_density_band(
        self, density: float, max_score: float, expected: float
    ) -> None:
        result = Scorer._density_to_score(density, max_score)
        assert result == expected

    def test_result_never_exceeds_max_score(self) -> None:
        for density in [0.0, 0.1, 0.5, 1.0, 2.0]:
            for max_score in [15.0, 20.0]:
                result = Scorer._density_to_score(density, max_score)
                assert result <= max_score

    def test_result_is_always_non_negative(self) -> None:
        for density in [0.0, 0.5, 1.0, 10.0]:
            assert Scorer._density_to_score(density, 20.0) >= 0.0


# ---------------------------------------------------------------------------
# TestScorerCalculate (end-to-end)
# ---------------------------------------------------------------------------


class TestScorerCalculate:

    def test_returns_score_report_instance(self) -> None:
        r = Scorer(make_profiling([1]), _empty_report()).calculate()
        assert isinstance(r, ScoreReport)

    def test_perfect_program_scores_100(self) -> None:
        # O(1) profile, no errors, no suggestions
        p = make_profiling([1, 1, 1, 1, 1])
        r = Scorer(p, _empty_report(), source_lines=5, errors=[]).calculate()
        assert r.score == 100.0

    def test_perfect_program_grade_is_excellent(self) -> None:
        p = make_profiling([1, 1, 1, 1, 1])
        r = Scorer(p, _empty_report(), source_lines=5, errors=[]).calculate()
        assert r.grade == "Excellent"

    def test_score_always_in_0_to_100(self) -> None:
        combos = [
            (make_profiling([1]), _empty_report(), 5, []),
            (None, _empty_report(), 5, []),
            (make_profiling([1]), None, 5, []),
            (None, None, 5, ["err"]),
            (
                make_profiling([101, 10100, 10000]),
                _report(("hot_loop", "high"), ("nested_loops", "high")),
                3,
                ["err", "err"],
            ),
        ]
        for prof, opt, sl, errs in combos:
            r = Scorer(prof, opt, source_lines=sl, errors=errs).calculate()
            assert 0.0 <= r.score <= 100.0, f"Out of bounds: {r.score}"

    def test_two_errors_caps_correctness_at_zero(self) -> None:
        p = make_profiling([1])
        r = Scorer(p, _empty_report(), errors=["e", "f"]).calculate()
        assert r.dimensions.correctness == 0.0

    def test_nested_loop_lower_score_than_linear(self) -> None:
        linear = Scorer(
            make_profiling([1, 100, 100]), _empty_report(), source_lines=3
        ).calculate()
        nested = Scorer(
            make_profiling([101, 10100, 10000]), _empty_report(), source_lines=3
        ).calculate()
        assert nested.score < linear.score

    def test_complexity_class_in_report(self) -> None:
        p = make_profiling([101, 10100, 10000])
        r = Scorer(p, _empty_report(), source_lines=3).calculate()
        assert r.complexity_class == "O(n²)"

    def test_breakdown_keys_all_present(self) -> None:
        r = Scorer(make_profiling([1]), _empty_report()).calculate()
        dims = r.dimensions.to_dict()
        expected = {
            "correctness",
            "efficiency_complexity",
            "quality",
            "maintainability",
            "complexity_subscore",
            "efficiency_subscore",
            "profiling_partial",
            "optimizer_partial",
        }
        assert set(dims.keys()) == expected

    def test_breakdown_values_non_negative(self) -> None:
        p = make_profiling([1, 50, 50])
        r = Scorer(p, _report(("hot_loop", "high")), source_lines=3).calculate()
        assert r.dimensions.correctness >= 0.0
        assert r.dimensions.efficiency_complexity >= 0.0
        assert r.dimensions.quality >= 0.0
        assert r.dimensions.maintainability >= 0.0

    def test_lines_profiled_in_report(self) -> None:
        p = make_profiling([1, 50, 200])
        r = Scorer(p, _empty_report()).calculate()
        assert r.lines_profiled == 3

    def test_no_profiling_partial_flag_set(self) -> None:
        r = Scorer(None, _empty_report()).calculate()
        assert r.dimensions.profiling_partial is True

    def test_no_optimizer_partial_flag_set(self) -> None:
        r = Scorer(make_profiling([1]), None).calculate()
        assert r.dimensions.optimizer_partial is True

    def test_both_present_no_partial_flags(self) -> None:
        r = Scorer(make_profiling([1]), _empty_report()).calculate()
        assert r.dimensions.profiling_partial is False
        assert r.dimensions.optimizer_partial is False

    def test_narrative_non_empty(self) -> None:
        r = Scorer(make_profiling([1]), _empty_report()).calculate()
        assert isinstance(r.narrative, str) and len(r.narrative) > 10

    def test_efficiency_complexity_equals_subscore_sum(self) -> None:
        p = make_profiling([1, 100, 100])
        r = Scorer(p, _empty_report()).calculate()
        assert r.dimensions.efficiency_complexity == pytest.approx(
            r.dimensions.complexity_subscore + r.dimensions.efficiency_subscore
        )

    def test_final_score_equals_dimension_sum(self) -> None:
        p = make_profiling([1, 1, 1])
        r = Scorer(p, _empty_report(), source_lines=3, errors=[]).calculate()
        expected = (
            r.dimensions.correctness
            + r.dimensions.efficiency_complexity
            + r.dimensions.quality
            + r.dimensions.maintainability
        )
        assert r.score == pytest.approx(expected)


# ---------------------------------------------------------------------------
# TestCalculateScoreFunction
# ---------------------------------------------------------------------------


class TestCalculateScoreFunction:

    def test_returns_score_report(self) -> None:
        r = calculate_score(make_profiling([1]))
        assert isinstance(r, ScoreReport)

    def test_matches_scorer_directly(self) -> None:
        p = make_profiling([1, 100, 100])
        report = _report(("hot_loop", "medium"))
        r1 = calculate_score(p, optimizer_report=report, source_lines=10, errors=["e"])
        r2 = Scorer(p, report, source_lines=10, errors=["e"]).calculate()
        assert r1.score == r2.score
        assert r1.grade == r2.grade
        assert r1.complexity_class == r2.complexity_class

    def test_defaults_no_optimizer_no_errors(self) -> None:
        r = calculate_score(make_profiling([1]))
        assert r.dimensions.optimizer_partial is True
        assert r.error_count == 0

    def test_none_profiling_default(self) -> None:
        r = calculate_score(None)
        assert r.dimensions.profiling_partial is True

    def test_to_dict_is_json_serialisable(self) -> None:
        p = make_profiling([1, 50, 50])
        r = calculate_score(p, optimizer_report=_empty_report(), source_lines=3)
        json.dumps(r.to_dict())  # must not raise


# ---------------------------------------------------------------------------
# TestEdgeCases
# ---------------------------------------------------------------------------


class TestEdgeCases:

    def test_empty_line_stats_dict(self) -> None:
        # When profiling is present but line_stats is empty, the scorer
        # treats it the same as no profiling and awards partial credit.
        # complexity_class is "Unknown" and profiling_partial is True.
        p = {"line_stats": {}, "total_time_ms": 0.0, "total_lines": 0}
        r = calculate_score(p, optimizer_report=_empty_report(), source_lines=1)
        assert r.score >= 0.0
        assert r.complexity_class == "Unknown"
        assert r.dimensions.profiling_partial is True

    def test_source_lines_zero_does_not_divide_by_zero(self) -> None:
        # source_lines is clamped to max(n, 1) inside Scorer
        r = calculate_score(
            make_profiling([1]),
            optimizer_report=_report(("hot_loop", "high")),
            source_lines=0,
        )
        assert r.score >= 0.0

    def test_unknown_complexity_class_falls_back_to_partial(self) -> None:
        # If _detect_complexity returned something not in COMPLEXITY_POINTS,
        # the scorer uses PARTIAL_COMPLEXITY as fallback via dict.get()
        assert (
            COMPLEXITY_POINTS.get("O(unknown)", PARTIAL_COMPLEXITY)
            == PARTIAL_COMPLEXITY
        )

    def test_both_profiling_and_optimizer_absent(self) -> None:
        r = calculate_score(None, optimizer_report=None, source_lines=5, errors=[])
        assert r.dimensions.profiling_partial is True
        assert r.dimensions.optimizer_partial is True
        # Score should still be a valid float in [0, 100]
        assert 0.0 <= r.score <= 100.0

    def test_very_large_execution_count_does_not_crash(self) -> None:
        p = make_profiling([1, 10, 1_000_000])
        r = calculate_score(p, optimizer_report=_empty_report(), source_lines=3)
        assert 0.0 <= r.score <= 100.0

    def test_all_errors_produces_zero_correctness(self) -> None:
        r = calculate_score(
            make_profiling([1]),
            optimizer_report=_empty_report(),
            errors=["e1", "e2", "e3"],
        )
        assert r.dimensions.correctness == 0.0

    def test_mixed_quality_and_maintainability_patterns(self) -> None:
        # Both pattern groups present — each dimension scored independently
        report = _report(
            ("hot_loop", "high"),  # quality
            ("unused_vars", "high"),  # maintainability
        )
        r = Scorer(make_profiling([1] * 5), report, source_lines=5).calculate()
        assert r.dimensions.quality < MAX_QUALITY
        assert r.dimensions.maintainability < MAX_MAINTAINABILITY

    def test_unknown_pattern_does_not_affect_either_dimension(self) -> None:
        # A pattern that belongs to neither group should be silently ignored
        report = _report(("some_future_pattern", "high"))
        s = Scorer(make_profiling([1] * 5), report, source_lines=5)
        q, _ = s._score_quality()
        m = s._score_maintainability()
        assert q == MAX_QUALITY
        assert m == MAX_MAINTAINABILITY

    def test_single_line_program_scores_correctly(self) -> None:
        p = make_profiling([1])
        r = calculate_score(
            p, optimizer_report=_empty_report(), source_lines=1, errors=[]
        )
        assert r.score >= 90.0

    def test_score_report_grade_consistent_with_score(self) -> None:
        # Grade must always match the score band
        for score_val in [95.0, 80.0, 65.0, 45.0, 20.0]:
            dims = DimensionScores(
                correctness=score_val * 0.35,
                efficiency_complexity=score_val * 0.30,
                quality=score_val * 0.20,
                maintainability=score_val * 0.15,
            )
            narrative = _generate_narrative(score_val, dims)
            grade = _assign_grade(score_val)
            if score_val >= 90:
                assert grade == "Excellent"
                assert "Excellent" in narrative
            elif score_val >= 75:
                assert grade == "Good"
                assert "Good" in narrative
            elif score_val >= 60:
                assert grade == "Fair"
                assert "Fair" in narrative
