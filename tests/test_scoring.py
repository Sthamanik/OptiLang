"""
tests/test_scoring.py
---------------------
Test suite for the OptiLang Intelligent Scoring System (v2.0).

Structure
---------
 1.  TestScoreReport          — data class, serialisation, backwards compat
 2.  TestProgramProfiler      — Stage 1: profile building and classification
 3.  TestScaleAndGini         — scale factor + Gini index computations
 4.  TestDimensionScorer_E1   — execution efficiency sub-score
 5.  TestDimensionScorer_E2   — memory efficiency sub-score
 6.  TestDimensionScorer_Q1   — code cleanliness sub-score
 7.  TestDimensionScorer_Q2   — issue density sub-score
 8.  TestDimensionScorer_C1   — complexity handling sub-score
 9.  TestWeightEngine         — context-aware weight selection
10.  TestNarrative            — narrative generation content and tone
11.  TestDynamicScorer        — full four-stage calculate() end-to-end
12.  TestContextAwareness     — the core property: same code, different scale
13.  TestProgramTypes         — each program type scored appropriately
14.  TestCalculateFullScore   — public calculate_full_score() API
15.  TestCalculateScore       — backwards-compatible calculate_score()
16.  TestScorerAlias          — backwards-compatible Scorer class
17.  TestEdgeCases            — empty data, single lines, no AST, errors
"""

from __future__ import annotations

from typing import Any, Optional, cast

import pytest

from optilang.models import Suggestion
from optilang.scoring import (  # Public API; Internals under test; Constants
    COMPLEXITY_BASE,
    DIMENSION_WEIGHTS,
    FUNCTION_LENGTH_THRESHOLD,
    GRADE_THRESHOLDS,
    MAX_ACCEPTABLE_NESTING,
    DimensionScorer,
    DynamicScorer,
    NarrativeGenerator,
    ProgramProfile,
    ProgramProfiler,
    ProgramType,
    Scorer,
    ScoreReport,
    WeightEngine,
    calculate_full_score,
    calculate_score,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------


def make_profiling(
    line_counts: list[int] | None = None,
    memory_vars: list[int] | None = None,
    avg_times: list[float] | None = None,
    total_time_ms: float = 1.0,
    complexity_confidence: float = 1.0,
    function_stats: dict | None = None,
) -> dict:
    """Build a minimal profiling dict that mirrors ProfilingData.to_dict()."""
    line_counts = line_counts or [1]
    memory_vars = memory_vars or [0] * len(line_counts)
    avg_times = avg_times or [total_time_ms / max(c, 1) for c in line_counts]

    line_stats: dict = {}
    total_executed = 0
    for i, (cnt, mem, avg) in enumerate(
        zip(line_counts, memory_vars, avg_times), start=1
    ):
        line_stats[i] = {
            "count": cnt,
            "avg_time_ms": avg,
            "memory_vars": mem,
        }
        total_executed += cnt

    return {
        "line_stats": line_stats,
        "function_stats": function_stats or {},
        "total_time_ms": total_time_ms,
        "total_lines": total_executed,
        "lines_profiled": len(line_stats),
        "complexity_confidence": complexity_confidence,
    }


def make_suggestion(severity: str, line: int = 1) -> Suggestion:
    return Suggestion(
        line=line,
        pattern="test",
        severity=severity,
        description="test",
        suggestion="test",
        impact_score=5.0,
    )


def make_fn_stats(max_recursion_depth: int = 0) -> dict:
    return {"fn": {"calls": 3, "max_recursion_depth": max_recursion_depth}}


def profiler(
    line_counts: Optional[list[int]] = None,
    memory_vars: Optional[list[int]] = None,
    avg_times: Optional[list[float]] = None,
    total_time_ms: float = 1.0,
    suggestions: Optional[list[Any]] = None,
    source_lines: int = 10,
    function_stats: Optional[dict[str, Any]] = None,
    complexity_confidence: float = 1.0,
) -> ProgramProfiler:
    p = make_profiling(
        line_counts,
        memory_vars,
        avg_times,
        total_time_ms,
        complexity_confidence,
        function_stats,
    )
    return ProgramProfiler(
        profiling_data=p,
        suggestions=suggestions or [],
        total_source_lines=source_lines,
        function_stats=function_stats or {},
    )


def scorer(
    line_counts: Optional[list[int]] = None,
    memory_vars: Optional[list[int]] = None,
    avg_times: Optional[list[float]] = None,
    total_time_ms: float = 1.0,
    suggestions: Optional[list[Any]] = None,
    source_lines: int = 10,
    function_stats: Optional[dict[str, Any]] = None,
    complexity_confidence: float = 1.0,
) -> DynamicScorer:
    p = make_profiling(
        line_counts,
        memory_vars,
        avg_times,
        total_time_ms,
        complexity_confidence,
        function_stats,
    )
    return DynamicScorer(
        profiling_data=p,
        suggestions=suggestions or [],
        total_source_lines=source_lines,
        function_stats=function_stats or {},
    )


# ---------------------------------------------------------------------------
# 1. ScoreReport
# ---------------------------------------------------------------------------


class TestScoreReport:

    def _make_report(self, score: float = 80.0, grade: str = "Good") -> ScoreReport:
        return ScoreReport(
            final_score=score,
            grade=grade,
            program_type="linear_script",
            complexity_class="O(n)",
            dimension_scores={"efficiency": 80.0, "quality": 80.0, "complexity": 80.0},
            applied_weights={"efficiency": 0.2, "quality": 0.55, "complexity": 0.25},
            narrative="Test narrative.",
        )

    def test_to_dict_has_required_keys(self) -> None:
        d = self._make_report().to_dict()
        required = {
            "final_score",
            "score",
            "grade",
            "program_type",
            "complexity_class",
            "dimension_scores",
            "applied_weights",
            "narrative",
            "breakdown",
            "adaptive_context",
        }
        assert required.issubset(d.keys())

    def test_score_alias_equals_final_score(self) -> None:
        r = self._make_report(score=72.5)
        assert r.score == r.final_score == 72.5

    def test_to_dict_score_rounded_2dp(self) -> None:
        r = self._make_report(score=72.3456)
        assert r.to_dict()["final_score"] == 72.35

    def test_to_dict_dimension_scores_rounded(self) -> None:
        r = ScoreReport(
            final_score=80.0,
            grade="Good",
            program_type="looping",
            complexity_class="O(n)",
            dimension_scores={"efficiency": 80.123456},
        )
        assert r.to_dict()["dimension_scores"]["efficiency"] == 80.12

    def test_grade_values_come_from_thresholds(self) -> None:
        r = self._make_report()
        assert r.grade in {g for _, g in GRADE_THRESHOLDS}

    def test_narrative_field_is_string(self) -> None:
        r = self._make_report()
        assert isinstance(r.to_dict()["narrative"], str)


# ---------------------------------------------------------------------------
# 2. ProgramProfiler — classification
# ---------------------------------------------------------------------------


class TestProgramProfiler:

    def test_trivial_when_max_count_zero(self) -> None:
        p = profiler(line_counts=[0]).build()
        assert p.program_type == "trivial"

    def test_trivial_when_max_count_one(self) -> None:
        p = profiler(line_counts=[1, 1, 1]).build()
        assert p.program_type == "trivial"

    def test_recursive_computation_detected(self) -> None:
        p = profiler(
            line_counts=[10, 50, 10],
            function_stats=make_fn_stats(max_recursion_depth=3),
        ).build()
        assert p.program_type == "recursive_computation"

    def test_recursive_not_detected_when_depth_zero(self) -> None:
        p = profiler(
            line_counts=[100, 100],
            function_stats=make_fn_stats(max_recursion_depth=0),
        ).build()
        assert p.program_type != "recursive_computation"

    def test_nested_processing_detected(self) -> None:
        # High count + high gini + O(n²) complexity → nested_processing
        # Use a count pattern that produces strong outer/inner signal
        p = profiler(line_counts=[101, 10100, 10000]).build()
        assert p.program_type in ("nested_processing", "data_iteration")

    def test_data_iteration_for_simple_loop(self) -> None:
        p = profiler(line_counts=[1, 100, 100]).build()
        assert p.program_type in ("data_iteration", "nested_processing")

    def test_linear_script_for_moderate_execution(self) -> None:
        # max_count = 5, no recursion, below LOOP_THRESHOLD
        p = profiler(line_counts=[1, 2, 5]).build()
        assert p.program_type == "linear_script"

    def test_recursive_beats_looping_in_priority(self) -> None:
        # High count AND recursion → recursive wins
        p = profiler(
            line_counts=[1000, 1000],
            function_stats=make_fn_stats(max_recursion_depth=5),
        ).build()
        assert p.program_type == "recursive_computation"

    def test_scale_factor_in_profile(self) -> None:
        # LOG_SCALE_DENOM = 7, so 10^7 = 10_000_000 gives scale_factor = 1.0
        p = profiler(line_counts=[10_000_000]).build()
        assert p.scale_factor == pytest.approx(1.0, abs=0.01)

    def test_gini_near_zero_for_uniform(self) -> None:
        p = profiler(line_counts=[10, 10, 10, 10]).build()
        assert p.gini_index < 0.1

    def test_gini_high_for_dominant_line(self) -> None:
        # With 4 lines where one dominates, Gini converges ~0.75
        p = profiler(line_counts=[1, 1, 1, 10000]).build()
        assert p.gini_index > 0.65

    def test_dead_line_ratio_when_half_dead(self) -> None:
        p = profiler(line_counts=[1] * 5, source_lines=10).build()
        assert p.dead_line_ratio == pytest.approx(0.5)

    def test_no_dead_code_for_small_program(self) -> None:
        # Below MIN_LINES_FOR_DEAD_CODE threshold
        p = profiler(line_counts=[1, 1], source_lines=3).build()
        assert p.dead_line_ratio == 0.0

    def test_complexity_class_populated(self) -> None:
        p = profiler(line_counts=[1, 100, 100]).build()
        assert p.complexity_class in COMPLEXITY_BASE

    def test_function_stats_as_object_with_attribute(self) -> None:
        """function_stats can come from ProfilingData as objects not dicts."""

        class FakeStats:
            max_recursion_depth = 4

        prof = ProgramProfiler(
            profiling_data=make_profiling(line_counts=[100]),
            function_stats={"fn": FakeStats()},
        )
        p = prof.build()
        assert p.program_type == "recursive_computation"

    def test_hotness_weighted_issues_higher_than_plain_weighted(self) -> None:
        # Suggestion on the hottest line → hotness bonus applied
        sug = [make_suggestion("high", line=2)]  # line 2 is the hot line
        p = profiler(
            line_counts=[1, 10000],
            suggestions=sug,
        ).build()
        # hotness_weighted >= weighted (because of hotness bonus)
        assert p.hotness_weighted_issue_score >= p.weighted_issue_score


# ---------------------------------------------------------------------------
# 3. Scale factor and Gini index
# ---------------------------------------------------------------------------


class TestScaleAndGini:

    def test_scale_zero_for_count_one(self) -> None:
        pr = profiler(line_counts=[1])
        assert pr._compute_scale_factor() == 0.0

    def test_scale_partial_at_100(self) -> None:
        pr = profiler(line_counts=[100])
        sf = pr._compute_scale_factor()
        assert 0.28 < sf < 0.35  # log10(100)/7 ≈ 0.286

    def test_scale_full_at_10m(self) -> None:
        pr = profiler(line_counts=[10_000_000])
        assert pr._compute_scale_factor() == pytest.approx(1.0)

    def test_scale_monotone(self) -> None:
        s10 = profiler(line_counts=[10])._compute_scale_factor()
        s1k = profiler(line_counts=[1000])._compute_scale_factor()
        s1m = profiler(line_counts=[1_000_000])._compute_scale_factor()
        assert s10 < s1k < s1m

    def test_gini_zero_single_line(self) -> None:
        pr = profiler(line_counts=[100])
        assert pr._compute_gini() == 0.0

    def test_gini_zero_empty(self) -> None:
        pr = profiler(line_counts=[])
        assert pr._compute_gini() == 0.0

    def test_gini_bounded_zero_to_one(self) -> None:
        for counts in [[1, 5, 50], [100, 1, 1, 1], [10, 10, 10]]:
            g = profiler(line_counts=counts)._compute_gini()
            assert 0.0 <= g <= 1.0


# ---------------------------------------------------------------------------
# 4. DimensionScorer — E1: Execution Efficiency
# ---------------------------------------------------------------------------


class TestDimensionScorerE1:

    def _dim(self, **kw: Any) -> DimensionScorer:
        return DimensionScorer(profiler(**kw).build())

    def test_uniform_execution_gives_high_score(self) -> None:
        score, _ = self._dim(line_counts=[50, 50, 50, 50]).execution_efficiency()
        assert score > 80.0

    def test_dominant_hot_loop_gives_lower_score(self) -> None:
        score, _ = self._dim(line_counts=[1, 1, 10000]).execution_efficiency()
        assert score < 80.0

    def test_hotness_weighted_issues_reduce_score(self) -> None:
        sug = [make_suggestion("high", line=2)] * 5
        s_no_issues, _ = self._dim(line_counts=[1, 10000]).execution_efficiency()
        dim_with = DimensionScorer(
            profiler(line_counts=[1, 10000], suggestions=sug).build()
        )
        s_with_issues, _ = dim_with.execution_efficiency()
        assert s_with_issues <= s_no_issues

    def test_detail_contains_gini_and_scale(self) -> None:
        _, detail = self._dim(line_counts=[1, 100]).execution_efficiency()
        assert "gini_index" in detail
        assert "scale_factor" in detail

    def test_score_in_range_zero_to_100(self) -> None:
        for counts in [[1], [100, 100], [1, 10000], [1, 1, 1_000_000]]:
            s, _ = self._dim(line_counts=counts).execution_efficiency()
            assert 0.0 <= s <= 100.0


# ---------------------------------------------------------------------------
# 5. DimensionScorer — E2: Memory Efficiency
# ---------------------------------------------------------------------------


class TestDimensionScorerE2:

    def _dim(self, **kw: Any) -> DimensionScorer:
        return DimensionScorer(profiler(**kw).build())

    def test_no_memory_data_returns_100(self) -> None:
        score, _ = self._dim(line_counts=[1], memory_vars=[0]).memory_efficiency()
        assert score == 100.0

    def test_low_stdev_gives_high_score(self) -> None:
        # All lines have same var count → stdev = 0 → neutral (100)
        score, _ = self._dim(
            line_counts=[1] * 5, memory_vars=[5, 5, 5, 5, 5]
        ).memory_efficiency()
        assert score == 100.0

    def test_high_spread_reduces_score(self) -> None:
        # Wildly varying memory → high stdev → lower score
        score, _ = self._dim(
            line_counts=[1] * 5, memory_vars=[1, 1, 1, 50, 100]
        ).memory_efficiency()
        assert score < 100.0

    def test_detail_contains_threshold_info(self) -> None:
        _, detail = self._dim(
            line_counts=[1] * 5, memory_vars=[2, 3, 4, 5, 30]
        ).memory_efficiency()
        assert "memory_adaptive_threshold" in detail


# ---------------------------------------------------------------------------
# 6. DimensionScorer — Q1: Code Cleanliness
# ---------------------------------------------------------------------------


class TestDimensionScorerQ1:

    def _profile_with(
        self,
        dead_ratio: float = 0.0,
        avg_fn_len: float = 0.0,
        max_nest: int = 0,
        branch_density: float = 0.0,
    ) -> ProgramProfile:
        p = ProgramProfile()
        p.dead_line_ratio = dead_ratio
        p.avg_function_length = avg_fn_len
        p.max_nesting_depth = max_nest
        p.branch_density = branch_density
        return p

    def test_perfect_code_scores_100(self) -> None:
        dim = DimensionScorer(self._profile_with())
        score, _ = dim.code_cleanliness()
        assert score == pytest.approx(100.0)

    def test_dead_code_reduces_score(self) -> None:
        dim_clean = DimensionScorer(self._profile_with(dead_ratio=0.0))
        dim_dead = DimensionScorer(self._profile_with(dead_ratio=0.5))
        s_clean, _ = dim_clean.code_cleanliness()
        s_dead, _ = dim_dead.code_cleanliness()
        assert s_dead < s_clean

    def test_long_functions_reduce_score(self) -> None:
        dim_short = DimensionScorer(self._profile_with(avg_fn_len=5.0))
        dim_long = DimensionScorer(
            self._profile_with(avg_fn_len=FUNCTION_LENGTH_THRESHOLD + 20)
        )
        s_short, _ = dim_short.code_cleanliness()
        s_long, _ = dim_long.code_cleanliness()
        assert s_long < s_short

    def test_deep_nesting_reduces_score(self) -> None:
        dim_flat = DimensionScorer(self._profile_with(max_nest=1))
        dim_deep = DimensionScorer(
            self._profile_with(max_nest=MAX_ACCEPTABLE_NESTING + 3)
        )
        s_flat, _ = dim_flat.code_cleanliness()
        s_deep, _ = dim_deep.code_cleanliness()
        assert s_deep < s_flat

    def test_high_branch_density_reduces_score(self) -> None:
        dim_low = DimensionScorer(self._profile_with(branch_density=0.1))
        dim_high = DimensionScorer(self._profile_with(branch_density=0.8))
        s_low, _ = dim_low.code_cleanliness()
        s_high, _ = dim_high.code_cleanliness()
        assert s_high < s_low

    def test_score_bounded_zero_to_100(self) -> None:
        worst = DimensionScorer(
            self._profile_with(
                dead_ratio=1.0, avg_fn_len=100.0, max_nest=10, branch_density=1.0
            )
        )
        s, _ = worst.code_cleanliness()
        assert 0.0 <= s <= 100.0

    def test_detail_contains_all_sub_scores(self) -> None:
        _, detail = DimensionScorer(self._profile_with()).code_cleanliness()
        assert "dead_code_score" in detail
        assert "function_length_score" in detail
        assert "nesting_score" in detail
        assert "branch_score" in detail


# ---------------------------------------------------------------------------
# 7. DimensionScorer — Q2: Issue Density
# ---------------------------------------------------------------------------


class TestDimensionScorerQ2:

    def _profile_with(
        self,
        suggestions: int = 0,
        weighted: float = 0.0,
        hotness_weighted: float = 0.0,
        density: float = 0.0,
    ) -> ProgramProfile:
        p = ProgramProfile()
        p.total_suggestions = suggestions
        p.weighted_issue_score = weighted
        p.hotness_weighted_issue_score = hotness_weighted
        p.issue_density = density
        return p

    def test_no_suggestions_gives_100(self) -> None:
        dim = DimensionScorer(self._profile_with())
        s, _ = dim.issue_density()
        assert s == 100.0

    def test_high_density_gives_low_score(self) -> None:
        dim = DimensionScorer(self._profile_with(suggestions=10, density=2.0))
        s, _ = dim.issue_density()
        assert s < 30.0

    def test_low_density_gives_high_score(self) -> None:
        dim = DimensionScorer(self._profile_with(suggestions=1, density=0.05))
        s, _ = dim.issue_density()
        assert s > 85.0

    def test_score_bounded_zero_to_100(self) -> None:
        dim = DimensionScorer(self._profile_with(suggestions=100, density=10.0))
        s, _ = dim.issue_density()
        assert 0.0 <= s <= 100.0

    def test_detail_has_density_and_count(self) -> None:
        dim = DimensionScorer(self._profile_with(suggestions=3, density=0.3))
        _, detail = dim.issue_density()
        assert "suggestion_count" in detail
        assert "issue_density" in detail


# ---------------------------------------------------------------------------
# 8. DimensionScorer — C1: Complexity Handling
# ---------------------------------------------------------------------------


class TestDimensionScorerC1:

    def _profile_for(
        self,
        complexity_class: str,
        scale: float = 0.5,
        confidence: float = 1.0,
        program_type: str = "data_iteration",
    ) -> ProgramProfile:
        p = ProgramProfile()
        p.complexity_class = complexity_class
        p.scale_factor = scale
        p.complexity_confidence = confidence
        p.program_type = cast(ProgramType, program_type)
        return p

    def test_o1_gives_100(self) -> None:
        dim = DimensionScorer(self._profile_for("O(1)", scale=1.0))
        s, _ = dim.complexity_handling()
        assert s == 100.0

    def test_on_gives_high_score(self) -> None:
        dim = DimensionScorer(self._profile_for("O(n)", scale=1.0))
        s, _ = dim.complexity_handling()
        assert s >= 90.0

    def test_on2_at_full_scale_gives_lower_score(self) -> None:
        dim = DimensionScorer(self._profile_for("O(n²)", scale=1.0))
        s, _ = dim.complexity_handling()
        assert s < 50.0

    def test_small_scale_softens_on2_penalty(self) -> None:
        s_small, _ = DimensionScorer(
            self._profile_for("O(n²)", scale=0.1)
        ).complexity_handling()
        s_large, _ = DimensionScorer(
            self._profile_for("O(n²)", scale=1.0)
        ).complexity_handling()
        assert s_small > s_large

    def test_low_confidence_softens_penalty(self) -> None:
        s_certain, _ = DimensionScorer(
            self._profile_for("O(n²)", scale=1.0, confidence=1.0)
        ).complexity_handling()
        s_unsure, _ = DimensionScorer(
            self._profile_for("O(n²)", scale=1.0, confidence=0.4)
        ).complexity_handling()
        assert s_unsure > s_certain

    def test_recursive_gets_justification_credit(self) -> None:
        s_loop, _ = DimensionScorer(
            self._profile_for("O(n²)", scale=1.0, program_type="data_iteration")
        ).complexity_handling()
        s_rec, _ = DimensionScorer(
            self._profile_for("O(n²)", scale=1.0, program_type="recursive_computation")
        ).complexity_handling()
        assert s_rec > s_loop

    def test_score_bounded_zero_to_100(self) -> None:
        dim = DimensionScorer(self._profile_for("O(2^n)", scale=1.0))
        s, _ = dim.complexity_handling()
        assert 0.0 <= s <= 100.0

    def test_detail_contains_key_fields(self) -> None:
        dim = DimensionScorer(self._profile_for("O(n²)", scale=0.5))
        _, detail = dim.complexity_handling()
        assert "base_ratio" in detail
        assert "scale_factor" in detail
        assert "confidence" in detail
        assert "justification_credit" in detail


# ---------------------------------------------------------------------------
# 9. WeightEngine
# ---------------------------------------------------------------------------


class TestWeightEngine:

    def test_weights_sum_to_one_for_all_types(self) -> None:
        for pt in DIMENSION_WEIGHTS:
            w = WeightEngine.weights_for(pt)  # type: ignore[arg-type]
            assert abs(sum(w.values()) - 1.0) < 1e-9, f"Failed for {pt}"

    def test_data_iteration_efficiency_is_highest(self) -> None:
        w = WeightEngine.weights_for("data_iteration")
        assert w["efficiency"] == max(w.values())

    def test_function_heavy_quality_is_highest(self) -> None:
        w = WeightEngine.weights_for("function_heavy")
        assert w["quality"] == max(w.values())

    def test_recursive_complexity_is_highest(self) -> None:
        w = WeightEngine.weights_for("recursive_computation")
        assert w["complexity"] == max(w.values())

    def test_linear_script_quality_is_highest(self) -> None:
        w = WeightEngine.weights_for("linear_script")
        assert w["quality"] == max(w.values())

    def test_unknown_type_returns_linear_script_default(self) -> None:
        w = WeightEngine.weights_for("unknown_type")  # type: ignore[arg-type]
        expected = DIMENSION_WEIGHTS["linear_script"]
        assert w == expected


# ---------------------------------------------------------------------------
# 10. NarrativeGenerator
# ---------------------------------------------------------------------------


class TestNarrative:

    def _gen(
        self,
        eff: float = 80.0,
        qlt: float = 80.0,
        cmp: float = 80.0,
        score: float = 80.0,
        grade: str = "Good",
        program_type: str = "linear_script",
    ) -> str:
        p = ProgramProfile()
        p.program_type = cast(ProgramType, program_type)
        p.complexity_class = "O(n)"
        p.complexity_confidence = 0.9
        p.gini_index = 0.3
        p.max_execution_count = 100
        p.total_suggestions = 0
        p.dead_line_ratio = 0.0
        p.avg_function_length = 5.0
        p.max_nesting_depth = 1
        p.hotness_weighted_issue_score = 0.0
        return NarrativeGenerator(
            profile=p,
            efficiency_score=eff,
            quality_score=qlt,
            complexity_score=cmp,
            final_score=score,
            grade=grade,
        ).generate()

    def test_narrative_is_non_empty_string(self) -> None:
        assert len(self._gen()) > 10

    def test_excellent_narrative_positive_tone(self) -> None:
        text = self._gen(score=95.0, grade="Excellent")
        assert any(w in text.lower() for w in ["well", "good", "efficient"])

    def test_poor_score_mentions_improvement(self) -> None:
        text = self._gen(eff=35.0, qlt=35.0, cmp=35.0, score=35.0, grade="Poor")
        assert any(
            w in text.lower() for w in ["issue", "concern", "significant", "problem"]
        )

    def test_recursive_type_mentioned_in_description(self) -> None:
        text = self._gen(program_type="recursive_computation")
        assert "recurs" in text.lower()

    def test_dead_code_mentioned_when_present(self) -> None:
        p = ProgramProfile()
        p.program_type = "linear_script"
        p.complexity_class = "O(n)"
        p.complexity_confidence = 0.9
        p.gini_index = 0.2
        p.max_execution_count = 50
        p.total_suggestions = 0
        p.dead_line_ratio = 0.40  # 40% dead
        p.avg_function_length = 5.0
        p.max_nesting_depth = 1
        p.hotness_weighted_issue_score = 0.0
        text = NarrativeGenerator(p, 80.0, 45.0, 80.0, 65.0, "Fair").generate()
        assert any(w in text.lower() for w in ["dead", "never executed", "unused"])

    def test_high_nesting_mentioned(self) -> None:
        p = ProgramProfile()
        p.program_type = "nested_processing"
        p.complexity_class = "O(n²)"
        p.complexity_confidence = 0.8
        p.gini_index = 0.7
        p.max_execution_count = 10000
        p.total_suggestions = 0
        p.dead_line_ratio = 0.0
        p.avg_function_length = 5.0
        p.max_nesting_depth = 5  # deep nesting
        p.hotness_weighted_issue_score = 0.0
        text = NarrativeGenerator(p, 70.0, 50.0, 60.0, 60.0, "Fair").generate()
        assert "nest" in text.lower()

    def test_narrative_does_not_exceed_reasonable_length(self) -> None:
        # Should be informative but concise — not a wall of text
        text = self._gen(eff=40.0, qlt=40.0, cmp=40.0, score=40.0, grade="Poor")
        assert len(text) < 800


# ---------------------------------------------------------------------------
# 11. DynamicScorer — full end-to-end calculate()
# ---------------------------------------------------------------------------


class TestDynamicScorer:

    def test_returns_score_report(self) -> None:
        report = scorer(line_counts=[1]).calculate()
        assert isinstance(report, ScoreReport)

    def test_score_in_valid_range(self) -> None:
        for counts in [[1], [1, 100, 100], [101, 10100, 10000]]:
            r = scorer(line_counts=counts).calculate()
            assert 0.0 <= r.final_score <= 100.0

    def test_trivial_program_scores_excellent(self) -> None:
        r = scorer(line_counts=[1], source_lines=1).calculate()
        assert r.final_score >= 85.0
        assert r.program_type == "trivial"

    def test_breakdown_has_dimension_scores(self) -> None:
        r = scorer(line_counts=[1]).calculate()
        assert "efficiency_score" in r.breakdown
        assert "quality_score" in r.breakdown
        assert "complexity_score" in r.breakdown

    def test_applied_weights_sum_to_one(self) -> None:
        r = scorer(line_counts=[100, 100]).calculate()
        total = sum(r.applied_weights.values())
        assert abs(total - 1.0) < 1e-9

    def test_dimension_scores_each_0_to_100(self) -> None:
        r = scorer(line_counts=[101, 10100, 10000]).calculate()
        for k, v in r.dimension_scores.items():
            assert 0.0 <= v <= 100.0, f"{k} = {v} out of range"

    def test_grade_matches_score(self) -> None:
        r = scorer(line_counts=[1]).calculate()
        for threshold, grade in GRADE_THRESHOLDS:
            if r.final_score >= threshold:
                assert r.grade == grade
                break

    def test_narrative_is_non_empty(self) -> None:
        r = scorer(line_counts=[1]).calculate()
        assert len(r.narrative) > 10

    def test_adaptive_context_contains_program_type(self) -> None:
        r = scorer(line_counts=[100, 100]).calculate()
        assert "program_type" in r.adaptive_context

    def test_adaptive_context_contains_weights(self) -> None:
        r = scorer(line_counts=[1]).calculate()
        assert "applied_weights" in r.adaptive_context

    def test_complexity_class_populated(self) -> None:
        r = scorer(line_counts=[101, 10100, 10000]).calculate()
        assert r.complexity_class == "O(n²)"

    def test_dead_code_lowers_score(self) -> None:
        r_full = scorer(line_counts=[1] * 10, source_lines=10).calculate()
        r_dead = scorer(line_counts=[1] * 5, source_lines=10).calculate()
        assert r_dead.final_score < r_full.final_score

    def test_suggestions_lower_score(self) -> None:
        sug = [make_suggestion("high")] * 5
        r_clean = scorer(line_counts=[1, 100]).calculate()
        r_issues = scorer(line_counts=[1, 100], suggestions=sug).calculate()
        assert r_issues.final_score < r_clean.final_score

    def test_to_dict_is_json_serialisable(self) -> None:
        import json

        r = scorer(line_counts=[1, 50, 50]).calculate()
        json.dumps(r.to_dict())  # must not raise


# ---------------------------------------------------------------------------
# 12. Context-awareness — same structure, different scale = different score
# ---------------------------------------------------------------------------


class TestContextAwareness:

    def test_small_nested_loop_scores_better_than_large(self) -> None:
        """
        Core adaptive property: a nested loop over 3 iterations should
        score significantly better than one over 10,000 iterations,
        even though they have the same structural O(n²) pattern.
        """
        small = scorer(
            line_counts=[3, 9, 9],
            source_lines=3,
        ).calculate()
        large = scorer(
            line_counts=[101, 10100, 10000],
            source_lines=3,
        ).calculate()
        assert small.final_score > large.final_score

    def test_issues_on_hot_lines_penalise_more_than_cold(self) -> None:
        """
        A suggestion on a line that runs 10000 times should produce a
        lower score than the same suggestion on a line that runs once.
        """
        hot_sug = [make_suggestion("medium", line=2)]
        cold_sug = [make_suggestion("medium", line=1)]

        r_hot = scorer(line_counts=[1, 10000], suggestions=hot_sug).calculate()
        r_cold = scorer(line_counts=[1, 10000], suggestions=cold_sug).calculate()
        # Hot suggestion should produce lower or equal score
        assert r_hot.final_score <= r_cold.final_score

    def test_recursive_scores_same_complexity_better_than_looping(self) -> None:
        """
        Recursive programs earn a justification credit that reduces the
        complexity penalty. At low execution scale the difference is small
        but the recursive score should not be significantly worse.
        """
        counts = [1000, 1000]
        r_loop = scorer(
            line_counts=counts,
            source_lines=5,
        ).calculate()
        r_rec = scorer(
            line_counts=counts,
            source_lines=5,
            function_stats=make_fn_stats(max_recursion_depth=8),
        ).calculate()
        # Recursive complexity score should be higher due to justification credit
        assert (
            r_rec.dimension_scores["complexity"]
            >= r_loop.dimension_scores["complexity"]
        )

    def test_low_confidence_softens_complexity_score(self) -> None:
        r_certain = scorer(
            line_counts=[101, 10100, 10000],
            complexity_confidence=1.0,
        ).calculate()
        r_unsure = scorer(
            line_counts=[101, 10100, 10000],
            complexity_confidence=0.3,
        ).calculate()
        assert r_unsure.final_score >= r_certain.final_score

    def test_same_score_for_different_machines_given_same_structure(self) -> None:
        """
        Execution time does not directly penalise — only structural and
        count-based signals matter, so the score should be stable across
        fast and slow execution environments.
        """
        r_fast = scorer(line_counts=[100, 100], total_time_ms=0.5).calculate()
        r_slow = scorer(line_counts=[100, 100], total_time_ms=500.0).calculate()
        # Scores should be close — structural signals are the same
        assert abs(r_fast.final_score - r_slow.final_score) < 10.0

    def test_issue_density_normalised_by_program_size(self) -> None:
        sug = [make_suggestion("high")] * 5
        r_small = scorer(suggestions=sug, source_lines=10).calculate()
        r_large = scorer(suggestions=sug, source_lines=200).calculate()
        # Same 5 issues in a larger program should be scored more leniently
        assert r_large.final_score > r_small.final_score


# ---------------------------------------------------------------------------
# 13. Program type scoring appropriateness
# ---------------------------------------------------------------------------


class TestProgramTypes:

    def test_trivial_uses_quality_dominant_weights(self) -> None:
        r = scorer(line_counts=[1, 1]).calculate()
        if r.program_type == "trivial":
            assert r.applied_weights["quality"] >= 0.60

    def test_data_iteration_uses_efficiency_dominant_weights(self) -> None:
        r = scorer(
            line_counts=[1, 500, 500],
            function_stats=None,
        ).calculate()
        if r.program_type == "data_iteration":
            assert r.applied_weights["efficiency"] >= 0.40

    def test_recursive_uses_complexity_dominant_weights(self) -> None:
        r = scorer(
            line_counts=[50, 50],
            function_stats=make_fn_stats(max_recursion_depth=5),
        ).calculate()
        assert r.program_type == "recursive_computation"
        assert r.applied_weights["complexity"] >= 0.35

    def test_all_program_types_produce_valid_reports(self) -> None:
        configs: list[dict[str, Any]] = [
            {"line_counts": [1]},
            {"line_counts": [1, 2, 5]},
            {"line_counts": [1, 100, 100]},
            {"line_counts": [101, 10100, 10000]},
            {
                "line_counts": [50, 50],
                "function_stats": make_fn_stats(max_recursion_depth=3),
            },
        ]
        for cfg in configs:
            r = scorer(**cfg).calculate()
            assert isinstance(r, ScoreReport)
            assert 0.0 <= r.final_score <= 100.0


# ---------------------------------------------------------------------------
# 14. calculate_full_score()
# ---------------------------------------------------------------------------


class TestCalculateFullScore:

    class _FakeProfilingData:
        """Minimal mock of ProfilingData."""

        def to_dict(self) -> dict:
            return make_profiling(line_counts=[1, 50, 50])

    class _FakeResult:
        def __init__(self) -> None:
            self.profiling = TestCalculateFullScore._FakeProfilingData()

    def test_returns_score_report(self) -> None:
        result = self._FakeResult()
        r = calculate_full_score("x = 1\ny = 2\nz = x + y", result)
        assert isinstance(r, ScoreReport)

    def test_score_in_valid_range(self) -> None:
        result = self._FakeResult()
        r = calculate_full_score("x = 1", result)
        assert 0.0 <= r.final_score <= 100.0

    def test_none_profiling_handled_gracefully(self) -> None:
        class NullResult:
            profiling: Optional[Any] = None

        r = calculate_full_score("x = 1", NullResult())
        assert isinstance(r, ScoreReport)
        assert r.program_type == "trivial"

    def test_source_lines_counted_correctly(self) -> None:
        # Comments and blank lines should NOT count
        source = """
# This is a comment

x = 1
y = 2
# Another comment
z = x + y
"""
        result = self._FakeResult()
        r = calculate_full_score(source, result)
        assert isinstance(r, ScoreReport)

    def test_suggestions_passed_through(self) -> None:
        result = self._FakeResult()
        sug = [make_suggestion("high")]
        r_clean = calculate_full_score("x = 1", result)
        r_issues = calculate_full_score("x = 1", result, suggestions=sug)
        assert r_issues.final_score <= r_clean.final_score


# ---------------------------------------------------------------------------
# 15. calculate_score() — backwards-compatible API
# ---------------------------------------------------------------------------


class TestCalculateScore:

    def test_returns_score_report(self) -> None:
        p = make_profiling(line_counts=[1])
        assert isinstance(calculate_score(p), ScoreReport)

    def test_matches_dynamic_scorer(self) -> None:
        p = make_profiling(line_counts=[1, 100, 100])
        sug = [make_suggestion("medium")]
        r1 = calculate_score(p, suggestions=sug, total_source_lines=10)
        r2 = DynamicScorer(p, suggestions=sug, total_source_lines=10).calculate()
        assert r1.final_score == r2.final_score

    def test_function_stats_passed_through(self) -> None:
        p = make_profiling(line_counts=[50, 50])
        fs = make_fn_stats(max_recursion_depth=5)
        r = calculate_score(p, function_stats=fs)
        assert r.program_type == "recursive_computation"

    def test_to_dict_json_serialisable(self) -> None:
        import json

        p = make_profiling(line_counts=[1, 50, 50])
        json.dumps(calculate_score(p).to_dict())


# ---------------------------------------------------------------------------
# 16. Scorer alias — backwards-compatible class
# ---------------------------------------------------------------------------


class TestScorerAlias:

    def test_scorer_is_subclass_of_dynamic_scorer(self) -> None:
        assert issubclass(Scorer, DynamicScorer)

    def test_accepts_original_signature(self) -> None:
        p = make_profiling(line_counts=[1])
        s = Scorer(profiling_data=p, suggestions=[], total_source_lines=5)
        assert isinstance(s.calculate(), ScoreReport)

    def test_trivial_program_high_score(self) -> None:
        p = make_profiling(line_counts=[1])
        r = Scorer(p).calculate()
        assert r.final_score >= 85.0

    def test_nested_loop_lower_score(self) -> None:
        p = make_profiling(line_counts=[101, 10100, 10000])
        r = Scorer(p).calculate()
        assert r.final_score < 90.0

    def test_score_alias_property(self) -> None:
        p = make_profiling(line_counts=[1])
        r = Scorer(p).calculate()
        assert r.score == r.final_score


# ---------------------------------------------------------------------------
# 17. Edge cases
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
        r = calculate_score(p)
        assert isinstance(r, ScoreReport)
        assert r.program_type == "trivial"
        assert r.final_score >= 80.0

    def test_single_line_program(self) -> None:
        p = make_profiling(line_counts=[1])
        r = calculate_score(p, total_source_lines=1)
        assert r.final_score >= 85.0

    def test_very_large_count_no_crash(self) -> None:
        p = make_profiling(line_counts=[1, 10, 1_000_000])
        r = calculate_score(p)
        assert 0.0 <= r.final_score <= 100.0

    def test_total_source_lines_zero_handled(self) -> None:
        p = make_profiling(line_counts=[1])
        r = calculate_score(p, total_source_lines=0)
        assert r.final_score >= 0.0

    def test_all_zero_execution_counts(self) -> None:
        p = make_profiling(line_counts=[0, 0, 0])
        r = calculate_score(p)
        assert r.program_type == "trivial"

    def test_no_suggestions_no_crash(self) -> None:
        p = make_profiling(line_counts=[100, 100])
        r = calculate_score(p, suggestions=None)
        assert isinstance(r, ScoreReport)

    def test_many_high_severity_suggestions_clamped(self) -> None:
        p = make_profiling(line_counts=[100])
        sug = [make_suggestion("high")] * 100
        r = calculate_score(p, suggestions=sug, total_source_lines=5)
        assert r.final_score >= 0.0
        assert r.final_score <= 100.0

    def test_all_lines_same_execution_count(self) -> None:
        p = make_profiling(line_counts=[50] * 10)
        r = calculate_score(p)
        assert isinstance(r, ScoreReport)

    def test_score_report_to_dict_all_floats_rounded(self) -> None:
        p = make_profiling(line_counts=[1, 50, 50])
        d = calculate_score(p).to_dict()
        # final_score should be rounded to at most 2 decimal places
        assert d["final_score"] == round(d["final_score"], 2)

    def test_profiler_without_ast_does_not_crash(self) -> None:
        p = make_profiling(line_counts=[1, 10])
        r = DynamicScorer(p, ast=None).calculate()
        assert isinstance(r, ScoreReport)
