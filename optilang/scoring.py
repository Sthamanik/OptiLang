"""
optilang/scoring.py
-------------------
Scoring system for OptiLang.

Calculates an overall optimization score (0–100) across four dimensions:

    Dimension               Max     Source
    ────────────────────────────
    Correctness              35     result.errors
    Efficiency + Complexity  30     complexity: Big-O class (15)
                                    efficiency: optimizer patterns (15)
    Quality                  20     optimizer suggestions (runtime patterns)
    Maintainability          15     optimizer suggestions (style patterns)
    ────────────────────────────
    Total                   100

Design principles:
    - Score-earned model: each dimension contributes positively (not penalty-based).
    - Dynamic scoring: calculations are relative to the program being analysed,
      not fixed global thresholds. A 5-line program and a 50-line program are
      judged on their own terms.
    - Graceful degradation: if profiling or optimizer data is unavailable,
      partial credit (~50 % of max) is awarded for that dimension rather than
      zero, reflecting genuine uncertainty rather than failure.
    - Smooth scoring: linear interpolation between density anchor points
      prevents score cliffs where two nearly identical programs score very
      differently because they straddle a fixed band boundary.

Dimension breakdown:

    Correctness (0–35)
        Uses error density = errors / source_lines, fed through the same
        linear-interpolated density-to-score function as Quality and
        Maintainability. A 100-line program with 4 errors (density 0.04)
        is judged far more leniently than a 5-line program with 3 errors
        (density 0.60). Zero errors always yields full marks (35.0).

    Efficiency + Complexity (0–30), two independent sub-scores:

        Complexity sub-score (0–15)
            Coverage-weighted: base_points × hot_coverage +
                               15.0 × (1 - hot_coverage)
            where hot_coverage = hot_lines / total_lines and hot_lines are
            those executing more than sqrt(max_count) times.
            A tiny quadratic loop in a 200-line program barely dents the
            score; a program that is quadratic wall-to-wall pays the full
            penalty. O(1) and O(log n) always return 15.0.
            Requires profiling data; falls back to PARTIAL_COMPLEXITY (7.0).

        Efficiency sub-score (0–15)
            Sourced from optimizer suggestions in EFFICIENCY_PATTERNS:
                hot_loop, loop_invariant, repeated_computation, expensive_calls.
            Uses linear-interpolated density scoring.
            Measures WHETHER THE PROGRAM AVOIDS UNNECESSARY WORK within its
            complexity class.
            Requires optimizer data; falls back to PARTIAL_EFFICIENCY (8.0).

            The two sub-scores are genuinely independent:
                - O(n) complexity but loop-invariant recomputed every iteration
                  → good complexity sub-score, poor efficiency sub-score.
                - O(n²) nested loop with no wasted work per iteration
                  → poor complexity sub-score, good efficiency sub-score.

    Quality (0–20)
        From optimizer suggestions that affect runtime behaviour beyond loop
        structure: dead_code, string_concat_loop.
        Linear-interpolated weighted density, mapped to 0–20.

    Maintainability (0–15)
        From optimizer suggestions that affect readability and structural
        clarity: unused_vars, early_return, nested_loops, constant_folding.
        constant_folding belongs here (not Quality) because it is a write-time
        concern — replacing '3 * 4' with '12' is a code-clarity improvement,
        not a runtime fix.
        Linear-interpolated weighted density, mapped to 0–15.

Fixes applied vs. previous version:
    1. optimizer_partial flag now set correctly when efficiency sub-score
       falls back to partial credit (optimizer absent).
    2. Partial-credit narrative notes accurately describe which sub-scores
       were affected and why, including the cascading effect when both
       profiling and optimizer data are absent.
    3. constant_folding moved from QUALITY_PATTERNS → MAINTAINABILITY_PATTERNS
       (write-time concern, not runtime behaviour).
    4. Maintainability narrative hint now explicitly mentions early_return
       and constant_folding, matching the patterns actually detected.
    5. Correctness scale smoothed: 35/25/15/5/0 across 0/1/2/3/4+ errors,
       eliminating the previous 25-point cliff for a single error.
    6. _density_to_score replaced step bands with linear interpolation between
       anchor points, eliminating score cliffs at band boundaries.
    7. _lowest_dimension tie-breaks by highest dimension max (most room to
       improve) instead of relying on arbitrary dict key ordering.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Pattern classification
# ---------------------------------------------------------------------------

# Patterns that measure wasted work within the complexity class.
# → Efficiency sub-score (inside Efficiency + Complexity dimension)
EFFICIENCY_PATTERNS = frozenset(
    {
        "hot_loop",
        "loop_invariant",
        "repeated_computation",
        "expensive_calls",
    }
)

# Patterns that affect runtime behaviour beyond loop structure.
# → Quality dimension
QUALITY_PATTERNS = frozenset(
    {
        "dead_code",
        "string_concat_loop",
    }
)

# Patterns that affect readability and structural clarity.
# → Maintainability dimension
# constant_folding is here because replacing '3 * 4' with '12' is a
# write-time code-clarity improvement, not a runtime performance fix.
MAINTAINABILITY_PATTERNS = frozenset(
    {
        "unused_vars",
        "early_return",
        "nested_loops",
        "constant_folding",
    }
)

# ---------------------------------------------------------------------------
# Complexity class → sub-score mapping (out of 15)
# ---------------------------------------------------------------------------

COMPLEXITY_POINTS: Dict[str, float] = {
    "O(1)": 15.0,
    "O(log n)": 15.0,
    "O(n)": 13.0,
    "O(n log n)": 10.0,
    "O(n²)": 6.0,
    "O(n³)": 3.0,
    "O(n^k)": 1.0,
    "O(2^n)": 0.0,
}

# ---------------------------------------------------------------------------
# Density → score: linear-interpolation anchor points
#
# Each entry is (density_threshold, score_fraction).
# Between two adjacent anchors the score fraction is linearly interpolated.
# Beyond the last anchor the fraction is clamped to the last value (0.10).
#
# This replaces the previous fixed step-bands which caused up to a 20 %
# score cliff for a 0.02 density difference straddling a band boundary.
# ---------------------------------------------------------------------------

_DENSITY_ANCHORS: List[Tuple[float, float]] = [
    (0.0, 1.00),  # zero issues      → full marks
    (0.3, 0.80),  # sparse issues    → 80 %
    (0.6, 0.60),  # moderate issues  → 60 %
    (1.0, 0.35),  # heavy issues     → 35 %
    (2.0, 0.10),  # very heavy       → 10 %  (clamped beyond this)
]

# ---------------------------------------------------------------------------
# Dimension maximums
# ---------------------------------------------------------------------------

MAX_CORRECTNESS: float = 35.0
MAX_EFFICIENCY_COMPLEXITY: float = 30.0  # 15 complexity + 15 efficiency
MAX_QUALITY: float = 20.0
MAX_MAINTAINABILITY: float = 15.0

# Partial credit (~50 %) awarded when required data is absent
PARTIAL_COMPLEXITY: float = 7.0  # ≈ half of 15
PARTIAL_EFFICIENCY: float = 8.0  # ≈ half of 15
PARTIAL_QUALITY: float = 10.0  # half of 20
PARTIAL_MAINTAINABILITY: float = 7.0  # ≈ half of 15

# Minimum lines needed for a meaningful CV calculation
MIN_LINES_FOR_CV: int = 2

# ---------------------------------------------------------------------------
# Grade thresholds
# ---------------------------------------------------------------------------

GRADE_THRESHOLDS: List[Tuple[float, str]] = [
    (90.0, "Excellent"),
    (75.0, "Good"),
    (60.0, "Fair"),
    (40.0, "Poor"),
    (0.0, "Critical"),
]

# ---------------------------------------------------------------------------
# Output data classes
# ---------------------------------------------------------------------------


@dataclass
class DimensionScores:
    """
    Scores for each individual dimension.

    All values are in the range [0, dimension_max].
    """

    correctness: float = 0.0  # 0–35
    efficiency_complexity: float = 0.0  # 0–30

    quality: float = 0.0  # 0–20
    maintainability: float = 0.0  # 0–15

    # Sub-scores within efficiency_complexity (informational)
    complexity_subscore: float = 0.0  # 0–15  (how the program scales)
    efficiency_subscore: float = 0.0  # 0–15  (wasted work within complexity class)

    # Flags that record whether partial credit was applied
    profiling_partial: bool = False  # True when profiling data was absent
    optimizer_partial: bool = False  # True when optimizer data was absent

    def to_dict(self) -> Dict[str, Any]:
        return {
            "correctness": round(self.correctness, 2),
            "efficiency_complexity": round(self.efficiency_complexity, 2),
            "quality": round(self.quality, 2),
            "maintainability": round(self.maintainability, 2),
            "complexity_subscore": round(self.complexity_subscore, 2),
            "efficiency_subscore": round(self.efficiency_subscore, 2),
            "profiling_partial": self.profiling_partial,
            "optimizer_partial": self.optimizer_partial,
        }


@dataclass
class ScoreReport:
    """
    Complete scoring report returned by the Scorer.

    Attributes:
        score:             Final score (0.0–100.0).
        grade:             Human-readable grade label.
        complexity_class:  Detected Big-O class string.
        dimensions:        Per-dimension score breakdown.
        narrative:         Beginner-friendly explanation of the score.
        error_count:       Number of errors in result.errors.
        lines_profiled:    Number of unique lines that were executed.
        cv:                Coefficient of variation of execution counts
                           (informational only — not used in scoring).
    """

    score: float
    grade: str
    complexity_class: str
    dimensions: DimensionScores
    narrative: str
    error_count: int = 0
    lines_profiled: int = 0
    cv: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": round(self.score, 2),
            "grade": self.grade,
            "complexity_class": self.complexity_class,
            "dimensions": self.dimensions.to_dict(),
            "narrative": self.narrative,
            "error_count": self.error_count,
            "lines_profiled": self.lines_profiled,
            "cv": round(self.cv, 4),
        }


# ---------------------------------------------------------------------------
# Internal complexity heuristic
# ---------------------------------------------------------------------------

EXPONENTIAL_COUNT_THRESHOLD: int = 1_000_000
MIN_N: int = 4


# Internal fallback only — used when profiling_data does not contain
# Primary path: Scorer._score_efficiency_complexity reads
# profiling_data["complexity_estimate"] set by Profiler.stop().
def _detect_complexity(line_stats: Dict[str, Any]) -> str:
    """
    Heuristic Big-O detection from a single execution trace.

    Reads execution counts from profiling line_stats and returns a
    standard complexity class string.

    Returns one of:
        "O(1)", "O(log n)", "O(n)", "O(n log n)",
        "O(n²)", "O(n³)", "O(n^k)", "O(2^n)"
    """
    if not line_stats:
        return "O(1)"

    execution_counts: List[int] = [int(s.get("count", 0)) for s in line_stats.values()]
    lines_profiled = len(execution_counts)
    max_c = max(execution_counts) if execution_counts else 0

    if max_c == 0:
        return "O(1)"

    HOT_RATIO = 0.5
    hot_counts = [c for c in execution_counts if c >= max_c * HOT_RATIO]
    non_hot = [c for c in execution_counts if c < max_c * HOT_RATIO]
    cluster_size = len(hot_counts)

    sqrt_max = math.sqrt(max_c)
    max_non_hot = max(non_hot) if non_hot else 0
    has_outer_loop = max_non_hot > sqrt_max * 0.5

    n = max(max_c, lines_profiled, MIN_N)

    if not has_outer_loop:
        return _classify_linear_or_below(max_c, n)

    if _is_exponential(max_c, n):
        return "O(2^n)"

    if cluster_size <= 2:
        return "O(n²)"
    elif cluster_size == 3:
        return "O(n³)"
    else:
        return "O(n^k)"


def _classify_linear_or_below(max_c: int, n: int) -> str:
    log_n = math.log2(n)
    if max_c <= 1:
        return "O(1)"
    if max_c <= log_n * 2:
        return "O(log n)"
    if max_c <= n * 2:
        return "O(n)"
    if max_c <= n * log_n * 3:
        return "O(n log n)"
    return "O(n²)"


def _is_exponential(max_c: int, n: int) -> bool:
    if max_c < EXPONENTIAL_COUNT_THRESHOLD:
        return False
    if n > 60:
        return False
    return bool(max_c >= (2**n) * 0.5)


# ---------------------------------------------------------------------------
# Density → score conversion (linear interpolation — no step cliffs)
# ---------------------------------------------------------------------------


def _density_to_score(density: float, max_score: float) -> float:
    """
    Map a weighted issue density value to a score in [0, max_score].

    Uses linear interpolation between the anchor points in _DENSITY_ANCHORS
    so that programs with nearly identical densities receive nearly identical
    scores. The old step-band approach produced up to a 20 % cliff for a
    0.02 density difference at a band boundary.

    Beyond the last anchor (density > 2.0) the score is clamped at 10 %
    of max_score.

    Args:
        density:   Weighted issue count / source_lines (≥ 0.0).
        max_score: The maximum possible score for this dimension.

    Returns:
        Score in [max_score * 0.10, max_score], rounded to 2 dp.
    """
    anchors = _DENSITY_ANCHORS

    # At or below the first anchor — full marks
    if density <= anchors[0][0]:
        return round(max_score * anchors[0][1], 2)

    # Beyond the last anchor — clamp to minimum fraction
    if density >= anchors[-1][0]:
        return round(max_score * anchors[-1][1], 2)

    # Find the two bracketing anchors and linearly interpolate
    for i in range(len(anchors) - 1):
        d_lo, f_lo = anchors[i]
        d_hi, f_hi = anchors[i + 1]
        if d_lo <= density <= d_hi:
            t = (density - d_lo) / (d_hi - d_lo)  # 0.0 … 1.0
            fraction = f_lo + t * (f_hi - f_lo)
            return round(max_score * fraction, 2)

    # Unreachable — safety fallback
    return round(max_score * anchors[-1][1], 2)


# ---------------------------------------------------------------------------
# Main Scorer class
# ---------------------------------------------------------------------------


def _coverage_weighted_complexity(
    complexity_class: str,
    line_stats: Dict[str, Any],
) -> float:
    """
    Return a complexity sub-score (0–15) that accounts for how much of
    the program is actually affected by the detected complexity class.

    Problem with a fixed lookup:
        COMPLEXITY_POINTS["O(n²)"] = 6.0 regardless of whether 2 lines
        or 200 lines are inside the quadratic hot path. A tiny nested loop
        in a 500-line program should not be penalised as harshly as a
        program that is quadratic from top to bottom.

    Algorithm:
        1. Start with the base points for the complexity class (same table
           as before — this represents the worst-case penalty for that class).
        2. Compute hot_coverage = hot_lines / total_lines, where hot_lines
           are those executing more than sqrt(max_count) times. This is a
           proxy for "what fraction of the program is inside the hot path".
        3. Blend: score = full_marks × (1 - hot_coverage) +
                          base_points × hot_coverage
           When hot_coverage → 0 (tiny hot path): score → full_marks (15).
           When hot_coverage → 1 (entire program is hot): score → base_points.

    This means:
        - A 200-line program with 2 quadratic lines → hot_coverage ≈ 0.01
          → score ≈ 14.9  (almost perfect, the problem is tiny)
        - A 5-line program where all lines are quadratic → hot_coverage ≈ 1.0
          → score ≈ 6.0   (full O(n²) penalty)
        - O(1) and O(log n) always return 15.0 regardless of coverage
          (no penalty to apply).

    Args:
        complexity_class: Big-O class string from _detect_complexity().
        line_stats:       Profiling line_stats dict.

    Returns:
        Complexity sub-score in [0.0, 15.0].
    """
    base = COMPLEXITY_POINTS.get(complexity_class, PARTIAL_COMPLEXITY)

    # Perfect classes — no penalty possible, skip coverage calculation
    if base >= 15.0:
        return 15.0

    counts = [int(s.get("count", 0)) for s in line_stats.values()]
    total_lines = len(counts)
    if total_lines == 0:
        return base

    max_count = max(counts) if counts else 0
    if max_count == 0:
        return 15.0

    # Lines executing more than sqrt(max_count) times are "in the hot path"
    threshold = math.sqrt(max_count)
    hot_lines = sum(1 for c in counts if c > threshold)
    hot_coverage = hot_lines / total_lines  # 0.0 … 1.0

    # Blend between full marks (no hot path) and base penalty (all hot)
    score = 15.0 * (1.0 - hot_coverage) + base * hot_coverage
    return round(max(0.0, min(15.0, score)), 2)


class Scorer:
    """
    Calculates a four-dimension optimization score (0–100).

    Args:
        profiling_data:    Dict from ``ProfilingData.to_dict()``.
                           Pass ``None`` when profiling is unavailable —
                           partial credit is awarded automatically.
        optimizer_report:  ``OptimizationReport`` from ``Optimizer.run()``.
                           Pass ``None`` when the optimizer has not run —
                           partial credit is awarded automatically.
        source_lines:      Number of lines in the original source code.
                           Used to normalise density-based sub-scores.
        errors:            List of error strings from ``ExecutionResult.errors``.
                           Defaults to empty list (perfect correctness).

    Usage::

        from optilang import execute
        from optilang.lexer import tokenize
        from optilang.parser import parse
        from optilang.optimizer import Optimizer
        from optilang.scoring import Scorer

        source = \"\"\"
        for i in range(100):
            for j in range(100):
                x = i + j
        \"\"\"

        result   = execute(source)
        ast      = parse(tokenize(source))
        report   = Optimizer(ast, result.profiling, result.symbol_table).run()

        scorer   = Scorer(
            profiling_data=result.profiling.to_dict() if result.profiling else None,
            optimizer_report=report,
            source_lines=source.count("\\n") + 1,
            errors=result.errors,
        )
        score_report = scorer.calculate()
        print(score_report.score)
        print(score_report.narrative)
    """

    def __init__(
        self,
        profiling_data: Optional[Dict[str, Any]],
        optimizer_report: Optional[Any],  # OptimizationReport | None
        source_lines: int = 1,
        errors: Optional[List[str]] = None,
    ) -> None:
        self._profiling = profiling_data
        self._optimizer = optimizer_report
        self._source_lines = max(source_lines, 1)
        self._errors = errors or []

        # Pre-extract line_stats for convenience
        self._line_stats: Dict[str, Any] = (
            self._profiling.get("line_stats", {}) if self._profiling else {}
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(self) -> ScoreReport:
        """
        Calculate scores for all four dimensions and return a ScoreReport.
        """
        dims = DimensionScores()

        # ── Correctness ───────────────────────
        dims.correctness = self._score_correctness()

        # ── Efficiency + Complexity ───────────────────
        # Returns five values: class string, two sub-scores, and two
        # separate partial-credit flags — one per data source.
        complexity_class, c_sub, e_sub, profiling_partial, optimizer_partial_eff = (
            self._score_efficiency_complexity()
        )
        dims.complexity_subscore = c_sub
        dims.efficiency_subscore = e_sub
        dims.efficiency_complexity = c_sub + e_sub
        dims.profiling_partial = profiling_partial

        # ── Quality ─────────────────────────
        q_score, optimizer_partial_quality = self._score_quality()
        dims.quality = q_score

        # ── Maintainability ──────────────────────
        dims.maintainability = self._score_maintainability()

        # optimizer_partial is True if ANY optimizer-dependent sub-score
        # used partial credit: efficiency sub-score, quality, or
        # maintainability all require the optimizer to have run.
        dims.optimizer_partial = optimizer_partial_eff or optimizer_partial_quality

        # ── Final score ───────────────────────
        total = (
            dims.correctness
            + dims.efficiency_complexity
            + dims.quality
            + dims.maintainability
        )
        final = max(0.0, min(100.0, total))

        grade = _assign_grade(final)

        # Collect all suggestions for dynamic narrative hint building
        all_suggestions: List[Any] = (
            list(getattr(self._optimizer, "suggestions", []))
            if self._optimizer is not None
            else []
        )
        narrative = _generate_narrative(
            final,
            dims,
            complexity_class=complexity_class,
            all_suggestions=all_suggestions,
        )

        # CV is kept for diagnostic/informational output only
        cv = self._compute_cv()
        lines_profiled = len(self._line_stats)

        return ScoreReport(
            score=round(final, 2),
            grade=grade,
            complexity_class=complexity_class,
            dimensions=dims,
            narrative=narrative,
            error_count=len(self._errors),
            lines_profiled=lines_profiled,
            cv=round(cv, 4),
        )

    # ------------------------------------------------------------------
    # Dimension calculations
    # ------------------------------------------------------------------

    def _score_correctness(self) -> float:
        """
        Correctness (0–35) based on error density relative to program size.

        Uses error density = errors / source_lines, fed through the same
        linear-interpolated density-to-score function as all other dimensions.
        This means a 100-line program with 4 errors (density 0.04) is judged
        far more leniently than a 5-line program with 3 errors (density 0.60),
        which is the correct behaviour — proportional impact matters.

        Special case: zero errors always yields full marks (35.0) regardless
        of program size.

        Density anchors (shared with _density_to_score):
            0.00  → 35.0  (no errors)
            0.30  → 28.0  (sparse errors  — 80 %)
            0.60  → 21.0  (moderate errors — 60 %)
            1.00  → 12.25 (heavy errors    — 35 %)
            2.00+ →  3.5  (critical        — 10 %, clamped)
        """
        n = len(self._errors)
        if n == 0:
            return MAX_CORRECTNESS

        density = n / self._source_lines
        return _density_to_score(density, MAX_CORRECTNESS)

    def _score_efficiency_complexity(
        self,
    ) -> Tuple[str, float, float, bool, bool]:
        """
        Efficiency + Complexity dimension (0–30).

        Complexity sub-score (0–15):
            Derived from Big-O class detected via execution count heuristic.
            Requires profiling data; falls back to PARTIAL_COMPLEXITY (7.0).

        Efficiency sub-score (0–15):
            Derived from EFFICIENCY_PATTERNS optimizer suggestions.
            Requires optimizer data; falls back to PARTIAL_EFFICIENCY (8.0).

        Both data-source failures are tracked independently so that
        calculate() can set dims.optimizer_partial correctly even when
        only the efficiency sub-score (not quality) needed partial credit.

        Returns:
            (complexity_class, complexity_subscore, efficiency_subscore,
             profiling_partial_flag, optimizer_partial_flag)
        """
        # Complexity sub-score — requires profiling data
        if not self._line_stats or self._profiling is None:
            c_sub = PARTIAL_COMPLEXITY
            complexity_class = "Unknown"
            profiling_partial = True
        else:
            complexity_class = self._profiling.get(
                "complexity_estimate"
            ) or _detect_complexity(self._line_stats)
            c_sub = _coverage_weighted_complexity(complexity_class, self._line_stats)
            profiling_partial = False

        # Efficiency sub-score — requires optimizer data
        if self._optimizer is None:
            e_sub = PARTIAL_EFFICIENCY
            optimizer_partial = True
        else:
            suggestions = getattr(self._optimizer, "suggestions", [])
            efficiency_suggestions = [
                s for s in suggestions if s.pattern in EFFICIENCY_PATTERNS
            ]
            density = self._weighted_density(efficiency_suggestions)
            e_sub = _density_to_score(density, 15.0)
            optimizer_partial = False

        return complexity_class, c_sub, e_sub, profiling_partial, optimizer_partial

    def _score_quality(self) -> Tuple[float, bool]:
        """
        Quality score (0–20) from optimizer suggestions that affect runtime
        behaviour beyond loop structure.

        Patterns: dead_code, string_concat_loop.

        Uses linear-interpolated weighted density scoring.

        Returns:
            (score, partial_credit_flag)
        """
        if self._optimizer is None:
            return PARTIAL_QUALITY, True

        suggestions = getattr(self._optimizer, "suggestions", [])
        quality_suggestions = [s for s in suggestions if s.pattern in QUALITY_PATTERNS]
        density = self._weighted_density(quality_suggestions)
        return _density_to_score(density, MAX_QUALITY), False

    def _score_maintainability(self) -> float:
        """
        Maintainability score (0–15) from optimizer suggestions affecting
        readability and structural clarity.

        Patterns: unused_vars, early_return, nested_loops, constant_folding.

        constant_folding is classified here (not Quality) because replacing
        a computable literal expression is a write-time code-clarity concern,
        not a runtime performance concern.

        Uses linear-interpolated weighted density scoring.

        Returns partial credit when optimizer is absent. The partial flag is
        already captured by _score_quality; no second flag needed here.
        """
        if self._optimizer is None:
            return PARTIAL_MAINTAINABILITY

        suggestions = getattr(self._optimizer, "suggestions", [])
        maint_suggestions = [
            s for s in suggestions if s.pattern in MAINTAINABILITY_PATTERNS
        ]
        density = self._weighted_density(maint_suggestions)
        return _density_to_score(density, MAX_MAINTAINABILITY)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _weighted_density(self, suggestions: List[Any]) -> float:
        """
        Compute weighted issue density relative to program size.

        weighted = sum(3×high + 2×medium + 1×low)
        density  = weighted / source_lines

        Program-size-agnostic: a 5-line and a 50-line program with the
        same number of issues produce different densities, reflecting the
        proportional impact of the issues on each program.
        """
        weighted = sum(
            3 if s.severity == "high" else 2 if s.severity == "medium" else 1
            for s in suggestions
        )
        return weighted / self._source_lines

    def _compute_cv(self) -> float:
        """
        Coefficient of Variation of execution counts.

        Informational only — no longer used in scoring. Retained for
        diagnostic output in ScoreReport.cv.
        """
        counts: List[int] = [
            int(s.get("count", 0))
            for s in self._line_stats.values()
            if s.get("count", 0) > 0
        ]
        if len(counts) < MIN_LINES_FOR_CV:
            return 0.0

        mean = sum(counts) / len(counts)
        if mean == 0.0:
            return 0.0

        variance = sum((c - mean) ** 2 for c in counts) / len(counts)
        std = math.sqrt(variance)
        return std / mean


# ---------------------------------------------------------------------------
# Grade and narrative (module-level helpers)
# ---------------------------------------------------------------------------


def _assign_grade(score: float) -> str:
    """Map a numeric score to a grade label."""
    for threshold, grade in GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return "Critical"


def _rank_dimensions(dims: DimensionScores) -> List[Tuple[str, float, float]]:
    """
    Rank all four dimensions by how much room for improvement they have,
    expressed as absolute points missing from their maximum.

    Returns a list of (name, ratio, points_missing) tuples sorted by
    points_missing descending (most room to improve first).

    Normalised ratio is used to detect whether a dimension is "perfect"
    (ratio == 1.0). Points missing is used as the primary sort key so
    that higher-weight dimensions (Correctness max=35) are surfaced above
    lower-weight ones (Maintainability max=15) when both are equally
    imperfect in percentage terms.

    Tie-breaking: when points_missing is equal, the dimension with the
    higher maximum wins — it represents a more significant absolute gap.
    """
    dimension_maxes: Dict[str, float] = {
        "Correctness": MAX_CORRECTNESS,
        "Efficiency & Complexity": MAX_EFFICIENCY_COMPLEXITY,
        "Quality": MAX_QUALITY,
        "Maintainability": MAX_MAINTAINABILITY,
    }
    raw_scores: Dict[str, float] = {
        "Correctness": dims.correctness,
        "Efficiency & Complexity": dims.efficiency_complexity,
        "Quality": dims.quality,
        "Maintainability": dims.maintainability,
    }
    ranked = []
    for name, max_val in dimension_maxes.items():
        raw = raw_scores[name]
        ratio = raw / max_val
        missing = max_val - raw
        ranked.append((name, ratio, missing))

    # Sort: most points missing first; break ties by higher max
    ranked.sort(key=lambda t: (-t[2], -dimension_maxes[t[0]]))
    return ranked


# Threshold below which a dimension is considered "actionable" (not perfect).
# A dimension scoring ≥ this fraction of its max is treated as fully healthy
# and will not be mentioned in the narrative as needing improvement.
_PERFECT_RATIO: float = 1.0  # exactly full marks
_HEALTHY_RATIO: float = 0.90  # within 10 % of full marks → no complaint

# ---------------------------------------------------------------------------
# Pattern → human-readable label (used in dynamic hints)
# ---------------------------------------------------------------------------

_PATTERN_LABELS: Dict[str, str] = {
    # Efficiency patterns
    "hot_loop": "a loop that dominates execution time",
    "loop_invariant": (
        "a computation inside a loop whose value never changes between iterations"
    ),
    "repeated_computation": (
        "the same expression computed more than once unnecessarily"
    ),
    "expensive_calls": "a slow function called repeatedly inside a loop",
    # Quality patterns
    "dead_code": "code that can never execute",
    "string_concat_loop": (
        "string concatenation with += inside a loop (creates O(n²) copies)"
    ),
    # Maintainability patterns
    "unused_vars": "a variable that is assigned but never used",
    "early_return": "a function that could return early to reduce nesting",
    "nested_loops": "a loop nested inside another loop",
    "constant_folding": (
        "a literal expression that could be pre-computed (e.g. '3 * 4' → '12')"
    ),
}

# ---------------------------------------------------------------------------
# Complexity class → plain-English explanation (used when complexity sub-score
# is the reason Efficiency & Complexity is below full marks)
# ---------------------------------------------------------------------------

_COMPLEXITY_EXPLANATIONS: Dict[str, str | None] = {
    "O(1)": None,  # perfect — no explanation needed
    "O(log n)": None,  # perfect — no explanation needed
    "O(n)": (
        "linear time (O(n)) — acceptable for most programs, "
        "but watch for unnecessary work inside the loop"
    ),
    "O(n log n)": (
        "O(n log n) time — typical of sorting algorithms; "
        "consider whether this is necessary"
    ),
    "O(n²)": (
        "quadratic time (O(n²)) — often caused by a loop nested inside "
        "another loop; consider restructuring"
    ),
    "O(n³)": (
        "cubic time (O(n³)) — caused by deeply nested loops; "
        "this will become very slow as input grows"
    ),
    "O(n^k)": (
        "polynomial time — caused by deeply nested loops; "
        "consider a more efficient algorithm"
    ),
    "O(2^n)": (
        "exponential time (O(2^n)) — this will be impractically slow "
        "for any non-trivial input; consider dynamic programming or memoization"
    ),
    "Unknown": None,  # no profiling data — nothing specific to say
}


def _build_dimension_hint(
    name: str,
    complexity_class: str,
    complexity_subscore: float,
    suggestions: List[Any],
) -> str:
    """
    Build a specific hint for one dimension based on what was actually found.

    For Correctness: always the same (fix your errors).
    For Efficiency & Complexity: combines complexity class explanation (if
        complexity sub-score < 15) with the actual efficiency suggestions found.
    For Quality and Maintainability: lists only the optimizer suggestions that
        belong to that dimension — no generic laundry list of everything possible.

    Args:
        name:               Dimension name.
        complexity_class:   Detected Big-O class string (from ScoreReport).
        complexity_subscore: Complexity sub-score (0–15).
        suggestions:        All optimizer suggestions for this dimension only.

    Returns:
        A concise, specific hint string.
    """
    seen_patterns: Set[str] = set()
    if name == "Correctness":
        return (
            "Fix the errors in your program first. "
            "A program that runs without errors is the foundation of everything else."
        )

    if name == "Efficiency & Complexity":
        parts: List[str] = []

        # Complexity part — only if complexity is not perfect
        if complexity_subscore < 15.0:
            explanation = _COMPLEXITY_EXPLANATIONS.get(complexity_class)
            if explanation:
                parts.append(f"Your program runs in {explanation}.")

        # Efficiency part — only the patterns actually detected
        if suggestions:
            seen_patterns.clear()
            pattern_lines = []
            for s in suggestions:
                if s.pattern in seen_patterns:
                    continue
                seen_patterns.add(s.pattern)
                label = _PATTERN_LABELS.get(s.pattern)
                if label:
                    pattern_lines.append(f"• {label.capitalize()}.")
            if pattern_lines:
                parts.append(
                    "The optimizer found the following efficiency issues:\n"
                    + "\n".join(pattern_lines)
                )

        if not parts:
            # complexity is perfect (15/15) and no efficiency suggestions —
            # this dimension is close to perfect, give generic encouragement
            return (
                "Your algorithmic complexity looks good. "
                "Keep an eye on loop-invariant computations and repeated "
                "expressions as your programs grow."
            )

        return " ".join(parts)

    # Quality and Maintainability — list only what was actually found
    if suggestions:
        pattern_lines = []
        seen_patterns.clear()
        for s in suggestions:
            if s.pattern in seen_patterns:
                continue
            seen_patterns.add(s.pattern)
            label = _PATTERN_LABELS.get(s.pattern)
            if label:
                pattern_lines.append(f"• {label.capitalize()}.")
        if pattern_lines:
            return "The optimizer found the following issues:\n" + "\n".join(
                pattern_lines
            )

    # Fallback: dimension scored below healthy but no specific suggestions
    # (can happen with partial credit or rounding)
    fallback: Dict[str, str] = {
        "Quality": (
            "Review your program for dead code that can never execute "
            "and string concatenation with += inside loops."
        ),
        "Maintainability": (
            "Review your program for unused variables, expressions that "
            "could be pre-computed, opportunities to return early from "
            "functions, and loops nested inside other loops."
        ),
    }
    return fallback.get(name, "Review the optimizer suggestions for this dimension.")


def _generate_narrative(
    score: float,
    dims: DimensionScores,
    complexity_class: str = "Unknown",
    all_suggestions: Optional[List[Any]] = None,
) -> str:
    """
    Generate a clear, honest, specific, beginner-friendly narrative.

    Logic:
        1. If every dimension is at 100 % → pure congratulation.
        2. Otherwise collect every dimension whose ratio < _HEALTHY_RATIO,
           ranked by most absolute points missing. All of them are mentioned
           with a SPECIFIC hint based on what was actually found — not a
           generic laundry list of every possible issue.
        3. If all dimensions are healthy (≥ 90 %) but not all perfect,
           give a gentle note on the single most-improvable dimension.
        4. Append an accurate partial-credit note when data was unavailable.

    Args:
        score:            Final score (0–100).
        dims:             DimensionScores from Scorer.calculate().
        complexity_class: Detected Big-O class (e.g. "O(n)").
        all_suggestions:  All optimizer suggestions (used to build specific hints).
    """
    all_suggestions = all_suggestions or []

    # Pre-partition suggestions by dimension so each hint only sees its own
    efficiency_suggs = [s for s in all_suggestions if s.pattern in EFFICIENCY_PATTERNS]
    quality_suggs = [s for s in all_suggestions if s.pattern in QUALITY_PATTERNS]
    maint_suggs = [s for s in all_suggestions if s.pattern in MAINTAINABILITY_PATTERNS]

    ranked = _rank_dimensions(dims)

    actionable = [
        (name, ratio, missing)
        for name, ratio, missing in ranked
        if ratio < _HEALTHY_RATIO
    ]
    near_perfect = [
        (name, ratio, missing)
        for name, ratio, missing in ranked
        if _HEALTHY_RATIO <= ratio < _PERFECT_RATIO
    ]
    all_perfect = all(ratio >= _PERFECT_RATIO for _, ratio, _ in ranked)

    def _hint_for(name: str) -> str:
        suggs_map = {
            "Efficiency & Complexity": efficiency_suggs,
            "Quality": quality_suggs,
            "Maintainability": maint_suggs,
            "Correctness": [],
        }
        return _build_dimension_hint(
            name,
            complexity_class,
            dims.complexity_subscore,
            suggs_map.get(name, []),
        )

    # ── Partial-credit note ──────────────────────
    partial_note = ""
    if dims.profiling_partial and dims.optimizer_partial:
        partial_note = (
            " Note: both profiling and optimizer data were unavailable. "
            "Partial credit was awarded for the Complexity sub-score "
            "(profiling absent) and for the Efficiency sub-score, Quality, "
            "and Maintainability (optimizer absent)."
        )
    elif dims.profiling_partial:
        partial_note = (
            " Note: profiling data was unavailable, so partial credit was "
            "awarded for the Complexity sub-score only. The Efficiency "
            "sub-score, Quality, and Maintainability were scored normally "
            "from the optimizer report."
        )
    elif dims.optimizer_partial:
        partial_note = (
            " Note: optimizer data was unavailable, so partial credit was "
            "awarded for the Efficiency sub-score, Quality, and "
            "Maintainability. The Complexity sub-score was scored normally "
            "from profiling data."
        )

    # ── Grade headline ────────────────────────
    if score >= 90:
        headline = (
            "Excellent work! Your program is correct, efficient, and "
            "well-structured. "
        )
    elif score >= 75:
        headline = "Good job! Your program runs well and shows solid structure. "
    elif score >= 60:
        headline = (
            "Fair result. Your program works, but there are clear "
            "opportunities to improve it. "
        )
    elif score >= 40:
        headline = "Your program needs some work. "
    else:
        headline = "Your program has significant issues that need to be resolved. "

    # ── Body ───────────────────────

    # Case 1: truly perfect — no improvement advice needed
    if all_perfect:
        body = (
            "Every dimension scores full marks. "
            "Keep writing clean, efficient code as your programs grow "
            "more complex."
        )

    # Case 2: one or more dimensions are meaningfully below healthy
    elif actionable:
        if len(actionable) == 1:
            name, _, _ = actionable[0]
            body = f"Your main area to improve is {name}. " f"{_hint_for(name)}"
        else:
            items = []
            for i, (name, ratio, missing) in enumerate(actionable, 1):
                pct = round((1.0 - ratio) * 100)
                items.append(
                    f"{i}. {name} ({pct}% below full marks) — {_hint_for(name)}"
                )
            body = (
                "Multiple areas need attention, listed from most to least "
                "impactful:\n" + "\n".join(items)
            )

    # Case 3: all healthy but not all perfect — gentle single note
    else:
        name, ratio, missing = near_perfect[0] if near_perfect else ranked[0]
        pct = round((1.0 - ratio) * 100)
        body = (
            f"Your program is in great shape. "
            f"If you want to squeeze out the last few points, "
            f"{name} is {pct}% below full marks. "
            f"{_hint_for(name)}"
        )

    return headline + body + partial_note


# ---------------------------------------------------------------------------
# Convenience function — public API
# ---------------------------------------------------------------------------


def calculate_score(
    profiling_data: Optional[Dict[str, Any]],
    optimizer_report: Optional[Any] = None,
    source_lines: int = 1,
    errors: Optional[List[str]] = None,
) -> ScoreReport:
    """
    Calculate a four-dimension optimization score.

    Primary public entry point for the scoring system.

    Args:
        profiling_data:    Dict from ``ProfilingData.to_dict()``, or None.
        optimizer_report:  ``OptimizationReport`` from ``Optimizer.run()``, or None.
        source_lines:      Number of lines in the original source code.
        errors:            List of error strings from ``ExecutionResult.errors``.

    Returns:
        ``ScoreReport`` with final score, grade, complexity class,
        per-dimension scores, and a beginner-friendly narrative.

    Example::

        from optilang import execute
        from optilang.lexer import tokenize
        from optilang.parser import parse
        from optilang.optimizer import Optimizer
        from optilang.scoring import calculate_score

        source = "for i in range(100):\\n    for j in range(100):\\n        x = i+j"
        result = execute(source)
        ast    = parse(tokenize(source))
        report = Optimizer(ast, result.profiling, result.symbol_table).run()

        score_report = calculate_score(
            profiling_data=result.profiling.to_dict() if result.profiling else None,
            optimizer_report=report,
            source_lines=source.count("\\n") + 1,
            errors=result.errors,
        )

        print(score_report.score)            # e.g. 68.5
        print(score_report.grade)            # "Fair"
        print(score_report.complexity_class) # "O(n²)"
        print(score_report.narrative)        # beginner-friendly explanation
        print(score_report.to_dict())        # full JSON-serialisable output
    """
    return Scorer(
        profiling_data=profiling_data,
        optimizer_report=optimizer_report,
        source_lines=source_lines,
        errors=errors,
    ).calculate()
