"""
optilang/scoring.py
-------------------
Scoring system for OptiLang.

Calculates an overall optimization score (0–100) across four dimensions:

    Dimension               Max     Source
    ─────────────────────────────────────────────────────────────────
    Correctness              35     result.errors
    Efficiency + Complexity  30     profiling.line_stats
    Quality                  20     optimizer suggestions (runtime patterns)
    Maintainability          15     optimizer suggestions (style patterns)
    ─────────────────────────────────────────────────────────────────
    Total                   100

Design principles:
    - Score-earned model: each dimension contributes positively (not penalty-based).
    - Dynamic scoring: calculations are relative to the program being analysed,
      not fixed global thresholds. A 5-line program and a 50-line program are
      judged on their own terms.
    - Graceful degradation: if profiling or optimizer data is unavailable,
      partial credit (50 % of max) is awarded for that dimension rather than
      zero, reflecting genuine uncertainty rather than failure.

Dimension breakdown:

    Correctness (0–35)
        Derived directly from result.errors. No algorithm needed.
        0 errors → 35 | 1 error → 10 | 2+ errors → 0

    Efficiency + Complexity (0–30), two equal sub-scores:

        Complexity sub-score (0–15)
            Reuses the existing _detect_complexity() heuristic which reads
            execution counts from profiling.line_stats and returns a standard
            Big-O class string. That string is mapped to points.

        Efficiency sub-score (0–15)
            Uses the Coefficient of Variation (CV = std / mean) of line
            execution counts. CV is dimensionless and program-size-agnostic:
            a perfectly flat execution profile → CV ≈ 0 (efficient);
            a spiked nested-loop profile → CV >> 1 (inefficient).

    Quality (0–20)
        From optimizer suggestions whose patterns affect runtime behaviour:
        hot_loop, loop_invariant, repeated_computation, expensive_calls,
        dead_code, constant_folding.
        Weighted issue density (weighted_count / source_lines) is mapped
        to points so that a 5-line and 50-line program with the same number
        of issues are scored proportionally.

    Maintainability (0–15)
        From optimizer suggestions whose patterns affect readability:
        unused_vars, early_return, string_concat_loop, nested_loops.
        Same weighted density formula, mapped to 0–15.

Narrative:
    A short human-readable explanation is generated from the final score
    and the lowest-scoring dimension, aimed at beginners. It names one
    concrete area to focus on rather than listing every problem.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Pattern classification
# ---------------------------------------------------------------------------

# Patterns that affect runtime behaviour → Quality dimension
QUALITY_PATTERNS = frozenset(
    {
        "hot_loop",
        "loop_invariant",
        "repeated_computation",
        "expensive_calls",
        "dead_code",
        "constant_folding",
    }
)

# Patterns that affect readability / structure → Maintainability dimension
MAINTAINABILITY_PATTERNS = frozenset(
    {
        "unused_vars",
        "early_return",
        "string_concat_loop",
        "nested_loops",
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
# Dimension maximums
# ---------------------------------------------------------------------------

MAX_CORRECTNESS: float = 35.0
MAX_EFFICIENCY_COMPLEXITY: float = 30.0  # 15 complexity + 15 efficiency
MAX_QUALITY: float = 20.0
MAX_MAINTAINABILITY: float = 15.0

# Partial credit (50 %) awarded when the required data is absent
PARTIAL_COMPLEXITY: float = 7.0  # half of 15, rounded up
PARTIAL_EFFICIENCY: float = 8.0  # half of 15, rounded up
PARTIAL_QUALITY: float = 10.0  # half of 20
PARTIAL_MAINTAINABILITY: float = 7.0  # half of 15, rounded up

# Minimum number of lines to avoid division-by-zero in CV calculation
MIN_LINES_FOR_CV: int = 2

# ---------------------------------------------------------------------------
# Grade thresholds
# ---------------------------------------------------------------------------

GRADE_THRESHOLDS: List[tuple[float, str]] = [
    (90.0, "Excellent"),
    (75.0, "Good"),
    (60.0, "Fair"),
    (40.0, "Poor"),
    (0.0, "Critical"),
]

# ---------------------------------------------------------------------------
# Output data class
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
    complexity_subscore: float = 0.0  # 0–15
    efficiency_subscore: float = 0.0  # 0–15

    # Flags that record whether partial credit was applied
    profiling_partial: bool = False
    optimizer_partial: bool = False

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
        cv:                Coefficient of variation of execution counts.
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
# Internal complexity heuristic (extracted from original Scorer)
# ---------------------------------------------------------------------------

# These constants are unchanged from the original scoring.py heuristic.
EXPONENTIAL_COUNT_THRESHOLD: int = 1_000_000
MIN_N: int = 4


def _detect_complexity(line_stats: Dict[str, Any]) -> str:
    """
    Heuristic Big-O detection from a single execution trace.

    Reads execution counts from profiling line_stats and returns a
    standard complexity class string. This is the same algorithm as
    the original Scorer._detect_complexity() and is kept here as a
    module-level function so it can be used without instantiating Scorer.

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
# Main Scorer class
# ---------------------------------------------------------------------------


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
                           Used to normalise quality/maintainability density.
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

        # ── Correctness ───────────────────────────────────────────────
        dims.correctness = self._score_correctness()

        # ── Efficiency + Complexity ───────────────────────────────────
        complexity_class, c_sub, e_sub, profiling_partial = (
            self._score_efficiency_complexity()
        )
        dims.complexity_subscore = c_sub
        dims.efficiency_subscore = e_sub
        dims.efficiency_complexity = c_sub + e_sub
        dims.profiling_partial = profiling_partial

        # ── Quality ───────────────────────────────────────────────────
        q_score, opt_partial = self._score_quality()
        dims.quality = q_score
        dims.optimizer_partial = opt_partial

        # ── Maintainability ───────────────────────────────────────────
        # optimizer_partial is already determined from quality calculation
        dims.maintainability = self._score_maintainability()

        # ── Final score ───────────────────────────────────────────────
        total = (
            dims.correctness
            + dims.efficiency_complexity
            + dims.quality
            + dims.maintainability
        )
        final = max(0.0, min(100.0, total))

        grade = _assign_grade(final)
        narrative = _generate_narrative(final, dims)

        # Supplementary info
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
        Correctness (0–35) derived directly from result.errors.

        Scale:
            0 errors  → 35.0
            1 error   → 10.0
            2+ errors →  0.0
        """
        n = len(self._errors)
        if n == 0:
            return 35.0
        if n == 1:
            return 10.0
        return 0.0

    def _score_efficiency_complexity(
        self,
    ) -> tuple[str, float, float, bool]:
        """
        Efficiency + Complexity dimension (0–30).

        Returns:
            (complexity_class, complexity_subscore, efficiency_subscore,
             profiling_partial_flag)
        """
        if not self._line_stats:
            # Profiling unavailable — partial credit
            return "Unknown", PARTIAL_COMPLEXITY, PARTIAL_EFFICIENCY, True

        complexity_class = _detect_complexity(self._line_stats)
        c_sub = COMPLEXITY_POINTS.get(complexity_class, PARTIAL_COMPLEXITY)
        e_sub = self._compute_efficiency_subscore()

        return complexity_class, c_sub, e_sub, False

    def _compute_efficiency_subscore(self) -> float:
        """
        Efficiency sub-score (0–15) using Coefficient of Variation.

        CV = std / mean of line execution counts.
        A flat execution profile (CV ≈ 0) scores maximum points.
        A heavily spiked profile (high CV) scores minimum points.

        Scale:
            CV < 0.5  → 15
            CV < 1.0  → 12
            CV < 2.0  →  8
            CV < 4.0  →  4
            CV ≥ 4.0  →  0
        """
        cv = self._compute_cv()

        if cv < 0.5:
            return 15.0
        if cv < 1.0:
            return 12.0
        if cv < 2.0:
            return 8.0
        if cv < 4.0:
            return 4.0
        return 0.0

    def _compute_cv(self) -> float:
        """
        Coefficient of Variation of execution counts across all profiled lines.

        Returns 0.0 when fewer than MIN_LINES_FOR_CV lines are available
        (not enough data for a meaningful spread measure).
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

    def _score_quality(self) -> tuple[float, bool]:
        """
        Quality score (0–20) from optimizer suggestions that affect runtime.

        Patterns: hot_loop, loop_invariant, repeated_computation,
                  expensive_calls, dead_code, constant_folding.

        Uses weighted density (weighted_issues / source_lines):
            density = 0      → 20
            density ≤ 0.3    → 16
            density ≤ 0.6    → 12
            density ≤ 1.0    →  7
            density > 1.0    →  3

        Returns:
            (score, partial_credit_flag)
        """
        if self._optimizer is None:
            return PARTIAL_QUALITY, True

        suggestions = getattr(self._optimizer, "suggestions", [])
        quality_suggestions = [s for s in suggestions if s.pattern in QUALITY_PATTERNS]

        density = self._weighted_density(quality_suggestions)
        return self._density_to_score(density, MAX_QUALITY), False

    def _score_maintainability(self) -> float:
        """
        Maintainability score (0–15) from optimizer suggestions affecting
        readability.

        Patterns: unused_vars, early_return, string_concat_loop, nested_loops.

        Same weighted density formula, mapped to 0–15.

        Returns partial credit when optimizer is absent (already flagged by
        _score_quality so no second flag needed here).
        """
        if self._optimizer is None:
            return PARTIAL_MAINTAINABILITY

        suggestions = getattr(self._optimizer, "suggestions", [])
        maint_suggestions = [
            s for s in suggestions if s.pattern in MAINTAINABILITY_PATTERNS
        ]

        density = self._weighted_density(maint_suggestions)
        return self._density_to_score(density, MAX_MAINTAINABILITY)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _weighted_density(self, suggestions: List[Any]) -> float:
        """
        Compute weighted issue density relative to program size.

        weighted = sum(3×high + 2×medium + 1×low)
        density  = weighted / source_lines

        The density is program-size-agnostic: a 5-line program and a
        50-line program with the same number of issues produce different
        densities, reflecting the proportional impact of the issues.
        """
        weighted = sum(
            3 if s.severity == "high" else 2 if s.severity == "medium" else 1
            for s in suggestions
        )
        return weighted / self._source_lines

    @staticmethod
    def _density_to_score(density: float, max_score: float) -> float:
        """
        Map a weighted issue density value to a score in [0, max_score].

        Thresholds are defined as fractions of max_score so the same
        mapping logic works for both Quality (max=20) and
        Maintainability (max=15).

        Density bands:
            0         → 100 % of max
            ≤ 0.3     →  80 %
            ≤ 0.6     →  60 %
            ≤ 1.0     →  35 %
            > 1.0     →  15 %
        """
        if density == 0.0:
            return max_score
        if density <= 0.3:
            return round(max_score * 0.80, 2)
        if density <= 0.6:
            return round(max_score * 0.60, 2)
        if density <= 1.0:
            return round(max_score * 0.35, 2)
        return round(max_score * 0.15, 2)


# ---------------------------------------------------------------------------
# Grade and narrative (module-level helpers)
# ---------------------------------------------------------------------------


def _assign_grade(score: float) -> str:
    """Map a numeric score to a grade label."""
    for threshold, grade in GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return "Critical"


def _lowest_dimension(dims: DimensionScores) -> str:
    """
    Return the name of the dimension with the lowest percentage of its maximum.

    Comparing raw scores directly would be unfair because the dimensions have
    different maximums (35, 30, 20, 15). Normalising to a percentage puts
    them on the same scale so the narrative points to the genuinely weakest
    area.
    """
    ratios = {
        "Correctness": dims.correctness / MAX_CORRECTNESS,
        "Efficiency & Complexity": dims.efficiency_complexity
        / MAX_EFFICIENCY_COMPLEXITY,
        "Quality": dims.quality / MAX_QUALITY,
        "Maintainability": dims.maintainability / MAX_MAINTAINABILITY,
    }
    return min(ratios, key=lambda k: ratios[k])


def _generate_narrative(score: float, dims: DimensionScores) -> str:
    """
    Generate a short, beginner-friendly explanation of the score.

    The narrative:
        1. Opens with a grade-appropriate headline.
        2. Names the single lowest-scoring dimension.
        3. Gives one concrete action to improve.
        4. Notes partial credit if applicable.
    """
    lowest = _lowest_dimension(dims)

    # Per-dimension improvement hints
    hints: Dict[str, str] = {
        "Correctness": (
            "Focus on fixing the errors in your program first. "
            "A program that runs without errors is the foundation of "
            "everything else."
        ),
        "Efficiency & Complexity": (
            "Look at how many times each line of your program runs. "
            "Nested loops (a loop inside a loop) cause lines to run "
            "exponentially more times and are the most common cause of "
            "poor efficiency."
        ),
        "Quality": (
            "Review the optimization suggestions — particularly any "
            "hot loops, loop-invariant computations, or repeated "
            "expressions. Moving work outside of loops is usually "
            "the highest-impact change you can make."
        ),
        "Maintainability": (
            "Look for unused variables, deeply nested loops, and "
            "opportunities to return early from functions. "
            "Cleaner structure makes code easier to reason about and "
            "often reveals further optimizations."
        ),
    }

    partial_note = ""
    if dims.profiling_partial and dims.optimizer_partial:
        partial_note = (
            " Note: profiling and optimizer data were unavailable, "
            "so partial credit was awarded for Efficiency, Complexity, "
            "Quality, and Maintainability."
        )
    elif dims.profiling_partial:
        partial_note = (
            " Note: profiling data was unavailable, so partial credit "
            "was awarded for Efficiency and Complexity."
        )
    elif dims.optimizer_partial:
        partial_note = (
            " Note: optimizer data was unavailable, so partial credit "
            "was awarded for Quality and Maintainability."
        )

    if score >= 90:
        headline = (
            "Excellent work! Your program is correct, efficient, and "
            "well-structured. "
        )
        body = (
            f"Your weakest area is {lowest} — keep it in mind as your "
            f"programs grow more complex."
        )

    elif score >= 75:
        headline = "Good job! Your program runs well and shows solid structure. "
        body = f"Your main area to improve is {lowest}. " f"{hints[lowest]}"

    elif score >= 60:
        headline = (
            "Fair result. Your program works, but there are clear "
            "opportunities to improve it. "
        )
        body = (
            f"Start with {lowest} — it had the biggest impact on your score. "
            f"{hints[lowest]}"
        )

    elif score >= 40:
        headline = "Your program needs some work. "
        body = (
            f"The most important thing to address right now is {lowest}. "
            f"{hints[lowest]}"
        )

    else:
        headline = "Your program has significant issues that need to be resolved. "
        body = f"Begin with {lowest} — this is the foundation. " f"{hints[lowest]}"

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

    This is the primary public entry point for the scoring system.

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

        print(score_report.score)            # e.g. 61.0
        print(score_report.grade)            # "Fair"
        print(score_report.complexity_class) # "O(n²)"
        print(score_report.narrative)        # beginner explanation
        print(score_report.to_dict())        # full JSON-serialisable output
    """
    return Scorer(
        profiling_data=profiling_data,
        optimizer_report=optimizer_report,
        source_lines=source_lines,
        errors=errors,
    ).calculate()
