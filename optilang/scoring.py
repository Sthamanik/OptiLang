"""
optilang/scoring.py
-------------------
Optimization scoring system for OptiLang.

Takes profiling data (from Profiler) and suggestions (from Optimizer, Sprint 3)
and produces a quantitative score (0-100) with a full breakdown.

Complexity detection uses a nesting-depth heuristic combined with absolute
execution counts — since we only have a single execution trace (not multiple
runs at varying input sizes), true Big-O derivation is not possible. This
approach provides a reasonable educational approximation.

Note: Weights and thresholds are empirically tuned based on benchmark programs
(loops, recursion, nested structures). See CONTRIBUTING.md for tuning guide.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Tunable constants — adjust these based on empirical benchmarking
# ---------------------------------------------------------------------------

# Severity weights (used once Optimizer is available in Sprint 3)
HIGH_SEVERITY_WEIGHT: int = 3
MEDIUM_SEVERITY_WEIGHT: int = 2
LOW_SEVERITY_WEIGHT: int = 1

# Maximum penalty each component can contribute
MAX_SEVERITY_PENALTY: float = 30.0
MAX_COMPLEXITY_PENALTY: float = 30.0
MAX_PERFORMANCE_PENALTY: float = 30.0
MAX_MEMORY_PENALTY: float = 10.0

# Performance baseline: expected milliseconds per single line execution
# A well-written program should take ~0.01ms per line execution on average
BASELINE_TIME_PER_LINE_MS: float = 0.01

# Memory: number of variables in scope considered "high" for a single line
HIGH_VAR_THRESHOLD: int = 10

# Complexity detection constants
# HOT_RATIO: a line is in the "inner hot cluster" if count >= max_count × HOT_RATIO
# OUTER_LOOP_RATIO: outer loop is detected
# if max_non_hot >= sqrt(max_count) × OUTER_LOOP_RATIO
# These are defined inline in _detect_complexity for clarity but documented here.
# HOT_RATIO = 0.5   (within 2× of peak)
# OUTER_LOOP_RATIO = 0.5  (generous lower bound for triangular/imperfect loops)

# Exponential detection: if max_count exceeds this absolute value AND
# nesting level is low, suspect O(2^n) rather than O(n^k)
EXPONENTIAL_COUNT_THRESHOLD: int = 1_000_000

# Minimum value of n used in calculations to avoid log(0) or divide-by-zero
MIN_N: int = 4

# Complexity class → penalty points mapping
COMPLEXITY_PENALTIES: Dict[str, float] = {
    "O(1)": 0.0,
    "O(log n)": 0.0,
    "O(n)": 0.0,
    "O(n log n)": 5.0,
    "O(n²)": 15.0,
    "O(n³)": 22.0,
    "O(n^k)": 28.0,
    "O(2^n)": 35.0,  # will be clamped to MAX_COMPLEXITY_PENALTY
}

# Score → grade thresholds
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
class ScoreReport:
    """
    Complete scoring report returned by the Scorer.

    Attributes:
        score:            Final optimization score (0.0 – 100.0).
        grade:            Human-readable grade label.
        complexity_class: Detected time complexity string e.g. "O(n²)".
        breakdown:        Per-component penalty values that produced the score.
        max_execution_count: Highest line execution count observed.
        lines_profiled:   Number of unique lines that were executed.
        baseline_time_ms: Estimated baseline execution time used for comparison.
    """

    score: float
    grade: str
    complexity_class: str
    breakdown: Dict[str, float] = field(default_factory=dict)
    max_execution_count: int = 0
    lines_profiled: int = 0
    baseline_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain dictionary (JSON-safe)."""
        return {
            "score": round(self.score, 2),
            "grade": self.grade,
            "complexity_class": self.complexity_class,
            "breakdown": {k: round(v, 4) for k, v in self.breakdown.items()},
            "max_execution_count": self.max_execution_count,
            "lines_profiled": self.lines_profiled,
            "baseline_time_ms": round(self.baseline_time_ms, 4),
        }


# ---------------------------------------------------------------------------
# Main Scorer class
# ---------------------------------------------------------------------------


class Scorer:
    """
    Calculates an optimization score (0–100) from profiling data and
    (optionally) a list of optimization suggestions.

    Usage::

        from optilang.scoring import Scorer

        # With profiling data only (v0.2.0 — no optimizer yet)
        scorer = Scorer(profiling_data=result.profiling)
        report = scorer.calculate()

        # With suggestions (Sprint 3 — once Optimizer is available)
        scorer = Scorer(
            profiling_data=result.profiling,
            suggestions=report.suggestions,
            total_source_lines=20,
        )
        report = scorer.calculate()

    Args:
        profiling_data:    Dict returned by ``ProfilingData.to_dict()``.
        suggestions:       List of ``Suggestion`` objects from the Optimizer.
                           Defaults to empty list (severity_penalty = 0).
        total_source_lines: Number of lines in the original source code.
                           Used to normalise the severity penalty. Defaults to 1.
    """

    def __init__(
        self,
        profiling_data: Dict[str, Any],
        suggestions: Optional[List[Any]] = None,
        total_source_lines: int = 1,
    ) -> None:
        self._profiling = profiling_data
        self._suggestions = suggestions or []
        self._total_source_lines = max(total_source_lines, 1)

        # Pre-compute frequently used values from profiling data
        line_stats: Dict[str, Any] = self._profiling.get("line_stats", {})

        self._lines_profiled: int = len(line_stats)

        # execution counts per line — keyed by line number (int or str)
        self._execution_counts: List[int] = [
            s.get("count", 0) for s in line_stats.values()
        ]

        self._max_execution_count: int = (
            max(self._execution_counts) if self._execution_counts else 0
        )

        # memory_vars per line
        self._memory_vars: List[int] = [s.get("memory", 0) for s in line_stats.values()]

        self._total_time_ms: float = self._profiling.get("total_time_ms", 0.0)
        self._total_lines_executed: int = self._profiling.get("total_lines", 0)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def calculate(self) -> ScoreReport:
        """
        Run all penalty calculations and return a ``ScoreReport``.

        Returns:
            ScoreReport with score, grade, complexity class, and breakdown.
        """
        severity_p = self._severity_penalty()
        complexity_p = self._complexity_penalty()
        performance_p = self._performance_penalty()
        memory_p = self._memory_penalty()

        total_penalty = severity_p + complexity_p + performance_p + memory_p
        raw_score = 100.0 - total_penalty
        final_score = max(0.0, min(100.0, raw_score))  # clamp to [0, 100]

        complexity_class = self._detect_complexity()
        grade = self._assign_grade(final_score)

        baseline = self._estimate_baseline()

        return ScoreReport(
            score=round(final_score, 2),
            grade=grade,
            complexity_class=complexity_class,
            breakdown={
                "severity_penalty": round(severity_p, 4),
                "complexity_penalty": round(complexity_p, 4),
                "performance_penalty": round(performance_p, 4),
                "memory_penalty": round(memory_p, 4),
            },
            max_execution_count=self._max_execution_count,
            lines_profiled=self._lines_profiled,
            baseline_time_ms=round(baseline, 4),
        )

    # ------------------------------------------------------------------
    # Penalty calculations (private)
    # ------------------------------------------------------------------

    def _severity_penalty(self) -> float:
        """
        Penalty based on optimization suggestions from the Optimizer.

        Formula:
            raw = (high×3 + medium×2 + low×1) / max(10, total_source_lines) × 10
            clamped to MAX_SEVERITY_PENALTY

        Returns 0.0 when no suggestions are provided (pre-Sprint-3).
        """
        if not self._suggestions:
            return 0.0

        high = sum(1 for s in self._suggestions if s.severity == "high")
        medium = sum(1 for s in self._suggestions if s.severity == "medium")
        low = sum(1 for s in self._suggestions if s.severity == "low")

        weighted = (
            high * HIGH_SEVERITY_WEIGHT
            + medium * MEDIUM_SEVERITY_WEIGHT
            + low * LOW_SEVERITY_WEIGHT
        )

        normaliser = max(10, self._total_source_lines)
        raw = (weighted / normaliser) * 10.0

        return min(raw, MAX_SEVERITY_PENALTY)

    def _complexity_penalty(self) -> float:
        """
        Penalty based on detected time complexity class.

        Detects complexity using nesting depth heuristic, then maps it
        to a penalty value. Clamped to MAX_COMPLEXITY_PENALTY.
        """
        complexity_class = self._detect_complexity()
        raw = COMPLEXITY_PENALTIES.get(complexity_class, 0.0)
        return min(raw, MAX_COMPLEXITY_PENALTY)

    def _performance_penalty(self) -> float:
        """
        Penalty based on how much slower the program ran vs the baseline.

        Formula:
            baseline = total_lines_executed × BASELINE_TIME_PER_LINE_MS
            raw = (actual_time / baseline - 1) × 3
            clamped to MAX_PERFORMANCE_PENALTY

        If actual time is at or below baseline, penalty is 0.
        """
        if self._total_lines_executed == 0 or self._total_time_ms == 0.0:
            return 0.0

        baseline = self._estimate_baseline()

        if baseline <= 0:
            return 0.0

        ratio = self._total_time_ms / baseline
        raw = (ratio - 1.0) * 3.0

        return min(max(raw, 0.0), MAX_PERFORMANCE_PENALTY)

    def _memory_penalty(self) -> float:
        """
        Penalty based on lines with an unusually high number of variables
        in scope — a proxy for memory pressure.

        Formula:
            high_var_lines = lines where memory_vars > HIGH_VAR_THRESHOLD
            raw = (high_var_lines / max(lines_profiled, 1)) × MAX_MEMORY_PENALTY
        """
        if not self._memory_vars:
            return 0.0

        high_var_lines = sum(1 for v in self._memory_vars if v > HIGH_VAR_THRESHOLD)

        ratio = high_var_lines / max(self._lines_profiled, 1)
        return ratio * MAX_MEMORY_PENALTY

    # ------------------------------------------------------------------
    # Complexity detection (private)
    # ------------------------------------------------------------------

    def _detect_complexity(self) -> str:
        """
        Heuristic complexity detection from a single execution trace.

        Strategy:
            1. Identify the "hot cluster" — lines that executed at least 50%
               as often as the peak line. The cluster size approximates the
               number of lines inside the innermost nested loop.

            2. Detect an "outer loop signal" by checking whether any non-hot
               line ran at least sqrt(max_count) × 0.5 times. In a nested
               loop the outer header runs ~n times while the inner body runs
               ~n², so sqrt(n²) = n separates the two tiers naturally.

            3. Combine cluster size and outer-loop presence:
               - no outer loop signal  →  flat/linear structure  →  O(n) or below
               - cluster 1–2 + outer   →  O(n²)
               - cluster 3   + outer   →  O(n³)
               - cluster ≥ 4 + outer   →  O(n^k)

            4. Cross-check with absolute max_count for O(2^n) detection.

        Limitation:
            Single-trace analysis cannot determine true asymptotic complexity.
            This is an educational approximation; weights are empirically tuned
            on benchmark programs and documented in the technical report.

        Returns:
            One of: "O(1)", "O(log n)", "O(n)", "O(n log n)",
            "O(n²)", "O(n³)", "O(n^k)", "O(2^n)".
        """
        if self._max_execution_count == 0:
            return "O(1)"

        max_c = self._max_execution_count

        # ── Step 1: hot cluster ───────────────────────────────────────────
        # Lines running at >= 50% of peak form the "inner hot group".
        HOT_RATIO = 0.5
        hot_counts = [c for c in self._execution_counts if c >= max_c * HOT_RATIO]
        non_hot = [c for c in self._execution_counts if c < max_c * HOT_RATIO]
        cluster_size = len(hot_counts)

        # ── Step 2: outer loop signal ─────────────────────────────────────
        # In a 2-level nested loop:  outer runs ~n, inner runs ~n²
        # → sqrt(n²) = n ≈ max_non_hot
        # We use 0.5× as a generous lower bound to handle triangular loops
        # and loop headers that add fractional overhead.
        sqrt_max = math.sqrt(max_c)
        max_non_hot = max(non_hot) if non_hot else 0
        has_outer_loop = max_non_hot > sqrt_max * 0.5

        # ── Step 3: classify ──────────────────────────────────────────────
        # Use max_c as the best proxy for the loop range (n), since for a
        # loop iterating n times, the body executes exactly n times.
        n = max(max_c, self._lines_profiled, MIN_N)

        if not has_outer_loop:
            # No outer loop → flat / linear structure
            return self._classify_linear_or_below(max_c, n)

        # Has outer loop → polynomial nesting detected
        if self._is_exponential(max_c, n):
            return "O(2^n)"

        if cluster_size <= 2:
            return "O(n²)"
        elif cluster_size == 3:
            return "O(n³)"
        else:
            return "O(n^k)"

    def _classify_linear_or_below(self, max_c: int, n: int) -> str:
        """
        Distinguish between O(1), O(log n), O(n), and O(n log n)
        when no outer-loop signal is detected.

        n is passed in as max(max_count, lines_profiled, MIN_N), which uses
        the execution count itself as the best proxy for the loop range —
        a loop iterating k times produces max_count ≈ k.

        Args:
            max_c: Maximum execution count across all lines.
            n:     Loop-range proxy = max(max_count, lines_profiled, MIN_N).

        Returns:
            Complexity string for sub-quadratic complexity classes.
        """
        log_n = math.log2(n)

        # O(1): the hottest line ran only once (or a constant small number)
        if max_c <= 1:
            return "O(1)"

        # O(log n): count grows slowly — within 2× of log2(n)
        # Multiplier 2 accounts for loop header overhead
        if max_c <= log_n * 2:
            return "O(log n)"

        # O(n): count is proportional to n — within 2× of n
        # Multiplier 2 accounts for loop header (n+1 header executions + n body)
        if max_c <= n * 2:
            return "O(n)"

        # O(n log n): count exceeds linear but within n×log(n)×3
        # Multiplier 3 gives buffer for merge-sort-style divide-and-conquer
        if max_c <= n * log_n * 3:
            return "O(n log n)"

        # Count exceeds n log n but nesting_level is 1 — unusual.
        # Could be a large constant factor or a loop with variable range.
        # Conservatively report O(n²) rather than silently under-reporting.
        return "O(n²)"

    def _is_exponential(self, max_c: int, n: int) -> bool:
        """
        Heuristic check for O(2^n) behaviour.

        Two conditions must both be true:
            1. max_count exceeds EXPONENTIAL_COUNT_THRESHOLD (absolute floor).
            2. max_count is at least half of 2^n  (relative check).
               Skipped when n > 60 to avoid integer overflow / meaningless
               comparison (2^60 ≈ 10^18, beyond any realistic execution count).

        Args:
            max_c: Maximum execution count.
            n:     Number of unique profiled lines used as n proxy.

        Returns:
            True if exponential behaviour is suspected.
        """
        if max_c < EXPONENTIAL_COUNT_THRESHOLD:
            return False

        if n > 60:
            # n is too large; 2^n comparison is meaningless
            return False

        expected_2n = 2**n
        return bool(max_c >= expected_2n * 0.5)

    # ------------------------------------------------------------------
    # Helpers (private)
    # ------------------------------------------------------------------

    def _estimate_baseline(self) -> float:
        """
        Estimate the baseline (ideal) execution time in milliseconds.

        Baseline = total_lines_executed × BASELINE_TIME_PER_LINE_MS

        This represents how long a perfectly efficient program of the same
        size should take, assuming ~0.01ms overhead per line execution.
        """
        return self._total_lines_executed * BASELINE_TIME_PER_LINE_MS

    @staticmethod
    def _assign_grade(score: float) -> str:
        """
        Map a numeric score to a human-readable grade label.

        Args:
            score: Final clamped score in [0.0, 100.0].

        Returns:
            Grade string: "Excellent", "Good", "Fair", "Poor", or "Critical".
        """
        for threshold, grade in GRADE_THRESHOLDS:
            if score >= threshold:
                return grade
        return "Critical"


# ---------------------------------------------------------------------------
# Convenience function — public API
# ---------------------------------------------------------------------------


def calculate_score(
    profiling_data: Dict[str, Any],
    suggestions: Optional[List[Any]] = None,
    total_source_lines: int = 1,
) -> ScoreReport:
    """
    Calculate an optimization score from profiling data and suggestions.

    This is the primary public entry point for the scoring system.
    Internally creates a ``Scorer`` instance and calls ``calculate()``.

    Args:
        profiling_data:     Dict from ``ProfilingData.to_dict()``.
                            Must contain keys: ``line_stats``, ``total_time_ms``,
                            ``total_lines``.
        suggestions:        List of ``Suggestion`` objects from the Optimizer.
                            Pass ``None`` or ``[]`` when no optimizer is available.
        total_source_lines: Number of lines in the original source code.
                            Used to normalise severity penalty.

    Returns:
        ``ScoreReport`` with score (0-100), grade, complexity class, and
        per-component penalty breakdown.

    Example::

        from optilang import execute
        from optilang.scoring import calculate_score

        result = execute(
                "for i in range(100):
                    for j in range(100):
                        x = i+j")
        report = calculate_score(result.profiling)

        print(report.score)           # e.g. 73.5
        print(report.grade)           # "Fair"
        print(report.complexity_class)# "O(n²)"
        print(report.to_dict())       # full JSON-serialisable breakdown
    """
    return Scorer(
        profiling_data=profiling_data,
        suggestions=suggestions,
        total_source_lines=total_source_lines,
    ).calculate()
