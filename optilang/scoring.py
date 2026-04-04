"""
optilang/scoring.py
-------------------
OptiLang Intelligent Scoring System  (v2.0)

WHAT THIS SCORE MEANS
=====================
The score is not a checklist of penalties. It is an earned measure of how
well a program is written *relative to what it is trying to do*.

A score of 85 for a recursive Fibonacci program means something different
from a score of 85 for a nested-loop matrix processor — and that is by
design. The system first understands the program, then judges it on its
own terms.

The score reflects two fundamental qualities a programmer should aim for:

    Efficiency   — does the program use computation and memory well?
    Quality      — is the code clean, intentional, and well-structured?

Both qualities are measured relative to the program's own execution data,
so a loop that ran 5 times is never penalised the same as one that ran
5 million times.

FOUR-STAGE PIPELINE
===================
Stage 1 — ProgramProfiler
    Reads the AST, execution trace, and optimizer suggestions to build
    a ProgramProfile describing *what kind of program this is*.

Stage 2 — DimensionScorer
    Scores each of six sub-dimensions on a 0–100 scale:
        Efficiency group (max 100 each):
            E1  Execution Efficiency    — wasted vs. useful work
            E2  Memory Efficiency       — adaptive pressure threshold
        Quality group (max 100 each):
            Q1  Code Cleanliness        — dead code, function length,
                                          nesting depth, branch density
            Q2  Issue Density           — hotness-weighted suggestions
        Cross-cutting:
            C1  Complexity Handling     — class × scale × confidence
            C2  Structure               — overall code organisation score

Stage 3 — WeightEngine
    Selects context-aware weights for the three top-level dimensions
    (Efficiency, Quality, Complexity) based on the program type.
    Weights sum to 1.0.

Stage 4 — NarrativeGenerator
    Builds a plain-language explanation of the score that is useful to
    both beginners and experienced programmers.

PUBLIC API
==========
    calculate_full_score(source, result)   → ScoreReport   (primary)
    calculate_score(profiling_data, ...)   → ScoreReport   (backwards-compatible)
    Scorer(profiling_data, ...)            → .calculate()  (backwards-compatible)

The ScoreReport carries:
    final_score        — 0–100
    grade              — Excellent / Good / Fair / Poor / Critical
    program_type       — what kind of program was detected
    dimension_scores   — Efficiency / Quality / Complexity each 0–100
    applied_weights    — the weights used for this program type
    narrative          — plain-language explanation (audience-aware)
    breakdown          — every sub-score and intermediate value
    adaptive_context   — all computed thresholds (for UI transparency)
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Tuple

# ---------------------------------------------------------------------------
# Program type
# ---------------------------------------------------------------------------

ProgramType = Literal[
    "trivial",
    "linear_script",
    "recursive_computation",
    "data_iteration",
    "nested_processing",
    "function_heavy",
]

# ---------------------------------------------------------------------------
# Constants — all documented with rationale
# ---------------------------------------------------------------------------

# ── Classification thresholds ────────────────────────────────────────────────
# A loop body must execute at least this many times to be "looping"
LOOP_THRESHOLD: int = 20
# A function with max_recursion_depth >= this is "recursive"
RECURSIVE_DEPTH_MIN: int = 1
# A program with >= this many user functions (relative to lines) is "function_heavy"
FUNCTION_HEAVY_RATIO: float = 0.15  # 15% of source lines are function defs

# ── Complexity class base ratios  [0.0 = best … 1.0 = worst] ────────────────
# These are the MAXIMUM ratios — actual penalty is modulated by scale × confidence
COMPLEXITY_BASE: Dict[str, float] = {
    "O(1)": 0.00,
    "O(log n)": 0.05,
    "O(n)": 0.10,
    "O(n log n)": 0.28,
    "O(n²)": 0.62,
    "O(n³)": 0.82,
    "O(n^k)": 0.93,
    "O(2^n)": 1.00,
}

# ── Scale normalisation ───────────────────────────────────────────────────────
# log₁₀ denominator: a loop running 10^7 times gets full (1.0) scale factor
# A loop running 100 times: log₁₀(100)/7 ≈ 0.29 → only 29% of base penalty
LOG_SCALE_DENOM: float = 7.0

# ── Justification credits ─────────────────────────────────────────────────────
# Recursive programs earn a credit that reduces the complexity penalty
# because high call counts are *expected* in recursive algorithms
RECURSIVE_JUSTIFICATION_CREDIT: float = 0.35

# ── Quality sub-dimension thresholds ─────────────────────────────────────────
# Functions longer than this (in lines) start receiving a length penalty
FUNCTION_LENGTH_THRESHOLD: int = 20
# Maximum nesting depth before penalty starts (0-indexed: 0=top-level)
MAX_ACCEPTABLE_NESTING: int = 2
# Branch density (decision points / source lines) above which penalty starts
MAX_ACCEPTABLE_BRANCH_DENSITY: float = 0.30

# ── Hotness weighting ─────────────────────────────────────────────────────────
# Suggestions on lines that execute frequently should count more
# hotness_factor = log(1 + execution_count) / log(1 + max_count)
# This is applied per suggestion when the optimizer provides line numbers

# ── Minimum data guards ───────────────────────────────────────────────────────
MIN_LINES_FOR_STDEV: int = 3
MIN_LINES_FOR_DEAD_CODE: int = 5

# ── Context-aware dimension weights ──────────────────────────────────────────
# Weights between Efficiency (E), Quality (Q), Complexity (C)
# They shift based on program type because different types have different
# "primary concerns" for what makes code good.
#
# Rationale per type:
#   trivial            → quality is all that matters (nothing to be efficient about)
#   linear_script      → quality dominates; complexity and efficiency are low stakes
#   recursive_computation → complexity handling is the primary skill; quality second
#   data_iteration     → efficiency dominates; this is where runtime waste shows
#   nested_processing  → efficiency + complexity are both primary concerns
#   function_heavy     → quality (decomposition, length) is the main skill signal

DIMENSION_WEIGHTS: Dict[str, Dict[str, float]] = {
    "trivial": {
        "efficiency": 0.15,
        "quality": 0.70,
        "complexity": 0.15,
    },
    "linear_script": {
        "efficiency": 0.20,
        "quality": 0.55,
        "complexity": 0.25,
    },
    "recursive_computation": {
        "efficiency": 0.25,
        "quality": 0.35,
        "complexity": 0.40,
    },
    "data_iteration": {
        "efficiency": 0.45,
        "quality": 0.30,
        "complexity": 0.25,
    },
    "nested_processing": {
        "efficiency": 0.38,
        "quality": 0.25,
        "complexity": 0.37,
    },
    "function_heavy": {
        "efficiency": 0.20,
        "quality": 0.50,
        "complexity": 0.30,
    },
}

# ── Grade thresholds ──────────────────────────────────────────────────────────
GRADE_THRESHOLDS: List[Tuple[float, str]] = [
    (90.0, "Excellent"),
    (75.0, "Good"),
    (60.0, "Fair"),
    (40.0, "Poor"),
    (0.0, "Critical"),
]

# Severity weights for issue density calculation
SEVERITY_WEIGHTS: Dict[str, int] = {"high": 3, "medium": 2, "low": 1}

# Maps optimizer pattern IDs to the scoring dimension they affect.
# Used by the frontend to connect the score panel to the optimization panel.
PATTERN_TO_DIMENSION: Dict[str, str] = {
    "hot_loop": "efficiency",
    "loop_invariant": "efficiency",
    "repeated_computation": "efficiency",
    "string_concat_loop": "efficiency",
    "expensive_calls": "efficiency",
    "nested_loops": "efficiency",
    "unused_vars": "quality",
    "dead_code": "quality",
    "early_return": "quality",
    "constant_folding": "quality",
}


# ---------------------------------------------------------------------------
# Stage 1 — Program Profile
# ---------------------------------------------------------------------------


@dataclass
class ProgramProfile:
    """
    A description of the program built before any scoring begins.

    This is the foundation of context-aware scoring. Every penalty
    and weight decision is made relative to this profile, not against
    a universal standard.
    """

    # Classification
    program_type: ProgramType = "linear_script"

    # Execution shape
    max_execution_count: int = 0
    total_lines_executed: int = 0
    lines_profiled: int = 0
    scale_factor: float = 0.0  # log-magnitude normalisation [0, 1]
    gini_index: float = 0.0  # execution concentration [0, 1]
    complexity_class: str = "O(1)"
    complexity_confidence: float = 1.0

    # Structural shape (from AST when available)
    total_source_lines: int = 1
    function_count: int = 0
    avg_function_length: float = 0.0
    max_nesting_depth: int = 0
    branch_count: int = 0
    branch_density: float = 0.0  # decision points / source lines
    has_dead_code: bool = False
    dead_line_ratio: float = 0.0

    # Memory shape
    memory_vars_mean: float = 0.0
    memory_vars_stdev: float = 0.0
    memory_adaptive_threshold: float = 0.0

    # Issue shape
    total_suggestions: int = 0
    weighted_issue_score: float = 0.0
    hotness_weighted_issue_score: float = 0.0
    issue_density: float = 0.0


class ProgramProfiler:
    """
    Stage 1: Builds a ProgramProfile from all available data sources.

    Accepts:
        profiling_data  — dict from ProfilingData.to_dict()
        suggestions     — list of Suggestion objects (optional)
        total_source_lines — line count of original source
        function_stats  — from profiling_data["function_stats"] (optional)
        ast             — ProgramNode (optional, enables structural analysis)
    """

    def __init__(
        self,
        profiling_data: Dict[str, Any],
        suggestions: Optional[List[Any]] = None,
        total_source_lines: int = 1,
        function_stats: Optional[Dict[str, Any]] = None,
        ast: Optional[Any] = None,
    ) -> None:
        self._profiling = profiling_data
        self._suggestions = suggestions or []
        self._tagged_suggestions: List[Dict[str, Any]] = []
        self._source_lines = max(total_source_lines, 1)
        self._function_stats = function_stats or {}
        self._ast = ast

        # Pre-extract from profiling dict
        line_stats = self._profiling.get("line_stats", {})
        self._line_stats = line_stats
        self._exec_counts: List[int] = [s.get("count", 0) for s in line_stats.values()]
        self._memory_vars: List[int] = [
            s.get("memory_vars", 0) for s in line_stats.values()
        ]
        self._avg_times: List[float] = [
            s.get("avg_time_ms", 0.0) for s in line_stats.values()
        ]
        self._max_count = max(self._exec_counts) if self._exec_counts else 0
        self._total_executed = self._profiling.get("total_lines", 0)
        self._lines_profiled = len(line_stats)

    def build(self) -> ProgramProfile:
        profile = ProgramProfile()

        # Execution shape
        profile.max_execution_count = self._max_count
        profile.total_lines_executed = self._total_executed
        profile.lines_profiled = self._lines_profiled
        profile.scale_factor = self._compute_scale_factor()
        profile.gini_index = self._compute_gini()
        profile.complexity_class = self._detect_complexity()
        profile.complexity_confidence = float(
            self._profiling.get("complexity_confidence", 1.0)
        )

        # Source shape
        profile.total_source_lines = self._source_lines

        # Structural shape — from AST if available
        ast_metrics = self._extract_ast_metrics()
        profile.function_count = ast_metrics["function_count"]
        profile.avg_function_length = ast_metrics["avg_function_length"]
        profile.max_nesting_depth = ast_metrics["max_nesting_depth"]
        profile.branch_count = ast_metrics["branch_count"]
        profile.branch_density = ast_metrics["branch_density"]

        # Dead code
        profile.has_dead_code, profile.dead_line_ratio = self._dead_code()

        # Memory
        m_mean, m_stdev, m_thresh = self._memory_stats()
        profile.memory_vars_mean = m_mean
        profile.memory_vars_stdev = m_stdev
        profile.memory_adaptive_threshold = m_thresh

        # Issues
        i_total, i_weighted, i_hot, i_density = self._issue_stats()
        profile.total_suggestions = i_total
        profile.weighted_issue_score = i_weighted
        profile.hotness_weighted_issue_score = i_hot
        profile.issue_density = i_density

        # Classification — must come last (uses fields set above)
        profile.program_type = self._classify(profile)

        return profile

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _compute_scale_factor(self) -> float:
        """
        Log-magnitude normalisation of execution count.
        log₁₀(max_count) / LOG_SCALE_DENOM → [0, 1]
        A loop running 100 times → ~0.29; 10M times → 1.0
        """
        if self._max_count < 2:
            return 0.0
        return min(math.log10(self._max_count) / LOG_SCALE_DENOM, 1.0)

    def _compute_gini(self) -> float:
        """
        Gini coefficient of execution count distribution.
        0 = perfectly even; 1 = all executions on one line.
        Uses sorted-array formula for efficiency.
        """
        counts = [c for c in self._exec_counts if c >= 0]
        n = len(counts)
        if n < 2:
            return 0.0
        total = sum(counts)
        if total == 0:
            return 0.0
        sorted_c = sorted(counts)
        weighted = sum((i + 1) * x for i, x in enumerate(sorted_c))
        gini = (2 * weighted) / (n * total) - (n + 1) / n
        return max(0.0, min(gini, 1.0))

    def _dead_code(self) -> Tuple[bool, float]:
        if self._source_lines < MIN_LINES_FOR_DEAD_CODE:
            return False, 0.0
        dead = max(0, self._source_lines - self._lines_profiled)
        ratio = min(dead / self._source_lines, 1.0)
        return ratio > 0.0, ratio

    def _memory_stats(self) -> Tuple[float, float, float]:
        non_zero = [v for v in self._memory_vars if v > 0]
        if len(non_zero) < MIN_LINES_FOR_STDEV:
            mean = statistics.mean(self._memory_vars) if self._memory_vars else 0.0
            return mean, 0.0, mean * 1.5 if mean > 0 else 10.0
        mean = statistics.mean(non_zero)
        stdev = statistics.stdev(non_zero)
        return mean, stdev, mean + stdev

    def _issue_stats(self) -> Tuple[int, float, float, float]:
        if not self._suggestions:
            return 0, 0.0, 0.0, 0.0

        # Build a line → max execution count lookup
        line_counts = {
            int(line): s.get("count", 0) for line, s in self._line_stats.items()
        }
        max_c = max(self._exec_counts) if self._exec_counts else 1

        weighted_total = 0.0
        hotness_total = 0.0

        for s in self._suggestions:
            sev = getattr(s, "severity", None) or s.get("severity", "low")
            w = SEVERITY_WEIGHTS.get(sev, 1)
            weighted_total += w

            line_no = getattr(s, "line", None) or s.get("line", 0)
            exec_count = line_counts.get(int(line_no), 0) if line_no else 0
            if max_c > 1:
                hotness = math.log(1 + exec_count) / math.log(1 + max_c)
            else:
                hotness = 0.0
            hotness_total += w * (1 + hotness)

        density = weighted_total / max(self._source_lines, 10)

        # Tag each suggestion with the dimension it affects
        self._tagged_suggestions = [
            {
                "line": getattr(s, "line", None) or s.get("line", 0),
                "pattern": getattr(s, "pattern", None) or s.get("pattern", ""),
                "severity": getattr(s, "severity", None) or s.get("severity", "low"),
                "description": getattr(s, "description", None)
                or s.get("description", ""),
                "suggestion": getattr(s, "suggestion", None) or s.get("suggestion", ""),
                "impact_score": getattr(s, "impact_score", None)
                or s.get("impact_score", 0.0),
                "affects": PATTERN_TO_DIMENSION.get(
                    getattr(s, "pattern", None) or s.get("pattern", ""), "quality"
                ),
                "hotness": (
                    round(
                        math.log(
                            1
                            + line_counts.get(
                                int(getattr(s, "line", None) or s.get("line", 0)), 0
                            )
                        )
                        / math.log(1 + max_c + 1),
                        4,
                    )
                    if max_c > 1
                    else 0.0
                ),
            }
            for s in self._suggestions
        ]

        return len(self._suggestions), weighted_total, hotness_total, density

    def _extract_ast_metrics(self) -> Dict[str, Any]:
        """
        Walk the AST for structural metrics when available.
        Returns neutral defaults when no AST is provided.
        """
        defaults: Dict[str, Any] = {
            "function_count": 0,
            "avg_function_length": 0.0,
            "max_nesting_depth": 0,
            "branch_count": 0,
            "branch_density": 0.0,
        }
        if self._ast is None:
            return defaults

        try:
            import dataclasses as _dc

            from optilang.ast_nodes import (
                ForNode,
                FunctionDefNode,
                IfNode,
                TryNode,
                WhileNode,
            )

            function_lengths: List[int] = []
            max_depth = 0
            branch_count = 0

            def walk(node: Any, depth: int = 0) -> None:
                nonlocal max_depth, branch_count

                if isinstance(node, FunctionDefNode):
                    function_lengths.append(len(node.body))

                if isinstance(node, (IfNode, WhileNode, ForNode, TryNode)):
                    branch_count += 1
                    max_depth = max(max_depth, depth)

                for f in _dc.fields(node):
                    val = getattr(node, f.name)
                    if hasattr(val, "__dataclass_fields__"):
                        walk(
                            val,
                            (
                                depth + 1
                                if isinstance(
                                    node,
                                    (
                                        IfNode,
                                        WhileNode,
                                        ForNode,
                                        TryNode,
                                        FunctionDefNode,
                                    ),
                                )
                                else depth
                            ),
                        )
                    elif isinstance(val, list):
                        for item in val:
                            if hasattr(item, "__dataclass_fields__"):
                                walk(
                                    item,
                                    (
                                        depth + 1
                                        if isinstance(
                                            node,
                                            (
                                                IfNode,
                                                WhileNode,
                                                ForNode,
                                                TryNode,
                                                FunctionDefNode,
                                            ),
                                        )
                                        else depth
                                    ),
                                )
                            elif isinstance(item, tuple):
                                for elem in item:
                                    if hasattr(elem, "__dataclass_fields__"):
                                        walk(elem, depth)

            walk(self._ast)

            avg_len = statistics.mean(function_lengths) if function_lengths else 0.0
            density = branch_count / max(self._source_lines, 1)

            return {
                "function_count": len(function_lengths),
                "avg_function_length": round(avg_len, 2),
                "max_nesting_depth": max_depth,
                "branch_count": branch_count,
                "branch_density": round(density, 4),
            }
        except Exception:
            return defaults

    def _classify(self, p: ProgramProfile) -> ProgramType:
        """
        Classify into one of six program types using a priority chain.

        Priority order (first match wins):
            trivial            → max execution count ≤ 1
            recursive          → any function with recursion depth ≥ threshold
            data_iteration     → high max count + low function count
            nested_processing  → high max count + gini shows heavy hotspot
            function_heavy     → many functions relative to source lines
            linear_script      → everything else (default)
        """
        if p.max_execution_count <= 1:
            return "trivial"

        # Check for recursion in function stats
        for stats in self._function_stats.values():
            depth = (
                stats.get("max_recursion_depth", 0)
                if isinstance(stats, dict)
                else getattr(stats, "max_recursion_depth", 0)
            )
            if depth >= RECURSIVE_DEPTH_MIN:
                return "recursive_computation"

        # Nested processing: high execution + strong hotspot concentration
        if (
            p.max_execution_count > LOOP_THRESHOLD
            and p.gini_index > 0.6
            and p.complexity_class in ("O(n²)", "O(n³)", "O(n^k)")
        ):
            return "nested_processing"

        # Data iteration: significant looping but not heavily nested
        if p.max_execution_count > LOOP_THRESHOLD:
            return "data_iteration"

        # Function heavy: many functions relative to code size
        fn_ratio = p.function_count / max(p.total_source_lines, 1)
        if fn_ratio >= FUNCTION_HEAVY_RATIO and p.function_count >= 2:
            return "function_heavy"

        return "linear_script"

    # ── Complexity detection (proven heuristic, unchanged) ────────────────────

    def _detect_complexity(self) -> str:
        if self._max_count == 0:
            return "O(1)"
        max_c = self._max_count
        hot = [c for c in self._exec_counts if c >= max_c * 0.5]
        n_hot = [c for c in self._exec_counts if c < max_c * 0.5]
        cluster = len(hot)
        sqrt_max = math.sqrt(max_c)
        max_nh = max(n_hot) if n_hot else 0
        has_outer = max_nh > sqrt_max * 0.5
        n = max(max_c, self._lines_profiled, 4)
        if not has_outer:
            return self._linear_or_below(max_c, n)
        if self._is_exponential(max_c, n):
            return "O(2^n)"
        if cluster <= 2:
            return "O(n²)"
        if cluster == 3:
            return "O(n³)"
        return "O(n^k)"

    def _linear_or_below(self, max_c: int, n: int) -> str:
        lg = math.log2(n)
        if max_c <= 1:
            return "O(1)"
        if max_c <= lg * 2:
            return "O(log n)"
        if max_c <= n * 2:
            return "O(n)"
        if max_c <= n * lg * 3:
            return "O(n log n)"

        return "O(n²)"

    def _is_exponential(self, max_c: int, n: int) -> bool:
        return max_c >= 1_000_000 and n <= 60 and max_c >= (2**n) * 0.5


# ---------------------------------------------------------------------------
# Stage 2 — Dimension Scorer
# ---------------------------------------------------------------------------


class DimensionScorer:
    """
    Stage 2: Scores each sub-dimension on a 0–100 scale.

    Each method returns (score_0_to_100, detail_dict).
    Score of 100 = perfect; 0 = worst possible for that dimension.
    """

    def __init__(self, profile: ProgramProfile) -> None:
        self._p = profile

    # ── E1: Execution Efficiency ──────────────────────────────────────────────

    def execution_efficiency(self) -> Tuple[float, Dict[str, Any]]:
        """
        How well does the program use its execution cycles?

        Key signal: the Gini index of execution distribution.
        A Gini near 0 means all lines run equally (efficient).
        A Gini near 1 means one line dominates (potential hotspot).

        Adjusted by:
        - scale_factor: a tiny loop is not a concern even if it dominates
        - complexity class: some concentration is expected in O(n) loops
        - hotness-weighted issue score: issues on hot lines penalise harder

        Formula:
            base_waste   = gini × scale_factor
            issue_factor = tanh(hotness_weighted_issues / max(lines, 10) × 2)
            combined     = (base_waste × 0.6) + (issue_factor × 0.4)
            score        = (1 − combined) × 100
        """
        p = self._p
        base_waste = p.gini_index * p.scale_factor

        # Issue contribution — problems on hot lines matter more
        max_issues = max(p.total_source_lines, 10)
        issue_factor = math.tanh((p.hotness_weighted_issue_score / max_issues) * 2.0)

        combined = (base_waste * 0.6) + (issue_factor * 0.4)
        combined = max(0.0, min(combined, 1.0))
        score = (1.0 - combined) * 100.0

        detail = {
            "gini_index": round(p.gini_index, 4),
            "scale_factor": round(p.scale_factor, 4),
            "base_waste": round(base_waste, 4),
            "hotness_weighted_issues": round(p.hotness_weighted_issue_score, 4),
            "issue_factor": round(issue_factor, 4),
        }
        return round(score, 2), detail

    # ── E2: Memory Efficiency ─────────────────────────────────────────────────

    def memory_efficiency(self) -> Tuple[float, Dict[str, Any]]:
        """
        How well does the program manage its variable scope?

        Uses the program's own mean+σ as the 'high' threshold rather than
        a fixed constant, so every program judges itself against its own
        baseline. A line above the adaptive threshold is "high pressure".

        Formula:
            high_pressure_lines = lines where memory_vars > adaptive_threshold
            pressure_ratio      = high_pressure_lines / lines_profiled
            score               = (1 − pressure_ratio) × 100
        """
        p = self._p
        mem_vars = p.memory_vars_mean  # we only have aggregate here # noqa: F841

        # If no meaningful memory data, return neutral
        if p.lines_profiled == 0 or p.memory_vars_stdev == 0.0:
            return 100.0, {
                "memory_adaptive_threshold": round(p.memory_adaptive_threshold, 2),
                "memory_mean_vars": round(p.memory_vars_mean, 2),
                "memory_stdev_vars": round(p.memory_vars_stdev, 2),
                "pressure_ratio": 0.0,
                "note": "Insufficient data for adaptive threshold",
            }

        # Estimate pressure ratio from mean and stdev:
        # If mean is well below threshold, pressure is low.
        # We approximate using how far mean is below threshold.
        threshold = p.memory_adaptive_threshold
        if threshold <= 0:
            pressure_ratio = 0.0
        else:
            # How much does the mean exceed the threshold?
            # By definition threshold = mean + stdev, so mean is always below.
            # We use stdev/mean as a proxy for spread — high spread means
            # some lines are significantly above the threshold.
            spread = p.memory_vars_stdev / max(p.memory_vars_mean, 1.0)
            # Normalize: spread > 1 means significant high-memory outliers
            pressure_ratio = min(spread / 3.0, 1.0)

        score = (1.0 - pressure_ratio) * 100.0

        detail = {
            "memory_adaptive_threshold": round(threshold, 2),
            "memory_mean_vars": round(p.memory_vars_mean, 2),
            "memory_stdev_vars": round(p.memory_vars_stdev, 2),
            "pressure_ratio": round(pressure_ratio, 4),
        }
        return round(score, 2), detail

    # ── Q1: Code Cleanliness ──────────────────────────────────────────────────

    def code_cleanliness(self) -> Tuple[float, Dict[str, Any]]:
        """
        How clean and well-structured is the code?

        Four sub-signals, averaged:

        dead_code_score:
            dead_ratio = (source_lines - profiled_lines) / source_lines
            score = (1 - dead_ratio) × 100

        function_length_score:
            Sigmoid penalty for average function length above threshold.
            score = 100 × sigmoid(-k × (avg_len - threshold))
            where k controls how quickly the penalty grows.

        nesting_score:
            Linear penalty above MAX_ACCEPTABLE_NESTING depth.
            score = max(0, 1 - excess / MAX_ACCEPTABLE_NESTING) × 100

        branching_score:
            Penalty when branch density exceeds MAX_ACCEPTABLE_BRANCH_DENSITY.
            score = max(0, 1 - excess_density / MAX_ACCEPTABLE_BRANCH_DENSITY) × 100
        """
        p = self._p

        # Dead code
        dead_score = (1.0 - p.dead_line_ratio) * 100.0

        # Function length
        if p.avg_function_length == 0.0:
            fn_score = 100.0  # no functions or no AST → no penalty
        else:
            excess = max(0.0, p.avg_function_length - FUNCTION_LENGTH_THRESHOLD)
            # Sigmoid: at threshold+10 lines → score ~50; at threshold+30 → ~20
            fn_score = 100.0 / (1.0 + math.exp(0.15 * excess - 1.5))

        # Nesting depth
        if p.max_nesting_depth == 0:
            nesting_score = 100.0
        else:
            excess_depth = max(0, p.max_nesting_depth - MAX_ACCEPTABLE_NESTING)
            nesting_score = max(
                0.0, (1.0 - excess_depth / max(MAX_ACCEPTABLE_NESTING, 1)) * 100.0
            )

        # Branch density
        if p.branch_density == 0.0:
            branch_score = 100.0
        else:
            excess_density = max(0.0, p.branch_density - MAX_ACCEPTABLE_BRANCH_DENSITY)
            branch_score = max(
                0.0, (1.0 - excess_density / MAX_ACCEPTABLE_BRANCH_DENSITY) * 100.0
            )

        # Weighted average — dead code and nesting matter most
        cleanliness = (
            dead_score * 0.35
            + fn_score * 0.25
            + nesting_score * 0.25
            + branch_score * 0.15
        )

        detail = {
            "dead_line_ratio": round(p.dead_line_ratio, 4),
            "dead_code_score": round(dead_score, 2),
            "avg_function_length": round(p.avg_function_length, 2),
            "function_length_score": round(fn_score, 2),
            "max_nesting_depth": p.max_nesting_depth,
            "nesting_score": round(nesting_score, 2),
            "branch_density": round(p.branch_density, 4),
            "branch_score": round(branch_score, 2),
        }
        return round(cleanliness, 2), detail

    # ── Q2: Issue Density ─────────────────────────────────────────────────────

    def issue_density(self) -> Tuple[float, Dict[str, Any]]:
        """
        How many optimization problems were found, weighted by severity
        and normalized by program size?

        Uses tanh to keep the score bounded even for heavily-flagged programs.
        Issues on hot lines (hotness_weighted) count more than cold ones.

        Formula:
            density_score = tanh(issue_density × 3)
            score         = (1 − density_score) × 100
        """
        p = self._p
        if p.total_suggestions == 0:
            return 100.0, {
                "suggestion_count": 0,
                "weighted_issue_score": 0.0,
                "issue_density": 0.0,
            }

        density_score = math.tanh(p.issue_density * 3.0)
        score = (1.0 - density_score) * 100.0

        detail = {
            "suggestion_count": p.total_suggestions,
            "weighted_issue_score": round(p.weighted_issue_score, 4),
            "hotness_weighted_score": round(p.hotness_weighted_issue_score, 4),
            "issue_density": round(p.issue_density, 4),
            "density_score": round(density_score, 4),
        }
        return round(score, 2), detail

    # ── C1: Complexity Handling ───────────────────────────────────────────────

    def complexity_handling(self) -> Tuple[float, Dict[str, Any]]:
        """
        How appropriate is the program's complexity for its goal?

        Formula:
            base_ratio    = COMPLEXITY_BASE[class]
            scaled_ratio  = base_ratio × scale_factor × confidence
            credit        = RECURSIVE_JUSTIFICATION_CREDIT if recursive
            final_ratio   = scaled_ratio × (1 − credit)
            score         = (1 − final_ratio) × 100

        Rationale:
        - base_ratio: O(n²) is inherently worse than O(n)
        - scale_factor: a small nested loop barely matters at runtime
        - confidence: if the profiler is 60% sure, the penalty is 60%
        - credit: recursion justifies higher complexity — it's intentional
        """
        p = self._p
        base = COMPLEXITY_BASE.get(p.complexity_class, 0.0)
        scaled = base * p.scale_factor * p.complexity_confidence

        credit = 0.0
        if p.program_type == "recursive_computation":
            credit = RECURSIVE_JUSTIFICATION_CREDIT

        final_ratio = scaled * (1.0 - credit)
        final_ratio = max(0.0, min(final_ratio, 1.0))
        score = (1.0 - final_ratio) * 100.0

        detail = {
            "complexity_class": p.complexity_class,
            "base_ratio": round(base, 4),
            "scale_factor": round(p.scale_factor, 4),
            "confidence": round(p.complexity_confidence, 4),
            "justification_credit": round(credit, 4),
            "final_complexity_ratio": round(final_ratio, 4),
        }
        return round(score, 2), detail


# ---------------------------------------------------------------------------
# Stage 3 — Weight Engine
# ---------------------------------------------------------------------------


class WeightEngine:
    """
    Stage 3: Selects dimension weights based on program type.

    Returns a dict with keys "efficiency", "quality", "complexity"
    summing to 1.0.
    """

    @staticmethod
    def weights_for(program_type: ProgramType) -> Dict[str, float]:
        return DIMENSION_WEIGHTS.get(
            program_type, DIMENSION_WEIGHTS["linear_script"]  # safe default
        )


# ---------------------------------------------------------------------------
# Stage 4 — Narrative Generator
# ---------------------------------------------------------------------------


class NarrativeGenerator:
    """
    Stage 4: Builds a plain-language explanation of the score.

    The narrative is designed for two audiences simultaneously:
        - Beginners / learners / educators: understand what the score means
          and know what to improve first
        - Experienced programmers: get specific, actionable information
          without unnecessary padding

    Each narrative is built from three parts:
        1. What the program is (one sentence)
        2. What the score reflects (one sentence per weak dimension)
        3. The single most important thing to fix (one sentence)
    """

    def __init__(
        self,
        profile: ProgramProfile,
        efficiency_score: float,
        quality_score: float,
        complexity_score: float,
        final_score: float,
        grade: str,
    ) -> None:
        self._p = profile
        self._eff = efficiency_score
        self._qlt = quality_score
        self._cmp = complexity_score
        self._score = final_score
        self._grade = grade

    def generate(self) -> str:
        parts: List[str] = []

        # Part 1 — What the program is
        parts.append(self._program_description())

        # Part 2 — What each dimension reflects (only mention weak ones)
        dim_feedback = self._dimension_feedback()
        if dim_feedback:
            parts.append(dim_feedback)

        # Part 3 — Overall verdict + top recommendation
        parts.append(self._verdict())

        return " ".join(parts)

    def _program_description(self) -> str:
        p = self._p
        type_descriptions = {
            "trivial": "This is a simple, single-pass program.",
            "linear_script": "This is a linear script with straightforward"
            " execution flow.",
            "recursive_computation": "This program uses recursion as its primary"
            " computational approach.",
            "data_iteration": "This program processes data using loops and"
            " iteration.",
            "nested_processing": "This program uses nested loops to process"
            " data, which creates higher computational"
            " demands.",
            "function_heavy": "This program is structured around multiple"
            " function definitions.",
        }
        base = type_descriptions.get(p.program_type, "This program was analysed.")

        # Add complexity if it's notable
        if p.complexity_class not in ("O(1)", "O(log n)", "O(n)"):
            base += (
                f" Its time complexity is estimated at {p.complexity_class}"
                f" (confidence: {int(p.complexity_confidence * 100)}%)."
            )

        return base

    def _dimension_feedback(self) -> str:
        feedbacks: List[str] = []
        p = self._p

        # Efficiency
        if self._eff < 70:
            if p.gini_index > 0.7:
                feedbacks.append(
                    f"Execution is highly concentrated — "
                    f"{int(p.gini_index * 100)}% of work is done by a small"
                    f" number of lines, suggesting a potential hotspot."
                )
            elif p.hotness_weighted_issue_score > 5:
                feedbacks.append(
                    "Several optimization issues were found on frequently"
                    " executed lines, which amplifies their performance impact."
                )

        # Quality
        if self._qlt < 70:
            if p.dead_line_ratio > 0.15:
                feedbacks.append(
                    f"About {int(p.dead_line_ratio * 100)}% of the source lines"
                    f" were never executed — this may indicate dead branches or"
                    f" unused code paths."
                )
            if p.avg_function_length > FUNCTION_LENGTH_THRESHOLD:
                feedbacks.append(
                    f"Functions average {p.avg_function_length:.0f} lines,"
                    f" which is above the recommended {FUNCTION_LENGTH_THRESHOLD}."
                    f" Shorter functions are easier to test and optimise."
                )
            if p.max_nesting_depth > MAX_ACCEPTABLE_NESTING:
                feedbacks.append(
                    f"The code reaches a nesting depth of {p.max_nesting_depth}."
                    f" Deep nesting makes logic harder to follow and"
                    f" often hides optimisation opportunities."
                )
            if p.total_suggestions > 0 and self._qlt < 60:
                feedbacks.append(
                    f"{p.total_suggestions} optimization issue"
                    f"{'s were' if p.total_suggestions > 1 else ' was'}"
                    f" detected across the program."
                )

        # Complexity
        if self._cmp < 70:
            if p.scale_factor > 0.5:
                feedbacks.append(
                    f"The {p.complexity_class} complexity is significant at this"
                    f" execution scale — the hottest line ran"
                    f" {p.max_execution_count:,} times."
                )

        return " ".join(feedbacks)

    def _verdict(self) -> str:
        p = self._p  # noqa: F841
        score = self._score

        # Find the weakest dimension to prioritize
        dims = {
            "efficiency": self._eff,
            "quality": self._qlt,
            "complexity": self._cmp,
        }
        weakest = min(dims, key=dims.get)  # type: ignore[arg-type]

        if score >= 90:
            return (
                "Overall, this is well-written, efficient code."
                " Keep up the good work."
            )

        if score >= 75:
            recommendations = {
                "efficiency": "Focus on reducing work in the most-executed lines.",
                "quality": "Refactor for cleaner structure and remove any"
                " unused code.",
                "complexity": "Consider whether the algorithm can be simplified"
                " for the input sizes you typically use.",
            }
            return f"The code is generally good." f" {recommendations.get(weakest, '')}"

        if score >= 60:
            recommendations = {
                "efficiency": "The primary concern is execution efficiency —"
                " review the hottest lines for unnecessary work.",
                "quality": "The primary concern is code quality — dead code,"
                " long functions, or deep nesting are dragging"
                " the score down.",
                "complexity": "The primary concern is computational complexity —"
                " consider a more efficient algorithm.",
            }
            return recommendations.get(weakest, "There is room for improvement.")

        if score >= 40:
            return (
                "There are significant issues across multiple dimensions."
                " Start by addressing the highest-severity suggestions,"
                " then revisit the overall structure of the program."
            )

        return (
            "This program has fundamental efficiency or quality problems."
            " Consider reviewing the algorithm design, removing dead code,"
            " and addressing all high-severity suggestions before"
            " optimising further."
        )


# ---------------------------------------------------------------------------
# DynamicScorer — orchestrates all four stages
# ---------------------------------------------------------------------------


class DynamicScorer:
    """
    Orchestrates the full four-stage scoring pipeline.

    Usage (primary path)::

        from optilang.scoring import calculate_full_score
        report = calculate_full_score(source, result)

    Usage (direct)::

        scorer = DynamicScorer(
            profiling_data=result.profiling.to_dict(),
            suggestions=optimizer_report.suggestions,
            total_source_lines=len(source.splitlines()),
            function_stats=result.profiling.to_dict()["function_stats"],
            ast=parsed_ast,
        )
        report = scorer.calculate()
    """

    def __init__(
        self,
        profiling_data: Dict[str, Any],
        suggestions: Optional[List[Any]] = None,
        total_source_lines: int = 1,
        function_stats: Optional[Dict[str, Any]] = None,
        ast: Optional[Any] = None,
    ) -> None:
        self._profiling = profiling_data
        self._suggestions = suggestions or []
        self._source_lines = max(total_source_lines, 1)
        self._function_stats = function_stats or {}
        self._ast = ast

    def calculate(self) -> "ScoreReport":
        # Stage 1 — Profile
        profiler = ProgramProfiler(
            profiling_data=self._profiling,
            suggestions=self._suggestions,
            total_source_lines=self._source_lines,
            function_stats=self._function_stats,
            ast=self._ast,
        )
        profile = profiler.build()

        # Stage 2 — Dimension scores
        dim = DimensionScorer(profile)

        eff_exec_score, eff_exec_detail = dim.execution_efficiency()
        eff_mem_score, eff_mem_detail = dim.memory_efficiency()
        qlt_clean_score, qlt_clean_detail = dim.code_cleanliness()
        qlt_issue_score, qlt_issue_detail = dim.issue_density()
        cmp_score, cmp_detail = dim.complexity_handling()

        # Aggregate top-level dimension scores
        efficiency_score = eff_exec_score * 0.65 + eff_mem_score * 0.35
        quality_score = qlt_clean_score * 0.60 + qlt_issue_score * 0.40
        complexity_score = cmp_score

        # Stage 3 — Weights
        weights = WeightEngine.weights_for(profile.program_type)
        w_e = weights["efficiency"]
        w_q = weights["quality"]
        w_c = weights["complexity"]

        final_score = (
            efficiency_score * w_e + quality_score * w_q + complexity_score * w_c
        )
        final_score = max(0.0, min(100.0, round(final_score, 2)))
        grade = _assign_grade(final_score)

        # Stage 4 — Narrative
        narrative = NarrativeGenerator(
            profile=profile,
            efficiency_score=efficiency_score,
            quality_score=quality_score,
            complexity_score=complexity_score,
            final_score=final_score,
            grade=grade,
        ).generate()

        # Collect tagged suggestions from the profiler
        tagged = profiler._tagged_suggestions

        # Assemble full breakdown
        breakdown: Dict[str, Any] = {
            "efficiency_score": round(efficiency_score, 2),
            "quality_score": round(quality_score, 2),
            "complexity_score": round(complexity_score, 2),
            "execution_efficiency_score": round(eff_exec_score, 2),
            "memory_efficiency_score": round(eff_mem_score, 2),
            "code_cleanliness_score": round(qlt_clean_score, 2),
            "issue_density_score": round(qlt_issue_score, 2),
            "execution_efficiency_detail": eff_exec_detail,
            "memory_efficiency_detail": eff_mem_detail,
            "code_cleanliness_detail": qlt_clean_detail,
            "issue_density_detail": qlt_issue_detail,
            "complexity_detail": cmp_detail,
            # Per-dimension explanation: why each dimension scored as it did
            "dimension_detail": {
                "efficiency": {
                    "score": round(efficiency_score, 2),
                    "loss": round(100.0 - efficiency_score, 2),
                    "suggestions": [s for s in tagged if s["affects"] == "efficiency"],
                },
                "quality": {
                    "score": round(quality_score, 2),
                    "loss": round(100.0 - quality_score, 2),
                    "suggestions": [s for s in tagged if s["affects"] == "quality"],
                },
                "complexity": {
                    "score": round(complexity_score, 2),
                    "loss": round(100.0 - complexity_score, 2),
                    "suggestions": [],  # complexity is driven by execution data
                },
            },
            # All suggestions tagged with dimension and hotness, sorted by impact
            "tagged_suggestions": sorted(
                tagged,
                key=lambda s: s["impact_score"],
                reverse=True,
            ),
        }

        adaptive_context: Dict[str, Any] = {
            "program_type": profile.program_type,
            "scale_factor": round(profile.scale_factor, 4),
            "gini_index": round(profile.gini_index, 4),
            "complexity_class": profile.complexity_class,
            "complexity_confidence": round(profile.complexity_confidence, 4),
            "max_execution_count": profile.max_execution_count,
            "lines_profiled": profile.lines_profiled,
            "dead_line_ratio": round(profile.dead_line_ratio, 4),
            "memory_adaptive_threshold": round(profile.memory_adaptive_threshold, 2),
            "memory_mean_vars": round(profile.memory_vars_mean, 2),
            "memory_stdev_vars": round(profile.memory_vars_stdev, 2),
            "total_suggestions": profile.total_suggestions,
            "issue_density": round(profile.issue_density, 4),
            "applied_weights": {
                "efficiency": w_e,
                "quality": w_q,
                "complexity": w_c,
            },
        }

        return ScoreReport(
            final_score=final_score,
            grade=grade,
            program_type=profile.program_type,
            complexity_class=profile.complexity_class,
            dimension_scores={
                "efficiency": round(efficiency_score, 2),
                "quality": round(quality_score, 2),
                "complexity": round(complexity_score, 2),
            },
            applied_weights={
                "efficiency": w_e,
                "quality": w_q,
                "complexity": w_c,
            },
            narrative=narrative,
            breakdown=breakdown,
            adaptive_context=adaptive_context,
        )


# ---------------------------------------------------------------------------
# ScoreReport — output data class
# ---------------------------------------------------------------------------


@dataclass
class ScoreReport:
    """
    Complete scoring report.

    The score reflects how well the program is written *relative to what
    it is trying to do* — not a checklist of universal penalties.

    Fields
    ------
    final_score      : Overall score 0–100. Higher = better.
    grade            : Excellent / Good / Fair / Poor / Critical
    program_type     : What kind of program was detected
    complexity_class : Estimated time complexity
    dimension_scores : Efficiency / Quality / Complexity each 0–100
    applied_weights  : Weights used for this program type (sum to 1.0)
    narrative        : Plain-language explanation for any audience
    breakdown        : All sub-scores and intermediate values
    adaptive_context : All computed thresholds (for UI transparency)
    """

    final_score: float
    grade: str
    program_type: str
    complexity_class: str
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    applied_weights: Dict[str, float] = field(default_factory=dict)
    narrative: str = ""
    breakdown: Dict[str, Any] = field(default_factory=dict)
    adaptive_context: Dict[str, Any] = field(default_factory=dict)

    # ── Backwards-compatible aliases ─────────────────────────────────────────
    @property
    def score(self) -> float:
        """Alias for final_score — backwards compatible with old ScoreReport."""
        return self.final_score

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a plain, JSON-safe dictionary."""
        return {
            "final_score": round(self.final_score, 2),
            "score": round(self.final_score, 2),  # alias
            "grade": self.grade,
            "program_type": self.program_type,
            "complexity_class": self.complexity_class,
            "dimension_scores": {
                k: round(v, 2) for k, v in self.dimension_scores.items()
            },
            "applied_weights": self.applied_weights,
            "narrative": self.narrative,
            "breakdown": _round_dict(self.breakdown),
            "adaptive_context": _round_dict(self.adaptive_context),
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assign_grade(score: float) -> str:
    for threshold, grade in GRADE_THRESHOLDS:
        if score >= threshold:
            return grade
    return "Critical"


def _round_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively round floats in a dict for JSON serialisation."""
    result: Dict[str, Any] = {}
    for k, v in d.items():
        if isinstance(v, float):
            result[k] = round(v, 4)
        elif isinstance(v, dict):
            result[k] = _round_dict(v)
        else:
            result[k] = v
    return result


# ---------------------------------------------------------------------------
# Public API — primary entry point
# ---------------------------------------------------------------------------


def calculate_full_score(
    source: str,
    result: Any,
    suggestions: Optional[List[Any]] = None,
    ast: Optional[Any] = None,
) -> ScoreReport:
    """
    Calculate a full, context-aware score from source code and execution result.

    This is the primary public entry point. It handles everything internally:
    profiling data extraction, AST structural analysis, and all four scoring stages.

    Parameters
    ----------
    source      : Original PyLite source code string
    result      : ExecutionResult from execute(source)
    suggestions : List of Suggestion objects from the Optimizer.
                  When None, the scorer runs without issue-density data.
    ast         : Parsed ProgramNode (optional). When provided, enables full
                  structural analysis (function length, nesting, branching).
                  When None, structural sub-scores return neutral values.

    Returns
    -------
    ScoreReport with final_score (0–100), grade, program_type,
    dimension_scores, applied_weights, narrative, and full breakdown.

    Example
    -------
    ::
        from optilang import execute
        from optilang.scoring import calculate_full_score

        source = \"\"\"
        def factorial(n):
            if n <= 1:
                return 1
            return n * factorial(n - 1)
        print(factorial(10))
        \"\"\"
        result = execute(source)
        report = calculate_full_score(source, result)

        print(report.final_score)      # e.g. 82.4
        print(report.grade)            # "Good"
        print(report.program_type)     # "recursive_computation"
        print(report.dimension_scores) # {
                                            "efficiency": 88,
                                            "quality": 79,
                                            "complexity": 80
                                        }
        print(report.applied_weights)  # {
                                            "efficiency": 0.25,
                                            "quality": 0.35,
                                            "complexity": 0.40
                                        }
        print(report.narrative)        # plain-language explanation
        print(report.to_dict())        # full JSON-serialisable output
    """
    profiling_data = (
        result.profiling.to_dict()
        if result.profiling is not None
        else {
            "line_stats": {},
            "function_stats": {},
            "total_time_ms": 0.0,
            "total_lines": 0,
            "lines_profiled": 0,
        }
    )

    # Count non-blank, non-comment source lines
    source_lines = sum(
        1
        for line in source.splitlines()
        if line.strip() and not line.strip().startswith("#")
    )

    function_stats = profiling_data.get("function_stats", {})

    return DynamicScorer(
        profiling_data=profiling_data,
        suggestions=suggestions or [],
        total_source_lines=max(source_lines, 1),
        function_stats=function_stats,
        ast=ast,
    ).calculate()


# ---------------------------------------------------------------------------
# Backwards-compatible API
# ---------------------------------------------------------------------------


def calculate_score(
    profiling_data: Dict[str, Any],
    suggestions: Optional[List[Any]] = None,
    total_source_lines: int = 1,
    function_stats: Optional[Dict[str, Any]] = None,
    ast: Optional[Any] = None,
) -> ScoreReport:
    """
    Backwards-compatible scoring entry point.

    Accepts the same arguments as the original calculate_score() so that
    all existing code continues to work without modification.

    For new code, prefer calculate_full_score(source, result) which handles
    all setup automatically.
    """
    return DynamicScorer(
        profiling_data=profiling_data,
        suggestions=suggestions,
        total_source_lines=total_source_lines,
        function_stats=function_stats,
        ast=ast,
    ).calculate()


class Scorer(DynamicScorer):
    """
    Backwards-compatible Scorer class.

    Preserves the original Scorer(profiling_data, suggestions,
    total_source_lines).calculate() interface so that existing tests
    and calling code need no changes.
    """

    def __init__(
        self,
        profiling_data: Dict[str, Any],
        suggestions: Optional[List[Any]] = None,
        total_source_lines: int = 1,
        function_stats: Optional[Dict[str, Any]] = None,
        ast: Optional[Any] = None,
    ) -> None:
        super().__init__(
            profiling_data=profiling_data,
            suggestions=suggestions,
            total_source_lines=total_source_lines,
            function_stats=function_stats,
            ast=ast,
        )
