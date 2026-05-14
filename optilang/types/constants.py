"""
Shared constants for OptiLang complexity detection and scoring.

All complexity class strings are defined here once to avoid inconsistencies
between profiler.py and scoring.py.
"""

# ── Complexity Class Labels ──────────────────────────────────────────────────

COMPLEXITY_O1 = "O(1)"
COMPLEXITY_LOGN = "O(log n)"
COMPLEXITY_N = "O(n)"
COMPLEXITY_NLOGN = "O(n log n)"
COMPLEXITY_N2 = "O(n²)"
COMPLEXITY_N2LOGN = "O(n² log n)"
COMPLEXITY_N3 = "O(n³)"
COMPLEXITY_N4 = "O(n⁴)"
COMPLEXITY_NK = "O(n^k)"
COMPLEXITY_EXP = "O(2^n)"
COMPLEXITY_N3_PLUS = "O(n^3) or worse"


# ── Scoring Points (out of 15 per complexity class) ─────────────────────────

COMPLEXITY_POINTS: dict[str, float] = {
    COMPLEXITY_O1: 15.0,
    COMPLEXITY_LOGN: 15.0,
    COMPLEXITY_N: 13.0,
    COMPLEXITY_NLOGN: 10.0,
    COMPLEXITY_N2: 6.0,
    COMPLEXITY_N2LOGN: 4.0,
    COMPLEXITY_N3: 3.0,
    COMPLEXITY_N4: 2.0,
    COMPLEXITY_NK: 1.0,
    COMPLEXITY_EXP: 0.0,
    # Note: COMPLEXITY_N3_PLUS is defined above but not included here
    # because it's only used as a fallback label, not a scored class
}
