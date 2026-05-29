# Backward compatibility: re-export profiler from runtime
# This allows: from optilang.profiler import ProfilingData
import sys as _sys
from ..runtime.profiler import (
    Profiler,
    ProfilerConfig,
    ProfilingData,
    FunctionStats,
    LineStats,
    detect_complexity,
    detect_complexity_with_confidence,
    estimate_memory_bytes,
    profile_execution,
    _estimate_deep_object_size,
    _safe_getsizeof,
)

# Re-export sys for backward compatibility (tests patch optilang.profiler.sys)
sys = _sys

__all__ = [
    "Profiler",
    "ProfilerConfig",
    "ProfilingData",
    "FunctionStats",
    "LineStats",
    "detect_complexity",
    "detect_complexity_with_confidence",
    "estimate_memory_bytes",
    "profile_execution",
    "_estimate_deep_object_size",
    "_safe_getsizeof",
    "sys",
]
