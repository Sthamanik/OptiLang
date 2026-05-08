"""
Tests for the OptiLang Profiler (Sprint 2 - Enhanced Profiling)

Covers:
- Basic line tracking (execution count, timing)
- Min/Max timing for lines
- Memory estimation (bytes)
- Function call tracking (count, timing, recursion depth, callers)
- Complexity detection (O(1), O(log n), O(n), O(n²), etc.)
- High-level summary output
- Hot lines / hot functions queries
- Profiler reset and enable/disable
- Integration with execute()
"""

import io
import runpy
import subprocess
import sys
import time
from contextlib import redirect_stdout
from pathlib import Path

import pytest

from optilang import execute
from optilang.profiler import (
    FunctionStats,
    LineStats,
    Profiler,
    ProfilerConfig,
    ProfilingData,
    _estimate_deep_object_size,
    _safe_getsizeof,
    detect_complexity,
    detect_complexity_with_confidence,
    estimate_memory_bytes,
    profile_execution,
)

# ─────────────────────────────────────────────────────────────────────────────
#  Unit Tests: estimate_memory_bytes
# ─────────────────────────────────────────────────────────────────────────────


class TestEstimateMemoryBytes:
    """Tests for the standalone memory estimation helper."""

    def test_empty_env_returns_zero(self) -> None:
        assert estimate_memory_bytes({}) == 0

    def test_integer_variable_has_positive_size(self) -> None:
        result = estimate_memory_bytes({"x": 42})
        assert result > 0

    def test_string_variable_larger_than_integer(self) -> None:
        int_size = estimate_memory_bytes({"x": 1})
        str_size = estimate_memory_bytes({"x": "hello world"})
        assert str_size > int_size

    def test_list_variable_accounts_for_elements(self) -> None:
        small = estimate_memory_bytes({"x": [1, 2]})
        large = estimate_memory_bytes({"x": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]})
        assert large > small

    def test_dict_variable_accounts_for_pairs(self) -> None:
        small = estimate_memory_bytes({"x": {"a": 1}})
        large = estimate_memory_bytes({"x": {"a": 1, "b": 2, "c": 3, "d": 4}})
        assert large > small

    def test_multiple_variables_accumulate(self) -> None:
        one = estimate_memory_bytes({"x": 1})
        two = estimate_memory_bytes({"x": 1, "y": 2})
        assert two > one

    def test_does_not_raise_on_unusual_types(self) -> None:
        # Should not crash on None, bool, or nested lists
        result = estimate_memory_bytes({"a": None, "b": True, "c": [[1, 2], [3, 4]]})
        assert result >= 0

    def test_memory_mode_off_returns_zero(self) -> None:
        result = estimate_memory_bytes({"x": [1, 2, 3]}, mode="off")
        assert result == 0

    def test_deep_memory_mode_accounts_for_nested_objects(self) -> None:
        nested = {"x": [{"a": [1, 2, 3, 4]}]}
        shallow = estimate_memory_bytes(nested, mode="shallow")
        deep = estimate_memory_bytes(nested, mode="deep")
        assert deep >= shallow

    def test_deep_memory_mode_respects_depth_limit(self) -> None:
        nested = {"x": [{"a": [1, 2, 3, 4]}]}
        deep_low = estimate_memory_bytes(
            nested, mode="deep", deep_max_depth=1, deep_max_items=100
        )
        deep_high = estimate_memory_bytes(
            nested, mode="deep", deep_max_depth=5, deep_max_items=100
        )
        assert deep_high >= deep_low

    def test_invalid_mode_falls_back_to_shallow(self) -> None:
        env = {"x": [1, 2, 3]}
        shallow = estimate_memory_bytes(env, mode="shallow")
        invalid = estimate_memory_bytes(env, mode="unknown-mode")
        assert invalid == shallow

    def test_deep_mode_zero_item_budget_still_counts_container(self) -> None:
        env = {"x": [1, 2, 3, 4, 5]}
        deep_zero_items = estimate_memory_bytes(
            env, mode="deep", deep_max_depth=10, deep_max_items=0
        )
        assert deep_zero_items > 0

    def test_deep_mode_handles_objects_with_dict(self) -> None:
        class Box:
            def __init__(self) -> None:
                self.payload = [1, 2, 3, 4]

        env = {"obj": Box()}
        shallow = estimate_memory_bytes(env, mode="shallow")
        deep = estimate_memory_bytes(
            env, mode="deep", deep_max_depth=5, deep_max_items=100
        )
        assert deep >= shallow

    def test_safe_getsizeof_falls_back_on_type_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(value: object) -> int:
            raise TypeError("boom")

        monkeypatch.setattr("optilang.profiler.sys.getsizeof", boom)
        assert _safe_getsizeof(object()) == 28

    def test_deep_mode_handles_cycles(self) -> None:
        cyclic: list[object] = []
        cyclic.append(cyclic)
        assert (
            estimate_memory_bytes(
                {"x": cyclic}, mode="deep", deep_max_depth=10, deep_max_items=10
            )
            > 0
        )

    def test_deep_mode_dict_budget_breaks_on_next_iteration(self) -> None:
        env = {"x": {"a": 1, "b": 2}}
        assert (
            estimate_memory_bytes(env, mode="deep", deep_max_depth=5, deep_max_items=2)
            > 0
        )

    def test_deep_mode_dict_budget_breaks_after_key_before_value(self) -> None:
        assert _estimate_deep_object_size({"a": 1}, max_depth=5, max_items=1) > 0

    def test_deep_mode_list_budget_breaks_on_next_iteration(self) -> None:
        env = {"x": [1, 2, 3]}
        assert (
            estimate_memory_bytes(env, mode="deep", deep_max_depth=5, deep_max_items=1)
            > 0
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Unit Tests: detect_complexity
# ─────────────────────────────────────────────────────────────────────────────


class TestDetectComplexity:
    """Tests for the complexity detection heuristic."""

    def _make_stats(self, counts: list[int]) -> dict[int, LineStats]:
        """Helper: build a line_stats dict from a list of execution counts."""
        return {
            i: LineStats(line_number=i, execution_count=c)
            for i, c in enumerate(counts, start=1)
        }

    def test_empty_returns_O1(self) -> None:
        assert detect_complexity({}) == "O(1)"

    def test_all_once_returns_O1(self) -> None:
        stats = self._make_stats([1, 1, 1])
        assert detect_complexity(stats) == "O(1)"

    def test_small_repeat_returns_Ologn(self) -> None:
        # Max count of 8 → O(log n)
        stats = self._make_stats([1, 8, 3])
        assert detect_complexity(stats) == "O(log n)"

    def test_moderate_repeat_returns_On(self) -> None:
        # Max count of 500 → O(n)
        stats = self._make_stats([1, 500, 10])
        assert detect_complexity(stats) == "O(n)"

    def test_high_repeat_returns_On2(self) -> None:
        # Max count of 10,000 → O(n²)
        stats = self._make_stats([1, 10_000])
        assert detect_complexity(stats) in ("O(n^2)", "O(n log n)")

    def test_very_high_repeat_returns_On2(self) -> None:
        # Max count of 100,000 → O(n²)
        stats = self._make_stats([100_000])
        assert detect_complexity(stats) == "O(n^2)"

    def test_extreme_repeat_returns_On3(self) -> None:
        # Max count > 1,000,000 → O(n³) or worse
        stats = self._make_stats([2_000_000])
        assert detect_complexity(stats) == "O(n^3) or worse"

    def test_detect_complexity_with_confidence_range(self) -> None:
        stats = self._make_stats([1, 200, 10])
        complexity, confidence = detect_complexity_with_confidence(stats)
        assert complexity == "O(n)"
        assert 0.0 <= confidence <= 1.0

    def test_detect_complexity_with_confidence_nlogn_branch(self) -> None:
        stats = self._make_stats([10_000] * 200)
        complexity, confidence = detect_complexity_with_confidence(stats)
        assert complexity == "O(n log n)"
        assert 0.0 <= confidence <= 1.0


class TestProfilerConfig:
    """Normalization and clamping behavior for ProfilerConfig."""

    def test_memory_mode_invalid_defaults_to_shallow(self) -> None:
        cfg = ProfilerConfig(memory_mode="invalid")
        assert cfg.normalized_memory_mode() == "shallow"

    def test_sampling_rate_clamps_low(self) -> None:
        cfg = ProfilerConfig(line_sampling_rate=-0.2)
        assert cfg.normalized_sampling_rate() == 0.0

    def test_sampling_rate_clamps_high(self) -> None:
        cfg = ProfilerConfig(line_sampling_rate=1.7)
        assert cfg.normalized_sampling_rate() == 1.0

    def test_deep_bounds_clamp_non_negative(self) -> None:
        cfg = ProfilerConfig(deep_max_depth=-2, deep_max_items=-10)
        assert cfg.normalized_deep_max_depth() == 0
        assert cfg.normalized_deep_max_items() == 0


# ─────────────────────────────────────────────────────────────────────────────
#  Unit Tests: LineStats
# ─────────────────────────────────────────────────────────────────────────────


class TestLineStats:
    """Tests for LineStats data class and its update_time method."""

    def test_initial_values(self) -> None:
        stats = LineStats(line_number=5)
        assert stats.execution_count == 0
        assert stats.total_time_ms == 0.0
        assert stats.avg_time_ms == 0.0
        assert stats.min_time_ms == float("inf")
        assert stats.max_time_ms == 0.0

    def test_single_update(self) -> None:
        stats = LineStats(line_number=1)
        stats.update_time(10.0)
        assert stats.execution_count == 1
        assert stats.total_time_ms == 10.0
        assert stats.avg_time_ms == 10.0
        assert stats.min_time_ms == 10.0
        assert stats.max_time_ms == 10.0

    def test_multiple_updates_track_min_max(self) -> None:
        stats = LineStats(line_number=1)
        stats.update_time(5.0)
        stats.update_time(15.0)
        stats.update_time(10.0)
        assert stats.execution_count == 3
        assert stats.min_time_ms == 5.0
        assert stats.max_time_ms == 15.0
        assert abs(stats.avg_time_ms - 10.0) < 0.001

    def test_to_dict_keys(self) -> None:
        stats = LineStats(line_number=3)
        stats.update_time(2.5)
        stats.memory_vars = 4
        stats.memory_bytes = 256
        d = stats.to_dict()
        assert d["line"] == 3
        assert d["count"] == 1
        assert d["total_time_ms"] == 2.5
        assert d["memory_vars"] == 4
        assert d["memory_bytes"] == 256
        assert "min_time_ms" in d
        assert "max_time_ms" in d


# ─────────────────────────────────────────────────────────────────────────────
#  Unit Tests: FunctionStats
# ─────────────────────────────────────────────────────────────────────────────


class TestFunctionStats:
    """Tests for FunctionStats data class and its record_call method."""

    def test_initial_values(self) -> None:
        stats = FunctionStats(name="foo")
        assert stats.call_count == 0
        assert stats.total_time_ms == 0.0
        assert stats.max_recursion_depth == 0
        assert stats.callers == {}

    def test_single_call(self) -> None:
        stats = FunctionStats(name="foo")
        stats.record_call(20.0, depth=0, caller=None)
        assert stats.call_count == 1
        assert stats.total_time_ms == 20.0
        assert stats.avg_time_ms == 20.0
        assert stats.min_time_ms == 20.0
        assert stats.max_time_ms == 20.0

    def test_multiple_calls_avg(self) -> None:
        stats = FunctionStats(name="foo")
        stats.record_call(10.0)
        stats.record_call(30.0)
        assert stats.call_count == 2
        assert stats.avg_time_ms == 20.0
        assert stats.min_time_ms == 10.0
        assert stats.max_time_ms == 30.0

    def test_recursion_depth_tracked(self) -> None:
        stats = FunctionStats(name="factorial")
        stats.record_call(1.0, depth=0)
        stats.record_call(1.0, depth=3)
        stats.record_call(1.0, depth=1)
        assert stats.max_recursion_depth == 3

    def test_caller_tracking(self) -> None:
        stats = FunctionStats(name="helper")
        stats.record_call(1.0, caller="main")
        stats.record_call(1.0, caller="main")
        stats.record_call(1.0, caller="other")
        assert stats.callers["main"] == 2
        assert stats.callers["other"] == 1

    def test_to_dict_keys(self) -> None:
        stats = FunctionStats(name="bar")
        stats.record_call(5.0, depth=2, caller="baz")
        d = stats.to_dict()
        assert d["name"] == "bar"
        assert d["calls"] == 1
        assert d["max_recursion_depth"] == 2
        assert d["callers"] == {"baz": 1}
        assert "min_time_ms" in d
        assert "max_time_ms" in d


# ─────────────────────────────────────────────────────────────────────────────
#  Unit Tests: Profiler (direct API)
# ─────────────────────────────────────────────────────────────────────────────


class TestProfilerDirect:
    """Tests for Profiler class using its API directly (no executor)."""

    def _run_simple_session(
        self, profiler: Profiler, lines: int = 5, sleep: float = 0.001
    ) -> None:
        """Helper: simulate a simple N-line execution session."""
        profiler.start()
        env = {"x": 1, "y": "hello"}
        for i in range(1, lines + 1):
            profiler.start_line(i, env_values=env)
            time.sleep(sleep)
            profiler.end_line(i)
        profiler.stop()

    # ── Session control ──────────────────────────────────────────────────

    def test_start_stop_sets_times(self) -> None:
        p = Profiler()
        p.start()
        time.sleep(0.01)
        p.stop()
        assert p.data.start_time is not None
        assert p.data.end_time is not None
        assert p.data.total_execution_time_ms > 0

    def test_reset_clears_data(self) -> None:
        p = Profiler()
        self._run_simple_session(p)
        p.reset()
        assert p.data.line_stats == {}
        assert p.data.function_stats == {}
        assert p.data.total_execution_time_ms == 0.0

    # ── Line tracking ────────────────────────────────────────────────────

    def test_line_stats_created(self) -> None:
        p = Profiler()
        self._run_simple_session(p, lines=3)
        assert 1 in p.data.line_stats
        assert 2 in p.data.line_stats
        assert 3 in p.data.line_stats

    def test_execution_count_increments(self) -> None:
        p = Profiler()
        p.start()
        env = {"a": 1}
        for _ in range(7):
            p.start_line(1, env)
            p.end_line(1)
        p.stop()
        assert p.data.line_stats[1].execution_count == 7

    def test_total_lines_executed_sum(self) -> None:
        p = Profiler()
        p.start()
        env: dict[str, object] = {}
        for _ in range(3):
            p.start_line(1, env)
            p.end_line(1)
        for _ in range(5):
            p.start_line(2, env)
            p.end_line(2)
        p.stop()
        assert p.data.total_lines_executed == 8

    def test_timing_is_positive(self) -> None:
        p = Profiler()
        p.start()
        p.start_line(1, {"x": 0})
        time.sleep(0.002)
        p.end_line(1)
        p.stop()
        assert p.data.line_stats[1].total_time_ms > 0

    def test_memory_bytes_recorded(self) -> None:
        p = Profiler()
        p.start()
        env = {"x": 1, "y": [1, 2, 3, 4, 5]}
        p.start_line(1, env)
        p.end_line(1)
        p.stop()
        assert p.data.line_stats[1].memory_bytes > 0

    def test_memory_vars_count(self) -> None:
        p = Profiler()
        p.start()
        env = {"a": 1, "b": 2, "c": 3}
        p.start_line(1, env)
        p.end_line(1)
        p.stop()
        assert p.data.line_stats[1].memory_vars == 3

    def test_peak_memory_tracked(self) -> None:
        p = Profiler()
        p.start()
        p.start_line(1, {"x": 1})
        p.end_line(1)
        p.start_line(2, {"x": 1, "y": "a longer string here", "z": [1, 2, 3]})
        p.end_line(2)
        p.stop()
        assert p.data.peak_memory_bytes > 0

    def test_min_max_line_timing(self) -> None:
        p = Profiler()
        p.start()
        env: dict[str, object] = {}
        p.start_line(1, env)
        time.sleep(0.001)
        p.end_line(1)
        p.start_line(1, env)
        time.sleep(0.005)
        p.end_line(1)
        p.stop()
        stats = p.data.line_stats[1]
        assert stats.min_time_ms < stats.max_time_ms
        assert stats.min_time_ms > 0

    # ── Function tracking ────────────────────────────────────────────────

    def test_function_stats_created(self) -> None:
        p = Profiler()
        p.start()
        p.start_function_call("my_func")
        time.sleep(0.001)
        p.end_function_call("my_func")
        p.stop()
        assert "my_func" in p.data.function_stats

    def test_function_call_count(self) -> None:
        p = Profiler()
        p.start()
        for _ in range(4):
            p.start_function_call("add")
            p.end_function_call("add")
        p.stop()
        assert p.data.function_stats["add"].call_count == 4

    def test_function_recursion_depth(self) -> None:
        p = Profiler()
        p.start()
        # Simulate 3 levels of recursion
        p.start_function_call("fact")
        p.start_function_call("fact")
        p.start_function_call("fact")
        p.end_function_call("fact")
        p.end_function_call("fact")
        p.end_function_call("fact")
        p.stop()
        assert p.data.function_stats["fact"].max_recursion_depth >= 1

    def test_function_caller_tracking(self) -> None:
        p = Profiler()
        p.start()
        p.start_function_call("helper", caller="main")
        p.end_function_call("helper")
        p.start_function_call("helper", caller="main")
        p.end_function_call("helper")
        p.start_function_call("helper", caller="setup")
        p.end_function_call("helper")
        p.stop()
        callers = p.data.function_stats["helper"].callers
        assert callers["main"] == 2
        assert callers["setup"] == 1

    def test_function_min_max_timing(self) -> None:
        p = Profiler()
        p.start()
        p.start_function_call("slow")
        time.sleep(0.005)
        p.end_function_call("slow")
        p.start_function_call("slow")
        time.sleep(0.001)
        p.end_function_call("slow")
        p.stop()
        stats = p.data.function_stats["slow"]
        assert stats.min_time_ms < stats.max_time_ms

    # ── Complexity detection ─────────────────────────────────────────────

    def test_complexity_O1_after_single_pass(self) -> None:
        p = Profiler()
        p.start()
        for line in range(1, 5):
            p.start_line(line, {})
            p.end_line(line)
        p.stop()
        assert p.data.complexity_estimate == "O(1)"

    def test_complexity_On_after_loop(self) -> None:
        p = Profiler()
        p.start()
        for _ in range(200):
            p.start_line(2, {})
            p.end_line(2)
        p.stop()
        assert p.data.complexity_estimate == "O(n)"

    def test_complexity_On2_after_nested_loop(self) -> None:
        p = Profiler()
        p.start()
        for _ in range(10_000):
            p.start_line(3, {})
            p.end_line(3)
        p.stop()
        assert p.data.complexity_estimate in ("O(n^2)", "O(n log n)")

    # ── Query methods ────────────────────────────────────────────────────

    def test_get_hottest_lines(self) -> None:
        p = Profiler()
        p.start()
        for _ in range(3):
            p.start_line(1, {})
            time.sleep(0.001)
            p.end_line(1)
        p.start_line(2, {})
        time.sleep(0.010)  # line 2 is slowest overall
        p.end_line(2)
        p.stop()
        hottest = p.get_hottest_lines(top_n=1)
        assert len(hottest) == 1
        assert hottest[0].line_number == 2

    def test_get_most_executed_lines(self) -> None:
        p = Profiler()
        p.start()
        for _ in range(10):
            p.start_line(5, {})
            p.end_line(5)
        p.start_line(6, {})
        p.end_line(6)
        p.stop()
        most = p.get_most_executed_lines(top_n=1)
        assert most[0].line_number == 5

    def test_get_hottest_functions(self) -> None:
        p = Profiler()
        p.start()
        p.start_function_call("fast")
        time.sleep(0.001)
        p.end_function_call("fast")
        p.start_function_call("slow")
        time.sleep(0.010)
        p.end_function_call("slow")
        p.stop()
        hottest = p.get_hottest_functions(top_n=1)
        assert hottest[0].name == "slow"

    def test_get_most_called_functions(self) -> None:
        p = Profiler()
        p.start()
        for _ in range(5):
            p.start_function_call("frequent")
            p.end_function_call("frequent")
        p.start_function_call("rare")
        p.end_function_call("rare")
        p.stop()
        most = p.get_most_called_functions(top_n=1)
        assert most[0].name == "frequent"

    # ── Summary ──────────────────────────────────────────────────────────

    def test_get_summary_keys_present(self) -> None:
        p = Profiler()
        self._run_simple_session(p, lines=3)
        summary = p.get_summary()
        expected_keys = [
            "total_time_ms",
            "total_lines_executed",
            "unique_lines_profiled",
            "peak_memory_bytes",
            "peak_memory_kb",
            "complexity_estimate",
            "complexity_method",
            "complexity_confidence",
            "functions_called",
            "total_function_calls",
            "sampled_lines",
            "skipped_lines",
            "line_sampling_rate",
            "memory_mode",
            "hottest_lines",
            "most_executed_lines",
            "hottest_functions",
        ]
        for key in expected_keys:
            assert key in summary, f"Missing key: {key}"

    def test_get_summary_values_sensible(self) -> None:
        p = Profiler()
        self._run_simple_session(p, lines=3)
        summary = p.get_summary()
        assert summary["total_time_ms"] > 0
        assert summary["total_lines_executed"] == 3
        assert summary["unique_lines_profiled"] == 3
        assert summary["complexity_estimate"] == "O(1)"
        assert summary["complexity_method"] == "heuristic"
        assert 0.0 <= summary["complexity_confidence"] <= 1.0
        assert isinstance(summary["hottest_lines"], list)
        assert isinstance(summary["hottest_functions"], list)

    def test_summary_peak_memory_kb_conversion(self) -> None:
        p = Profiler()
        p.start()
        p.start_line(1, {"big": list(range(100))})
        p.end_line(1)
        p.stop()
        summary = p.get_summary()
        assert summary["peak_memory_kb"] == round(
            summary["peak_memory_bytes"] / 1024, 2
        )

    def test_summary_includes_config_metadata(self) -> None:
        p = Profiler(config=ProfilerConfig(memory_mode="deep", line_sampling_rate=0.5))
        p.start()
        p.start_line(1, {"x": [1, 2, 3]})
        p.end_line(1)
        p.stop()
        summary = p.get_summary()
        assert summary["memory_mode"] == "deep"
        assert summary["line_sampling_rate"] == 0.5

    # ── Enable / Disable ─────────────────────────────────────────────────

    def test_disabled_profiler_records_nothing(self) -> None:
        p = Profiler()
        p.disable()
        p.start()
        p.start_line(1, {"x": 1})
        time.sleep(0.001)
        p.end_line(1)
        p.stop()
        assert p.data.line_stats == {}

    def test_re_enable_resumes_recording(self) -> None:
        p = Profiler()
        p.disable()
        p.enable()
        p.start()
        p.start_line(1, {})
        p.end_line(1)
        p.stop()
        assert 1 in p.data.line_stats

    def test_sampling_zero_keeps_counts_without_timing(self) -> None:
        p = Profiler(config=ProfilerConfig(line_sampling_rate=0.0))
        p.start()
        for _ in range(5):
            p.start_line(1, {"x": 1})
            p.end_line(1)
        p.stop()
        stats = p.data.line_stats[1]
        assert stats.execution_count == 5
        assert stats.total_time_ms == 0.0
        assert p.data.sampled_lines == 0
        assert p.data.skipped_lines == 5

    def test_memory_mode_off_skips_memory_tracking(self) -> None:
        p = Profiler(config=ProfilerConfig(memory_mode="off"))
        p.start()
        p.start_line(1, {"big": list(range(100))})
        p.end_line(1)
        p.stop()
        stats = p.data.line_stats[1]
        assert stats.memory_vars == 0
        assert stats.memory_bytes == 0
        assert p.data.peak_memory_bytes == 0

    def test_memory_mode_deep_can_exceed_shallow(self) -> None:
        env = {"x": [{"a": [1, 2, 3, 4, 5]}]}

        shallow = Profiler(config=ProfilerConfig(memory_mode="shallow"))
        shallow.start()
        shallow.start_line(1, env)
        shallow.end_line(1)
        shallow.stop()

        deep = Profiler(
            config=ProfilerConfig(
                memory_mode="deep",
                deep_max_depth=5,
                deep_max_items=200,
            )
        )
        deep.start()
        deep.start_line(1, env)
        deep.end_line(1)
        deep.stop()
        assert (
            deep.data.line_stats[1].memory_bytes
            >= shallow.data.line_stats[1].memory_bytes
        )


# ─────────────────────────────────────────────────────────────────────────────
#  Integration Tests: Profiler via execute()
# ─────────────────────────────────────────────────────────────────────────────


class TestProfilerIntegration:
    """End-to-end tests using the public execute() API."""

    def test_execute_returns_profiling_data(self) -> None:
        result = execute("x = 1\ny = 2\nprint(x + y)")
        assert result.profiling is not None

    def test_execute_profiling_has_line_stats(self) -> None:
        result = execute("x = 1\ny = 2")
        assert result.profiling is not None
        assert len(result.profiling.line_stats) > 0

    def test_execute_loop_inflates_execution_count(self) -> None:
        result = execute("total = 0\n" "for i in range(50):\n" "    total += i")
        assert result.profiling is not None
        counts = [s.execution_count for s in result.profiling.line_stats.values()]
        assert max(counts) >= 50

    def test_execute_nested_loop_detected_as_quadratic(self) -> None:
        result = execute(
            "for i in range(100):\n" "    for j in range(100):\n" "        x = i * j"
        )
        assert result.profiling is not None
        complexity = result.profiling.complexity_estimate
        assert complexity in ("O(n^2)", "O(n log n)")

    def test_execute_simple_program_detected_as_O1(self) -> None:
        result = execute("x = 1\ny = 2\nz = x + y")
        assert result.profiling is not None
        assert result.profiling.complexity_estimate == "O(1)"

    def test_execute_function_tracked(self) -> None:
        result = execute(
            "def greet(name):\n" "    return name\n" "print(greet('world'))"
        )
        assert result.profiling is not None
        assert "greet" in result.profiling.function_stats
        assert result.profiling.function_stats["greet"].call_count == 1

    def test_execute_recursive_function_depth(self) -> None:
        result = execute(
            "def factorial(n):\n"
            "    if n <= 1:\n"
            "        return 1\n"
            "    return n * factorial(n - 1)\n"
            "print(factorial(5))"
        )
        assert result.errors == []
        assert result.output == "120"
        assert result.profiling is not None
        stats = result.profiling.function_stats.get("factorial")
        assert stats is not None
        assert stats.call_count == 5
        assert stats.max_recursion_depth >= 1

    def test_execute_peak_memory_positive(self) -> None:
        result = execute(
            "items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]\n"
            "total = 0\n"
            "for i in items:\n"
            "    total += i"
        )
        assert result.profiling is not None
        assert result.profiling.peak_memory_bytes > 0

    def test_execute_total_time_positive(self) -> None:
        result = execute("x = 1 + 1")
        assert result.profiling is not None
        assert result.profiling.total_execution_time_ms > 0

    def test_execute_profiling_disabled(self) -> None:
        result = execute("x = 1", enable_profiling=False)
        assert result.profiling is None

    def test_execute_errors_do_not_crash_profiler(self) -> None:
        result = execute("print(undefined_variable)")
        assert len(result.errors) == 1
        # Profiling data should still be returned (partial session is fine)
        # No assertion on profiling content — just must not raise

    def test_execute_caller_tracking_in_mutual_calls(self) -> None:
        result = execute(
            "def add(a, b):\n"
            "    return a + b\n"
            "def compute(x):\n"
            "    return add(x, x)\n"
            "print(compute(5))"
        )
        assert result.errors == []
        assert result.output == "10"
        assert result.profiling is not None
        # Both functions should be tracked
        assert "add" in result.profiling.function_stats
        assert "compute" in result.profiling.function_stats
        assert result.profiling.function_stats["add"].call_count == 1
        assert result.profiling.function_stats["compute"].call_count == 1

    def test_execute_profiling_data_serializable(self) -> None:
        import json

        result = execute(
            "def square(n):\n"
            "    return n * n\n"
            "for i in range(5):\n"
            "    x = square(i)"
        )
        # to_dict should produce JSON-serializable output
        assert result.profiling is not None
        data_dict = result.profiling.to_dict()
        serialized = json.dumps(data_dict)
        assert isinstance(serialized, str)
        assert len(serialized) > 0


# ─────────────────────────────────────────────────────────────────────────────
#  Unit Tests: get_call_stack()
# ─────────────────────────────────────────────────────────────────────────────


class TestGetCallStack:
    """Tests for the public get_call_stack() method."""

    def test_empty_stack_when_no_calls(self) -> None:
        p = Profiler()
        assert p.get_call_stack() == []

    def test_single_function_on_stack(self) -> None:
        p = Profiler()
        p.start()
        p.start_function_call("main")
        assert p.get_call_stack() == ["main"]
        p.end_function_call("main")
        p.stop()

    def test_nested_functions_on_stack(self) -> None:
        p = Profiler()
        p.start()
        p.start_function_call("outer")
        p.start_function_call("inner")
        stack = p.get_call_stack()
        assert stack[0] == "outer"
        assert stack[1] == "inner"
        p.end_function_call("inner")
        p.end_function_call("outer")
        p.stop()

    def test_stack_shrinks_after_return(self) -> None:
        p = Profiler()
        p.start()
        p.start_function_call("a")
        p.start_function_call("b")
        p.end_function_call("b")
        assert p.get_call_stack() == ["a"]
        p.end_function_call("a")
        assert p.get_call_stack() == []
        p.stop()

    def test_stack_empty_after_all_returns(self) -> None:
        p = Profiler()
        p.start()
        p.start_function_call("foo")
        p.end_function_call("foo")
        p.stop()
        assert p.get_call_stack() == []


# ─────────────────────────────────────────────────────────────────────────────
#  Unit Tests: ProfilingData.to_dict()
# ─────────────────────────────────────────────────────────────────────────────


class TestProfilingDataToDict:
    """Tests for ProfilingData serialization."""

    def _run_session(self) -> ProfilingData:
        p = Profiler()
        p.start()
        p.start_line(1, {"x": 1})
        p.end_line(1)
        p.start_function_call("foo")
        p.end_function_call("foo")
        p.stop()
        return p.get_data()

    def test_to_dict_has_required_keys(self) -> None:
        data = self._run_session()
        d = data.to_dict()
        for key in [
            "line_stats",
            "function_stats",
            "total_time_ms",
            "total_lines_executed",
            "lines_profiled",
            "peak_memory_bytes",
            "complexity_estimate",
            "complexity_method",
            "complexity_confidence",
            "sampled_lines",
            "skipped_lines",
            "line_sampling_rate",
            "memory_mode",
        ]:
            assert key in d, f"Missing key: {key}"

    def test_to_dict_line_stats_sorted(self) -> None:
        p = Profiler()
        p.start()
        for line in [5, 2, 8, 1]:
            p.start_line(line, {})
            p.end_line(line)
        p.stop()
        d = p.get_data().to_dict()
        keys = list(d["line_stats"].keys())
        assert keys == sorted(keys)

    def test_to_dict_is_json_serializable(self) -> None:
        import json

        data = self._run_session()
        serialized = json.dumps(data.to_dict())
        assert isinstance(serialized, str)

    def test_to_dict_total_time_non_negative(self) -> None:
        data = self._run_session()
        assert data.to_dict()["total_time_ms"] >= 0

    def test_to_dict_lines_profiled_count(self) -> None:
        p = Profiler()
        p.start()
        for line in [1, 2, 3]:
            p.start_line(line, {})
            p.end_line(line)
        p.stop()
        d = p.get_data().to_dict()
        assert d["lines_profiled"] == 3

    def test_to_dict_complexity_is_string(self) -> None:
        data = self._run_session()
        assert isinstance(data.to_dict()["complexity_estimate"], str)


# ─────────────────────────────────────────────────────────────────────────────
#  Edge Case Tests
# ─────────────────────────────────────────────────────────────────────────────


class TestProfilerEdgeCases:
    """Edge cases and boundary conditions."""

    def test_end_line_without_start_does_not_crash(self) -> None:
        p = Profiler()
        p.start()
        p.end_line(99)  # no matching start_line — should not raise
        p.stop()

    def test_end_function_without_start_does_not_crash(self) -> None:
        p = Profiler()
        p.start()
        p.end_function_call("ghost")  # no matching start — should not raise
        p.stop()

    def test_start_line_none_env_skips_memory(self) -> None:
        p = Profiler()
        p.start()
        p.start_line(1, None)
        p.end_line(1)
        p.stop()
        assert p.data.line_stats[1].memory_vars == 0
        assert p.data.line_stats[1].memory_bytes == 0

    def test_same_line_executed_multiple_times_accumulates(self) -> None:
        p = Profiler()
        p.start()
        for _ in range(5):
            p.start_line(10, {})
            p.end_line(10)
        p.stop()
        assert p.data.line_stats[10].execution_count == 5

    def test_reset_then_reuse(self) -> None:
        p = Profiler()
        p.start()
        p.start_line(1, {"x": 1})
        p.end_line(1)
        p.stop()
        p.reset()
        p.start()
        p.start_line(2, {"y": 2})
        p.end_line(2)
        p.stop()
        assert 1 not in p.data.line_stats
        assert 2 in p.data.line_stats

    def test_get_hottest_lines_top_n_respected(self) -> None:
        p = Profiler()
        p.start()
        for line in range(1, 11):  # 10 lines
            p.start_line(line, {})
            p.end_line(line)
        p.stop()
        assert len(p.get_hottest_lines(top_n=3)) == 3

    def test_get_most_executed_lines_top_n_respected(self) -> None:
        p = Profiler()
        p.start()
        for line in range(1, 8):
            p.start_line(line, {})
            p.end_line(line)
        p.stop()
        assert len(p.get_most_executed_lines(top_n=5)) == 5

    def test_get_hottest_functions_top_n_respected(self) -> None:
        p = Profiler()
        p.start()
        for name in ["a", "b", "c", "d", "e", "f"]:
            p.start_function_call(name)
            p.end_function_call(name)
        p.stop()
        assert len(p.get_hottest_functions(top_n=2)) == 2

    def test_profiler_handles_zero_time_lines(self) -> None:
        # Lines that execute so fast time is ~0 should not crash
        p = Profiler()
        p.start()
        p.start_line(1, {})
        p.end_line(1)
        p.stop()
        stats = p.data.line_stats[1]
        assert stats.total_time_ms >= 0
        assert stats.min_time_ms >= 0

    def test_end_line_recreates_missing_line_stats_entry(self) -> None:
        p = Profiler()
        p.start()
        p._current_line_number = 7
        p._current_line_sampled = False
        p._current_line_start = None
        p.data.line_stats.pop(7, None)

        p.end_line(7)

        assert p.data.line_stats[7].execution_count == 1

    def test_end_line_with_missing_start_resets_state(self) -> None:
        p = Profiler()
        p.start()
        p._current_line_number = 8
        p._current_line_sampled = True
        p._current_line_start = None

        p.end_line(8)

        assert p._current_line_number is None
        assert p._current_line_sampled is False

    def test_start_function_call_returns_immediately_when_disabled(self) -> None:
        p = Profiler()
        p.disable()

        p.start_function_call("compute")

        assert p.get_call_stack() == []


# ─────────────────────────────────────────────────────────────────────────────
#  Unit Tests: profile_execution() convenience function
# ─────────────────────────────────────────────────────────────────────────────


class TestProfileExecution:
    """Tests for the profile_execution() convenience wrapper."""

    def test_returns_tuple(self) -> None:
        def fake_executor(
            code: str, profiler: Profiler, scale: int = 1
        ) -> dict[str, object]:
            profiler.start_line(1, {"x": 1})
            profiler.end_line(1)
            return {"code": code, "scale": scale}

        result, profiling = profile_execution(fake_executor, "x = 1", scale=2)
        assert isinstance(result, dict)
        assert result["code"] == "x = 1"
        assert result["scale"] == 2
        assert isinstance(profiling, ProfilingData)

    def test_wrapper_collects_profiler_stats(self) -> None:
        def fake_executor(code: str, profiler: Profiler) -> str:
            profiler.start_function_call("inner", caller="outer")
            profiler.start_line(2, {"code": code})
            profiler.end_line(2)
            profiler.end_function_call("inner")
            return "ok"

        result, profiling = profile_execution(fake_executor, "print(1)")
        assert result == "ok"
        assert profiling.total_lines_executed == 1
        assert "inner" in profiling.function_stats
        assert profiling.function_stats["inner"].call_count == 1


class TestProfilerModuleScript:
    """Tests for running profiler module as a script."""

    def test_run_as_module_subprocess(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        completed = subprocess.run(
            [sys.executable, "-m", "optilang.profiler"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
        assert completed.returncode == 0
        assert "PROFILING RESULTS" in completed.stdout
        assert "SUMMARY:" in completed.stdout

    def test_run_as_module_runpy(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        profiler_path = repo_root / "optilang" / "profiler.py"
        out = io.StringIO()
        with redirect_stdout(out):
            runpy.run_path(str(profiler_path), run_name="__main__")
        output = out.getvalue()
        assert "PROFILING RESULTS" in output
        assert "LINE STATS:" in output
        assert "FUNCTION STATS:" in output
