"""
Comprehensive test suite for OptiLang Profiler (optilang/profiler.py)

Tests cover:
- Line execution tracking (count, timing)
- Memory (variable count) tracking
- Function call tracking
- Profiler controls (enable, disable, reset)
- Hotspot detection (hottest lines, most executed lines)
- Integration with executor via execute()
- ProfilingData structure and serialization
- LineStats and FunctionStats dataclasses
- NFR: profiling overhead is reasonable
"""

import time
import pytest
from optilang.profiler import Profiler, LineStats, FunctionStats, ProfilingData
from optilang import execute


# ===========================================================================
# 1. PROFILER INITIALIZATION
# ===========================================================================

class TestProfilerInit:

    def test_profiler_creates_empty_line_stats(self):
        profiler = Profiler()
        assert profiler.data.line_stats == {}

    def test_profiler_creates_empty_function_stats(self):
        profiler = Profiler()
        assert profiler.data.function_stats == {}

    def test_profiler_enabled_by_default(self):
        profiler = Profiler()
        assert profiler._enabled is True

    def test_profiler_call_stack_empty_initially(self):
        profiler = Profiler()
        assert profiler._function_call_stack == []


# ===========================================================================
# 2. LINE TRACKING — BASIC
# ===========================================================================

class TestLineTracking:

    def test_executed_line_appears_in_stats(self):
        profiler = Profiler()
        profiler.start()
        profiler.start_line(1, variable_count=0)
        profiler.end_line(1)
        profiler.stop()
        assert 1 in profiler.data.line_stats

    def test_single_execution_count_is_1(self):
        profiler = Profiler()
        profiler.start()
        profiler.start_line(1, variable_count=0)
        profiler.end_line(1)
        profiler.stop()
        assert profiler.data.line_stats[1].execution_count == 1

    def test_multiple_executions_of_same_line(self):
        profiler = Profiler()
        profiler.start()
        for _ in range(5):
            profiler.start_line(3, variable_count=0)
            profiler.end_line(3)
        profiler.stop()
        assert profiler.data.line_stats[3].execution_count == 5

    def test_never_executed_line_not_in_stats(self):
        profiler = Profiler()
        profiler.start()
        profiler.start_line(1, variable_count=0)
        profiler.end_line(1)
        profiler.stop()
        assert 99 not in profiler.data.line_stats

    def test_multiple_different_lines_tracked(self):
        profiler = Profiler()
        profiler.start()
        for line in [1, 2, 3]:
            profiler.start_line(line, variable_count=0)
            profiler.end_line(line)
        profiler.stop()
        assert 1 in profiler.data.line_stats
        assert 2 in profiler.data.line_stats
        assert 3 in profiler.data.line_stats

    def test_total_lines_executed_correct(self):
        profiler = Profiler()
        profiler.start()
        for _ in range(3):
            profiler.start_line(1, variable_count=0)
            profiler.end_line(1)
        profiler.start_line(2, variable_count=0)
        profiler.end_line(2)
        profiler.stop()
        assert profiler.data.total_lines_executed == 4


# ===========================================================================
# 3. LINE TIMING
# ===========================================================================

class TestLineTiming:

    def test_line_total_time_greater_than_zero(self):
        profiler = Profiler()
        profiler.start()
        profiler.start_line(1, variable_count=0)
        time.sleep(0.001)
        profiler.end_line(1)
        profiler.stop()
        assert profiler.data.line_stats[1].total_time_ms > 0

    def test_avg_time_equals_total_for_single_execution(self):
        profiler = Profiler()
        profiler.start()
        profiler.start_line(1, variable_count=0)
        time.sleep(0.001)
        profiler.end_line(1)
        profiler.stop()
        stats = profiler.data.line_stats[1]
        assert abs(stats.avg_time_ms - stats.total_time_ms) < 0.001

    def test_avg_time_calculated_correctly(self):
        profiler = Profiler()
        profiler.start()
        for _ in range(4):
            profiler.start_line(1, variable_count=0)
            profiler.end_line(1)
        profiler.stop()
        stats = profiler.data.line_stats[1]
        expected_avg = stats.total_time_ms / stats.execution_count
        assert abs(stats.avg_time_ms - expected_avg) < 0.001

    def test_total_time_increases_with_executions(self):
        profiler = Profiler()
        profiler.start()
        profiler.start_line(1, variable_count=0)
        time.sleep(0.001)
        profiler.end_line(1)
        first_total = profiler.data.line_stats[1].total_time_ms
        profiler.start_line(1, variable_count=0)
        time.sleep(0.001)
        profiler.end_line(1)
        second_total = profiler.data.line_stats[1].total_time_ms
        assert second_total > first_total


# ===========================================================================
# 4. MEMORY / VARIABLE COUNT TRACKING
# ===========================================================================

class TestMemoryTracking:

    def test_variable_count_recorded(self):
        profiler = Profiler()
        profiler.start()
        profiler.start_line(1, variable_count=3)
        profiler.end_line(1)
        profiler.stop()
        assert profiler.data.line_stats[1].memory_vars == 3

    def test_variable_count_zero(self):
        profiler = Profiler()
        profiler.start()
        profiler.start_line(1, variable_count=0)
        profiler.end_line(1)
        profiler.stop()
        assert profiler.data.line_stats[1].memory_vars == 0

    def test_variable_count_updated_on_repeat(self):
        profiler = Profiler()
        profiler.start()
        profiler.start_line(1, variable_count=1)
        profiler.end_line(1)
        profiler.start_line(1, variable_count=5)
        profiler.end_line(1)
        profiler.stop()
        assert profiler.data.line_stats[1].memory_vars == 5


# ===========================================================================
# 5. FUNCTION CALL TRACKING
# ===========================================================================

class TestFunctionTracking:

    def test_called_function_appears_in_stats(self):
        profiler = Profiler()
        profiler.start()
        profiler.start_function_call("my_func")
        profiler.end_function_call("my_func")
        profiler.stop()
        assert "my_func" in profiler.data.function_stats

    def test_call_count_is_1_for_single_call(self):
        profiler = Profiler()
        profiler.start()
        profiler.start_function_call("foo")
        profiler.end_function_call("foo")
        profiler.stop()
        assert profiler.data.function_stats["foo"].call_count == 1

    def test_call_count_increments_on_multiple_calls(self):
        profiler = Profiler()
        profiler.start()
        for _ in range(3):
            profiler.start_function_call("foo")
            profiler.end_function_call("foo")
        profiler.stop()
        assert profiler.data.function_stats["foo"].call_count == 3

    def test_function_total_time_is_positive(self):
        profiler = Profiler()
        profiler.start()
        profiler.start_function_call("foo")
        time.sleep(0.001)
        profiler.end_function_call("foo")
        profiler.stop()
        assert profiler.data.function_stats["foo"].total_time_ms > 0

    def test_uncalled_function_not_in_stats(self):
        profiler = Profiler()
        profiler.start()
        profiler.start_function_call("called")
        profiler.end_function_call("called")
        profiler.stop()
        assert "not_called" not in profiler.data.function_stats

    def test_nested_call_depth_tracked(self):
        profiler = Profiler()
        profiler.start()
        profiler.start_function_call("outer")
        profiler.start_function_call("inner")
        profiler.end_function_call("inner")
        profiler.end_function_call("outer")
        profiler.stop()
        assert profiler.data.function_stats["inner"].max_recursion_depth >= 1

    def test_multiple_different_functions_tracked(self):
        profiler = Profiler()
        profiler.start()
        profiler.start_function_call("foo")
        profiler.end_function_call("foo")
        profiler.start_function_call("bar")
        profiler.end_function_call("bar")
        profiler.stop()
        assert "foo" in profiler.data.function_stats
        assert "bar" in profiler.data.function_stats


# ===========================================================================
# 6. PROFILER CONTROLS
# ===========================================================================

class TestProfilerControls:

    def test_disable_stops_line_tracking(self):
        profiler = Profiler()
        profiler.start()
        profiler.disable()
        profiler.start_line(1, variable_count=0)
        profiler.end_line(1)
        profiler.stop()
        assert 1 not in profiler.data.line_stats

    def test_disable_stops_function_tracking(self):
        profiler = Profiler()
        profiler.start()
        profiler.disable()
        profiler.start_function_call("foo")
        profiler.end_function_call("foo")
        profiler.stop()
        assert "foo" not in profiler.data.function_stats

    def test_enable_resumes_tracking(self):
        profiler = Profiler()
        profiler.start()
        profiler.disable()
        profiler.enable()
        profiler.start_line(1, variable_count=0)
        profiler.end_line(1)
        profiler.stop()
        assert 1 in profiler.data.line_stats

    def test_reset_clears_line_stats(self):
        profiler = Profiler()
        profiler.start()
        profiler.start_line(1, variable_count=0)
        profiler.end_line(1)
        profiler.stop()
        profiler.reset()
        assert profiler.data.line_stats == {}

    def test_reset_clears_function_stats(self):
        profiler = Profiler()
        profiler.start()
        profiler.start_function_call("foo")
        profiler.end_function_call("foo")
        profiler.stop()
        profiler.reset()
        assert profiler.data.function_stats == {}

    def test_reset_clears_call_stack(self):
        profiler = Profiler()
        profiler.reset()
        assert profiler._function_call_stack == []


# ===========================================================================
# 7. HOTSPOT DETECTION
# ===========================================================================

class TestHotspotDetection:

    def _make_profiler_with_lines(self):
        profiler = Profiler()
        profiler.start()
        for _ in range(10):
            profiler.start_line(1, variable_count=0)
            time.sleep(0.001)
            profiler.end_line(1)
        for _ in range(5):
            profiler.start_line(2, variable_count=0)
            time.sleep(0.002)
            profiler.end_line(2)
        for _ in range(1):
            profiler.start_line(3, variable_count=0)
            time.sleep(0.001)
            profiler.end_line(3)
        profiler.stop()
        return profiler

    def test_get_hottest_lines_returns_list(self):
        profiler = self._make_profiler_with_lines()
        assert isinstance(profiler.get_hottest_lines(), list)

    def test_get_hottest_lines_sorted_by_time_descending(self):
        profiler = self._make_profiler_with_lines()
        result = profiler.get_hottest_lines()
        times = [s.total_time_ms for s in result]
        assert times == sorted(times, reverse=True)

    def test_get_hottest_lines_top_n_respected(self):
        profiler = self._make_profiler_with_lines()
        result = profiler.get_hottest_lines(top_n=2)
        assert len(result) <= 2

    def test_get_most_executed_lines_sorted_by_count_descending(self):
        profiler = self._make_profiler_with_lines()
        result = profiler.get_most_executed_lines()
        counts = [s.execution_count for s in result]
        assert counts == sorted(counts, reverse=True)

    def test_most_executed_line_is_line_1(self):
        profiler = self._make_profiler_with_lines()
        result = profiler.get_most_executed_lines(top_n=1)
        assert result[0].line_number == 1

    def test_get_most_executed_top_n_respected(self):
        profiler = self._make_profiler_with_lines()
        result = profiler.get_most_executed_lines(top_n=1)
        assert len(result) == 1


# ===========================================================================
# 8. PROFILING DATA STRUCTURE
# ===========================================================================

class TestProfilingData:

    def test_start_time_set_on_start(self):
        profiler = Profiler()
        profiler.start()
        assert profiler.data.start_time is not None

    def test_end_time_set_on_stop(self):
        profiler = Profiler()
        profiler.start()
        profiler.stop()
        assert profiler.data.end_time is not None

    def test_total_execution_time_positive_after_stop(self):
        profiler = Profiler()
        profiler.start()
        time.sleep(0.001)
        profiler.stop()
        assert profiler.data.total_execution_time_ms > 0

    def test_to_dict_returns_dict(self):
        profiler = Profiler()
        profiler.start()
        profiler.stop()
        assert isinstance(profiler.data.to_dict(), dict)

    def test_to_dict_has_required_keys(self):
        profiler = Profiler()
        profiler.start()
        profiler.stop()
        d = profiler.data.to_dict()
        for key in ["line_stats", "function_stats", "total_time_ms", "total_lines"]:
            assert key in d


# ===========================================================================
# 9. LINESTATS DATACLASS
# ===========================================================================

class TestLineStats:

    def test_initial_execution_count_zero(self):
        assert LineStats(line_number=1).execution_count == 0

    def test_initial_total_time_zero(self):
        assert LineStats(line_number=1).total_time_ms == 0.0

    def test_initial_avg_time_zero(self):
        assert LineStats(line_number=1).avg_time_ms == 0.0

    def test_update_time_increments_count(self):
        stats = LineStats(line_number=1)
        stats.update_time(5.0)
        assert stats.execution_count == 1

    def test_update_time_accumulates_total(self):
        stats = LineStats(line_number=1)
        stats.update_time(3.0)
        stats.update_time(7.0)
        assert stats.total_time_ms == 10.0

    def test_update_time_calculates_avg(self):
        stats = LineStats(line_number=1)
        stats.update_time(4.0)
        stats.update_time(6.0)
        assert stats.avg_time_ms == 5.0

    def test_to_dict_contains_all_keys(self):
        d = LineStats(line_number=5).to_dict()
        for key in ["line", "count", "total_time", "avg_time", "memory_vars"]:
            assert key in d

    def test_to_dict_line_number_correct(self):
        assert LineStats(line_number=42).to_dict()["line"] == 42


# ===========================================================================
# 10. FUNCTIONSTATS DATACLASS
# ===========================================================================

class TestFunctionStats:

    def test_initial_call_count_zero(self):
        assert FunctionStats(name="foo").call_count == 0

    def test_record_call_increments_count(self):
        stats = FunctionStats(name="foo")
        stats.record_call(5.0)
        assert stats.call_count == 1

    def test_record_call_accumulates_time(self):
        stats = FunctionStats(name="foo")
        stats.record_call(3.0)
        stats.record_call(7.0)
        assert stats.total_time_ms == 10.0

    def test_record_call_calculates_avg(self):
        stats = FunctionStats(name="foo")
        stats.record_call(4.0)
        stats.record_call(6.0)
        assert stats.avg_time_ms == 5.0

    def test_record_call_tracks_max_depth(self):
        stats = FunctionStats(name="foo")
        stats.record_call(1.0, depth=3)
        stats.record_call(1.0, depth=1)
        assert stats.max_recursion_depth == 3

    def test_to_dict_contains_all_keys(self):
        d = FunctionStats(name="foo").to_dict()
        for key in ["name", "calls", "total_time", "avg_time", "max_depth"]:
            assert key in d


# ===========================================================================
# 11. INTEGRATION — PROFILER VIA EXECUTE()
# ===========================================================================

class TestProfilerIntegration:

    def test_profiling_enabled_by_default(self):
        result = execute("x = 1")
        assert result.profiling is not None

    def test_profiling_disabled_returns_none(self):
        result = execute("x = 1", enable_profiling=False)
        assert result.profiling is None

    def test_profiling_dict_has_line_stats(self):
        result = execute("x = 1")
        assert "line_stats" in result.profiling

    def test_profiling_dict_has_function_stats(self):
        result = execute("x = 1")
        assert "function_stats" in result.profiling

    def test_profiling_dict_has_total_time(self):
        result = execute("x = 1")
        assert "total_time_ms" in result.profiling

    def test_loop_line_has_high_execution_count(self):
        source = "total = 0\nfor i in range(10):\n    total += i"
        result = execute(source)
        counts = [v["count"] for v in result.profiling["line_stats"].values()]
        assert max(counts) >= 10

    def test_function_tracked_after_call(self):
        source = "def add(x, y):\n    return x + y\nadd(1, 2)"
        result = execute(source)
        assert "add" in result.profiling["function_stats"]

    def test_function_call_count_correct(self):
        source = "def foo():\n    pass\nfoo()\nfoo()\nfoo()"
        result = execute(source)
        assert result.profiling["function_stats"]["foo"]["calls"] == 3

    def test_total_time_ms_non_negative(self):
        result = execute("x = 1")
        assert result.profiling["total_time_ms"] >= 0

    def test_profiling_none_on_lexer_error(self):
        result = execute('print("unterminated')
        assert result.profiling is None

    def test_profiling_overhead_is_reasonable(self):
        """
        NFR Test: Profiling should not cause extreme slowdown.
        Overhead should be well under 10x for a simple loop.
        """
        source = "total = 0\nfor i in range(100):\n    total += i"

        start = time.perf_counter()
        execute(source, enable_profiling=False)
        time_without = time.perf_counter() - start

        start = time.perf_counter()
        execute(source, enable_profiling=True)
        time_with = time.perf_counter() - start

        # Very generous threshold — just checking it doesn't hang
        assert time_with < time_without * 10 + 1.0