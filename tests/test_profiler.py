"""
Tests for optilang/profiler.py
"""

import time
import pytest
from optilang.profiler import Profiler, LineStats, FunctionStats


def make_profiler():
    p = Profiler()
    p.start()
    return p


def run_line(p, line, var_count=0, sleep=0.0):
    p.start_line(line, var_count)
    if sleep:
        time.sleep(sleep)
    p.end_line(line)


def run_func(p, name, sleep=0.0, depth=0):
    p.start_function_call(name)
    if sleep:
        time.sleep(sleep)
    p.end_function_call(name)


# ── Initialization ────────────────────────────────────────────────────────────


def test_initial_line_stats_empty():
    assert Profiler().data.line_stats == {}


def test_initial_function_stats_empty():
    assert Profiler().data.function_stats == {}


def test_initial_enabled():
    assert Profiler()._enabled is True


def test_initial_call_stack_empty():
    assert Profiler()._function_call_stack == []


# ── Start / Stop ──────────────────────────────────────────────────────────────


def test_start_sets_start_time():
    p = Profiler()
    p.start()
    assert p.data.start_time is not None


def test_stop_sets_end_time():
    p = make_profiler()
    p.stop()
    assert p.data.end_time is not None


def test_stop_total_time_positive():
    p = make_profiler()
    time.sleep(0.01)
    p.stop()
    assert p.data.total_execution_time_ms > 0


def test_stop_total_lines_executed():
    p = make_profiler()
    run_line(p, 1)
    run_line(p, 1)
    run_line(p, 2)
    p.stop()
    assert p.data.total_lines_executed == 3


# ── Line Profiling ────────────────────────────────────────────────────────────


def test_line_stats_created():
    p = make_profiler()
    run_line(p, 1)
    assert 1 in p.data.line_stats


def test_line_number_stored():
    p = make_profiler()
    run_line(p, 5)
    assert p.data.line_stats[5].line_number == 5


def test_execution_count_increments():
    p = make_profiler()
    run_line(p, 1)
    run_line(p, 1)
    assert p.data.line_stats[1].execution_count == 2


def test_total_time_positive():
    p = make_profiler()
    run_line(p, 1, sleep=0.001)
    assert p.data.line_stats[1].total_time_ms > 0


def test_avg_time_after_first_call():
    p = make_profiler()
    run_line(p, 1, sleep=0.001)
    s = p.data.line_stats[1]
    assert s.avg_time_ms == pytest.approx(s.total_time_ms, rel=0.1)


def test_avg_time_recalculates():
    p = make_profiler()
    run_line(p, 1, sleep=0.001)
    run_line(p, 1, sleep=0.001)
    s = p.data.line_stats[1]
    assert s.avg_time_ms == pytest.approx(s.total_time_ms / 2, rel=0.2)


def test_memory_vars_stored():
    p = make_profiler()
    run_line(p, 1, var_count=4)
    assert p.data.line_stats[1].memory_vars == 4


def test_end_line_without_start_no_crash():
    p = make_profiler()
    p.end_line(99)  # should not raise


def test_loop_execution_count():
    p = make_profiler()
    for _ in range(5):
        run_line(p, 10)
    assert p.data.line_stats[10].execution_count == 5


# ── Function Profiling ────────────────────────────────────────────────────────


def test_function_stats_created():
    p = make_profiler()
    run_func(p, "foo")
    assert "foo" in p.data.function_stats


def test_function_call_count():
    p = make_profiler()
    run_func(p, "foo")
    run_func(p, "foo")
    assert p.data.function_stats["foo"].call_count == 2


def test_function_total_time_positive():
    p = make_profiler()
    run_func(p, "foo", sleep=0.001)
    assert p.data.function_stats["foo"].total_time_ms > 0


def test_function_avg_time():
    p = make_profiler()
    run_func(p, "foo", sleep=0.001)
    run_func(p, "foo", sleep=0.001)
    s = p.data.function_stats["foo"]
    assert s.avg_time_ms == pytest.approx(s.total_time_ms / 2, rel=0.3)


def test_function_max_recursion_depth():
    p = make_profiler()
    p.start_function_call("outer")
    p.start_function_call("inner")
    p.end_function_call("inner")
    p.end_function_call("outer")
    assert p.data.function_stats["inner"].max_recursion_depth >= 1


def test_mismatched_end_function_no_crash():
    p = make_profiler()
    p.start_function_call("foo")
    p.end_function_call("bar")  # mismatch — should not raise


# ── LineStats Methods ─────────────────────────────────────────────────────────


def test_linestats_update_time():
    s = LineStats(line_number=1)
    s.update_time(5.0)
    assert s.execution_count == 1
    assert s.total_time_ms == pytest.approx(5.0)
    assert s.avg_time_ms == pytest.approx(5.0)


def test_linestats_to_dict_keys():
    s = LineStats(line_number=3)
    s.update_time(2.123456)
    d = s.to_dict()
    assert set(d.keys()) == {"line", "count", "total_time", "avg_time", "memory_vars"}


def test_linestats_to_dict_rounding():
    s = LineStats(line_number=1)
    s.update_time(1.123456789)
    d = s.to_dict()
    assert d["total_time"] == round(s.total_time_ms, 3)


# ── FunctionStats Methods ─────────────────────────────────────────────────────


def test_functionstats_record_call():
    s = FunctionStats(name="f")
    s.record_call(3.0, depth=0)
    assert s.call_count == 1
    assert s.total_time_ms == pytest.approx(3.0)


def test_functionstats_max_depth_not_decrease():
    s = FunctionStats(name="f")
    s.record_call(1.0, depth=5)
    s.record_call(1.0, depth=2)
    assert s.max_recursion_depth == 5


def test_functionstats_to_dict_keys():
    s = FunctionStats(name="f")
    s.record_call(1.0, 0)
    d = s.to_dict()
    assert set(d.keys()) == {"name", "calls", "total_time", "avg_time", "max_depth"}


# ── ProfilingData.to_dict ─────────────────────────────────────────────────────


def test_profilingdata_to_dict_keys():
    p = make_profiler()
    run_line(p, 1)
    p.stop()
    d = p.data.to_dict()
    assert set(d.keys()) == {
        "line_stats",
        "function_stats",
        "total_time_ms",
        "total_lines",
        "lines_profiled",
    }


def test_profilingdata_lines_profiled():
    p = make_profiler()
    run_line(p, 1)
    run_line(p, 2)
    run_line(p, 1)
    p.stop()
    assert p.data.to_dict()["lines_profiled"] == 2


def test_profilingdata_line_stats_sorted():
    p = make_profiler()
    run_line(p, 5)
    run_line(p, 2)
    run_line(p, 8)
    p.stop()
    keys = list(p.data.to_dict()["line_stats"].keys())
    assert keys == sorted(keys)


# ── get_hottest_lines / get_most_executed_lines ───────────────────────────────


def test_get_hottest_lines_sorted():
    p = make_profiler()
    run_line(p, 1, sleep=0.002)
    run_line(p, 2, sleep=0.001)
    run_line(p, 3, sleep=0.003)
    result = p.get_hottest_lines(3)
    times = [s.total_time_ms for s in result]
    assert times == sorted(times, reverse=True)


def test_get_hottest_lines_limit():
    p = make_profiler()
    for i in range(5):
        run_line(p, i)
    result = p.get_hottest_lines(3)
    assert len(result) <= 3


def test_get_hottest_lines_all_if_fewer():
    p = make_profiler()
    run_line(p, 1)
    run_line(p, 2)
    result = p.get_hottest_lines(100)
    assert len(result) == 2


def test_get_most_executed_lines_sorted():
    p = make_profiler()
    for _ in range(3):
        run_line(p, 1)
    for _ in range(7):
        run_line(p, 2)
    run_line(p, 3)
    result = p.get_most_executed_lines(3)
    counts = [s.execution_count for s in result]
    assert counts == sorted(counts, reverse=True)


# ── Enable / Disable / Reset ──────────────────────────────────────────────────


def test_disable_stops_recording():
    p = make_profiler()
    p.disable()
    run_line(p, 1)
    assert p.data.line_stats == {}


def test_enable_resumes_recording():
    p = make_profiler()
    p.disable()
    p.enable()
    run_line(p, 1)
    assert 1 in p.data.line_stats


def test_reset_clears_all():
    p = make_profiler()
    run_line(p, 1)
    run_func(p, "f")
    p.reset()
    assert p.data.line_stats == {}
    assert p.data.function_stats == {}
    assert p._function_call_stack == []
    assert p._current_line_start is None


# ── Integration with execute() ────────────────────────────────────────────────


def test_execute_profiling_not_none():
    from optilang import execute

    result = execute("x = 1")
    assert result.profiling is not None


def test_execute_profiling_disabled():
    from optilang import execute

    result = execute("x = 1", enable_profiling=False)
    # When disabled, executor returns None for profiling
    assert result.profiling is None


def test_execute_loop_execution_count():
    from optilang import execute

    result = execute("for i in range(5):\n    x = i")
    # result.profiling is already a dict returned by ProfilingData.to_dict()
    line_counts = [s["count"] for s in result.profiling["line_stats"].values()]
    assert any(count == 5 for count in line_counts)


def test_execute_function_call_count():
    from optilang import execute

    src = "def f():\n    pass\nf()\nf()\nf()"
    result = execute(src)
    # result.profiling is a dict, so access with string keys
    assert result.profiling["function_stats"]["f"]["calls"] == 3
