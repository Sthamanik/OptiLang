from __future__ import annotations

import textwrap

from optilang.executor import execute
from optilang.lexer import tokenize
from optilang.ml.src.extractor import extract
from optilang.ml.src.storage import EXECUTION_FIELDNAMES
from optilang.models import ExecutionResult, OptimizationReport, Suggestion
from optilang.parser import parse
from optilang.scoring import ScoreReport, calculate_score


def _score(result: ExecutionResult, report: OptimizationReport, source: str) -> ScoreReport:
    return calculate_score(
        profiling_data=result.profiling.to_dict() if result.profiling else None,
        optimizer_report=report,
        source_lines=len(source.splitlines()),
        errors=result.errors,
    )


def test_extract_emits_expanded_storage_compatible_rows() -> None:
    source = textwrap.dedent(
        """\
        def fact(n):
            if n <= 1:
                return 1
            return fact(n - 1)
        try:
            data = [1, 2, 3]
            table = {"a": 1}
        except:
            pass
        total = 0
        for i in range(3):
            total += i
        print(fact(3))
        """
    )
    result = execute(source)
    assert result.errors == []
    ast = parse(tokenize(source))
    report = OptimizationReport(
        suggestions=[
            Suggestion(
                line=7,
                pattern="unused_vars",
                severity="low",
                description="unused table",
                suggestion="remove table",
                impact_score=3.0,
            ),
            Suggestion(
                line=12,
                pattern="hot_loop",
                severity="high",
                description="hot loop",
                suggestion="reduce work",
                impact_score=18.0,
            ),
        ]
    )

    rows = extract(
        source=source,
        result=result,
        report=report,
        score=_score(result, report, source),
        ast=ast,
    )

    assert len(rows) == 2
    for row in rows:
        assert set(EXECUTION_FIELDNAMES).issubset(row.keys())
        assert row["line_number"] in {7, 12}
        assert row["score"] >= 0
        assert "execution_id" not in row
        assert "program_id" not in row
        assert "variant" not in row
        assert "source_path" not in row
        assert row["token_count"] > 0
        assert row["ast_node_count"] > 0
        assert row["function_count"] == 1
        assert row["loop_count"] == 1
        assert row["if_count"] == 1
        assert row["try_count"] == 1
        assert row["assignment_count"] >= 3
        assert row["call_count"] >= 2
        assert row["binary_op_count"] >= 2
        assert row["uses_lists"] is True
        assert row["uses_dicts"] is True
        assert row["uses_recursion"] is True
        assert row["uses_exceptions"] is True
        assert row["total_suggestions"] == 2
        assert row["high_severity_count"] == 1
        assert row["low_severity_count"] == 1
        assert row["static_suggestion_count"] == 1
        assert row["dynamic_suggestion_count"] == 1
        assert row["count_unused_vars"] == 1
        assert row["count_hot_loop"] == 1

    table_row = next(row for row in rows if row["line_number"] == 7)
    assert table_row["pattern"] == "unused_vars"
    assert table_row["severity_ordinal"] == 1
    assert table_row["detector_family"] == "static"
    assert table_row["score_dimension"] == "maintainability"
    assert table_row["inside_try"] is True
    assert table_row["node_type_at_line"] in {"AssignmentNode", "DictNode"}

    loop_row = next(row for row in rows if row["line_number"] == 12)
    assert loop_row["pattern"] == "hot_loop"
    assert loop_row["severity_ordinal"] == 3
    assert loop_row["detector_family"] == "dynamic"
    assert loop_row["score_dimension"] == "efficiency"
    assert loop_row["inside_loop"] is True
    assert loop_row["is_inside_loop"] is True
    assert loop_row["loop_depth"] == 1
    assert loop_row["execution_count_at_line"] == 3
    assert loop_row["line_execution_rank"] > 0
    assert loop_row["line_time_rank"] > 0


def test_extract_defaults_profiling_fields_when_line_stat_missing() -> None:
    source = "x = 1\nprint(x)"
    result = execute(source)
    assert result.errors == []
    ast = parse(tokenize(source))
    report = OptimizationReport(
        suggestions=[
            Suggestion(
                line=99,
                pattern="constant_folding",
                severity="low",
                description="fold",
                suggestion="replace",
                impact_score=2.0,
            )
        ]
    )

    rows = extract(
        source=source,
        result=result,
        report=report,
        score=_score(result, report, source),
        ast=ast,
    )

    assert len(rows) == 1
    row = rows[0]
    assert row["execution_count_at_line"] == 0
    assert row["avg_time_ms_at_line"] == 0.0
    assert row["total_time_ms_at_line"] == 0.0
    assert row["min_time_ms_at_line"] == 0.0
    assert row["max_time_ms_at_line"] == 0.0
    assert row["memory_vars_at_line"] == 0
    assert row["memory_bytes_at_line"] == 0
    assert row["line_execution_rank"] == 0
    assert row["line_time_rank"] == 0


def test_extract_skips_errored_executions() -> None:
    result = ExecutionResult(output="", errors=["boom"])
    report = OptimizationReport(
        suggestions=[
            Suggestion(
                line=1,
                pattern="unused_vars",
                severity="low",
                description="unused",
                suggestion="remove",
                impact_score=3.0,
            )
        ]
    )
    score = calculate_score(
        profiling_data=None,
        optimizer_report=report,
        source_lines=1,
        errors=result.errors,
    )

    assert (
        extract(
            source="x = 1",
            result=result,
            report=report,
            score=score,
            ast=None,
        )
        == []
    )
