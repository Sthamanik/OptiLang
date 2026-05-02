"""CSV storage helpers for the OptiLang ML dataset."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List


ML_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ML_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
EXECUTIONS_CSV = DATA_DIR / "executions.csv"


EXECUTION_FIELDNAMES: List[str] = [
    # --- Identifiers / ground truth ---
    "execution_id",
    "family",
    "strategy",
    # --- Core suggestion features ---
    "pattern",
    "severity",
    "impact_score",
    # --- Structural (AST) ---
    "loop_depth",
    "is_inside_loop",
    "relative_line_position",
    "co_occurring_patterns",
    # --- Dynamic — line level ---
    "execution_count_at_line",
    "avg_time_ms_at_line",
    "total_time_ms_at_line",
    "line_dominance",
    # --- Dynamic — function level ---
    "function_call_count",
    "max_recursion_depth",
    # --- Program-level context ---
    "source_lines",
    "complexity_class",
    "complexity_ordinal",
    "execution_time_ms",
    "peak_memory_bytes",
    "total_suggestions",
]


def ensure_data_dirs() -> None:
    """Create the ML data directory if it does not already exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def append_executions(rows: Iterable[Dict[str, object]]) -> int:
    """Append suggestion-level rows to executions.csv.

    Writes the header only when the file is new or empty.
    Returns the number of rows written.
    """
    ensure_data_dirs()
    write_header = not EXECUTIONS_CSV.exists() or EXECUTIONS_CSV.stat().st_size == 0
    count = 0
    with EXECUTIONS_CSV.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=EXECUTION_FIELDNAMES,
            extrasaction="ignore",
        )
        if write_header:
            writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def read_executions() -> List[Dict[str, str]]:
    """Read all rows from executions.csv.

    Returns an empty list if the file does not exist yet.
    """
    if not EXECUTIONS_CSV.exists():
        return []
    with EXECUTIONS_CSV.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def reset_executions() -> None:
    """Delete executions.csv — use when you want a clean dataset rebuild."""
    if EXECUTIONS_CSV.exists():
        EXECUTIONS_CSV.unlink()