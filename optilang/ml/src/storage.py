"""CSV storage helpers for the OptiLang ML dataset."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


ML_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = ML_DIR / "data"
EXECUTIONS_CSV = DATA_DIR / "executions.csv"


EXECUTION_FIELDNAMES: List[str] = [
    "execution_id",
    "program_id",
    "variant",
    "family",
    "strategy",
    "expected_patterns",
    "pattern",
    "severity",
    "impact_score",
    "line_number",
    "loop_depth",
    "is_inside_loop",
    "co_occurring_patterns",
    "source_lines",
    "complexity_class",
    "error_count",
    "execution_time_ms",
    "total_suggestions",
    "score",
    "grade",
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