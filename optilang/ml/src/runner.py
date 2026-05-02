"""Batch runner for OptiLang ML fixtures."""

from __future__ import annotations

import argparse
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from optilang.executor import execute
from optilang.lexer import tokenize
from optilang.models import OptimizationReport
from optilang.optimizer import analyze
from optilang.parser import parse
from optilang.scoring import ScoreReport, calculate_score

from .extractor import extract
from .storage import (
    EXECUTION_FIELDNAMES,
    EXECUTIONS_CSV,
    RAW_DIR,
    append_executions,
)


# ---------------------------------------------------------------------------
# Family-level metadata tables
# ---------------------------------------------------------------------------

# The dominant fix strategy for each program family.
# Used as a coarse label for the classify stage.
_FAMILY_STRATEGY: Dict[str, str] = {
    "simple":        "fold_and_prune",       # constant folding + unused var removal
    "loops":         "restructure_loop",     # hoist invariants, flatten nested loops
    "recursive":     "no_action",            # recursive programs rarely get loop opts
    "mixed":         "mixed",                # multiple strategies apply
    "pathological":  "aggressive_prune",     # dead code + unused vars at scale
}

# Pipe-separated patterns expected to appear in each family.
# Drives the expected_patterns column — ground truth for EDA / eval.
_FAMILY_EXPECTED_PATTERNS: Dict[str, str] = {
    "simple":        "constant_folding|unused_vars",
    "loops":         "nested_loops|loop_invariant|unused_vars",
    "recursive":     "unused_vars",
    "mixed":         "constant_folding|unused_vars|nested_loops|repeated_computation",
    "pathological":  "dead_code|unused_vars|repeated_computation|hot_loop",
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _source_line_count(source: str) -> int:
    return len(source.splitlines()) if source else 0


def _collect_sources(raw_dir: Path) -> List[Path]:
    """Recursively collect all .py files under raw_dir."""
    return sorted(raw_dir.rglob("*.py"))


def _build_manifest_row(source_path: Path) -> Dict[str, str]:
    """Derive manifest metadata from file path and name."""
    stem = source_path.stem                       # e.g. bubble_sort_v1
    parts = stem.rsplit("_v", 1)
    program_id = parts[0]                         # e.g. bubble_sort
    variant = f"v{parts[1]}" if len(parts) == 2 else "v1"
    family = source_path.parent.name              # simple / loops / recursive / mixed / pathological

    return {
        "program_id":        program_id,
        "variant":           variant,
        "family":            family,
        "strategy":          _FAMILY_STRATEGY.get(family, "unknown"),
        "patterns":          _FAMILY_EXPECTED_PATTERNS.get(family, ""),
        "pathological":      str(family == "pathological"),
        "source_path":       str(source_path),
    }


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def run_one(
    source_path: Path,
    timeout_seconds: float = 5.0,
) -> List[Dict[str, object]]:
    """Execute one source file and return flat suggestion rows."""

    source = source_path.read_text(encoding="utf-8")
    execution_id = str(uuid.uuid4())

    result = execute(source, timeout_seconds=timeout_seconds)
    ast = None
    report: Optional[OptimizationReport] = None

    if not result.errors:
        try:
            ast = parse(tokenize(source))
            report = analyze(ast, result.profiling, result.symbol_table)
        except Exception as exc:
            result.errors.append(str(exc))

    score: ScoreReport = calculate_score(
        profiling_data=result.profiling.to_dict() if result.profiling else None,
        optimizer_report=report,
        source_lines=_source_line_count(source),
        errors=result.errors,
    )

    manifest_row = _build_manifest_row(source_path)

    return extract(
        source=source,
        result=result,
        report=report,
        score=score,
        manifest_row=manifest_row,
        execution_id=execution_id,
        ast=ast,
    )


def run_all(
    raw_dir: Path = RAW_DIR,
    limit: Optional[int] = None,
    skip_pathological: bool = False,
    timeout_seconds: float = 5.0,
) -> List[Dict[str, object]]:
    """Run all source files and return flat suggestion rows."""

    sources = _collect_sources(raw_dir)

    if skip_pathological:
        sources = [s for s in sources if "pathological" not in s.parts]

    if limit is not None:
        sources = sources[:limit]

    all_rows: List[Dict[str, object]] = []
    failed: List[str] = []

    for i, source_path in enumerate(sources, 1):
        try:
            rows = run_one(source_path, timeout_seconds=timeout_seconds)
            all_rows.extend(rows)
            print(f"[{i}/{len(sources)}] OK   {source_path.name}  →  {len(rows)} rows")
        except Exception as exc:
            failed.append(source_path.name)
            print(f"[{i}/{len(sources)}] FAIL {source_path.name}  →  {exc}")

    if failed:
        print(f"\nFailed ({len(failed)}): {', '.join(failed)}")

    return all_rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run OptiLang ML fixtures.")
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=RAW_DIR,
        help="Directory containing raw .py program files.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Run only the first N programs.",
    )
    parser.add_argument(
        "--skip-pathological",
        action="store_true",
        help="Skip programs inside the pathological/ subdirectory.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Per-program execution timeout in seconds.",
    )
    args = parser.parse_args(argv)

    rows = run_all(
        raw_dir=args.raw_dir,
        limit=args.limit,
        skip_pathological=args.skip_pathological,
        timeout_seconds=args.timeout,
    )

    count = append_executions(rows)
    print(f"\nWrote {count} suggestion rows → {EXECUTIONS_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())